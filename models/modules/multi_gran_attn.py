""" multi head multi granularity attn """

import math
import torch
from torch import nn

from .multi_head_attn import gen_rel_dist_matrix


class MultiHeadMultiGranAttention(nn.Module):
    """ Multi-Head Self Attention """
    def __init__(self, dim, num_heads, dropout=0.1, rel_range=0, use_negative=True):
        super(MultiHeadMultiGranAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, 3 * dim)
        self.output = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.rel_range = rel_range
        self.use_negative = use_negative

        if rel_range > 0:
            stok_vocab_size = rel_range * 2 + 1 if self.use_negative else rel_range + 1
            stm_vocab_size = 16 * 2 + 1 
            self.rel_embeddings_k = nn.Embedding(stok_vocab_size, self.head_dim)
            self.rel_embeddings_v = nn.Embedding(stok_vocab_size, self.head_dim)
            self.rel_stm_emb_k = nn.Embedding(stm_vocab_size, self.head_dim)
            self.rel_stm_emb_v = nn.Embedding(stm_vocab_size, self.head_dim)

    def forward(self, stok_seq, stm_seq, stok_pad_mask, stm_pad_mask):
        """ Compute the context vector and the attention vectors.
        Args:
            stok_seq (FloatTensor): shape as `[batch, stok, dim]`
            stm_seq (FloatTensor): shape as `[batch, stm, dim]`
            stok_pad_mask (BoolTensor): 1/0 -> 0/~0  attn value, shape as  `[batch, 1, seq_len]`
            stm_pad_mask (BoolTensor): ...
        Returns:
           (FloatTensor, FloatTensor):
           - context, shape as `[batch, seq_len, dim]`
           - attn distribution, shape as `[batch, seq_len, seq_len]`
        """
        # merge data and mask
        x = torch.cat((stok_seq, stm_seq), dim=1)
        mask = torch.cat((stok_pad_mask, stm_pad_mask), dim=-1)

        batch_size, seq_len = x.shape[:2]
        stok_len = stok_seq.size(1)
        rel_pe_keys, rel_pe_values = None, None

        # project `x` to `key`, `value`, and `query`
        # all shapes as `[batch, head, seq_len, head_dim]`
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        # get relative distance embeddings: `rel_keys`, `rel_values`
        # all shapes as `[seq_len, seq_len, head_dim]`
        if self.rel_range > 0:
            # stok rel matrix
            rel_dist_matrix = gen_rel_dist_matrix(stok_len, self.rel_range, self.use_negative)
            rel_dist_matrix = rel_dist_matrix.to(x.device)  # `[stok_len, stok_len]`
            rel_stok_pe_keys = self.rel_embeddings_k(rel_dist_matrix)
            rel_stok_pe_values = self.rel_embeddings_v(rel_dist_matrix)

            # stm rel matrix
            stm_len = seq_len - stok_len
            rel_dist_matrix = gen_rel_dist_matrix(stm_len, 16, self.use_negative)
            rel_dist_matrix = rel_dist_matrix.to(x.device)  # `[stm_len, stm_len]`
            rel_stm_pe_keys = self.rel_stm_emb_k(rel_dist_matrix)
            rel_stm_pe_values = self.rel_stm_emb_v(rel_dist_matrix)

            # merge rel pos embedding
            rel_pe_keys = torch.zeros(seq_len, seq_len, self.head_dim, device=x.device)
            rel_pe_values = rel_pe_keys.clone()
            rel_pe_keys[:stok_len, :stok_len] = rel_stok_pe_keys
            rel_pe_keys[stok_len:, stok_len:] = rel_stm_pe_keys
            rel_pe_values[:stok_len, :stok_len] = rel_stok_pe_values
            rel_pe_values[stok_len:, stok_len:] = rel_stm_pe_values

        # scaled dot product attn, shape as `[batch, head, seq_len, seq_len]`
        # relative attn: <https://arxiv.org/abs/1803.02155>
        # attn_score_{ij} = query_i * (key_j + rel_pe_keys_{ij})^T / scale
        # context_i = \sum_j attn_{ij} (value_j + rel_pe_values_{ij})
        query = query / math.sqrt(self.head_dim)
        attn_score = query @ key.transpose(-1, -2)  # `[batch, head, seq_len, seq_len]`
        if self.rel_range > 0:  # matrix of query_i * rel_keys_{i,j}^T
            query_t = query.permute(2, 0, 1, 3).reshape(seq_len, -1, self.head_dim)
            rel_attn_bias = query_t @ rel_pe_keys.transpose(-1, -2)  # `[seq_len, -1, seq_len]`
            rel_attn_bias = rel_attn_bias.reshape(seq_len, batch_size, self.num_heads, -1)
            attn_score = attn_score + rel_attn_bias.permute(1, 2, 0, 3)

        # mask
        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            attn_score = attn_score.masked_fill(mask, -1e18)
        attn_score[:, :, stok_len:, :stok_len] = -1e18  # mask out stm2stok

        # attn distribution and get the context of attn-weighted over value
        attn = self.dropout(self.softmax(attn_score))  # `[batch, head, seq_len, seq_len]`
        context = attn @ value  # `[batch, head, seq_len, head_dim]`
        if self.rel_range > 0:  # matrix of \sum_j attn_{ij} * rel_pe_values_{ij}
            attn_t = attn.permute(2, 0, 1, 3).reshape(seq_len, -1, seq_len)  # -1 => batch * head
            rel_context_bias = attn_t @ rel_pe_values
            rel_context_bias = rel_context_bias.reshape(seq_len, batch_size, -1, self.head_dim)
            context = context + rel_context_bias.permute(1, 2, 0, 3)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.dim)

        # output projection
        final_output = self.output(context)
        attn_per_head = [attn.squeeze(1) for attn in attn.chunk(self.num_heads, dim=1)]

        return final_output, attn_per_head

    def update_dropout(self, dropout):
        self.dropout.p = dropout

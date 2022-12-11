""" hierarchical attention transformer considering:
- `stok_seq` (textual information)
- `tok_seq` (grammar information)
- `stm_seq` (logical information)
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

from utils import PAD
from .build import MODELS
from .base_model import BaseModel
from .modules import CopyAttention, merge_copy_dist, gen_rel_dist_matrix
from .transformer import SrcEmbedding, TgtEmbedding
from .transformer import PositionwiseFeedForward
from .transformer import TransformerDecoder


class MultiHeadMultiGran3Attention(nn.Module):
    """ Multi-Head Self Attention """
    def __init__(self, dim, num_heads, dropout=0.1, rel_range=0, use_negative=True):
        super(MultiHeadMultiGran3Attention, self).__init__()
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

            # stok relative pos embedding
            self.rel_stok_emb_k = nn.Embedding(stok_vocab_size, self.head_dim)
            self.rel_stok_emb_v = nn.Embedding(stok_vocab_size, self.head_dim)

            # tok relative pos embedding
            self.rel_tok_emb_k = nn.Embedding(stok_vocab_size, self.head_dim)
            self.rel_tok_emb_v = nn.Embedding(stok_vocab_size, self.head_dim)

            # stm relative pos embedding
            self.rel_stm_emb_k = nn.Embedding(stm_vocab_size, self.head_dim)
            self.rel_stm_emb_v = nn.Embedding(stm_vocab_size, self.head_dim)

    def forward(self, stok_seq, tok_seq, stm_seq, stok_pad_mask, tok_pad_mask, stm_pad_mask):
        """ Compute the context vector and the attention vectors.
        Args:
            stok_seq (FloatTensor): shape as `[batch, stok, dim]`
            tok_seq (FloatTensor): shape as `[batch, tok, dim]`
            stm_seq (FloatTensor): shape as `[batch, stm, dim]`
            stok_pad_mask (BoolTensor): 1/0 -> 0/~0  attn value, shape as  `[batch, 1, seq_len]`
            tok_pad_mask (BoolTensor): ...
            stm_pad_mask (BoolTensor): ...
        Returns:
           (FloatTensor, FloatTensor):
           - context, shape as `[batch, seq_len, dim]`
        """
        # merge data and mask
        x = torch.cat((stok_seq, tok_seq, stm_seq), dim=1)
        mask = torch.cat((stok_pad_mask, tok_pad_mask, stm_pad_mask), dim=-1)

        batch_size, seq_len = x.shape[:2]
        stok_len, tok_len, stm_len = stok_seq.size(1), tok_seq.size(1), stm_seq.size(1)
        rel_pe_keys, rel_pe_values = None, None

        # project `x` to `key`, `value`, and `query`
        # all shapes as `[batch, head, seq_len, head_dim]`
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        # get relative distance embeddings: `rel_keys`, `rel_values`
        # all shapes as `[seq_len, seq_len, head_dim]`
        max_len = max(stok_len, tok_len, stm_len)
        if self.rel_range > 0:
            # rel matrix
            rel_dist_matrix = gen_rel_dist_matrix(max_len, self.rel_range, self.use_negative)
            rel_dist_matrix = rel_dist_matrix.to(x.device)  # `[stok_len, stok_len]`

            # stok rel matrix
            rel_stok = rel_dist_matrix[:stok_len, :stok_len].contiguous()
            rel_stok_pe_keys = self.rel_stok_emb_k(rel_stok)
            rel_stok_pe_values = self.rel_stok_emb_v(rel_stok)

            # tok rel matrix
            rel_tok = rel_dist_matrix[:tok_len, :tok_len].contiguous()
            rel_tok_pe_keys = self.rel_tok_emb_k(rel_tok)
            rel_tok_pe_values = self.rel_tok_emb_v(rel_tok)

            # stm rel matrix
            rel_stm = rel_dist_matrix[:stm_len, :stm_len].contiguous()
            rel_stm = torch.clamp(rel_stm, min=-16, max=16)
            rel_stm_pe_keys = self.rel_stm_emb_k(rel_stm)
            rel_stm_pe_values = self.rel_stm_emb_v(rel_stm)

            # merge rel pos embedding
            stok_tok_len = stok_len + tok_len
            rel_pe_keys = torch.zeros(seq_len, seq_len, self.head_dim, device=x.device)
            rel_pe_values = rel_pe_keys.clone()
            # merge keys
            rel_pe_keys[:stok_len, :stok_len] = rel_stok_pe_keys
            rel_pe_keys[stok_len:stok_tok_len, stok_len:stok_tok_len] = rel_tok_pe_keys
            rel_pe_keys[stok_tok_len:, stok_tok_len:] = rel_stm_pe_keys
            # merge values
            rel_pe_values[:stok_len, :stok_len] = rel_stok_pe_values
            rel_pe_values[stok_len:stok_tok_len, stok_len:stok_tok_len] = rel_tok_pe_values
            rel_pe_values[stok_tok_len:, stok_tok_len:] = rel_stm_pe_values

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
        # attn_score[:, :, stok_len:, :stok_len] = -1e18  # mask out tok2stok
        # attn_score[:, :, stok_tok_len:, stok_len:stok_tok_len] = -1e18  # mask out tok2stok

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

        return final_output


class PAEncoderLayer(nn.Module):
    """ Encoder layer of transformer """
    def __init__(self, dim, num_heads, d_ff, dropout, rel_range=0, use_negative=True):
        super(PAEncoderLayer, self).__init__()
        self.use_negative = use_negative

        self.attn = MultiHeadMultiGran3Attention(dim, num_heads, dropout, rel_range)
        self.stok_attn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.tok_attn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.stm_attn_norm = nn.LayerNorm(dim, eps=1e-6)

        self.ffn = PositionwiseFeedForward(dim, d_ff, dropout)
        self.ffn_norm = nn.LayerNorm(dim, eps=1e-6)

        self.tok_proj = nn.Linear(dim, dim)
        self.tok_acc_norm = nn.LayerNorm(dim, eps=1e-6)

        self.stm_proj = nn.Linear(dim, dim)
        self.stm_acc_norm = nn.LayerNorm(dim, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

    def forward(self, stok_seq, tok_seq, stm_seq, stok2tok_mat, stok2stm_mat, stok_pad_mask, 
            tok_pad_mask, stm_pad_mask):
        """ Hierarchical Attention Encoder Layer.
        Args:
            stok_seq (FloatTensor): `[batch, stok, dim]`
            tok_seq (FloatTensor): `[batch, tok, dim]`
            stm_seq (FloatTensor): `[batch, stm, dim]`
            stok2tok_mat (FloatTensor): normalized mapping, `[batch, tok, stok]` 
            stok2stm_mat (FloatTensor): normalized mapping, `[batch, stm, stok]`
            stok_pad_mask (BoolTensor): `[batch, 1, stok]`
            tok_pad_mask (BoolTensor): `[batch, 1, tok]`
            stm_pad_mask (BoolTensor): `[batch, 1, stm]`
        Returns:
            FloatTensor:
            - context, shape as `[batch, seq_len, dim]`
        """
        len_group = stok_seq.size(1), tok_seq.size(1), stm_seq.size(1)

        # -> `[batch, tok, dim]`
        avg_pool_tok = stok2tok_mat @ stok_seq
        tok_seq = self.tok_acc_norm(self.tok_proj(avg_pool_tok) + tok_seq)

        # -> `[batch, stm, dim]`
        avg_pool_stm = stok2stm_mat @ stok_seq
        stm_seq = self.stm_acc_norm(self.stm_proj(avg_pool_stm) + stm_seq)

        hi_attn = self.attn(stok_seq, tok_seq, stm_seq, stok_pad_mask, tok_pad_mask, stm_pad_mask)
        hi_attn = self.dropout(hi_attn)
        hi_stok_seq, hi_tok_seq, hi_stm_seq = torch.split(hi_attn, len_group, dim=1)

        hi_stok_seq = self.stok_attn_norm(hi_stok_seq + stok_seq)
        hi_tok_seq = self.tok_attn_norm(hi_tok_seq + tok_seq)
        hi_stm_seq = self.stm_attn_norm(hi_stm_seq + stm_seq)

        hi_stok_seq = self.ffn_norm(self.ffn(hi_stok_seq) + hi_stok_seq)

        return hi_stok_seq, hi_tok_seq, hi_stm_seq


class Hi3AttnEncoder(nn.Module):
    """ Encoder of transformer """
    def __init__(self, num_layers, dim=512, num_heads=8, d_ff=2048, dropout=0.2,
                 rel_range=0, use_negative=True):
        super(Hi3AttnEncoder, self).__init__()
        self.num_layers = num_layers

        self.encoder_layers = nn.ModuleList([
            PAEncoderLayer(dim, num_heads, d_ff, dropout, rel_range, use_negative)
            for _ in range(num_layers)
        ])

    def forward(self, stok_seq, tok_seq, stm_seq, stok2tok_mat, stok2stm_mat, stok_pad_mask, 
            tok_pad_mask, stm_pad_mask):
        """ Hierarchical Attention Encoder.
        Args:
            stok_seq (FloatTensor): `[batch, stok, dim]`
            tok_seq (FloatTensor): `[batch, tok, dim]`
            stm_seq (FloatTensor): `[batch, stm, dim]`
            stok2tok_mat (FloatTensor): `[batch, tok, stok]`
            stok2stm_mat (FloatTensor): `[batch, stm, stok]`
            stok_pad_mask (BoolTensor): `[batch, stok]`
            tok_pad_mask (BoolTensor): `[batch, tok]`
            stm_pad_mask (BoolTensor): `[batch, stm]`
        Returns:
            FloatTensor:
            - context, shape as `[batch, seq_len, dim]`
        Returns:
            (list[FloatTensor], list[list(FloatTensor)]):
            - encoded context seq, shape as `[batch, seq_len, dim]`
            - attn distribution, shape as `[batch, head, seq_len, seq_len]`
        """
        stok_pad_mask = stok_pad_mask.unsqueeze(1)  # `[batch, 1, stok]`
        tok_pad_mask = tok_pad_mask.unsqueeze(1)  # `[batch, 1, tok]`
        stm_pad_mask = stm_pad_mask.unsqueeze(1)  # `[batch, 1, stm]`

        # normalize mapping
        stok2tok_mat /= (stok2tok_mat.sum(dim=-1, keepdim=True) + 1e-8)
        stok2stm_mat /= (stok2stm_mat.sum(dim=-1, keepdim=True) + 1e-8)

        mid_stok, mid_tok, mid_stm = stok_seq, tok_seq, stm_seq
        for layer in self.encoder_layers:
            mid_stok, mid_tok, mid_stm = layer(mid_stok, mid_tok, mid_stm, stok2tok_mat, 
                stok2stm_mat, stok_pad_mask, tok_pad_mask, stm_pad_mask)
        return mid_stok, mid_tok, mid_stm


@MODELS.register_module()
class PAformer(BaseModel):
    """ Transformer """
    def __init__(self, config):
        super(PAformer, self).__init__()
        self.name = 'Transformer'
        self.use_copy_attn = config.copy_attn
        self.num_heads = config.num_heads
        self.dim = config.dim
        self.head_dim = self.dim // self.num_heads

        self.src_stok_embedding = SrcEmbedding(
            dim=config.dim,
            vocab_size=config.embedding.stok_vocab_size,
            dropout=config.embedding.dropout
        )
        self.src_tok_embedding = nn.Embedding(
            num_embeddings=config.embedding.tok_vocab_size,
            embedding_dim=config.dim,
            padding_idx=0
        )
        self.src_stm_embedding = nn.Embedding(
            num_embeddings=config.embedding.stm_vocab_size,
            embedding_dim=config.dim,
            padding_idx=0
        )
        self.tgt_embedding = TgtEmbedding(
            dim=config.dim,
            vocab_size=config.embedding.tgt_vocab_size,
            max_tgt_len=config.max_tgt_len,
            pos_emb=config.embedding.tgt_pos_type == 'learn',
            dropout=config.embedding.dropout
        )

        self.encoder = Hi3AttnEncoder(
            num_layers=config.num_encoder_layers,
            dim=config.dim,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            dropout=config.attn_dropout,
            rel_range=config.embedding.rel_range,
            use_negative=config.embedding.use_negative
        )
        self.decoder = TransformerDecoder(
            num_layers=config.num_decoder_layers,
            dim=config.dim,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            dropout=config.attn_dropout
        )

        self.generator = nn.Linear(config.dim, config.embedding.tgt_vocab_size)

        if self.use_copy_attn:
            self.copy_attn = CopyAttention(config.dim, self.generator)

        if config.embedding.share_embedding:
            self.generator.weight = self.tgt_embedding.tgt_embedding.weight

    def forward(self, tgt_seq_rep, stok_seq_rep, tok_seq_rep, stm_seq_rep, 
            stok2tok_mat, stok2stm_mat, src_map):
        """ forward pass of transformer training process.
        Args:
            tgt_seq_rep (LongTensor): `[batch, tgt_len]`
            stok_seq_rep (LongTensor): `[batch, stok_len]`
            stm_seq_rep (LongTensor): `[batch, stm_len]`
            stok2stm_mat (FloatTensor): `[batch, stm_len, stok_len]`
            src_map (FloatTensor): `[batch, src_len, dynamic_vocab_size]`
        Returns:
            scores (FloatTensor): `[batch, tgt_len, (extend)_tgt_vocab_size]`
        """
        # encode
        # -> `[batch, src_len, dim]`, `[batch, src_len]`
        stok_seq_rep, stok_pad_mask = self.encode(stok_seq_rep, tok_seq_rep, stm_seq_rep, 
            stok2tok_mat, stok2stm_mat)

        # decode
        tgt_pad_mask = tgt_seq_rep == 0  # `[batch, tgt_len]`
        tgt_emb = self.tgt_embedding(tgt_seq_rep)
        dec_out = self.decoder(stok_seq_rep, stok_pad_mask, tgt_emb, tgt_pad_mask)

        if self.use_copy_attn:  # -> `[batch, tgt_len, extend_vocab_size]`, after softmax
            scores = self.copy_attn(dec_out, stok_seq_rep, stok_pad_mask, src_map)
        else:  # un-softmax
            scores = self.generator(dec_out)  # `[batch, tgt_len, vocab_size]`

        # `[batch, tgt_len - 1, vocab_size]`, get rid of `EOS`
        scores = scores[:, :-1, :].contiguous()

        return scores

    def encode(self, stok_seq_rep, tok_seq_rep, stm_seq_rep, stok2tok_mat, stok2stm_mat,
            return_dict=False):

        stok_pad_mask = stok_seq_rep == 0  # `[batch, stok]`
        tok_pad_mask = tok_seq_rep == 0  # `[batch, tok]`
        stm_pad_mask = stm_seq_rep == 0  # `[batch, stm]`

        stok_seq_emb = self.src_stok_embedding(stok_seq_rep)  # `[batch, stok, dim]`
        tok_seq_emb = self.src_tok_embedding(tok_seq_rep)  # `[batch, tok, dim]`
        stm_seq_emb = self.src_stm_embedding(stm_seq_rep)  # `[batch, stm, dim]`
        
        # -> `[batch, stok, dim]`, `[batch, stm, dim]`
        stok_seq_rep, _, _ = self.encoder(stok_seq_emb, tok_seq_emb, stm_seq_emb,
            stok2tok_mat, stok2stm_mat, stok_pad_mask, tok_pad_mask, stm_pad_mask)

        if return_dict:
            return {
                'stok_seq_rep': stok_seq_rep,
                'stok_pad_mask': stok_pad_mask,
            }

        return stok_seq_rep, stok_pad_mask

    def step_wise_decode(self, tgt_rep, stok_seq_rep, stok_pad_mask, src_map, shared_idxs, 
                        step, cache, beam_size=None, beam_idx=None, **kwargs):
        """ this function do one step of auto-regression decoding, which will be called
        by the decoding approach `greedy_decode`/`beam_search_decode` of `Trainer`.
        Args:
            tgt_rep (LongTensor): the batch of one-step input into decode part, `[batch]`.
            stok_seq_rep (FloatTensor): the out of encode part, `[batch, stok, dim]`.
            stok_pad_mask (LongTensor): `[batch, 1, stok]`
            src_map: encoded src_seq by local src_vocab, shape as
                `[batch, src_len, max_local_vocab_size]`, which maps the `copy scores`
                over the src_seq into the corresponding dynamic vocab.
            shared_idxs (List[(LongTensor, LongTensor)]): the shared word idxs in
                tgt_vocab and the dynamic_src_vocab each example, which are used to merge
                the concated distribution over extended vocab in the copy attention
                framework.
                - dy_idxs, `[num_shared_words_{i}]`
                - tgt_idxs, `[num_shared_words_{i}]`
            step (int): which step we have gone
            beam_size (int): beam search size
            cache (dict): cache for fast inference, which stores pre-computed keys/values
                - memory_keys
                - memory_values
                - self_keys
                - self_values
            beam_idx (LongTensor): `[batch_beam]`
        Returns:
            pred_dist: the predicted distribution over (extended)_tgt_vocab for each
                batch, shape as `[batch, (extended)_vocab_size]`, after softmax
            hidden: the updated hidden
        """
        tgt_rep = tgt_rep.unsqueeze(1)
        tgt = self.tgt_embedding(tgt_rep, step=step)
        tgt_pad_mask = tgt_rep == PAD  # `[batch, 1]`

        # update cache
        if beam_idx is not None:
            for i in range(len(self.decoder.decoder_layers)):
                layer_cache = cache[f'layer_{i}']
                layer_cache["memory_keys"] = layer_cache["memory_keys"][beam_idx]
                layer_cache['memory_values'] = layer_cache['memory_values'][beam_idx]
                layer_cache["self_keys"] = layer_cache["self_keys"][beam_idx]
                layer_cache["self_values"] = layer_cache["self_values"][beam_idx]

        # -> `[batch, 1, dim]`
        dec_out = self.decoder(stok_seq_rep, stok_pad_mask, tgt, tgt_pad_mask, cache, step)

        if self.use_copy_attn:
            # `[batch, 1, tgt_vocab_size + extra_words]`
            prediction = self.copy_attn(dec_out, stok_seq_rep, stok_pad_mask, src_map)
            prediction = prediction.squeeze(1)  # `[batch, tgt_vocab_size + extra_words]`
            prediction = merge_copy_dist(prediction, shared_idxs, beam_size)
        else:
            prediction = self.generator(dec_out.squeeze(1))
            prediction = F.softmax(prediction, dim=1)  # `[batch, tgt_vocab_size]`

        return prediction, {}

""" multi head attn: cross attn and self attn """

import math
import torch
from torch import nn


class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Head Self Attention """
    def __init__(self, dim, num_heads, dropout=0.1, rel_range=0, use_negative=True):
        super(MultiHeadedSelfAttention, self).__init__()
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
            vocab_size = rel_range * 2 + 1 if self.use_negative else rel_range + 1
            self.rel_embeddings_k = nn.Embedding(vocab_size, self.head_dim)
            self.rel_embeddings_v = nn.Embedding(vocab_size, self.head_dim)

    def forward(self, x, mask=None, layer_cache=None):
        """ Compute the context vector and the attention vectors.
        Args:
           x (FloatTensor): shape as `[batch, seq_len, dim]`
           mask (BoolTensor): 1/0 -> 0/~0  attn value, shape as  `[batch, seq_len, seq_len]`
           layer_cache (dict):
        Returns:
           (FloatTensor, FloatTensor):
           - context, shape as `[batch, seq_len, dim]`
           - attn distribution, shape as `[batch, seq_len, seq_len]`
        """
        batch_size, seq_len = x.shape[:2]
        rel_pe_keys, rel_pe_values = None, None

        # project `x` to `key`, `value`, and `query`
        # all shapes as `[batch, head, seq_len, head_dim]`
        if layer_cache is not None:  # used when decoding
            qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            query, key, value = qkv[0], qkv[1], qkv[2]

            if layer_cache["self_keys"] is not None:
                key = torch.cat([layer_cache["self_keys"], key], dim=2)
            if layer_cache["self_values"] is not None:
                value = torch.cat([layer_cache["self_values"], value], dim=2)

            layer_cache["self_keys"] = key
            layer_cache["self_values"] = value
        else:
            qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            query, key, value = qkv[0], qkv[1], qkv[2]

        # get relative distance embeddings: `rel_keys`, `rel_values`
        # all shapes as `[seq_len, seq_len, head_dim]`
        if self.rel_range > 0:
            key_len = key.size(2)
            rel_dist_matrix = gen_rel_dist_matrix(key_len, self.rel_range, self.use_negative,
                                                  layer_cache is not None)
            rel_dist_matrix = rel_dist_matrix.to(x.device)  # `[seq_len/1, seq_len]`
            rel_pe_keys = self.rel_embeddings_k(rel_dist_matrix)
            rel_pe_values = self.rel_embeddings_v(rel_dist_matrix)

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
            mask = mask.unsqueeze(1)  # `[batch, 1, 1, seq_len]`
            attn_score = attn_score.masked_fill(mask, -1e18)

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


class MultiHeadedCrossAttention(nn.Module):
    """ Multi-Head Cross Attention """
    def __init__(self, dim, num_heads, dropout=0.1):
        super(MultiHeadedCrossAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.query = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.output = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, key, query, mask=None, layer_cache=None):
        """ Compute the context vector and the attention vectors.
        Args:
            key (FloatTensor): shape as `[batch, key_len, dim]`
            query (FloatTensor): shape as `[batch, query_len, dim]`
            mask (BoolTensor): 1/0 -> 0/~0  attn value, shape as `[batch, query_len, key_len]`
            layer_cache (dict):
            - memory_keys (FloatTensor): proj_key of the key seq, `[batch, head, key_len, head_dim]`
            - memory_values (FloatTensor): proj_value *, `[batch, head, key_len, head_dim]`
        Returns:
            (FloatTensor, list[FloatTensor]):
            - context, shape as `[batch, query_len, dim]`
            - attn distribution, shape as `[batch, query_len, key_len]`
        """
        batch_size, key_len = key.shape[:2]

        # project `key` to `key`, `value` and `query` to `query`
        # shapes as `[batch, head, key_len/query_len, head_dim]`
        if layer_cache is not None:
            if layer_cache["memory_keys"] is None:
                kv = self.kv(key).reshape(batch_size, key_len, 2, self.num_heads, self.head_dim)
                kv = kv.permute(2, 0, 3, 1, 4)
                key, value = kv[0], kv[1]
                layer_cache["memory_keys"] = key
                layer_cache["memory_values"] = value
            else:
                key, value = layer_cache["memory_keys"], layer_cache["memory_values"]
        else:
            kv = self.kv(key).reshape(batch_size, key_len, 2, self.num_heads, self.head_dim)
            kv = kv.permute(2, 0, 3, 1, 4)
            key, value = kv[0], kv[1]
        query = self.query(query)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # scaled dot product attn, shape as `[batch, head, query_len, key_len]`
        query = query / math.sqrt(self.head_dim)
        attn_score = query @ key.transpose(-1, -2)

        # mask
        if mask is not None:
            mask = mask.unsqueeze(1)  # `[batch, 1, 1, seq_len]`
            attn_score = attn_score.masked_fill(mask, -1e18)

        # attn distribution and get the context of attn-weighted over value
        attn = self.dropout(self.softmax(attn_score))
        context = attn @ value  # `[batch, head, query_len, head_dim]`
        context = context.transpose(1, 2).reshape(batch_size, -1, self.dim)

        # output projection
        final_output = self.output(context)

        return final_output


class MultiHeadSerialCrossAttention(nn.Module):
    """ Multi-Head Serial Cross Attention 
    Libovický et al. Input combination strategies for multi-source transformer decoder. In WMT'2018.
    """
    def __init__(self, dim, num_heads, dropout=0.1, num_itr=2):
        super(MultiHeadSerialCrossAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.query = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.output = nn.Linear(dim, dim)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(dim, eps=1e-6) for _ in range(num_itr)])

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, src_mem_banks, src_pad_masks, cache=None):
        """ Compute the context vector and the attention vectors.
        Args:
            query (FloatTensor): shape as `[batch, query_len, dim]`
            src_mem_banks (list[FloatTensor]): list of src_mem_bank, `[batch, src_len_i, dim]`
            src_pad_masks (list[FloatTensor]): list of src_pad_mask, `[batch, 1, src_len_i]`
            layer_cache (dict):
            - mem_bank_key_{i} (FloatTensor): projected multi-head key of i-th src_mem_bank,
                `[batch, head, src_len_i, head_dim]`
            - mem_bank_value_{i} (FloatTensor): projected multi-head value of i-th src_mem_bank,
                `[batch, head, src_len_i, head_dim]`
        Returns:
            (FloatTensor): context, shape as `[batch, query_len, dim]`
        """
        ori_query = query
        for idx, (src_mem_bank, src_pad_mask) in enumerate(zip(src_mem_banks, src_pad_masks)):
            out = self.forward_per_src(query, src_mem_bank, src_pad_mask, idx, cache)
            query = self.layer_norms[idx](ori_query + self.dropout(out))

        return query

    def forward_per_src(self, query, src_mem_bank, src_pad_mask, src_idx, cache=None):
        # compute projected key and value, -> `[batch, head, src_len, head_dim]` 
        batch_size, src_len = src_mem_bank.shape[:2]
        if cache is not None:
            if cache.get(f'mem_bank_key_{src_idx}', None) is None:
                kv = self.kv(src_mem_bank).reshape(batch_size, src_len, 2, 
                    self.num_heads, self.head_dim)
                kv = kv.permute(2, 0, 3, 1, 4) # [2, batch, head, src_len, head_dim]
                key, value = kv[0], kv[1]
                cache[f'mem_bank_key_{src_idx}'] = key
                cache[f'mem_bank_value_{src_idx}'] = value
            else:
                key = cache[f'mem_bank_key_{src_idx}']
                value = cache[f'mem_bank_value_{src_idx}']
        else:
            kv = self.kv(src_mem_bank).reshape(batch_size, src_len, 2, 
                self.num_heads, self.head_dim)
            kv = kv.permute(2, 0, 3, 1, 4) # [2, batch, head, src_len, head_dim]
            key, value = kv[0], kv[1]
        
        # compute projected query, -> `[batch, head, query_len, head_dim]`
        query = self.query(query)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # scaled dot product attn, -> `[batch, head, query_len, src_len]`
        query /= math.sqrt(self.head_dim)
        attn_score = query @ key.transpose(-1, -2)

        # mask out attn scores w.r.t. src_pad_mask
        if src_pad_mask is not None:
            src_pad_mask = src_pad_mask.unsqueeze(1) # `[batch, 1, 1, src_len]`
            attn_score = attn_score.masked_fill(src_pad_mask, -1e18)
        
        # weighted-avg over value w.r.t. attn dist, -> `[batch, query_len, dim]`
        attn_dist = self.dropout(self.softmax(attn_score))
        context = attn_dist @ value  # -> `[batch, head, query_len, head_dim]`
        context = context.transpose(1, 2).reshape(batch_size, -1, self.dim)

        # final projection
        return self.output(context)


def gen_rel_dist_matrix(seq_len, rel_range, use_negative, cache=False):
    """Generate the clipped relative distance matrix
    Example: for seq_len=3, rel_range=2, use_negative=False/True
        [[0, 1, 2],        [[2, 3, 4],
         [1, 0, 1],   or    [1, 2, 3],
         [2, 1, 0]]         [0, 1, 2]]
    """
    if cache:
        dist_mat = torch.arange(-seq_len + 1, 1, 1).unsqueeze(0)
    else:
        range_vec = torch.arange(seq_len)
        range_mat = range_vec.unsqueeze(-1).expand(-1, seq_len).transpose(0, 1)
        dist_mat = range_mat - range_mat.transpose(0, 1)

    distance_mat_clipped = torch.clamp(dist_mat, min=-rel_range, max=rel_range)

    # shift values to be >= 0
    if use_negative:
        final_mat = distance_mat_clipped + rel_range
    else:
        final_mat = torch.abs(distance_mat_clipped)

    return final_mat


def add_future_mask(tgt_pad_mask):
    """ add the future mask for tgt_pad_mask
    Args:
        tgt_pad_mask (BoolTensor): shape as `[batch, 1, tgt_len]`
    Returns:
        (BoolTensor): tgt_pad_mask, shape as `[batch, tgt_len, tgt_len]`
    """
    tgt_len = tgt_pad_mask.size(-1)
    future_mask = torch.ones([tgt_len, tgt_len], dtype=torch.uint8)
    future_mask = future_mask.to(tgt_pad_mask.device)
    future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len)
    return torch.gt(tgt_pad_mask + future_mask, 0)

""" Transformer """

import torch
import torch.nn.functional as F
from torch import nn

from utils import PAD
from .build import MODELS
from .modules import CopyAttention, merge_copy_dist
from .modules import MultiHeadedSelfAttention, MultiHeadedCrossAttention
from .base_model import BaseModel


class PositionwiseFeedForward(nn.Module):
    """ two-layer Feed-Forward-Network with residual layer norm. """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(inplace=True)
        )

    def forward(self, x):
        x = self.ffn(x)

        return x


class SrcEmbedding(nn.Module):
    """ src embedding, vocab embedding + postion embedding """
    def __init__(self, dim, vocab_size, max_src_len=-1, pos_emb=False, dropout=0.1):
        super(SrcEmbedding, self).__init__()
        assert not pos_emb or max_src_len > 0
        self.pos_emb = pos_emb  # config.embedding.src_pos_type == 'learn'
        self.vocab_size = vocab_size  # config.embedding.src_vocab_size
        self.dim = dim  # config.dim

        self.src_embedding = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        if self.pos_emb:
            self.src_pos_embedding = nn.Embedding(max_src_len, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, seq):
        """ embedding forward pass """
        # seq (LongTensor): `[batch, seq_len]`
        # vocab embedding, -> `[batch, seq_len, dim]`
        word_emb = self.src_embedding(seq)

        # pos embedding
        if self.pos_emb:
            seq_len = seq.size(1)
            pos = torch.arange(start=0, end=seq_len).to(seq.device)

            pos_emb = self.src_pos_embedding(pos).unsqueeze(0)  # `[1, seq_len, dim]`
            word_emb = word_emb + pos_emb

        word_rep = self.dropout(word_emb)
        return word_rep


class TgtEmbedding(nn.Module):
    """ tgt embedding """
    def __init__(self, dim, vocab_size, max_tgt_len=-1, pos_emb=True, dropout=0.1):
        super(TgtEmbedding, self).__init__()
        assert not pos_emb or max_tgt_len > 0
        self.pos_emb = pos_emb  # config.embedding.tgt_pos_type == 'learn'
        self.vocab_size = vocab_size  # config.embedding.tgt_vocab_size
        self.dim = dim

        self.tgt_embedding = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        if self.pos_emb:
            self.tgt_pos_embedding = nn.Embedding(max_tgt_len + 2, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, seq, step=None):
        """ embedding forward pass """
        # vocab embedding, -> `[batch, seq_len, dim]`
        word_emb = self.tgt_embedding(seq)
        # pos embedding
        if self.pos_emb:
            seq_len = seq.size(1)
            if step is None:  # training
                pos = torch.arange(start=0, end=seq_len)
            else:  # inference
                pos = torch.LongTensor([step])
            pos = pos.to(seq.device)

            pos_emb = self.tgt_pos_embedding(pos).unsqueeze(0)  # `[1, 1/seq_len, dim]`
            word_emb = word_emb + pos_emb

        word_emb = self.dropout(word_emb)
        return word_emb


class TransformerEncoderLayer(nn.Module):
    """ Encoder layer of transformer """
    def __init__(self, dim, num_heads, d_ff, dropout, rel_range=0, use_negative=True,
                 pre_ln=False):
        super(TransformerEncoderLayer, self).__init__()
        self.pre_ln = pre_ln  # pre-LN(layer norm)

        self.self_attn = MultiHeadedSelfAttention(dim, num_heads, dropout, rel_range,
                                                  use_negative)
        self.ffn = PositionwiseFeedForward(dim, d_ff, dropout)

        self.self_attn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(dim, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """ Transformer Encoder Layer definition.
        Args:
            x (FloatTensor): `[batch, seq_len, dim]`
            mask (LongTensor): `[batch, seq_len, seq_len]`
        Returns:
            FloatTensor:
            - context, shape as `[batch, seq_len, dim]`
        """
        x_attn = self.self_attn(x, mask=mask)
        x = self.self_attn_norm(self.dropout(x_attn) + x)

        x = self.ffn_norm(self.ffn(x) + x)

        return x


class TransformerEncoder(nn.Module):
    """ Encoder of transformer """
    def __init__(self, num_layers, dim=512, num_heads=8, d_ff=2048, dropout=0.2,
                 rel_range=0, use_negative=True):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(dim, num_heads, d_ff, dropout, rel_range, use_negative)
            for _ in range(num_layers)
        ])

    def forward(self, x, pad_mask):
        """ Encoder forward
        Args:
            x (FloatTensor): input seq, shape as `[batch, seq_len, dim]`
            pad_mask (BoolTensor): length of each sequence, shape as `[batch, seq_len]`
        Returns:
            (list[FloatTensor], list[list(FloatTensor)]):
            - encoded context seq, shape as `[batch, seq_len, dim]`
            - attn distribution, shape as `[batch, head, seq_len, seq_len]`
        """
        pad_mask = pad_mask.unsqueeze(1)  # `[batch, 1, seq_len]`

        out = x
        for layer in self.encoder_layers:
            out = layer(out, pad_mask)

        return out


class TransformerDecoderLayer(nn.Module):
    """ Decoder layer of transformer """
    def __init__(self, dim, num_heads, d_ff, dropout, rel_range=0):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = MultiHeadedSelfAttention(dim, num_heads, dropout, rel_range)
        self.context_attn = MultiHeadedCrossAttention(dim, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(dim, d_ff, dropout)

        self.self_attn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.context_attn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(dim, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory_bank, src_pad_mask, tgt_mask, layer_cache=None):
        """ Decoder per layer forward, we use the improved version (pre-processing)
        following the blog <https://tunz.kr/post/4>
        pre-process: x + sub_layer(layer_norm(x))
        Args:
            tgt (FloatTensor): shape as `[batch, tgt_len, dim]`
            memory_bank (FloatTensor): out of encoder, shape as `[batch, src_len, dim]`
            src_pad_mask (LongTensor): shape as `[batch, 1, src_len]`
            tgt_mask (LongTensor): shape as `[batch, tgt_len, tgt_len]`
            layer_cache (dict):
        Returns:
            (FloatTensor, FloatTensor):
            - output `[batch, 1/tgt_len, dim]`
            - attn `[batch, 1/tgt_len, src_len]`
        """
        query = self.self_attn(tgt, tgt_mask, layer_cache)
        query_norm = self.self_attn_norm(self.dropout(query) + tgt)

        mid = self.context_attn(memory_bank, query_norm, src_pad_mask, layer_cache)
        mid_norm = self.context_attn_norm(self.dropout(mid) + query_norm)

        out = self.ffn_norm(self.ffn(mid_norm) + mid_norm)

        return out


class TransformerDecoder(nn.Module):
    """ Decoder of Transformer """
    def __init__(self, num_layers, dim=512, num_heads=8, d_ff=2048, dropout=0.2,
                 rel_range=0):
        super(TransformerDecoder, self).__init__()
        self.num_layers = num_layers

        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(dim, num_heads, d_ff, dropout, rel_range)
            for _ in range(num_layers)
        ])

    def forward(self, memory_bank, src_pad_mask, emb, tgt_pad_mask, cache=None,
                step=None):
        if step == 0:
            self._init_cache(cache)

        # mask of tgt_src seq
        tgt_src_mask = tgt_pad_mask
        if tgt_pad_mask is not None:
            tgt_pad_mask = tgt_pad_mask.unsqueeze(1)
            # training mode, add future mask
            tgt_src_mask = self.add_future_mask(tgt_pad_mask) if step is None \
                else tgt_pad_mask

        src_pad_mask = src_pad_mask.unsqueeze(1)
        out = emb
        for i, layer in enumerate(self.decoder_layers):
            layer_cache = cache[f"layer_{i}"] if cache is not None else None
            out = layer(out, memory_bank, src_pad_mask, tgt_src_mask, layer_cache)

        return out

    def _init_cache(self, cache):
        for i in range(self.num_layers):
            layer_cache = {}
            layer_cache["memory_keys"] = None
            layer_cache["memory_values"] = None
            layer_cache["self_keys"] = None
            layer_cache["self_values"] = None
            cache[f'layer_{i}'] = layer_cache

    @staticmethod
    def add_future_mask(tgt_src_pad_mask):
        """ add the future mask for tgt_src
        Args:
            tgt_src_pad_mask (BoolTensor): shape as `[batch, 1, tgt_src_len]`
        Returns:
            BoolTensor:
            - tgt_src_mask, shape as `[batch, tgt_src_len, tgt_src_len]`
        """
        tgt_len = tgt_src_pad_mask.size(-1)
        future_mask = torch.ones([tgt_len, tgt_len], dtype=torch.uint8)
        future_mask = future_mask.to(tgt_src_pad_mask.device)
        future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len)
        return torch.gt(tgt_src_pad_mask + future_mask, 0)


@MODELS.register_module()
class Transformer(BaseModel):
    """ Transformer """
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.name = 'Transformer'
        self.use_copy_attn = config.copy_attn

        self.src_embedding = SrcEmbedding(
            dim=config.dim,
            vocab_size=config.embedding.src_vocab_size,
            dropout=config.embedding.dropout
        )
        self.tgt_embedding = TgtEmbedding(
            dim=config.dim,
            vocab_size=config.embedding.tgt_vocab_size,
            max_tgt_len=config.max_tgt_len,
            pos_emb=config.embedding.tgt_pos_type == 'learn',
            dropout=config.embedding.dropout
        )

        self.encoder = TransformerEncoder(
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

    def forward(self, src_seq_rep, tgt_seq_rep, src_map=None):
        """ forward pass of transformer training process.
        Args:
            src_seq_rep (LongTensor): `[batch, src_len]`
            tgt_seq_rep (LongTensor): `[batch, tgt_len]`
            src_map (FloatTensor): `[batch, src_len, dynamic_vocab_size]`
        Returns:
            scores (FloatTensor): `[batch, tgt_len, (extend)_tgt_vocab_size]`
        """
        # encode
        # -> `[batch, src_len, dim]`, `[batch, src_len]`
        memory_bank, src_pad_mask = self.encode(src_seq_rep)

        # decode
        tgt_pad_mask = tgt_seq_rep == 0  # `[batch, tgt_len]`
        tgt_emb = self.tgt_embedding(tgt_seq_rep)
        dec_out = self.decoder(memory_bank, src_pad_mask, tgt_emb, tgt_pad_mask)

        if self.use_copy_attn:  # -> `[batch, tgt_len, extend_vocab_size]`, after softmax
            scores = self.copy_attn(dec_out, memory_bank, src_pad_mask, src_map)
        else:  # un-softmax
            scores = self.generator(dec_out)  # `[batch, tgt_len, vocab_size]`

        # `[batch, tgt_len - 1, vocab_size]`, get rid of `EOS`
        scores = scores[:, :-1, :].contiguous()

        return scores

    def encode(self, src_seq_rep, return_dict=False):
        # `[batch, src_len]`
        src_pad_mask = src_seq_rep == 0  # `[batch, src_len]`
        src_seq_emb = self.src_embedding(src_seq_rep)
        # `[batch, src_len, dim]`
        memory_bank = self.encoder(src_seq_emb, src_pad_mask)

        if return_dict:
            return {
                'memory_bank': memory_bank,
                'src_pad_mask': src_pad_mask
            }

        return memory_bank, src_pad_mask

    def step_wise_decode(self, tgt_rep, memory_bank, src_pad_mask, src_map, shared_idxs,
                         step, cache, beam_size=None, beam_idx=None, **kwargs):
        """ this function do one step of auto-regression decoding, which will be called
        by the decoding approach `greedy_decode`/`beam_search_decode` of `Trainer`.
        Args:
            tgt_rep (LongTensor): the batch of one-step input into decode part, `[batch]`.
            memory_bank (FloatTensor): the out of encode part, `[batch, src_len, dim]`.
            src_pad_mask (LongTensor): `[batch, src_len]`
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
                layer_cache['memory_keys'] = layer_cache['memory_keys'][beam_idx]
                layer_cache['memory_values'] = layer_cache['memory_values'][beam_idx]
                layer_cache['self_keys'] = layer_cache['self_keys'][beam_idx]
                layer_cache['self_values'] = layer_cache['self_values'][beam_idx]

        # -> `[batch, 1, dim]`
        dec_out = self.decoder(memory_bank, src_pad_mask, tgt, tgt_pad_mask, cache, step)

        if self.use_copy_attn:
            # `[batch, 1, tgt_vocab_size + extra_words]`
            prediction = self.copy_attn(dec_out, memory_bank, src_pad_mask, src_map)
            prediction = prediction.squeeze(1)  # `[batch, tgt_vocab_size + extra_words]`
            prediction = merge_copy_dist(prediction, shared_idxs, beam_size)
        else:
            prediction = self.generator(dec_out.squeeze(1))
            prediction = F.softmax(prediction, dim=1)  # `[batch, tgt_vocab_size]`

        return prediction, {}

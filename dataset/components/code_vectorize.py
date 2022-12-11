""" convert code data into tensors
- dynamic vocabs for copy mechanism
- dgl data for graph representation
"""

import torch
import torch.nn.functional as F

from utils import UNK
from ..vocabulary import Vocabulary


def gen_map_mat(a2b, max_a_len, max_b_len):
    """ generate mapping matrix M (`[stm, stok]`), which can map stok seq features 
    into stm seq feature
    Args: 
        a2b (LongTensor): shape as `[a_len]`
    example:
    >>> x = torch.tensor([0,0,1,1,2,3,2,1,0])
    >>> gen_map_mat(x)
    >>> out:
    >>> tensor([[1., 1., 0., 0., 0., 0., 0., 0., 1.],
    >>>         [0., 0., 1., 1., 0., 0., 0., 1., 0.],
    >>>         [0., 0., 0., 0., 1., 0., 1., 0., 0.],
    >>>         [0., 0., 0., 0., 0., 1., 0., 0., 0.]])
    """
    a2b = torch.LongTensor(a2b)
    num_stm = a2b.max() + 1
    num_stok = a2b.size(0)
    mapping = torch.arange(num_stm).unsqueeze(1).repeat(1, num_stok)
    x_a = a2b.unsqueeze(0)
    a2b_mat = torch.eq(mapping, x_a).float()

    # padding
    res_a = max_a_len - a2b_mat.size(1)
    res_b = max_b_len - a2b_mat.size(0)
    a2b_mat = F.pad(a2b_mat, (0, res_a, 0, res_b), mode='constant', value=0)  
    
    return a2b_mat



def encode_seq(seq, vocab):
    """ encode words in `seq` using the given vocab.
    Args:
        seq (list[str]): 
        vocab (X2Seq.Vocabulary): 
    """
    return torch.tensor([vocab[w] for w in seq], dtype=torch.long)


def gen_dynamic_vocab(ex, tgt_vocab, src_key='stok_seq'):
    """ for each (src, tgt), build dynamic vocab over `src` seq, find shared words between
    dynamic vocab and tgt vocab and return the dynamic encoded src and tgt.
    Args:
        ex (dict): data example
        tgt_vocab (Vocabulary): vocab over tgts
        src_key (str): use which element as `src` 
    """
    dynamic_vocab = Vocabulary(ex[src_key], no_special_token=True)
    ex['dynamic_vocab'] = dynamic_vocab
    ex['shared_idxs'] = _find_shared_words(dynamic_vocab, tgt_vocab)
    ex['src_dy_rep'] = torch.tensor([dynamic_vocab[w] for w in ex[src_key]], dtype=torch.long)
    ex['tgt_dy_rep'] = torch.tensor([dynamic_vocab[w] for w in ex['tgt_seq']], dtype=torch.long)


def _find_shared_words(dynamic_vocab, tgt_vocab):
    """ There are shared words between dynamic vocab and tgt_vocab, the method
    finds the shared words for each of them.
    Args:
        dynamic_vocab (vocab): the dynamic vocab w.r.t. src_stok_seq
    Returns:
        list[int], list[int]
        - the dynamic word idxs of shared words
        - the tgt word idxs of shared words
    """
    offset = len(tgt_vocab)
    shared_dy_idxs, shared_tgt_idxs = [], []
    for i in range(2, len(dynamic_vocab)):  # 2 for ignoring PAD and UNK
        src_word = dynamic_vocab[i]
        tgt_idx = tgt_vocab[src_word]
        if tgt_idx != UNK:
            shared_dy_idxs.append(offset + i)
            shared_tgt_idxs.append(tgt_idx)

    return shared_dy_idxs, shared_tgt_idxs

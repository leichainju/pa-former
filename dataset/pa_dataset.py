""" hierarchy dataset, which loads the `stok_seq`, `tok_stok_mat` and `stm_stok_mat`
"""

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from utils import UNK
from .vocabulary import Vocabulary
from .base_dataset import BaseDataset
from .build import DATASETS


@DATASETS.register_module()
class Hi3Dataset(BaseDataset):
    def __init__(self, config, examples, stok_vocab, tok_vocab, stm_vocab, tgt_vocab):
        super(Hi3Dataset, self).__init__(config, examples)

        # set vocabs
        self.stok_vocab = stok_vocab
        self.tok_vocab = tok_vocab
        self.stm_vocab = stm_vocab
        self.tgt_vocab = tgt_vocab
    
        # assert len(self.stm_vocab) == config.num_stm_cls + 2
        self.max_stok_len = config.max_stok_len
        self.max_tok_len = config.max_tok_len
        self.max_stm_len = config.max_stm_len

    def __getitem__(self, idx):
        ex = self.examples[idx]

        if 'dynamic_vocab' not in ex:
            dynamic_vocab = Vocabulary(ex['stok_seq'], no_special_token=True)
            ex['dynamic_vocab'] = dynamic_vocab
            ex['shared_idxs'] = self.find_shared_words(dynamic_vocab)
            ex['stok_seq_dy_rep'] = torch.tensor(
                [dynamic_vocab[w] for w in ex['stok_seq']], dtype=torch.long)
            ex['tgt_seq_dy_rep'] = torch.tensor(
                [dynamic_vocab[w] for w in ex['tgt_seq']], dtype=torch.long)
        if 'stok_seq_rep' not in ex:
            ex['stok_seq_rep'] = torch.tensor(
                [self.stok_vocab[w] for w in ex['stok_seq']], dtype=torch.long)
            ex['tok_seq_rep'] = torch.tensor(
                [self.tok_vocab[w] for w in ex['tok_seq']], dtype=torch.long)
            ex['stm_seq_rep'] = torch.tensor(
                [self.stm_vocab[w] for w in ex['stm_seq']], dtype=torch.long)
            ex['tgt_seq_rep'] = torch.tensor(
                [self.tgt_vocab[w] for w in ex['tgt_seq']], dtype=torch.long)
        if 'stok2stm_mat' not in ex:
            stok2tok_mat = self.gen_map_mat(torch.tensor(ex['stok2tok'], dtype=torch.long))
            assert ex['stok_seq_rep'].size(0) == stok2tok_mat.size(1)
            res_stok = self.max_stok_len - stok2tok_mat.size(1)
            res_tok = self.max_tok_len - stok2tok_mat.size(0)
            stok2tok_mat = F.pad(stok2tok_mat, (0, res_stok, 0, res_tok), mode='constant', value=0)  
            ex['stok2tok_mat'] = stok2tok_mat  # [max_tok_len, max_stok_len]

            stok2stm_mat = self.gen_map_mat(torch.tensor(ex['stok2stm'], dtype=torch.long))
            assert ex['stok_seq_rep'].size(0) == stok2stm_mat.size(1)
            res_stok = self.max_stok_len - stok2stm_mat.size(1)
            res_stm = self.max_stm_len - stok2stm_mat.size(0)
            stok2stm_mat = F.pad(stok2stm_mat, (0, res_stok, 0, res_stm), mode='constant', value=0)
            ex['stok2stm_mat'] = stok2stm_mat  # [max_stm_len, max_stok_len]

            del ex['stok2tok'], ex['stok2stm']
            
        return ex

    @staticmethod
    def gen_map_mat(a2b):
        """ generate mapping matrix M (`[stm, stok]`), which can map stok seq features 
        into stm seq feature
        Args: 
            a2b (LongTensor): shape as `[stok_len]`
        example:
        >>> x = torch.tensor([0,0,1,1,2,3,2,1,0])
        >>> gen_map_mat(x)
        >>> out:
        >>> tensor([[1., 1., 0., 0., 0., 0., 0., 0., 1.],
        >>>         [0., 0., 1., 1., 0., 0., 0., 1., 0.],
        >>>         [0., 0., 0., 0., 1., 0., 1., 0., 0.],
        >>>         [0., 0., 0., 0., 0., 1., 0., 0., 0.]])
        """
        num_stm = a2b.max() + 1
        num_stok = a2b.size(0)
        mapping = torch.arange(num_stm).unsqueeze(1).repeat(1, num_stok)
        x_stok = a2b.unsqueeze(0)
        return torch.eq(mapping, x_stok).float()

    def find_shared_words(self, dynamic_vocab):
        """ There are shared words between dynamic vocab and tgt_vocab, the method
        finds the shared words for each of them.
        Args:
            dynamic_vocab (vocab): the dynamic vocab w.r.t. src_stok_seq
        Returns:
            list[int], list[int]
            - the dynamic word idxs of shared words
            - the tgt word idxs of shared words
        """
        offset = len(self.tgt_vocab)
        shared_dy_idxs, shared_tgt_idxs = [], []
        for i in range(2, len(dynamic_vocab)):  # 2 for ignoring PAD and UNK
            src_word = dynamic_vocab[i]
            tgt_idx = self.tgt_vocab[src_word]
            if tgt_idx != UNK:
                shared_dy_idxs.append(offset + i)
                shared_tgt_idxs.append(tgt_idx)

        return shared_dy_idxs, shared_tgt_idxs

    def lengths(self):
        return [(len(ex['stok_seq']), len(ex['tgt_seq'])) for ex in self.examples]

    @staticmethod
    def collect_fn(batch: list):
        """ Gather a batch of individual examples into one batch."""
        batch_size = len(batch)
        copy_attn = batch[0]['dynamic_vocab'] is not None

        dynamic_vocabs, src_maps, shared_idxs = [], [], []  # for copy attn
        tgt_seq_reps, tgt_lens, tgt_seq_dy_reps = [], [], []  # tgt 
        stok_seq_reps = []
        tok_seq_reps, stok2tok_mats = [], []  
        stm_seq_reps, stok2stm_mats = [], []

        for ex in batch:
            # code information
            stok_seq_reps.append(ex['stok_seq_rep'])
            tok_seq_reps.append(ex['tok_seq_rep'])
            stm_seq_reps.append(ex['stm_seq_rep'])

            # mapping
            stok2tok_mats.append(ex['stok2tok_mat'])
            stok2stm_mats.append(ex['stok2stm_mat'])

            # summary information
            tgt_seq_reps.append(ex['tgt_seq_rep'])
            tgt_lens.append(len(ex['tgt_seq']))

            # for copy attn
            if copy_attn:
                dynamic_vocabs.append(ex['dynamic_vocab'])
                src_maps.append(ex['stok_seq_dy_rep'])
                shared_idxs.append(ex['shared_idxs'])
                tgt_seq_dy_reps.append(ex['tgt_seq_dy_rep'])

        stok_seq_rep = pad_sequence(stok_seq_reps, batch_first=True, padding_value=0)
        tok_seq_rep = pad_sequence(tok_seq_reps, batch_first=True, padding_value=0)
        stm_seq_rep = pad_sequence(stm_seq_reps, batch_first=True, padding_value=0)
        tgt_seq_rep = pad_sequence(tgt_seq_reps, batch_first=True, padding_value=0)
        tgt_seq_len = torch.LongTensor(tgt_lens)

        # structure information
        stok_len, tok_len, stm_len = stok_seq_rep.size(1), tok_seq_rep.size(1), stm_seq_rep.size(1) 
        stok2tok_mat = torch.stack(stok2tok_mats, dim=0)
        stok2tok_mat = stok2tok_mat[:, :tok_len, :stok_len].contiguous()
        stok2stm_mat = torch.stack(stok2stm_mats, dim=0)
        stok2stm_mat = stok2stm_mat[:, :stm_len, :stok_len].contiguous()

        # for copy attn
        src_map, tgt_seq_dy_rep = None, None
        if copy_attn:
            tgt_seq_dy_rep = pad_sequence(tgt_seq_dy_reps, batch_first=True, padding_value=0)
            # `[batch, src_len]`, (i,j) j-th tok's dy_idx
            src_map = pad_sequence(src_maps, batch_first=True, padding_value=0)

        return {
            'batch_size':       batch_size,
            'stok_seq_rep':     stok_seq_rep,
            'tok_seq_rep':      tok_seq_rep,
            'stm_seq_rep':      stm_seq_rep,
            'stok2tok_mat':     stok2tok_mat,
            'stok2stm_mat':     stok2stm_mat,
            'tgt_seq_rep':      tgt_seq_rep,
            'tgt_seq_len':      tgt_seq_len,
            'src_text':         [ex['src_text'] for ex in batch],
            'tgt_text':         [ex['tgt_text'] for ex in batch],
            'dynamic_vocabs':   dynamic_vocabs,
            'src_map':          src_map,
            'tgt_seq_dy_rep':   tgt_seq_dy_rep,
            'shared_idxs':      shared_idxs
        }

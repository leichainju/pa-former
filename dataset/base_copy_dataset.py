""" base sequential data dataset supporting copy mechanism.
----------------------------------------
Hard-encoded data in batch
- src_text
- tgt_text
- src_seq_rep
- tgt_seq_rep
----------------------------------------
Usage example in yaml
data:
  name: BaseCopyDataset
  copy_attn: True
  src_key: stok_seq
  tgt_key: tgt_seq
  vocabs:
    src_vocab: ...
    tgt_vocab: ...
"""

import torch
from torch.nn.utils.rnn import pad_sequence

from .components import gen_dynamic_vocab, encode_seq
from .base_dataset import BaseDataset
from .build import DATASETS


@DATASETS.register_module()
class BaseCopyDataset(BaseDataset):
    """ `src_seq`, `tgt_seq` and `copy information` """
    def __init__(self, config, examples, src_vocab=None, tgt_vocab=None):
        super(BaseCopyDataset, self).__init__(config, examples)
        self.copy_attn = config.get('copy_attn', False)
        self.src_key = config.get('src_key', 'stok_seq')
        self.tgt_key = config.get('tgt_key', 'tgt_seq')

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __getitem__(self, idx):
        ex = self.examples[idx]

        if self.copy_attn and 'dynamic_vocab' not in ex:
            # gen `dynamic_vocab`, `shared_idxs`, `src_dy_rep` and `tgt_dy_rep`
            gen_dynamic_vocab(ex, self.tgt_vocab, src_key=self.src_key)
        if 'src_seq_rep' not in ex:
            ex['src_seq_rep'] = encode_seq(ex[self.src_key], self.src_vocab)
            ex['tgt_seq_rep'] = encode_seq(ex[self.tgt_key], self.tgt_vocab)

        return {
            'src_text':            ex['src_text'],
            'tgt_text':            ex['tgt_text'],
            'src_seq_rep':         ex['src_seq_rep'],
            'tgt_seq':             ex['tgt_seq'],
            'tgt_seq_rep':         ex['tgt_seq_rep'],
            'dynamic_vocab':       ex['dynamic_vocab'] if self.copy_attn else None,
            'shared_idxs':         ex['shared_idxs'] if self.copy_attn else None,
            'src_seq_dy_rep':      ex['src_dy_rep'] if self.copy_attn else None,
            'tgt_seq_dy_rep':      ex['tgt_dy_rep'] if self.copy_attn else None
        }

    def lengths(self):
        return [(len(ex[self.src_key]), len(ex[self.tgt_key])) for ex in self.examples]

    @staticmethod
    def collect_fn(batch: list):
        """ Gather a batch of individual examples into one batch."""
        batch_size = len(batch)
        copy_attn = batch[0]['dynamic_vocab'] is not None

        src_seq_reps, dynamic_vocabs, src_maps = [], [], []
        tgt_seq_reps, tgt_lens, tgt_seq_dy_reps = [], [], []
        shared_idxs = []
        for ex in batch:
            # code information
            src_seq_reps.append(ex['src_seq_rep'])

            # summary information
            tgt_seq_reps.append(ex['tgt_seq_rep'])
            tgt_lens.append(len(ex['tgt_seq']))

            # for copy attn
            if copy_attn:
                dynamic_vocabs.append(ex['dynamic_vocab'])
                src_maps.append(ex['src_seq_dy_rep'])
                shared_idxs.append(ex['shared_idxs'])
                tgt_seq_dy_reps.append(ex['tgt_seq_dy_rep'])

        src_seq_rep = pad_sequence(src_seq_reps, batch_first=True, padding_value=0)
        tgt_seq_rep = pad_sequence(tgt_seq_reps, batch_first=True, padding_value=0)
        tgt_seq_len = torch.LongTensor(tgt_lens)

        src_map, tgt_seq_dy_rep = None, None
        if copy_attn:
            tgt_seq_dy_rep = pad_sequence(tgt_seq_dy_reps, batch_first=True, padding_value=0)
            src_map = pad_sequence(src_maps, batch_first=True, padding_value=0)

        return {
            'batch_size':       batch_size,
            'src_text':         [ex['src_text'] for ex in batch],
            'tgt_text':         [ex['tgt_text'] for ex in batch],
            'src_seq_rep':      src_seq_rep,
            'tgt_seq_rep':      tgt_seq_rep,
            'tgt_seq_len':      tgt_seq_len,
            'dynamic_vocabs':   dynamic_vocabs,
            'src_map':          src_map,
            'tgt_seq_dy_rep':   tgt_seq_dy_rep,
            'shared_idxs':      shared_idxs,
        }

""" processing base data
- summary_seq
- tok_seq
"""

import copy

from utils import BOS_WORD, EOS_WORD
from ..build import PIPELINES


@PIPELINES.register_module()
class PreprocessBase:
    """ load base information of data example, like `src_text`, `tgt_text` and `tgt_seq` """
    def __init__(self, config):
        self.init_cfg = copy.deepcopy(config)
        self.src_text_key = self.init_cfg.get('src_text_key', 'tok_seq')
        self.tgt_text_key = self.init_cfg.get('tgt_text_key', 'summary_seq')
    
    def __call__(self, raw_ex, tgt_ex=None):
        if tgt_ex is None:
            tgt_ex = {}

        tgt_seq = [tok.lower() for tok in raw_ex[self.tgt_text_key]]

        tgt_ex['tgt_seq'] = [BOS_WORD] + tgt_seq + [EOS_WORD]
        tgt_ex['tgt_text'] = ' '.join(tgt_seq)
        tgt_ex['src_text'] = ' '.join(raw_ex[self.src_text_key])

        return tgt_ex


@PIPELINES.register_module()
class PreprocessToken:
    """ process token """
    def __init__(self, config):
        self.init_cfg = copy.deepcopy(config)
        self.tok_seq_tok = self.init_cfg.get('tok_seq_key', 'tok_seq')

    def __call__(self, raw_ex, tgt_ex=None):
        if tgt_ex is None:
            tgt_ex = {}

        tgt_ex['tok_seq'] = raw_ex[self.tok_seq_tok]

        return tgt_ex


@PIPELINES.register_module()
class PreprocessSubToken:
    """ process sub-token """
    def __init__(self, config):
        self.init_cfg = copy.deepcopy(config)
        self.stok_seq_tok = self.init_cfg.get('stok_seq_key', 'stok_seq')

    def __call__(self, raw_ex, tgt_ex=None):
        if tgt_ex is None:
            tgt_ex = {}

        tgt_ex['stok_seq'] = raw_ex[self.stok_seq_tok]

        return tgt_ex

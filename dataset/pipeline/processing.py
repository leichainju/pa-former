""" processing methods """

import copy

from utils import PUNC
from ..build import PIPELINES


@PIPELINES.register_module()
class ProcessMultiHierarchy:
    """ generate the stm_cls, tok_grammar_types, stok2tok and stok2stm. """
    def __init__(self, config):
        self.init_cfg = copy.deepcopy(config)
        self.tok_key = config.get('tok_key', 'token_grammar_types')
        self.tok_no_punc = config.get('tok_no_punc', False)

    def __call__(self, raw_ex, tgt_ex=None):
        if tgt_ex is None:
            tgt_ex = {}
        
        tgt_ex['stm_seq'] = raw_ex['stm_seq']
        tok_seq = raw_ex[self.tok_key]
        if self.tok_no_punc:
            tok_seq = list(filter(lambda c: c not in PUNC , tok_seq))
        tgt_ex['tok_seq'] = tok_seq       

        tgt_ex['stok2tok'] = raw_ex['sub_token_to_token']
        stok2stm = self.link_map_list(
            raw_ex['sub_token_to_token'], raw_ex['token_to_stm'])
        tgt_ex['stok2stm'] = stok2stm

        return tgt_ex

    @staticmethod
    def link_map_list(a2b_map, b2c_map):
        a2c_map = [b2c_map[i] for i in a2b_map]
        return a2c_map

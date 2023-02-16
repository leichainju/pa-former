""" generate pyramid sequence of code """

import argparse
import treelib

from structure_parser import MultiLangStructureParser, get_subtoken
from vectorize_data import parallel_data_preprocess

PUNC = '!"#$%&\'(),.:;?@[\\]^_`{|}~'


def _linear_ast_leaves(ast: treelib.Tree):
    # print the tokenized code snippet corresponding to the input tree
    idxs = sorted([leaf.identifier for leaf in ast.leaves()])
    leaf_tokens, leaf_types, leaf_grammar_types = [], [], []
    for idx in idxs:
        node = ast.get_node(idx)

        # token types
        leaf_types.append(node.tag)

        # grammar type (replace the `identifier` using its parent's tag)
        if node.tag == 'identifier':
            p_node = ast.parent(node.identifier)
            leaf_grammar_types.append(p_node.tag)
        else:
            leaf_grammar_types.append(node.tag)

        # token text
        if node.tag == 'string':
            leaf_tokens.append('STR')
        else:
            leaf_tokens.append(ast.get_node(idx).data['token'])

    return idxs, leaf_tokens, leaf_types, leaf_grammar_types


def _gen_pyramid_data(ast, stm_subtrees):
    tok_ast_idxs, tok_seq, tok_types, tok_grammar_seq = _linear_ast_leaves(ast)
    stok_seq, tokt_seq, tokg_seq, tok_ast_seq = [], [], [], []
    stok2tok, tok_idx, stok2tok_all = [], 0, []
    for i, tok in enumerate(tok_seq):
        stoks = get_subtoken(tok)
        stok_seq += stoks
        stok2tok_all += [i] * len(stoks)
        if tok in PUNC:
            stok2tok += [-1] * len(stoks)
        else:
            tokt_seq.append(tok_types[i])
            tokg_seq.append(tok_grammar_seq[i])
            tok_ast_seq.append(tok_ast_idxs[i])
            stok2tok += [tok_idx] * len(stoks)
            tok_idx += 1
    
    map_ast_idx_to_stm_idx, stm_seq = {}, []
    for idx, stm_tree in enumerate(stm_subtrees):
        stm_seq.append(stm_tree.get_node(stm_tree.root).tag)
        ast_idxs = sorted([leaf.identifier for leaf in stm_tree.leaves()])
        for ast_idx in ast_idxs:
            map_ast_idx_to_stm_idx[ast_idx] = idx
    tok2stm = [map_ast_idx_to_stm_idx[ast_idx] for ast_idx in tok_ast_seq]
    tok2stm_all = [map_ast_idx_to_stm_idx[idx] for idx in tok_ast_idxs]
    stok2stm = [tok2stm_all[i] for i in stok2tok_all]
    
    return {
        'stok_seq': stok_seq,
        'tok_seq': tok_seq,
        'tokt_seq': tokt_seq,
        'tokg_seq': tokg_seq,
        'stm_seq': stm_seq,
        'stok2tok': stok2tok,
        'tok2stm': tok2stm,
        'stok2stm': stok2stm
    }


def gen_pyramid(examples, lang):
    parser = MultiLangStructureParser('java')

    # parse structure and linear the parsed structure
    pyramid_exs = []
    for ex in examples:
        try:
            code = ex['code']
            res = parser.parse_cfg(code)
            ast, cfg = res['ast'], res['cfg']
            stm_trees, cfg_edges = parser.split_statements(ast, cfg) 
            res = _gen_pyramid_data(ast, stm_trees)
            res['cfg_edges'] = cfg_edges
            res['summary_seq'] = ex['summary_seq']
            res['code'] = code
            pyramid_exs.append(res)
        except Exception as e:
            print(e, ex)

    return pyramid_exs


if __name__ == '__main__':
    datasets = ['java', 'python', 'funcom']
    splits = ['test', 'train']

    args_parser = argparse.ArgumentParser('preprocess sbt')
    args_parser.add_argument('--src_root', type=str, metavar="PATH", help='dataset root',
                             default='/home/chail/data/mmcodesum')
    args_parser.add_argument('--tgt_root', type=str, metavar="PATH", help='dir to cfg data',
                             default='/home/chail/data/mmcodesum')
    args_parser.add_argument('--split', type=str, choices=splits, help='split of dataset')
    args_parser.add_argument('--dataset', type=str, choices=datasets, help='which dataset')
    args = args_parser.parse_args()

    src_jsonl = f'{args.src_root}/{args.dataset}/filter_{args.split}.jsonl'
    tgt_jsonl = f'{args.src_root}/{args.dataset}/{args.split}_pyramid.jsonl'
    parallel_data_preprocess(gen_pyramid, src_jsonl, tgt_jsonl, args.dataset)

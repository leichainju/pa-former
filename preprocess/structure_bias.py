""" extract pair-wise relations from AST. to reduce the size of sequence length, remove all punctuations.
- pair-wise distance
- personalized page-rank
- 
"""

import argparse
import string
import treelib
import networkx as nx
import pickle

from structure_parser import MultiLangStructureParser, get_subtoken
from vectorize_data import parallel_data_preprocess


def _remove_many_repeat(path: list):
    curr_n = path[0]
    new_path = [curr_n]
    for n in path[1:]:
        if n != curr_n:
            curr_n = n
            new_path.append(n)
    return new_path


def _abs_path(ast: treelib.Tree, g_ast: nx.DiGraph, nt_vocab: dict):
    """ path (non-terminal tokens) from `root` -> `leaf` """
    register_path, path_assign, paths = {}, {}, []
    root = ast.root
    for leaf in ast.leaves():
        idx = leaf.identifier
        p = nx.shortest_path(g_ast, source=root, target=idx)[:-1] # remove leaf
        p_rep = _remove_many_repeat([nt_vocab[g_ast.nodes[n]['tag']] for n in p])
        ps = str(p_rep)
        if ps not in register_path:
            register_path[ps] = len(paths)
            paths.append(p_rep)
        path_assign[idx] = register_path[ps]

    return paths, path_assign


def _rel_path(ast: treelib.Tree, g_ast: nx.DiGraph, nt_vocab: dict):
    """ path (non-terminal tokens) from `leaf`1 -> `leaf`2 """
    un_ast = g_ast.to_undirected()
    leaf_idxs = list(map(lambda n: n.identifier, 
        filter(lambda n: not n.tag in string.punctuation, ast.leaves())))
    
    register_path, path_assign, paths = {}, {}, []
    for i in range(len(leaf_idxs)):
        src_idx = leaf_idxs[i]
        path_assign[src_idx] = {}
        for j in range(i + 1, len(leaf_idxs)):
            tgt_idx = leaf_idxs[j]
            p = nx.shortest_path(un_ast, source=src_idx, target=tgt_idx)[1: -1]
            if len(p) == 0:
                continue
            p_rep = _remove_many_repeat([nt_vocab[g_ast.nodes[n]['tag']] for n in p])
            ps = str(p_rep)
            if ps not in register_path:
                register_path[ps] = len(paths)
                paths.append(p_rep)
            path_assign[(src_idx, tgt_idx)] = path_assign[(tgt_idx, src_idx)] = register_path[ps]
    
    return paths, path_assign


def _gen_stok(ast):
    final_stok_seq = []
    idxs = sorted([leaf.identifier for leaf in ast.leaves()])
    stok_flatten_to_tok_idx, stok_idx = {}, 0
    for idx in idxs:
        node = ast.get_node(idx)
        stok_seq = ['STR'] if node.tag == 'string' else get_subtoken(node.data['token']) 
        for stok in stok_seq:
            final_stok_seq.append(stok)
            stok_flatten_to_tok_idx[stok_idx] = idx
            stok_idx += 1
    return final_stok_seq, stok_flatten_to_tok_idx # `flatten_stok` -> `ast_tok_idx`


def _gen_stok_abs_path_map(stok_to_tok_idx, tok_path_assign):
    stok_path_map = []  # [p_idx, ...]
    for i in range(len(stok_to_tok_idx)):
        tok_idx = stok_to_tok_idx[i]
        stok_path_map.append(tok_path_assign[tok_idx])
    return stok_path_map


def _gen_stok_rel_path_map(stok_to_tok_idx, tok_path_assign):
    stok_path_map = [] # (i, j, p_idx)
    for i in range(len(stok_to_tok_idx)):
        src_idx = stok_to_tok_idx[i]
        for j in range(i + 1, len(stok_to_tok_idx)):
            tgt_idx = stok_to_tok_idx[j]
            if (src_idx, tgt_idx) in tok_path_assign:
                path_idx = tok_path_assign[(src_idx, tgt_idx)]
                stok_path_map.append((i, j, path_idx))
    return stok_path_map


def gen_tree_path(examples, lang):
    parser = MultiLangStructureParser(lang)
    with open(vocab_path, 'rb') as f:
        non_terminal_vocab = pickle.load(f)

    tp_exs = []
    for ex in examples:
        code = ex['code']
        ast = parser.parse_ast(code)
        g_ast = parser.to_networkx(ast)
        abs_paths, tok_abs_path_assign = _abs_path(ast, g_ast, non_terminal_vocab)
        rel_paths, tok_rel_path_assign = _rel_path(ast, g_ast, non_terminal_vocab)
        stok_seq, stok_to_tok_idx = _gen_stok(ast)
        stok_abs_path_map = _gen_stok_abs_path_map(stok_to_tok_idx, tok_abs_path_assign)
        stok_rel_path_map = _gen_stok_rel_path_map(stok_to_tok_idx, tok_rel_path_assign)

        if len(abs_paths) == 0 or len(rel_paths) == 0:
            print(ex)
            print('------------')
            continue

        tp_exs.append({
            'code': code,
            'stok_seq': stok_seq,
            'abs_paths': abs_paths,
            'rel_paths': rel_paths,
            'abs_path_map': stok_abs_path_map,
            'rel_path_map': stok_rel_path_map,
            'summary_seq': ex['summary_seq']
        })

    return tp_exs


if __name__ == '__main__':
    datasets = ['java', 'python']
    splits = ['test', 'train']

    args_parser = argparse.ArgumentParser('preprocess')
    args_parser.add_argument('--src_root', type=str, metavar="PATH", help='dataset root',
                             default='/home/chail/data/mmcodesum')
    args_parser.add_argument('--tgt_root', type=str, metavar="PATH", help='dir to cfg data',
                             default='/home/chail/data/mmcodesum')
    args_parser.add_argument('--vocab', type=str, 
        default='/home/chail/data/mmcodesum/java/non_terminal_vocab.pkl')
    args_parser.add_argument('--split', type=str, choices=splits, help='split of dataset')
    args_parser.add_argument('--dataset', type=str, choices=datasets, help='which dataset')
    args_parser.add_argument('--task', type=str, choices=['vocab', 'data'], default='data')
    args = args_parser.parse_args()

    src_jsonl = f'{args.src_root}/{args.dataset}/filter_{args.split}.jsonl'
    tgt_jsonl = f'{args.src_root}/{args.dataset}/{args.split}_tree_path.jsonl'
    vocab_path = args.vocab
    parallel_data_preprocess(gen_tree_path, src_jsonl, tgt_jsonl, args.dataset)

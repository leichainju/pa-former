""" generate sub-token sequence and statement-level asts """

import argparse
import treelib
import copy

from structure_parser import MultiLangStructureParser, get_subtoken
from vectorize_data import parallel_data_preprocess


def _gen_subtrees(ast, cfg):
    """ split ast into statement-level sub-trees"""
    # collect stm-nodes' ast idx 
    stm_ast_idxs = [ast.root]
    for node in cfg.nodes:
        node_attr: dict = cfg.nodes[node]
        ast_idx = node_attr.get('ast_idx', None)
        if ast_idx is not None:
            stm_ast_idxs.append(ast_idx)
    
    # split sub_tree
    stm_subtrees, stm_ast = {}, treelib.Tree()
    for stm_root_idx in stm_ast_idxs:
        full_sub_tree = ast.subtree(stm_root_idx)
        stm_subtree = copy.deepcopy(full_sub_tree)
        stm_root = ast.get_node(stm_root_idx)
        if stm_ast.size() == 0: # the root stm node
            stm_ast.create_node(
                tag=stm_root.tag,
                identifier=stm_root_idx,
                parent=None
            )

        # remove statements in this statement and update stm_ast
        for idf in stm_ast_idxs:
            if idf != stm_root_idx and full_sub_tree.contains(idf):
                if stm_subtree.contains(idf):
                    stm_subtree.remove_subtree(idf)
                if stm_ast.contains(idf):  # add `stm_idf -> idf` to stm_ast
                    stm_ast.remove_node(idf)
                stm_node = ast.get_node(idf)
                stm_ast.create_node(
                    tag=stm_node.tag,
                    identifier=idf,
                    parent=stm_root_idx
                )

        for n in stm_subtree.all_nodes_itr():
            n.data.update({
                'stm_tag': stm_root.tag, 
                'stm_idx': stm_root.identifier
            })

        stm_subtrees[stm_root_idx] = stm_subtree
        
    return stm_subtrees, stm_ast


def _linear_ast(ast):
    u, v = [], []
    
    ast2flatten, idx, nodes = {}, 0, []
    for n in ast.all_nodes_itr():
        ast2flatten[n.identifier] = idx
        nodes.append(n.tag)
        idx += 1

    for n in ast.all_nodes_itr():
        for cn in ast.children(n.identifier):
            u.append(ast2flatten[n.identifier])
            v.append(ast2flatten[cn.identifier])
            
    return (u, v), nodes


def _linear_stm_ast(subtrees, stm_ast):
    ast_idxs = list(subtrees)
    nodes = [None] * len(ast_idxs)
    u, v = [], []
    
    ast2flatten = {}
    for idx, ast_idx in enumerate(ast_idxs):
        ast2flatten[ast_idx] = idx
        
    for n in stm_ast.all_nodes_itr():
        nodes[ast2flatten[n.identifier]] = n.tag
        for cn in stm_ast.children(n.identifier):
            u.append(ast2flatten[n.identifier])
            v.append(ast2flatten[cn.identifier])
            
    return (u, v), nodes


def _linear_subtrees(stm_subtrees: dict):
    subtree_edges, subtree_nodes = [], []
    for subtree in stm_subtrees.values():
        edges, nodes = _linear_ast(subtree)
        subtree_edges.append(edges)
        subtree_nodes.append(nodes)
    
    return subtree_edges, subtree_nodes


def _gen_stok(ast):
    stok_seq = []

    idxs = sorted([leaf.identifier for leaf in ast.leaves()])
    for idx in idxs:
        node = ast.get_node(idx)

        # token text
        if node.tag == 'string':
            stok_seq.append('STR')
        else:
            # tok_seq.append(node.data['token'])
            stok_seq += get_subtoken(node.data['token'])

    return stok_seq


def gen_cast(examples, lang):
    parser = MultiLangStructureParser(lang)

    cast_exs = []
    for ex in examples:
        code = ex['code']
        res = parser.parse_cfg(code)
        ast, cfg = res['ast'], res['cfg']
        stm_subtrees, stm_ast = _gen_subtrees(ast, cfg)
        subtree_edges, subtree_nodes = _linear_subtrees(stm_subtrees)
        stm_ast_edges, stm_ast_nodes = _linear_stm_ast(stm_subtrees, stm_ast) 
        assert stm_ast_nodes == [st[0] for st in subtree_nodes]
        stok_seq = _gen_stok(ast)
        cast_exs.append({
            'code': code,
            'stok_seq': stok_seq,
            'subtree_edges': subtree_edges,
            'subtree_nodes': subtree_nodes,
            'stm_ast_edges': stm_ast_edges,
            'stm_ast_nodes': stm_ast_nodes,
            'summary_seq': ex['summary_seq']
        })

    return cast_exs


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
    tgt_jsonl = f'{args.src_root}/{args.dataset}/{args.split}_cast.jsonl'
    parallel_data_preprocess(gen_cast, src_jsonl, tgt_jsonl, args.dataset)

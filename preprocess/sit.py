""" generate shallow relations for stok_seq """

import argparse

from structure_parser import MultiLangStructureParser
from vectorize_data import parallel_data_preprocess


def _remove_punctuation(ast):
    for n in ast.all_nodes():
        if n.tag in '!"#$%&\'(),.:;?@[\\]^_`{|}~':
            ast.remove_node(n.identifier)
    return ast

def _gen_stm_spans(stok2stm):
    idx, stm = 0, stok2stm[0]
    stm_spans = []
    for i in range(1, len(stok2stm)):
        if stok2stm[i] != stm:
            stm_spans.append((idx, i))
            idx, stm = i, stok2stm[i]
    stm_spans.append((idx, len(stok2stm)))
    return stm_spans

def _gen_dfg_edges(ast, dfg, linear_data):
    dfg_token = set()
    for e in dfg.edges:
        edge = dfg.edges[e]
        if edge['type'] in ['LexicalUse', 'LastRead', 'LastRead']:
            dfg_token.add((e[0], e[1]))
    
    ast2tok, tok2stok = {}, {}
    for stok_idx, tok_idx in enumerate(linear_data['sub_token_to_token']):
        if tok_idx in tok2stok: 
            tok2stok[tok_idx].append(stok_idx)
        else:
            tok2stok[tok_idx] = [stok_idx]
    tok_idxs = sorted([leaf.identifier for leaf in ast.leaves()])
    for tok_idx, ast_idx in enumerate(tok_idxs):
        ast2tok[ast_idx] = tok_idx
        
    dfg_edges = []
    for (s, t) in dfg_token:
        u, v = ast2tok[s], ast2tok[t]
        us, vs = tok2stok[u], tok2stok[v]
        for u, v in zip(us, vs):
            dfg_edges.append((u, v))
    
    return dfg_edges


def _link_map_list(a2b_map, b2c_map):
    a2c_map = [b2c_map[i] for i in a2b_map]
    return a2c_map


def gen_hybrid_sbt(examples, lang):
    parser = MultiLangStructureParser(lang)

    sbt_exs = []
    for ex in examples:
        try:
            res = parser.parse_code(ex['code'])
            ast, cfg, dfg = res['ast'], res['cfg'], res['dfg']
            ast = _remove_punctuation(ast)
            stm_trees, _ = parser.split_statements(ast, cfg)
            linear_data = parser.linear_structure_data(ast, stm_trees)
            stok2stm = _link_map_list(linear_data['sub_token_to_token'], linear_data['token_to_stm'])
            stm_spans = _gen_stm_spans(stok2stm)
            dfg_edges = _gen_dfg_edges(ast, dfg, linear_data)

            sbt_exs.append({
                'code': ex['code'],
                'tok_seq': linear_data['token_seq'],
                'stok_seq': linear_data['sub_token_seq'],
                'stm_spans': stm_spans,
                'dfg_edges': dfg_edges,
                'summary_seq': ex['summary_seq'],
            })
        except Exception:
            print(ex)

    return sbt_exs


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
    tgt_jsonl = f'{args.src_root}/{args.dataset}/{args.split}_sit.jsonl'
    parallel_data_preprocess(gen_hybrid_sbt, src_jsonl, tgt_jsonl, args.dataset)

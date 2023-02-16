""" generate sbt sequence """

import argparse

from structure_parser import MultiLangStructureParser, get_subtoken
from vectorize_data import parallel_data_preprocess


map_type2idx = {
    'NextToken': 1,
    'LastRead': 2,
    'LastWrite': 3,
    'ComputeFrom': 4,
    'LexicalUse': 5,
    'LeafCFG': 6
}


def _pack_dfg_to_seq(dfg, ast):
    """
    Args:
        dfg (nx.MultiDiGraph): 
        ast (treelib.Tree): 
    """
    ast_to_flatten = {}
    tok_seq = []
    for ft, ai in enumerate(dfg.nodes):
        ast_to_flatten[ai] = ft
        tok_seq.append(ast.get_node(ai).data['token'])

    stok_to_tok, stok_seq = [], []
    for idx, tok in enumerate(tok_seq):
        stoks = get_subtoken(tok)
        stok_seq += stoks
        stok_to_tok += [idx] * len(stoks)

    edges= []
    for u, v, t in dfg.edges:
        edge = dfg.edges[u, v, t]
        u_idx = ast_to_flatten[u]
        v_idx = ast_to_flatten[v]
        type_idx = map_type2idx[edge['type']]
        edges.append((u_idx, v_idx, type_idx))
    
    return {
        'tok_seq': tok_seq,
        'stok_seq': stok_seq,
        'stok2tok': stok_to_tok,
        'edges': edges
    }


def gen_great(examples, lang):
    parser = MultiLangStructureParser(lang)

    great_exs = []
    for ex in examples:
        try:
            code = ex['code']
            res = parser.parse_code(code)
            seq_res = _pack_dfg_to_seq(res['dfg'], res['ast'])
            seq_res.update({
                'summary_seq': ex['summary_seq']
            })
            great_exs.append(seq_res)
        except Exception as e:
            print(e, ex)

    return great_exs


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
    tgt_jsonl = f'{args.src_root}/{args.dataset}/{args.split}_great.jsonl'
    parallel_data_preprocess(gen_great, src_jsonl, tgt_jsonl, args.dataset)

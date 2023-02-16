""" generate sbt sequence """

import argparse

from structure_parser import MultiLangStructureParser, get_subtoken
from vectorize_data import parallel_data_preprocess


def _gen_sbt(ast):
    def reformat(tag):
        # example: `method_declaration` -> `MethodDeclaration`
        return ''.join([tok[0].upper() + tok[1:] for tok in tag.split('_')])

    def sbt(node, seq):
        tag = reformat(node.tag)
        if node.is_leaf():
            seq += ['(', 'SimpleName', ')', 'SimpleName']
        else:
            seq.append('(')
            seq.append(tag)
            for c_node in ast.children(node.identifier):
                sbt(c_node, seq)
            seq.append(')')
            seq.append(tag)

    sbt_seq = []
    sbt(ast.get_node(ast.root), sbt_seq)

    return sbt_seq


def _gen_stok(ast):
    tok_seq, stok_seq = [], []

    idxs = sorted([leaf.identifier for leaf in ast.leaves()])
    for idx in idxs:
        node = ast.get_node(idx)

        # token text
        if node.tag == 'string':
            tok_seq.append('STR')
            stok_seq.append('STR')
        else:
            tok_seq.append(node.data['token'])
            stok_seq += get_subtoken(node.data['token'])

    return tok_seq, stok_seq


def gen_hybrid_sbt(examples, lang):
    parser = MultiLangStructureParser('java')

    sbt_exs = []
    for ex in examples:
        try:
            code = ex['code']
            ast = parser.parse_ast(code)
            sbt_seq = _gen_sbt(ast)
            tok_seq, stok_seq = _gen_stok(ast)
            sbt_exs.append({
                'tok_seq': tok_seq,
                'stok_seq': stok_seq,
                'sbt_seq': sbt_seq,
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
    tgt_jsonl = f'{args.src_root}/{args.dataset}/{args.split}_hybrid_sbt.jsonl'
    parallel_data_preprocess(gen_hybrid_sbt, src_jsonl, tgt_jsonl, args.dataset)

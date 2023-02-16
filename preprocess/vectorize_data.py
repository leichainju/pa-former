"""build vocabulary and vectorize the sub-token sequence and structure
by Lei Chai, 2021-12-15
"""

import argparse
import concurrent.futures
import jsonlines
from structure_parser import MultiLangStructureParser, get_subtoken

SPE_CHAR = ['\\', '\n', '\'', '\"', '\a', '\b', '\000', '\v', '\t', '\r', '\f']


def parser_structure_data(data_type, examples: list, lang: str) -> list:
    if data_type == 'text':
        processed_exs = process_raw(examples, lang)
    elif data_type == 'sbt':
        processed_exs = process_sbt(examples, lang)
    elif data_type == 'tree':
        processed_exs = process_tree(examples, lang)
    elif data_type == 'path':
        processed_exs = process_path(examples, lang)
    else:
        raise RuntimeError(f'Un-support {data_type}')

    return processed_exs


def process_path(examples, lang):
    parser = MultiLangStructureParser(lang)

    # parse structure and linear the parsed structure
    structure_exs = []
    for ex in examples:
        code = ex['code']
        res = parser.parse_code(code)
        cfg, ast = res['cfg'], res['ast']
        stm_trees, cfg_edges = parser.split_statements(ast, cfg)
        linear_data = parser.linear_structure_data(ast, stm_trees)
        path_data = parser.get_root_path(ast, cfg)
        assert linear_data['token_types'] == path_data['tok_seq']
        assert linear_data['stm_seq'] == [path_data['nt_seq'][i] for i in path_data['stm_in_nt']]
        data = {
            'stok_seq': linear_data['sub_token_seq'],
            'tok_seq': linear_data['token_types'],
            'nt_seq': path_data['nt_seq'],
            'stm_seq': linear_data['stm_seq'],
            'stok2tok': linear_data['sub_token_to_token'],
            'tok2stm': linear_data['token_to_stm'],
            'stm_in_nt': path_data['stm_in_nt'],
            'path_assign': path_data['path_assign'],
            'path_list': path_data['path_list'],
            'cfg_edges': cfg_edges,
            'code': code,
            'summary_seq': ex['summary_seq']
        }
        structure_exs.append(data)

    return structure_exs


def process_tree(examples, lang):
    parser = MultiLangStructureParser('java')

    # parse structure and linear the parsed structure
    structure_exs = []
    for ex in examples:
        try:
            code = ex['code']
            structure_results = parser.parse_code(code)
            ast, cfg = structure_results['ast'], structure_results['cfg']
            stm_trees, cfg_edges = parser.split_statements(ast, cfg)
            linear_data = parser.linear_structure_data(ast, stm_trees)
            leaf_seq, non_terminal_seq, edges = parser.linear_ast(ast)
            assert leaf_seq == linear_data['token_seq']
            linear_data['cfg_edges'] = cfg_edges
            linear_data['summary_seq'] = ex['summary_seq']
            linear_data['code'] = code
            linear_data['non_terminal_seq'] = non_terminal_seq
            linear_data['ast_edges'] = edges
            structure_exs.append(linear_data)
        except Exception as e:
            print(e)
            print(ex)

    return structure_exs


def process_sbt(examples, lang):
    parser = MultiLangStructureParser(lang)

    structure_exs = []
    for ex in examples:
        try:
            code = ex['code']
            ast = parser.parse_ast(code)
            sbt_seq = parser.linear_ast_sbt(ast)
            _, token_seq, _, _ = parser.linear_ast_leaves(ast)
            linear_data = {
                'token_seq': token_seq,
                'sbt_seq': sbt_seq,
                'summary_seq': ex['summary_seq']
            }
            structure_exs.append(linear_data)
        except Exception:
            print(ex)

    return structure_exs


def process_raw(examples, lang):
    parser = MultiLangStructureParser('java')

    # parse structure and linear the parsed structure
    structure_exs = []
    for ex in examples:
        try:
            code = ex['code']
            structure_results = parser.parse_code(code)
            ast, cfg = structure_results['ast'], structure_results['cfg']
            stm_trees, cfg_edges = parser.split_statements(ast, cfg)
            linear_data = parser.linear_structure_data(ast, stm_trees)
            linear_data['cfg_edges'] = cfg_edges
            linear_data['summary_seq'] = ex['summary_seq']
            linear_data['code'] = code
            structure_exs.append(linear_data)
        except Exception as e:
            print(e, ex)

    return structure_exs


def clean_string(string):
    # remove tokens like '\\', '\t', ...
    new_str = ''
    for s in string:
        if s not in SPE_CHAR:
            new_str += s
    return new_str.strip()


def tokenize_summary(summary: str):
    tokens = clean_string(summary).split()
    new_tokens = []
    for token in tokens:
        new_tokens += get_subtoken(token)
    return new_tokens


def schedule_task(task_count, process_count) -> list:
    step_len = (task_count + process_count - 1) // process_count
    task_splits = []

    for i in range(process_count):
        left_idx = i * step_len
        right_idx = min((i + 1) * step_len, task_count)
        task_splits.append((left_idx, right_idx))

    return task_splits


def parallel_data_parsing(data_type, src_jsonl, tgt_jsonl, lang, process_count=20):
    src_len, tgt_len = 0, 0
    with jsonlines.open(src_jsonl, 'r') as reader, jsonlines.open(tgt_jsonl, 'w') as writer:
        src_list = list(reader.iter())
        task_splits = schedule_task(len(src_list), process_count)

        src_len = len(src_list)
        # multi-processing running
        with concurrent.futures.ProcessPoolExecutor(max_workers=process_count) as executor:
            futures = []
            for l_idx, r_idx in task_splits:
                futures.append(executor.submit(parser_structure_data, data_type,
                                               src_list[l_idx: r_idx], lang))

        # collect the results
        for future in concurrent.futures.as_completed(futures):
            for ex in future.result():
                writer.write(ex)
                tgt_len += 1

    return src_len, tgt_len


def parallel_data_preprocess(func, src_jsonl, tgt_jsonl, lang, process_count=20):
    src_len, tgt_len = 0, 0
    with jsonlines.open(src_jsonl, 'r') as reader, jsonlines.open(tgt_jsonl, 'w') as writer:
        src_list = list(reader.iter())
        task_splits = schedule_task(len(src_list), process_count)

        src_len = len(src_list)
        # multi-processing running
        with concurrent.futures.ProcessPoolExecutor(max_workers=process_count) as executor:
            futures = []
            for l_idx, r_idx in task_splits:
                futures.append(executor.submit(func, src_list[l_idx: r_idx], lang))

        # collect the results
        for future in concurrent.futures.as_completed(futures):
            for ex in future.result():
                writer.write(ex)
                tgt_len += 1

    print(f'save the parsed data into {tgt_jsonl}({tgt_len}) from {src_jsonl}({src_len})')


if __name__ == '__main__':
    datasets = ['java', 'python', 'funcom']
    splits = ['test', 'train']

    args_parser = argparse.ArgumentParser('vectorize dataset')
    args_parser.add_argument('--type', type=str, required=True)
    args_parser.add_argument('--src_root', type=str, metavar="PATH", help='dataset root',
                             default='/home/chail/data/mmcodesum')
    args_parser.add_argument('--tgt_root', type=str, metavar="PATH", help='dir to cfg data',
                             default='/home/chail/data/mmcodesum')
    args_parser.add_argument('--split', type=str, choices=splits, help='split of dataset')
    args_parser.add_argument('--dataset', type=str, choices=datasets, help='which dataset')
    args_parser.add_argument('--process_count', type=int, help='the number of process', default=20)
    args = args_parser.parse_args()

    data_type_ = args.type
    print(f'parse {data_type_} ...')

    if args.split is not None:
        splits = [args.split]
    if args.dataset is not None:
        datasets = [args.dataset]

    for dataset in datasets:
        for split in splits:
            src_jsonl_ = f'{args.src_root}/{dataset}/{split}.jsonl'
            tgt_jsonl_ = f'{args.src_root}/{dataset}/{split}_{data_type_}.jsonl'
            src_len_, tgt_len_ = parallel_data_parsing(data_type_, src_jsonl_, tgt_jsonl_,
                                                       dataset, args.process_count)
            print(f'save the parsed data into {tgt_jsonl_}({tgt_len_}) '
                  f'from {src_jsonl_}({src_len_})')

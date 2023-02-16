# filter the dataset according to the length
# by Lei Chai, 2022-01-28

import argparse
import jsonlines


LENS = {
    'python': [320, 256, 32, 48],
    'java': [196, 160, 16, 24],
    'funcom': [256, 196, 32, 32]
}


def fit_length(lengths, ex):
    fit = len(ex['sub_token_seq']) < lengths[0]
    fit = fit and (len(ex['token_seq']) < lengths[1])
    fit = fit and (len(ex['stm_seq']) < lengths[2])
    fit = fit and (len(ex['summary_seq']) < lengths[3])
    return fit


def filter_dataset(src_path, tgt_path, lengths):
    src_count, tgt_count = 0, 0

    with jsonlines.open(src_path, 'r') as reader, jsonlines.open(tgt_path, 'w') as writer:
        for ex in reader:
            src_count += 1
            if fit_length(lengths, ex):
                tgt_count += 1
                writer.write(ex)

    print(src_count, tgt_count)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser('filter dataset')
    args_parser.add_argument('--data_root', type=str, default='/data2/chail/mmcodesum/')
    args_parser.add_argument('--dataset', type=str, default='java')
    args_parser.add_argument('--split', type=str, default='train')
    args = args_parser.parse_args()

    print(f'{args.split} dataset')
    filter_dataset(f'{args.data_root}/{args.dataset}/{args.split}_text.jsonl',
                   f'{args.data_root}/{args.dataset}/filter_{args.split}.jsonl',
                   LENS[args.dataset])

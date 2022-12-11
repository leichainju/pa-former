import os
import sys
import time
import shutil
import datetime
import argparse
import numpy as np
from alive_progress import alive_bar

import torch
from torch.backends import cudnn

from config import get_config
from logger import build_logger
from utils import AverageMeter, log_eval
from dataset import build_loaders
from evaluation import calc_metrics
from models import build_model
from optimizer import build_optimizer
from lr_scheduler import build_scheduler
from criterion import build_criterion
from engine import Engine

torch.set_num_threads(3)


def parser_options():
    args_parser = argparse.ArgumentParser('script for MMCS training and testing',
                                          add_help=False)
    args_parser.add_argument('--cfg', type=str, metavar="FILE",
                             help='path to config yaml file')

    # modify some options just using command line
    args_parser.add_argument('--restart', action='store_true')
    args_parser.add_argument('--batch_size', type=int, help='batch size for training')
    args_parser.add_argument('--lr', type=float, help='init lr for optimizer')
    args_parser.add_argument('--data_root', type=str, help='root dir to data')
    args_parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    args_parser.add_argument('--tag', type=str, help='tag for exp')
    args_parser.add_argument('--group_tag', type=str, help='tag for group exps')
    args_parser.add_argument('--cuda', type=str, help='cuda device idxs')
    args_parser.add_argument('--mode', type=str, help='`test` or `train`')
    args_parser.add_argument('--ckpt', type=str, help='path of checkpoint file')
    args_parser.add_argument('--out_dir', type=str, metavar="PATH")
    args_parser.add_argument('--no_console_out', action='store_true')
    args_parser.add_argument('--aux_w', type=float, help='weight for aux task')
    args_parser.add_argument('--warmup', type=int, help='seed for this exp')
    args_parser.add_argument('--seed', type=int, help='seed for this exp')

    args, _ = args_parser.parse_known_args()
    config = get_config(args)

    return args, config


def train_one_epoch(config, engine, data_loader, epoch):
    """Run through one epoch of model training with the provided data loader."""
    loss_meter = AverageMeter()
    perpx_meter = AverageMeter()
    norm_meter = AverageMeter()
    batch_time = AverageMeter()
    num_steps = len(data_loader)

    start = time.time()
    end = time.time()
    for idx, ex in enumerate(data_loader):
        batch_size = ex['batch_size']

        loss, perplexity, grad_norm = engine.update(ex)
        loss_meter.update(loss, batch_size)
        perpx_meter.update(perplexity, batch_size)
        norm_meter.update(grad_norm, batch_size)

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.logging.freq_print == 0:
            lr = engine.optimizer.param_groups[0]['lr']
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch:02d}/{config.train.epochs}][{idx:04d}/{num_steps:04d}] '
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f} '
                f'time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                f'loss {loss_meter.val:7.3f} ({loss_meter.avg:7.3f}) '
                f'grad_norm {norm_meter.val:7.3f} ({norm_meter.avg:7.3f}) '
                f'perplexity {perpx_meter.val:7.3f} ({perpx_meter.avg:7.3f})'
            )

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def evaluation(config, data_loader, engine, epoch=-1, mode='train', policy='greedy'):
    """ validation over the given dataset """
    logger.info(f"evaluation: {f'epoch = {epoch}' if mode != 'test' else ''} ...")

    start_time = time.time()
    examples = 0
    sources, hypotheses, references = {}, {}, {}

    with alive_bar(len(data_loader), bar='classic', title=f'eval_{config.tag}') as bar:
        for idx, ex in enumerate(data_loader):
            batch_size = ex['batch_size']
            ex_ids = list(range(idx * batch_size, (idx * batch_size) + batch_size))
            predictions, targets = engine.predict(ex, policy=policy)

            src_sequences = ex['src_text']
            examples += batch_size
            for key, src, pred, tgt in zip(ex_ids, src_sequences, predictions, targets):
                hypotheses[key] = [pred]
                references[key] = tgt if isinstance(tgt, list) else [tgt]
                sources[key] = src
            bar()

    result = calc_metrics(hypotheses, references, sources, config.logging.eval_file, mode)
    result['examples'] = examples
    result['time(s)'] = datetime.timedelta(seconds=int(time.time() - start_time))
    result['updates'] = engine.update_steps
    result['epoch'] = epoch

    return result


def main(config):
    train_loader, dev_loader, vocabs = build_loaders(config, logger)

    if config.resume:  # train from the latest checkpoint or test using the best ckpt
        engine = Engine(config, logger)
        ckpt_name = 'latest' if config.mode == 'train' else 'best'
        engine.load_checkpoint(f"{config.out_dir}/{ckpt_name}.ckpt")
    else:  # init from scratch
        model = build_model(config.model)
        model.cuda()
        optimizer = build_optimizer(config, model)
        lr_scheduler = build_scheduler(config, optimizer)
        criterion = build_criterion(config)
        engine = Engine(config, logger, model, optimizer, lr_scheduler, criterion, vocabs)

    if config.mode == 'test':
        if not config.resume:
            assert os.path.exists(config.test.ckpt_file), f'No {config.test.ckpt_file}'
            engine.load_checkpoint(config.test.ckpt_file)
        dev_loader.dataset.set_vocabs(**engine.pack_vocabs())  # MUST!!!
        greedy_eval = evaluation(config, dev_loader, engine, engine.best_epoch,
                                 mode='test', policy='greedy')
        beam_eval = evaluation(config, dev_loader, engine, engine.best_epoch,
                               mode='test', policy='beam')
        log_eval(logger, engine, (greedy_eval, beam_eval), 0)
    else:
        engine.reset_state()
        metric = config.lr_scheduler.metric
        eval_results = []
        for epoch in range(config.train.epochs):
            # train
            train_one_epoch(config, engine, train_loader, epoch)

            # evaluation
            eval_result = evaluation(config, dev_loader, engine, epoch=epoch)
            eval_results.append(eval_result)
            log_eval(logger, engine, eval_results, epoch, writer=None)

            # lr update, partly according to the evaluation results
            if config.lr_scheduler.need_input:  # for ReduceLROnPlateau
                engine.lr_scheduler.step(eval_result[metric])
            else:
                engine.lr_scheduler.step()

            # update the state of engine according to evaluation results
            engine.update_state(eval_result)
            engine.save_checkpoint(f'{config.out_dir}/latest.ckpt', epoch)

            # backup the best one
            if engine.best_epoch == epoch:
                shutil.copyfile(f'{config.out_dir}/latest.ckpt',
                                f'{config.out_dir}/best.ckpt')
                shutil.copyfile(config.logging.eval_file, f'{config.out_dir}/best.json')

            if engine.no_improvement_epochs > config.train.early_stop:
                logger.info(f'Early stop after epoch = {epoch}')
                break

        # evaluation the best one using greedy and beam search
        engine.load_checkpoint(f'{config.out_dir}/best.ckpt')
        greedy_eval = evaluation(config, dev_loader, engine, engine.best_epoch,
                                 mode='test', policy='greedy')
        beam_eval = evaluation(config, dev_loader, engine, engine.best_epoch,
                               mode='test', policy='beam')
        log_eval(logger, engine, (greedy_eval, beam_eval), 0)


if __name__ == '__main__':
    _args, _config = parser_options()

    # set cuda device
    if _args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = _args.cuda

    # set correct cuda config
    _config.defrost()
    _config.cuda &= torch.cuda.is_available()
    _config.freeze()
    assert _config.cuda, 'need cuda !'

    # set random state and cudnn
    np.random.seed(_config.random_seed)
    torch.manual_seed(_config.random_seed)
    torch.cuda.manual_seed(_config.random_seed)
    cudnn.benchmark = True

    logger = build_logger(_config.logging.log_file, _config.tag, _config.console_out)
    logger.info(f"running command: {' '.join(sys.argv)}")
    logger.info(_config.dump())

    main(_config)

    logger.info(_config.dump())

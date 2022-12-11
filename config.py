""" the meta config for this project
adopted from swin-transformer
"""

import uuid
import time
import os
import sys
import shutil
import datetime
import yaml
import importlib
from yacs.config import CfgNode


meta_config = CfgNode()
meta_config.base = ['']  # inherit cfgs from cfg files listing here
meta_config.addition = ['']  # add cfg options from files

# -----------------------------------------------------------------------------
# dataset setting
# -----------------------------------------------------------------------------
meta_config.data = CfgNode()
# the root of dataset
meta_config.data.data_dir = ''
# which type of information we will use, e.g. token, sub-token, ast, cfg, ...
meta_config.data.options = 'sub_token'
meta_config.data.lang = 'python'
meta_config.data.train_file = 'final_train.jsonl'
meta_config.data.test_file = 'final_test.jsonl'
meta_config.data.src_vocab_size = 50000
meta_config.data.tgt_vocab_size = 30000
meta_config.data.max_src_subtok_len = 320
meta_config.data.max_src_tok_len = 160
meta_config.data.max_src_stm_len = 16
meta_config.data.max_tgt_len = 48
meta_config.data.copy_attn = False

# -----------------------------------------------------------------------------
# models setting
# -----------------------------------------------------------------------------
meta_config.model = CfgNode()
meta_config.model.name = 'transformer'
meta_config.model.copy_attn = False
meta_config.model.max_seq_len = 512
meta_config.model.num_encoder_layers = 6
meta_config.model.num_decoder_layers = 6
meta_config.model.dim = 512
meta_config.model.d_ff = 2048
meta_config.model.num_heads = 8
meta_config.model.dropout = 0.1
meta_config.model.attn_dropout = 0.1
meta_config.model.ffn_activation = 'relu'
meta_config.model.pre_ln = False
meta_config.model.max_tgt_len = 48

# -----------------------------------------------------------------------------
# embedding setting
# -----------------------------------------------------------------------------
meta_config.model.embedding = CfgNode()
meta_config.model.embedding.src_pos_type = 'rel'
meta_config.model.embedding.tgt_pos_type = 'learn'  # pos type: rel, learn, abs
meta_config.model.embedding.src_vocab_size = 50000
meta_config.model.embedding.tgt_vocab_size = 30000
meta_config.model.embedding.rel_range = 32  # max distance of relative pos_embedding
meta_config.model.embedding.use_negative = True
meta_config.model.embedding.share_embedding = True
meta_config.model.embedding.dropout = 0.1

# -----------------------------------------------------------------------------
# train setting
# -----------------------------------------------------------------------------
meta_config.train = CfgNode()
meta_config.train.start_epoch = 0
meta_config.train.epochs = 50
meta_config.train.warmup_steps = 0
meta_config.train.early_stop = 10
meta_config.train.batch_size = 32
meta_config.train.num_workers = 8
meta_config.train.label_smooth_eps = 0.1
meta_config.train.shuffle = True
meta_config.train.auto_resume = True

# -----------------------------------------------------------------------------
# train.optim setting
# -----------------------------------------------------------------------------
meta_config.optim = CfgNode()
meta_config.optim.name = 'Adam'
meta_config.optim.lr = 1e-4
meta_config.optim.clip_grad = 5.0
meta_config.optim.lr_decay = 0.99
meta_config.optim.weight_decay = 0.05
meta_config.optim.eps = 1e-8
meta_config.optim.betas = (0.9, 0.999)

# -----------------------------------------------------------------------------
# train.lr_scheduler setting
# -----------------------------------------------------------------------------
meta_config.lr_scheduler = CfgNode()
meta_config.lr_scheduler.name = 'multi_step'  # cos, ada, multi-step
meta_config.lr_scheduler.steps = [10, 15]
meta_config.lr_scheduler.gamma = 0.1
meta_config.lr_scheduler.min_lr = 1e-6
meta_config.lr_scheduler.metric = 'bleu'
meta_config.lr_scheduler.mode = 'max'
meta_config.lr_scheduler.need_input = False
meta_config.lr_scheduler.patience = 0.5
meta_config.lr_scheduler.t_max = 9
meta_config.lr_scheduler.t_mult = 2

# -----------------------------------------------------------------------------
# train.criterion setting
# -----------------------------------------------------------------------------
meta_config.criterion = CfgNode()
meta_config.criterion.eps = 0.1  # label smoothing loss
meta_config.criterion.idx_ignore = 0  # pad_idx in vocab
meta_config.criterion.reduction = 'mean'
meta_config.criterion.force_copy = False

# -----------------------------------------------------------------------------
# test setting
# -----------------------------------------------------------------------------
meta_config.test = CfgNode()
meta_config.test.batch_size = 64
meta_config.test.num_workers = 12
meta_config.test.shuffle = True
meta_config.test.decoding_strategy = 'greedy'
meta_config.test.beam_size = 3
meta_config.test.main_metric = 'bleu'
meta_config.test.ckpt_file = ''

# -----------------------------------------------------------------------------
# logging setting
# -----------------------------------------------------------------------------
meta_config.logging = CfgNode()
meta_config.logging.tb_dir = 'runs'
meta_config.logging.global_tb_dir = ''
meta_config.logging.eval_file = ''
meta_config.logging.log_file = ''
meta_config.logging.freq_print = 100
meta_config.logging.freq_save = 3

# -----------------------------------------------------------------------------
# others
# -----------------------------------------------------------------------------
meta_config.tag = ''
meta_config.out_dir = ''
meta_config.mode = 'train'  # 'train'/'test'
meta_config.cuda = True
meta_config.random_seed = 233
meta_config.console_out = True
meta_config.resume = False


def custom_import(config):
    custom_imports = config.get('custom_imports')
    if custom_imports:
        for m in custom_imports:
            importlib.import_module(m)
            print(f'import {m} successfully!')


def set_defaults(config):
    config.defrost()

    # set logging files
    config.logging.log_file = f'{config.out_dir}/{config.mode}.log'
    config.logging.eval_file = f'{config.out_dir}/curr_{config.mode}.json'
    
    # sync settings between data and model
    config.data.copy_attn = config.model.copy_attn
    config.freeze()

    print('set default done!')
    return config


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('base', ['']):
        if cfg:
            cfg_path = os.path.join(os.path.dirname(cfg_file), cfg)
            _update_config_from_file(config, cfg_path)

    print(f'=> add and merge configs from {cfg_file}')
    _add_config_from_dict(config, yaml_cfg)
    config.merge_from_file(cfg_file)
    config.freeze()


def _add_config_from_file(config, cfg_file):
    # the addition option
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for addition_cfg_file in yaml_cfg.setdefault('base', ['']):
        if addition_cfg_file:
            cfg_path = os.path.join(os.path.dirname(cfg_file), addition_cfg_file)
            _add_config_from_file(config, cfg_path)
            with open(cfg_path, 'r') as f:
                addition_yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
                _add_config_from_dict(config, addition_yaml_cfg)
            print(f'=> add new config attributes from {cfg_path}')

    config.freeze()


def _add_config_from_dict(sub_config, cfg_dict):
    for key, value in cfg_dict.items():
        if isinstance(value, dict):
            sub_cfg = sub_config.setdefault(key, CfgNode())
            assert isinstance(sub_cfg, CfgNode), f'get {type(sub_cfg)}'
            _add_config_from_dict(sub_cfg, value)
        else:
            sub_config.setdefault(key, value)


def update_config(config, args):
    # _add_config_from_file(config, args.cfg)
    _update_config_from_file(config, args.cfg)

    config.defrost()

    # merge from specific arguments
    if args.batch_size:
        config.train.batch_size = args.batch_size
    if args.lr:
        config.optim.lr = args.lr
    if args.data_root:
        config.data.data_root = args.data_root
    if args.resume:
        config.resume = args.resume
    if args.out_dir:
        config.out_dir = args.out_dir
    if args.tag:
        config.tag = args.tag
    if args.group_tag:
        config.logging.global_tb_dir = args.group_tag
    if args.mode:
        config.mode = args.mode
    if args.ckpt:
        config.test.ckpt_file = args.ckpt
    if args.no_console_out:
        config.console_out = False
    if args.aux_w:
        config.model.aux_weight = args.aux_w
    if args.warmup:
        config.train.warmup_steps = args.warmup
    if args.seed:
        config.random_seed = args.seed

    # output folder
    if not config.tag:  # we use the tag to identify the running of exps
        config.tag = time.strftime("%m%d-") + str(uuid.uuid4())[:4]
    config.out_dir = f'{config.out_dir}/{config.tag}'

    # reset the tag or over-write the data
    if not config.resume and os.path.exists(config.out_dir):
        while True:
            print(f'tag conflict: {config.tag} ...')
            command = input('overwrite [Y]/n: ').lower()
            if command in ['', 'Y']:
                shutil.rmtree(config.out_dir)
                break
            elif command == 'n':
                command = input('set a new tag (`n` for exit)\n>').lower()
                if command == 'n':
                    sys.exit()
                elif len(command) > 0 and command != config.tag:
                    config.tag = command
                    config.out_dir = f'{config.out_dir}/{config.tag}'
                    print(f'set a new tag: {config.tag}')
                    break
    if config.resume and not os.path.exists(config.out_dir):
        raise RuntimeError(f'resume from empty dir: {config.out_dir}')

    os.makedirs(config.out_dir, exist_ok=True)

    config.freeze()


def get_config(args=None, cfg=None):
    """ Get a yacs CfgNode object with default values. """
    config = meta_config.clone()
    if args is not None:
        update_config(config, args)
    elif cfg is not None:
        _add_config_from_file(config, cfg)
        _update_config_from_file(config, cfg)

    print('update cfg done! ')
    return set_defaults(config)


from torch.utils.data import DataLoader, sampler

from utils import Registry
from .base_dataset import build_dataset
from .samplers import SortedBatchSampler
from .vocabulary import build_vocab


DATASETS = Registry('dataset', build_fn=build_dataset)
PIPELINES = Registry('pipeline')


def build_loaders(config, logger):
    exs_loader = PIPELINES.build(config.data.loading)
    process_pipelines = PIPELINES.build(config.data.pipeline)
    train_exs, dev_exs = exs_loader(process_pipelines, logger)

    train_loader = None
    vocabs = {}
    if config.mode == 'train':
        # build vocabulary online
        vocabs = build_vocabs_from_cfg(config, logger, train_exs+dev_exs)

        train_dataset = DATASETS.build(config.data, train_exs, **vocabs)
        train_sampler = SortedBatchSampler(
            train_dataset.lengths(),
            config.train.batch_size,
            shuffle=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.train.batch_size,
            sampler=train_sampler,
            num_workers=config.train.num_workers,
            collate_fn=train_dataset.collect_fn,
            pin_memory=config.cuda,
            drop_last=False
        )

    dev_dataset = DATASETS.build(config.data, dev_exs, **vocabs)
    dev_sampler = sampler.SequentialSampler(dev_dataset)
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.test.batch_size,
        sampler=dev_sampler,
        num_workers=config.test.num_workers,
        collate_fn=dev_dataset.collect_fn,
        pin_memory=config.cuda,
        drop_last=False
    )

    return train_loader, dev_loader, vocabs


def build_vocabs_from_cfg(config, logger, examples):
    vocabs = {}
    
    config.defrost()
    for vocab_tag, cfg in config.data.vocabs.items():
        vocab = build_vocab(examples, cfg.fields, cfg.size, cfg.no_special_token)
        config.model.embedding.update({f'{vocab_tag}_size': len(vocab)})
        logger.info(f'build {vocab_tag}, vocab size as {len(vocab)}') 
        vocabs[vocab_tag] = vocab
    config.freeze()

    return vocabs


def build_vocabs(config, examples):
    src_vocab = build_vocab(examples=examples,
                            fields=config.data.src_vocab_fields,
                            dict_size=config.data.src_vocab_size,
                            no_special_token=True)
    tgt_vocab = build_vocab(examples=examples,
                            fields=config.data.tgt_vocab_fields,
                            dict_size=config.data.tgt_vocab_size,
                            no_special_token=False)
    return {
        'src_vocab': src_vocab,
        'tgt_vocab': tgt_vocab
    }

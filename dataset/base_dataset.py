""" the dataset for multiscale dataset
"""

from abc import ABCMeta, abstractmethod

from torch.utils.data import Dataset


def build_dataset(config, examples, registry, **kwargs):
    if not config.name:
        raise RuntimeError(f'the name of the cfg for {registry.name} is needed!')

    if not isinstance(config.name, str):
        raise RuntimeError(f'the name of the cfg for {registry.name} should be str !')

    cls = registry.get(config.name)
    try:
        return cls(config, examples, **kwargs)
    except Exception as e:
        raise type(e)(f'{cls.__name__}: {e}')


class BaseDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, config, examples, vocabs=None):
        self.config = config
        self.examples = examples

        # set vocabs
        if vocabs is not None:
            for k, v in vocabs.items():
                setattr(self, k, v)
        else:
            for vocab_k in config.vocabs:
                setattr(self, vocab_k, None)
                
        assert hasattr(self, 'tgt_vocab')

    def __len__(self):
        return len(self.examples)

    def set_vocabs(self, **kwargs):
        for vocab_k in self.config.vocabs:
            setattr(self, vocab_k, kwargs[vocab_k])

    @abstractmethod
    def lengths(self):
        pass

    @staticmethod
    @abstractmethod
    def collect_fn(batch: list):
        pass

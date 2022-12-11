""" root class of `DATASETS`, `MODELS`, `PIPELINES`
"""

import inspect
from prettytable import PrettyTable

PAD, PAD_WORD = 0, '<blank>'
UNK, UNK_WORD = 1, '<unk>'
BOS, BOS_WORD = 2, '<s>'
EOS, EOS_WORD = 3, '</s>'

PUNC = '!"#$%&\'(),.:;?@[\\]^_`{|}~'


class Registry:
    """ A registry to map str to classes, registered object could be built from registry.
    This is inspired by the projection `open-mmlab/mmcv`.

    Example:
        >>> MODELS = Registry(name='models')
        >>> @MODELS.register_module()
        >>> class Transformer:
        >>>     pass
        >>> transformer = MODELS.build(config='<cfg from `config.py`>')
    """
    def __init__(self, name=None, build_fn=None):
        """
        Args:
            name (str): the registered module's key
            build_fn (func, optional): build function to construct the instance from Registry, use
                `build_from_cfg` if not provided
        """
        self._name = name
        self._module_dict = dict()

        if build_fn is not None:
            self.build_fn = build_fn
        else:
            self.build_fn = build_from_cfg

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return key in self._module_dict

    def __repr__(self):
        format_str = f'{self.__class__.__name__}(name={self._name}, ' \
                     f'items={self._module_dict})'
        return format_str

    def get(self, key: str):
        assert key in self._module_dict, f'{key} not find'
        return self._module_dict.get(key)

    def build(self, *args, **kwargs):
        return self.build_fn(*args, **kwargs, registry=self)

    def _register_module(self, module_cls, name=None):
        if not inspect.isclass(module_cls):
            raise TypeError(f'module must be a class, but got {type(module_cls)}')

        if name is None:
            name = module_cls.__name__

        if name in self._module_dict:
            raise KeyError(f'{name} is already registered in {self.name}')

        self._module_dict[name] = module_cls

    def register_module(self, name=None):
        def _register_module(cls):
            self._register_module(module_cls=cls, name=name)
            return cls
        return _register_module

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict


def build_from_cfg(config, registry):
    if not config.name:
        raise RuntimeError(f'the name of the cfg for {registry.name} is needed!')

    if not isinstance(config.name, str):
        raise RuntimeError(f'the name of the cfg for {registry.name} should be str !')

    cls = registry.get(config.name)
    try:
        return cls(config)
    except Exception as e:
        raise type(e)(f'{cls.__name__}: {e}')


def log_eval(logger, trainer, eval_results, epoch, writer=None):
    best_eval_result = trainer.best_eval_result
    best_epoch = trainer.best_epoch

    if isinstance(eval_results, dict):
        eval_results = [eval_results]

    best_values = []
    metrics_table = PrettyTable(list(eval_results[0]))

    for key, value in eval_results[0].items():
        best_value = 0 if best_eval_result is None else best_eval_result[key]
        if isinstance(value, float):
            best_values.append(f'*{best_value :.3f}')
        else:
            best_values.append(f'*{best_value}')
    metrics_table.add_row(best_values)

    for eval_result in eval_results:
        curr_values = []
        for key, value in eval_result.items():
            if isinstance(value, float):
                curr_values.append(f'{value:.3f}')
            else:
                curr_values.append(f'{value}')
        metrics_table.add_row(curr_values)

    if writer is not None:
        for key, value in eval_results[-1].items():
            if isinstance(value, float):
                writer.add_scalar(f'metric/{key}', value, epoch)

    title = f'evaluation on epoch {epoch} (best on {best_epoch})'
    logger.info(f'{title}\n' + str(metrics_table))


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def tens2sent(t, word_dict=None, src_vocabs=None):
    # batch of preds, `[batch, seq_len]`
    sentences = []
    for idx, s in enumerate(t):
        sentence = []
        for wt in s:
            word = wt if isinstance(wt, int) else wt.item()
            if word in [BOS]:
                continue
            if word in [EOS]:
                break
            if word_dict and word < len(word_dict):
                sentence += [word_dict[word]]
            elif src_vocabs:
                word = word - len(word_dict)
                sentence += [src_vocabs[idx][word]]
            else:
                sentence += [str(word)]

        if len(sentence) == 0:
            sentence = [str(PAD)]

        sentence = ' '.join(sentence)
        sentences += [sentence]

    return sentences

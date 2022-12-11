from torch import nn
from utils import Registry


MODELS = Registry(name='models')


def build_model(config):
    """ build model """
    model: nn.Module = MODELS.build(config)

    return model

""" base model """

import copy
from abc import ABCMeta, abstractmethod

import torch.nn as nn


class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, config=None):
        super(BaseModel, self).__init__()
        self.init_cfg = copy.deepcopy(config)

    @abstractmethod
    def forward(self, **kwargs):
        pass

    @abstractmethod
    def encode(self, **kwargs):
        pass

    @abstractmethod
    def step_wise_decode(self, **kwargs):
        pass

    def __repr__(self):
        s = super().__repr__()
        if self.init_cfg:
            s += f'\ninit_cfg={self.init_cfg}'
        return s

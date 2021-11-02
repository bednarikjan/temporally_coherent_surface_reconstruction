""" Sampler of points from UV space.

Author: Jan Bednarik, jan.bednarik@epfl.ch
"""

# 3rd party
import torch
import torch.nn as nn

# Python std
from abc import ABC, abstractmethod

# Project files
from tcsr.models.common import Device


class FNSampler(ABC, nn.Module, Device):
    """ Abstract base sampler class. """
    def __init__(self, gpu=True):
        ABC.__init__(self)
        nn.Module.__init__(self)
        Device.__init__(self, gpu=gpu)

    @abstractmethod
    def forward(self, *input):
        raise NotImplementedError


class FNSampler2D(FNSampler):
    """ Abstract base class for sampling the 2D parametric space.

    Args:
        gpu (bool): Whether to use GPU.
        u_range (tuple): Range of u-axis, (u_min, u_max).
        v_range (tuple): Range of v-axis, (v_min, v_max).
    """
    def __init__(self, u_range, v_range, gpu=True):
        super(FNSampler2D, self).__init__(gpu=gpu)
        self.check_range(u_range)
        self.check_range(v_range)
        self._u_range = u_range
        self._v_range = v_range

    @abstractmethod
    def forward(self, *input):
        raise NotImplementedError

    @staticmethod
    def check_range(r):
        """ Checks that the given range `r` (min, max) is a 2-tuple and
        max >= min.

        Args:
            r (tuple): 2-tuple, range, (min, max), max >= min.
        """
        assert(len(r) == 2)
        assert(r[1] >= r[0])


class FNSamplerRandUniform(FNSampler2D):
    """ Random 2D grid points generator.

    Args:
        u_range (tuple): Range of u-axis, (u_min, u_max).
        v_range (tuple): Range of v-axis, (v_min, v_max).
        num_samples (int): # samples.
        gpu (bool): Whether to use gpu.
    """
    def __init__(self, u_range, v_range, num_samples, gpu=True):
        super(FNSamplerRandUniform, self).__init__(u_range, v_range, gpu=gpu)
        self._num_samples = num_samples

    def forward(self, B, num_samples=None, u_range=None, v_range=None):
        """
        Args:
            B (int): Current batch size.

        Returns:
            torch.Tensor: Randomly sampled 2D points,
                shape (B, `num_samples`, 2).
        """
        ns = (num_samples, self._num_samples)[num_samples is None]
        ur = (u_range, self._u_range)[u_range is None]
        vr = (v_range, self._v_range)[v_range is None]

        return torch.cat(
            [torch.empty((B, ns, 1)).uniform_(*ur),
             torch.empty((B, ns, 1)).uniform_(*vr)], dim=2).to(self.device)
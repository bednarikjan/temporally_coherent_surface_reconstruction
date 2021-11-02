# 3rd party
import torch


class Device:
    """ Creates the computation device.

    Args:
        gpu (bool): Whether to use gpu.
    """
    def __init__(self, gpu=True):
        if gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            if gpu:
                print('[WARNING] cuda not available, using CPU.')


def identity(x):
    """ Implements the identity function.
    """
    return x

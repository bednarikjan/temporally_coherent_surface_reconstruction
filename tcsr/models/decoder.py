""" Multipatch decoder corresponding to the one implemented in AtlasNet [1] with
the adjustment as in [2].

[1] Groueix Thibault et al. AtlasNet: A Papier-Mâché Approach to Learning 3D
    Surface Generation. CVPR 2018.
[2] Bednarik Jan et al. Shape Reconstruction by Learning Differentiable Surface
    Representations. CVPR 2020.

Author: Jan Bednarik, jan.bednarik@epfl.ch
"""

# 3rd party
import torch
import torch.nn as nn

# Project files.
from tcsr.models.common import Device, identity


class DecoderCounter:
    """ Keeps track of the id of instantiated decoders.

    """
    glob_id = 0

    def __init__(self):
        self.id = DecoderCounter.glob_id
        DecoderCounter.glob_id += 1

    @staticmethod
    def reset():
        DecoderCounter.glob_id = 0


class DecoderMultiPatch(nn.Module):
    """ Decoder, allows for having multiple patches, This class DOES NOT
    implement `forward`, it merely stores the `ModuleList` of the decoders and
    implements the `__getitem__` operator so that the separate patches can
    be accessed by standard indexing.

    Example:
        dec = DecoderMultiPatch(num_patches=4)
        for i in range(4):
            output_i = dec[i](input_i)

    Args:
        num_patches (int): # patches.
        decoder (class nn.Module): Decoder class.
        gpu (bool): Whether to use a gpu.
        kwargs: Named arguments for the decoder.
    """
    def __init__(self, num_patches, decoder, gpu=True, **kwargs):
        nn.Module.__init__(self)
        self._patch_decs = nn.ModuleList(
            [decoder(gpu=gpu, **kwargs) for _ in range(num_patches)])

    def __getitem__(self, i):
        return self._patch_decs[i]


class DecoderAtlasNet(nn.Module, Device, DecoderCounter):
    """ A modified decoder from AtlasNet [1]. Modifications:
        - softplus instead of relu
        - last layer has linear act. func. instead of a tanh.

        [1] T. Groueix et. al. AtlasNet: A Papier-Mâché Approach to Learning
        3D Surface Generation. CVPR 2018.

    Args:
        code (int): Dim. of a codeword.
        sample_dim (int): Dim. of samples concatenated to the CW.
        batch_norm (bool): Whether to use BN layers.
        activ_fns (str): What activation fnuctions to use in all but the last
            layer. One of {'relu', 'softplus'}.
        use_tanh (bool): Whtehr to use tanh or linear as a the last activation.
        gpu (bool): Whether to use a gpu.
    """

    activation_functions = {
        'relu': nn.ReLU,
        'softplus': nn.Softplus,
        'tanh': nn.Tanh,
        'prelu': nn.PReLU,
        'silu': nn.SiLU,
        'softsign': nn.Softsign,
        'tanhshrink': nn.Tanhshrink,
    }

    def __init__(self, code=1024, sample_dim=2, batch_norm=True,
                 activ_fns='relu', use_tanh=True, gpu=True, **kwargs):
        nn.Module.__init__(self)
        Device.__init__(self, gpu=gpu)
        DecoderCounter.__init__(self)

        self._code = code

        # Store histograms of outputs of each activation layer.
        self._act_hist = False
        if 'act_hist' in kwargs and kwargs['act_hist']:
            self._act_hist = True
            self._writer = kwargs['writer']
            self._act_hist_period = kwargs['act_hist_period']

        # Conv layers.
        code = code + sample_dim  # CW is concated with `sample_dim`-D samples.
        dec_layers = [code, code // 2, code // 4]
        if 'dec_layers' in kwargs:
            assert isinstance(kwargs['dec_layers'], (tuple, list)) and \
                   len(kwargs['dec_layers']) == 3
            dec_layers = kwargs['dec_layers']
        self.conv1 = torch.nn.Conv1d(code, dec_layers[0], 1)
        self.conv2 = torch.nn.Conv1d(dec_layers[0], dec_layers[1], 1)
        self.conv3 = torch.nn.Conv1d(dec_layers[1], dec_layers[2], 1)
        self.conv4 = torch.nn.Conv1d(dec_layers[2], 3, 1)

        # Batch norm layers.
        self.bn1, self.bn2, self.bn3 = [None] * 3
        if batch_norm:
            self.bn1 = torch.nn.BatchNorm1d(dec_layers[0])
            self.bn2 = torch.nn.BatchNorm1d(dec_layers[1])
            self.bn3 = torch.nn.BatchNorm1d(dec_layers[2])

        # Activation functions.
        self.act_func_all = self.activation_functions[activ_fns]()
        self.act_func_last = nn.Tanh() if use_tanh else identity

        # Send to device.
        self = self.to(self.device)

    def forward(self, x, **kwargs):
        """
        Args:
            x (torch.Tensor): Input, shape (B, N, D).

        Returns:
            torch.Tensor: Pcloud, shape (B, N, 3).
        """

        def layer_and_stats(x, layer, bn_layer, af, label='', it=None):
            """ Runs the layer, activation function and stores the distribution
            of activations.

            Args:
                x (torch.Tensor): Layer's input, shape (B, N, D)
                layer (torch.nn.Module): A layer.
                bn_layer (torh.nn.Module): BN layer.
                af (torch.nn.Module): Activation function.
                label (str): Label for tensorboard.
                it (int): Iteration.

            Returns:
                x (torch.Tensor): Layer's output.
            """
            x = af(bn_layer(layer(x))) if bn_layer is not None else af(layer(x))
            self._save_act_hist(x, label, it)
            return x

        it = kwargs.get('it', None)

        x = x.transpose(1, 2)
        x = layer_and_stats(x, self.conv1, self.bn1, self.act_func_all,
                            label='patch{}_fc1'.format(self.id), it=it)
        x = layer_and_stats(x, self.conv2, self.bn2, self.act_func_all,
                            label='patch{}_fc2'.format(self.id), it=it)
        x = layer_and_stats(x, self.conv3, self.bn3, self.act_func_all,
                            label='patch{}_fc3'.format(self.id), it=it)
        x = layer_and_stats(x, self.conv4, None, self.act_func_last,
                            label='patch{}_fc4'.format(self.id), it=it)
        x = x.transpose(1, 2)
        return x

    def _save_act_hist(self, data, name, it):
        if self._act_hist and it and it % self._act_hist_period == 0:
            self._writer.add_histogram(
                name, data[0].flatten().detach().cpu().numpy(), it)

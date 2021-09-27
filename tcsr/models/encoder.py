""" Implementation of PointNet encoder used in AtlasNet [1].

[1] Groueix Thibault et al. AtlasNet: A Papier-Mâché Approach to Learning 3D
    Surface Generation. CVPR 2018.
"""

# 3rd party
import torch
import torch.nn as nn
import torch.nn.functional as F

# Project files.
from tcsr.train.helpers import Device


class EncoderPointNet(nn.Module, Device):
    def __init__(self, nlatent=1024, dim_input=3, batch_norm=True, gpu=True):
        """
        PointNet Encoder adapted from
        https://github.com/ThibaultGROUEIX/AtlasNet (2020-09-29).
        """
        nn.Module.__init__(self)
        Device.__init__(self, gpu=gpu)

        self._batch_norm = batch_norm
        self.dim_input = dim_input
        self.conv1 = torch.nn.Conv1d(dim_input, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, nlatent, 1)
        self.lin1 = nn.Linear(nlatent, nlatent)
        self.lin2 = nn.Linear(nlatent, nlatent)

        self.bn1, self.bn2, self.bn3, self.bn4, self.bn5 = [None] * 5

        if self._batch_norm:
            self.bn1 = torch.nn.BatchNorm1d(64)
            self.bn2 = torch.nn.BatchNorm1d(128)
            self.bn3 = torch.nn.BatchNorm1d(nlatent)
            self.bn4 = torch.nn.BatchNorm1d(nlatent)
            self.bn5 = torch.nn.BatchNorm1d(nlatent)

        self.nlatent = nlatent

        self = self.to(self.device)

    def forward(self, x):
        if self._batch_norm:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.bn3(self.conv3(x))
            x, _ = torch.max(x, 2)
            x = x.view(-1, self.nlatent)
            x = F.relu(self.bn4(self.lin1(x).unsqueeze(-1)))
            x = F.relu(self.bn5(self.lin2(x.squeeze(2)).unsqueeze(-1)))
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.conv3(x)
            x, _ = torch.max(x, 2)
            x = x.view(-1, self.nlatent)
            x = F.relu(self.lin1(x).unsqueeze(-1))
            x = F.relu(self.lin2(x.squeeze(2)).unsqueeze(-1))
        return x.squeeze(2)

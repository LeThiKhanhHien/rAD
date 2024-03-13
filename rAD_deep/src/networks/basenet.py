#----------------------------------------------------------------------------
# © 2024 – UMONS
#
# Created By  : Sukanya Patra
# Created Date: 11-Mar-2024
# version ='1.0'
# ---------------------------------------------------------------------------

import torch

from abc import abstractmethod
from torch import nn

class BaseNet(nn.Module):

    def __init__(self):
        super(BaseNet, self).__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor):
        raise NotImplementedError
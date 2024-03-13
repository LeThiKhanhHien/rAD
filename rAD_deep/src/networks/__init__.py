#----------------------------------------------------------------------------
# © 2024 – UMONS
#
# Created By  : Sukanya Patra
# Created Date: 11-Mar-2024
# version ='1.0'
# ---------------------------------------------------------------------------

from .losses import rADLoss
from .model import MLP, CNN, FashionMNIST_CNN, MNIST_CNN
from .basenet import BaseNet

def get_model(model_name: str):

    available_models = {
        'cnn': CNN,
        'mlp': MLP,
        'fmnist_cnn': FashionMNIST_CNN,
        'mnist_cnn': MNIST_CNN
    }

    return available_models[model_name]
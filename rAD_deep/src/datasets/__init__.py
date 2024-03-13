#----------------------------------------------------------------------------
# © 2024 – UMONS
#
# Created By  : Sukanya Patra
# Created Date: 11-Mar-2024
# version ='1.0'
# ---------------------------------------------------------------------------

from .cifar10 import CIFAR10_Dataset
from .mnist import MNIST_Dataset
from .fmnist import FASHIONMNIST_Dataset
from .basedataset import BaseDataset

def get_dataset(dataset_name: str):

    available_datasets = {
        'cifar10': CIFAR10_Dataset,
        'mnist': MNIST_Dataset,
        'fmnist': FASHIONMNIST_Dataset,
    }

    return available_datasets[dataset_name]


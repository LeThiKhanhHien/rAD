#----------------------------------------------------------------------------
# © 2024 – UMONS
#
# Created By  : Sukanya Patra
# Created Date: 11-Mar-2024
# version ='1.0'
# ---------------------------------------------------------------------------

import random
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from utils import Config
from .basedataset import BaseDataset
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader, random_split


class FASHIONMNIST_Dataset(BaseDataset):

    def __init__(self, config: Config):

        self.normal_classes = tuple(config.settings["normal_classes"])
        self.anomalous_classes = list(range(0, 10))
        self.anomalous_classes = list(set(self.anomalous_classes) - set(self.normal_classes))
        self.root= config.settings["data_path"]
        self.pi_n = config.settings["pi_n"]
        self.gamma_l = config.settings["gamma_l"]
        self.val_set=None
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,),
                                                            (0.3081,))])

        self.train_set = CustomFASHIONMNIST(root=self.root, train=True, download=True,
                              transform=transform, normal_classes = self.normal_classes)


        test_set = CustomFASHIONMNIST(root=self.root, train=False, download=True,
                                  transform=transform, normal_classes = self.normal_classes)

        # choose percentage of anomalous samples

        train_data, train_targets = np.array(self.train_set.data), np.array(self.train_set.targets)

        num_ano = int(self.pi_n * len(np.where(train_targets == 1)[0]))

        del_ano = len(np.where(train_targets == -1)[0]) - num_ano

        del_indices = list(np.random.choice(np.where(train_targets == -1)[0], del_ano, replace=False))

        del_indices.extend(list(np.random.choice(np.where(train_targets == 1)[0], num_ano, replace=False)))

        train_data = list(np.delete(train_data, del_indices, axis=0))
        train_targets = list(np.delete(train_targets, del_indices, axis=0))

        self.train_set.data, self.train_set.targets = train_data, train_targets

        train_targets = np.array(train_targets)


        if config.settings["early_stopping"]:
            val_size = int(0.2 * len(test_set))
            test_size = len(test_set) - val_size
            self.test_set, self.val_set = random_split(test_set, [test_size, val_size])
        else:
            self.test_set=test_set


        self.train_set_nnpu=self.train_set
        self.num_labels = int(self.gamma_l * (len(self.train_set)))

        #common_positive_label=
        ano_labelled = int(self.pi_n * self.num_labels)
        self.train_set.labelled_samples=random.sample(list(np.where(train_targets == 1)[0]),
                                                                 (self.num_labels - ano_labelled))

        self.train_set_nnpu.labelled_samples=self.train_set.labelled_samples

        #if config.settings["method"] in ["nnPU", "uPU"]:
        temp_list=list(set(np.where(train_targets == 1)[0]) - set(self.train_set.labelled_samples))
        self.train_set_nnpu.labelled_samples.extend(random.sample(temp_list,
                                                            ano_labelled))
        #else:

        self.train_set.labelled_samples.extend(random.sample(list(np.where(train_targets == -1)[0]),
                                                            ano_labelled))
        # val_size = int(0.2 * len(test_set))
        # test_size = len(test_set) - val_size
        # self.test_set, self.val_set = random_split(test_set, [test_size, val_size])
        print('Train samples: {}  Test samples: {}'
                        .format(len(self.train_set),  len(self.test_set)))


    def get_prior(self):

      pi_p = round(float(np.sum(np.array(self.train_set.targets) == 1) / len(self.train_set.targets)), 2)

      print('Positive prior: {} Negative prior: {}'.format(pi_p, 1 - pi_p))

      return pi_p


    def loaders(self, batch_size: int, shuffle=True, num_workers: int = 0):
        """Initialise data loaders

        Args:
            batch_size (int): batch size
            shuffle (bool, optional): shuffle the data samples. Defaults to True.
            num_workers (int, optional): number of concurrent workers. Defaults to 0.

        Returns:
            train_dataLoader (DataLoader): train data loader
            test_dataLoader (DataLoader): test data loader
        """

        train_dataLoader = DataLoader(self.train_set, batch_size=batch_size,
                        shuffle=shuffle, num_workers=num_workers)
        val_dataLoader = DataLoader(self.val_set, batch_size=batch_size,
                         shuffle=shuffle, num_workers=num_workers)
        test_dataLoader = DataLoader(self.test_set, batch_size=batch_size,
                         shuffle=shuffle, num_workers=num_workers)
        return train_dataLoader, val_dataLoader, test_dataLoader



class CustomFASHIONMNIST(FashionMNIST):
    """Torchvision MNIST class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, *args, normal_classes, **kwargs):
        self.normal_classes = normal_classes
        super(CustomFASHIONMNIST, self).__init__(*args, **kwargs)

        self.labelled_samples = None
        self.targets = np.array([(2 * int(x in self.normal_classes) - 1) for x in self.targets])




    def __getitem__(self, index):
        """Override the original method of the CIFAR10 class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """

        img, target = self.data[index], self.targets[index]
        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.train and (index not in self.labelled_samples):
            target = 0
        return img, target, index

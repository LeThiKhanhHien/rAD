#----------------------------------------------------------------------------
# © 2024 – UMONS
#
# Created By  : Sukanya Patra
# Created Date: 11-Mar-2024
# version ='1.0'
# ---------------------------------------------------------------------------

import torch.nn.functional as F

from torch import nn
from .basenet import BaseNet

class MLP(BaseNet):
   """
   Basic Multi-layer perceptron as described in "Positive-Unlabeled Learning with Non-Negative Risk Estimator"
   """
   def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784,300, bias=False)
        self.bn1 = nn.BatchNorm1d(300)
        self.fc2 = nn.Linear(300,300, bias=False)
        self.bn2 = nn.BatchNorm1d(300)
        self.fc3 = nn.Linear(300,300, bias=False)
        self.bn3 = nn.BatchNorm1d(300)
        self.fc4 = nn.Linear(300,300, bias=False)
        self.bn4 = nn.BatchNorm1d(300)
        self.fc5 = nn.Linear(300,1)

   def forward(self, x):

        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.fc5(x)

        return x


class CNN(BaseNet):

     def __init__(self):
          super(CNN, self).__init__()

          self.net  = nn.Sequential(
               nn.Conv2d(3, 96, 3, padding=1),
               nn.BatchNorm2d(96),
               nn.ReLU(),
               nn.Conv2d(96, 96, 3, padding=1),
               nn.BatchNorm2d(96),
               nn.ReLU(),
               nn.Conv2d(96, 96, 3, padding=1, stride=2),
               nn.BatchNorm2d(96),
               nn.ReLU(),
               nn.Conv2d(96, 192, 3, padding=1),
               nn.BatchNorm2d(192),
               nn.ReLU(),
               nn.Conv2d(192, 192, 3, padding=1),
               nn.BatchNorm2d(192),
               nn.ReLU(),
               nn.Conv2d(192, 192, 3, padding=1, stride=2),
               nn.BatchNorm2d(192),
               nn.ReLU(),
               nn.Conv2d(192, 192, 3, padding=1),
               nn.BatchNorm2d(192),
               nn.ReLU(),
               nn.Conv2d(192, 192, 1),
               nn.BatchNorm2d(192),
               nn.ReLU(),
               nn.Conv2d(192, 10, 1),
               nn.BatchNorm2d(10),
               nn.ReLU()
          )

          self.fc = nn.Sequential(
               nn.Linear(640, 1000),
               nn.ReLU(),
               nn.Linear(1000, 1000),
               nn.ReLU(),
               nn.Linear(1000, 1)
          )

     def forward(self, x):

          feat = self.net(x)
          feat = feat.view(feat.size()[0], -1)
          outputs = self.fc(feat)

          return outputs


class FashionMNIST_CNN(BaseNet):

    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 16, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(16, 32, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(32 * 7 * 7, 128, bias=False)
        self.bn1d1 = nn.BatchNorm1d(128, eps=1e-04, affine=False)
        self.fc2 = nn.Linear(128, 64, bias=False)
        self.bn1d2 = nn.BatchNorm1d(64, eps=1e-04, affine=False)
        self.fc3 = nn.Linear(64,1, bias=False)


    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = x.view(int(x.size(0)), -1)
        x = F.leaky_relu(self.bn1d1(self.fc1(x)))
        x = F.leaky_relu(self.bn1d2(self.fc2(x)))
        x = self.fc3(x)
        return x


class MNIST_CNN(BaseNet):

    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 7 * 7, 32, bias=False)
        self.bn1d1 = nn.BatchNorm1d(32, eps=1e-04, affine=False)
        self.fc2 = nn.Linear(32, 1, bias=False)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(int(x.size(0)), -1)
        x = F.leaky_relu(self.bn1d1(self.fc1(x)))
        x = self.fc2(x)
        return x

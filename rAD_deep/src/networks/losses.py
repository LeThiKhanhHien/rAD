#----------------------------------------------------------------------------
# © 2024 – UMONS
#
# Created By  : Sukanya Patra
# Created Date: 11-Mar-2024
# version ='1.0'
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np

class rADLoss(nn.Module):

    def __init__(self, prior, loss='logistic', a=0.5, risk_estimator=1):
        super(rADLoss, self).__init__()

        self.prior = prior
        self.a = a
        self.loss = loss #logistic by default
        self.positive = 1
        self.unlabeled = 0
        self.negative = -1
        self.min_count = torch.tensor(1.)
        self.var_zero = torch.tensor(0.)

        self.risk_estimator = risk_estimator

    def loss_func(self, x):
        if self.loss=='sigmoid':
           return torch.sigmoid(-x)

        elif self.loss == 'logistic':
           return  torch.nn.functional.relu(-x) + torch.log(1. +torch.exp(-torch.abs(x)))

        elif self.loss == 'squared':
            return torch.square(x-1.)/2.

        elif self.loss== 'modified_huber':
            return torch.where(x<-1, -4*x, torch.square(torch.nn.functional.relu(1.-x)))
        else:
            raise ValueError("loss should be one of 'sigmoid','logistic','squared','modified_huber' ")


    def forward(self, outputs, targets):

        positive, unlabeled, negative = targets == self.positive, targets == self.unlabeled, targets == self.negative
        positive, unlabeled, negative = positive.type(torch.float), unlabeled.type(torch.float), negative.type(torch.float)

        if outputs.is_cuda:
            self.min_count = self.min_count.cuda()
            self.prior = self.prior.cuda()
            self.var_zero = self.var_zero.cuda()

        n_p, n_u, n_n = torch.max(self.min_count, torch.sum(positive)), torch.max(self.min_count, torch.sum(unlabeled)), torch.max(self.min_count, torch.sum(negative))

        y_positive = self.loss_func(positive*outputs) * positive
        y_negative = self.loss_func(-negative*outputs) * negative
        y_negative_inv = self.loss_func(negative*outputs) * negative
        y_unlabeled = self.loss_func(unlabeled*outputs) * unlabeled

        term1 = (torch.sum(y_unlabeled) / n_u) - ((1 - self.prior) * torch.sum(y_negative_inv) / n_n)

        negative_risk = ((1 - self.prior) * torch.sum(y_negative) / n_n)

        positive_risk = (1. - self.a) * self.prior * torch.sum(y_positive) / n_p

        if self.risk_estimator == 1:
            return positive_risk + negative_risk + torch.max(self.var_zero, (self.a * term1))
        else:
            return positive_risk + negative_risk + (self.a * term1)






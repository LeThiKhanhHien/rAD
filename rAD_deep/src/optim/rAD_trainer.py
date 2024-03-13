#----------------------------------------------------------------------------
# © 2024 – UMONS
#
# Created By  : Sukanya Patra
# Created Date: 11-Mar-2024
# version ='1.0'
# ---------------------------------------------------------------------------

import logging
import time
import torch
import json
import copy
import numpy as np
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt

from utils import Config
from pathlib import Path
from torch.utils.data import DataLoader
from networks import rADLoss, get_model
from sklearn.metrics import roc_auc_score, roc_curve

class rADTrainer():

    def __init__(self, config: Config):

        self.logger = logging.getLogger(self.__class__.__name__)

        self.config = config
        self.optimizer_name = config.settings['optimizer_name']
        self.lr = config.settings['lr']
        self.n_epochs = config.settings['n_epochs']
        self.batch_size = config.settings['batch_size']
        self.weight_decay = config.settings['weight_decay']
        self.device = config.settings['device']
        self.n_jobs_dataloader = config.settings['n_jobs_dataloader']
        self.risk_estimator = config.settings['risk_estimator']

        self.lr_milestones = config.settings['lr_milestones']

        self.prior = config.settings['prior']
        self.a = config.settings['a']

        self.model_name = self.config.settings['experiment']

        # If -1 calculate true prior
        if self.prior == -1:
            self.prior = self.dataset.get_prior()

        self.prior = torch.tensor(self.prior, device=self.device)
        if "num_normal_class" not in self.config.settings:
             self.config.settings["num_normal_class"]=len(self.config.settings["normal_classes"])

        if config.settings['dataset'] == 'fmnist':
           config.settings['model']= 'fmnist_' + config.settings['model']
        elif config.settings['dataset'] == 'mnist':
            config.settings['model']= 'mnist_' + config.settings['model']

        self.net = get_model(config.settings['model'])()

        if config.settings['load_model'] or not config.settings['train']:
            self.load_model()

        # Set device for network
        self.net = self.net.to(self.device)

    def train(self, train_set, val_set=None):
        """Train rAD

        Args:
            train_set (BaseDataset): Train dataset
            val_set (BaseDataset, optional): Validation dataset. Defaults to None.

        Raises:
            ValueError: _description_
        """

        if self.config.settings["early_stopping"]:
            if val_set==None:
                raise ValueError("needs validation set for early stopping")
            else:
                val_loader=DataLoader(val_set, batch_size=self.batch_size,
                                      shuffle=False, num_workers=self.n_jobs_dataloader, drop_last=True)
                patience_cnt = 0
                min_loss = 0
                best_model = None

        train_loader = DataLoader(train_set, batch_size=self.batch_size,
                                  shuffle=True, num_workers=self.n_jobs_dataloader, drop_last=True)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        self.logger.info('Starting training...')
        start_time = time.time()
        self.net.train()


        self.logger.info('Loss is {}'.format(self.config.settings['loss']))
        loss_fn = rADLoss(prior=self.prior, loss=self.config.settings['loss'], a = self.a, risk_estimator = self.risk_estimator)

        for epoch in range(self.n_epochs):

            loss_epoch = []

            if self.config.settings["early_stopping"]:
                loss_epoch_val = []

            epoch_start_time = time.time()
            for (inputs, targets, _) in train_loader:

                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = self.net(inputs)

                loss = loss_fn(outputs.view(-1), targets)

                loss.backward()
                optimizer.step()

                loss_epoch.append(loss.item())

            scheduler.step()

            if epoch in self.lr_milestones:
                self.logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))
            epoch_loss = sum(loss_epoch) / len(loss_epoch)

            # Validation
            if self.config.settings["early_stopping"]:
              with torch.no_grad():
                 for inputs_val, targets_val, _ in val_loader:
                    inputs_val, targets_val = inputs_val.to(self.device), targets_val.to(self.device)
                    outputs_val = self.net(inputs_val)
                    loss_val = loss_fn(outputs_val.view(-1), targets_val)
                    loss_epoch_val.append(loss_val.item())

              val_loss = sum(loss_epoch_val) / len(loss_epoch_val)

            # Log epoch statistics
            epoch_train_time = time.time() - epoch_start_time

            if self.config.settings["early_stopping"]:
                self.logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}\t Val_Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, epoch_loss, val_loss))
            else:
                self.logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}\t '
                        .format(epoch + 1, self.n_epochs, epoch_train_time, epoch_loss))

            # Implement early stopping
            if self.config.settings['early_stopping']:

                if epoch == 0 :
                    min_loss = val_loss
                    best_model = copy.deepcopy(self.net)
                    continue

                if val_loss < min_loss:
                    min_loss = val_loss
                    patience_cnt = 0

                    best_model = copy.deepcopy(self.net)
                else:
                    patience_cnt +=1
                    if patience_cnt == self.config.settings['patience']:
                        self.logger.info('Training stops at {} epoch'.format(epoch+1))
                        break

        if self.config.settings['early_stopping']:
            self.net = copy.deepcopy(best_model)

        self.save_model()

        train_time = time.time() - start_time
        self.logger.info('Training time: %.3f' % train_time)
        self.logger.info('Finished training.')


    def test(self, test_set):
        """Test rAD

        Args:
            test_set (BaseDataset): Test dataset
        """

        self.net.eval()
        test_loader = DataLoader(test_set, batch_size=self.batch_size,
                                 shuffle=False, num_workers=self.n_jobs_dataloader)
        # Testing
        self.logger.info('Starting testing...')
        start_time = time.time()

        # Calculate anomaly score for test samples
        test_labels = []
        true_labels = []
        test_output = []

        self.logger.info('Loss is {}'.format(self.config.settings['loss']))
        loss_fn = rADLoss(prior=self.prior, loss=self.config.settings['loss'], a = self.a, risk_estimator = self.risk_estimator)
        test_loss = []
        correct = 0

        with torch.no_grad():
            for (inputs, targets, _) in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.net(inputs)

                loss = loss_fn(outputs.view(-1), targets)
                test_loss.append(loss.item())

                pred = torch.where(outputs < 0,
                                   torch.tensor(-1, device=self.device),
                                   torch.tensor(1, device=self.device))
                correct += pred.eq(targets.view_as(pred)).sum().item()

                # Save tuples of (idx, score) in a list
                true_labels.extend(targets.cpu().data.numpy().tolist())
                test_labels.extend(pred.view(-1).cpu().data.numpy().tolist())
                test_output.extend(outputs.view(-1).cpu().data.numpy().tolist())

        auc_score=roc_auc_score(np.array(true_labels), np.array(test_output))

        test_time = time.time() - start_time

        self.logger.info('Testing Time: {:.3f}\t Loss: {:.8f}\t AUROC {:.8f}'
                    .format(test_time, sum(test_loss)/len(test_loss), auc_score))

        self.logger.info('Finished testing.')

    def load_model(self):
        """Load model from model_path."""

        config=self.config.settings
        model_name = f'{self.model_name}_rAD{str(config["risk_estimator"])}{config["loss"]}{str(config["num_normal_class"])}_{str(config["pi_n"])}.tar'

        import_path = Path(config['model_path']).joinpath(model_name)

        self.net.load_state_dict(torch.load(import_path))

    def save_model(self):
        """Save model to export_model."""

        config=self.config.settings
        if not Path.exists(Path(config['model_path'])):
            Path.mkdir(Path(config['model_path']), parents=True, exist_ok=True)

        model_name = f'{self.model_name}_rAD{str(config["risk_estimator"])}{config["loss"]}{str(config["num_normal_class"])}_{str(config["pi_n"])}.tar'
        export_path = Path(config['model_path']).joinpath(model_name)

        torch.save(self.net.state_dict(), export_path)
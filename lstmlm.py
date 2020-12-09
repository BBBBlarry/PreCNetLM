import os

import torch
import torchvision

import pytorch_lightning as pl
import torch.nn.functional as F

from collections import OrderedDict

from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning import loggers as pl_loggers

from torchvision import transforms


class LSTMLM(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMLM, self).__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, input_size)
        self.input_size = input_size
    
    def forward(self, x, state=None):
        if not state:
            out, state = self.lstm(x)
        else:
            out, state = self.lstm(x, state)
        out = self.linear(out)
        return out, state

    def training_step(self, batch, batch_idx):
        out, states = self(batch[:, :-1, :])
        label = batch[:, 1:, :].argmax(axis=-1)
        loss = F.cross_entropy(out.view(-1, self.input_size), label.view(-1), reduction='mean')
        self.log('Loss/train', loss, self.current_epoch)
        return loss

    def validation_step(self, batch, batch_idx):
        out, states = self(batch[:, :-1, :])
        label = batch[:, 1:, :].argmax(axis=-1)
        loss = F.cross_entropy(out.view(-1, self.input_size), label.view(-1), reduction='mean')
        self.log('Loss/val', loss, self.current_epoch)
        return loss

    def test_step(self, batch, batch_idx):
        out, states = self(batch[:, :-1, :])
        label = batch[:, 1:, :].argmax(axis=-1)
        loss = F.cross_entropy(out.view(-1, self.input_size), label.view(-1), reduction='mean')
        self.log('Loss/test', loss, self.current_epoch)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.75, patience=1)
        return {
            "optimizer": optimizer, 
            "lr_scheduler": scheduler, 
            "monitor": "Loss/val"
        }

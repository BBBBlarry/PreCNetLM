import os

import torch
import torchvision

import pytorch_lightning as pl
import torch.nn.functional as F

from collections import OrderedDict

from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from harry_potter_data_util import *


class PreCNetLM(pl.LightningModule):
    def __init__(self, vocabs_size, a_hat_stack_sizes, r_stack_sizes, 
        mu, error_activation='relu', a_hat_activation='relu',
        r_unit_type='lstm', extrap_start_time=None):

        assert len(a_hat_stack_sizes) == len(r_stack_sizes)
        assert len(a_hat_stack_sizes) > 1
        assert r_unit_type in ['lstm']
        assert a_hat_activation in ['relu']
        assert error_activation in ['relu']

        super(PreCNetLM, self).__init__()

        self.num_stacks = len(a_hat_stack_sizes)

        # a nn.ModuleDict of level -> (nn.ModuleDict of unit type -> nn.Module)
        self.units = {}
        for level in range(self.num_stacks):
            self.units[str(level)] = {}
            # append R unit
            if r_unit_type == 'lstm':
                hidden_size, num_layers = r_stack_sizes[level]

                if level == 0:
                    input_size = vocabs_size
                else:
                    input_size = r_stack_sizes[level - 1][0]
                input_size *= 2

                self.units[str(level)]['r'] = nn.LSTM(input_size, hidden_size, num_layers)

            # append A_hat unit
            hidden_sizes = a_hat_stack_sizes[level]
            num_layers = len(hidden_sizes) + 1
            input_size, _ = r_stack_sizes[level]
            input_sizes = [input_size] + hidden_sizes
            if level == 0:
                hidden_sizes = hidden_sizes + [vocabs_size]
            else:
                hidden_sizes = hidden_sizes + [r_stack_sizes[level - 1][0]]

            a_hat_unit = OrderedDict()
            for i in range(num_layers):
                a_hat_unit[f'a_hat_{level}_fc_{i}'] = nn.Linear(input_sizes[i], hidden_sizes[i])
                if i < num_layers - 1:
                    if a_hat_activation == 'relu':
                        a_hat_unit[f'a_hat_{level}_activation_{i}'] = nn.ReLU()
                    
            a_hat_unit = nn.Sequential(a_hat_unit)
            self.units[str(level)]['a_hat'] = a_hat_unit
            
            # append E affline unit
            # here we just add a simple affline layer to transfor E unit's
            # output's to pass on to the lower leverl
            if level > 0:
                input_size = r_stack_sizes[level - 1][0] * 2
                output_size = r_stack_sizes[level - 2][0] * 2 if level > 1 else vocabs_size * 2
                self.units[str(level)]['e_affline'] = nn.Linear(input_size, output_size)


        self.units = {level: nn.ModuleDict(units_at_level) for level, units_at_level in self.units.items()}
        self.units = nn.ModuleDict(self.units)
        self.error_activation='relu'
        self.extrap_start_time = extrap_start_time
        self.r_unit_type = r_unit_type
        self.r_stack_sizes = r_stack_sizes

        # training hyperparams
        self.mu = mu
    
    def forward(self, x, states):

        """
        current: time step t - 1
        updated: time step t
        """
        # we want: (batch size, seq len, embedding size)
        assert len(x.shape) == 3

        batch_size = x.shape[0]
        seq_length = x.shape[1]

        for seq_idx in range(seq_length):
            input = x[:, seq_idx, :]
            states_updated = {}

            # prediction phase
            for level in reversed(range(self.num_stacks)):
                state_updated = {}

                current_state = states[level] if level in states else {}
                units = self.units[str(level)]

                # calculate r
                r_unit = units['r']
                if level == self.num_stacks - 1:
                    if 'e' in current_state:
                        e_current = current_state['e']
                    else:
                        e_current = torch.zeros(batch_size, self.r_stack_sizes[level][0])

                    r_unit_input = e_current         
                else:
                    # get the E from upper layer (remember affline)
                    e_affline_unit = self.units[str(level + 1)]['e_affline']
                    e_top_down = states_updated[level + 1]['e']
                    r_unit_input = e_affline_unit(e_top_down)
                
                r_unit_input = torch.unsqueeze(r_unit_input, 0)

                if self.r_unit_type == 'lstm':
                    if 'r_internal' in current_state:
                        r, r_internal = r_unit(r_unit_input, current_state['r_internal'])
                    else:
                        r, r_internal = r_unit(r_unit_input)
                
                r = torch.squeeze(r, 0)

                state_updated['r'] = r
                state_updated['r_internal'] = r_internal

                # calculate a_hat
                a_hat_unit = units['a_hat']
                a_hat = a_hat_unit(r)

                state_updated['a_hat'] = a_hat

                # calculate e
                if level == 0:
                    actual = input
                else:
                    if level - 1 in states and 'r' in states[level-1]:
                        actual = states[level - 1]['r']
                    else:
                        actual = torch.zeros(batch_size, r_stack_sizes[level-1][0])
                prediction = a_hat

                e = F.relu(torch.cat((prediction - actual, actual - prediction), 1))

                state_updated['e'] = e

                # save the results into a map
                states_updated[level] = state_updated
                
            # correction phase 
            for level in range(self.num_stacks):
                # calculate e
                if level > 0:
                    actual = states_updated[level - 1]['r']
                    prediction = states_updated[level]['a_hat']
                    e = F.relu(torch.cat((prediction - actual, actual - prediction), 1))

                    states_updated[level]['e'] = e

                # calculate r
                if level < self.num_stacks - 1:
                    r_unit = self.units[str(level)]['r']
                    r_unit_input = states_updated[level]['e']
                    r_unit_input = torch.unsqueeze(r_unit_input, 0)

                    if self.r_unit_type == 'lstm':
                        r, r_internal = r_unit(r_unit_input, states_updated[level]['r_internal'])
                    
                    r = torch.squeeze(r, 0)

                    states_updated[level]['r'] = r
                    states_updated[level]['r_internal'] = r_internal
                    
            states = states_updated

        return states_updated[0]['a_hat'], states_updated

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        states = {}
        _, states = self(batch, states)

        loss = 0.0
        for level in states:
            sum_e = states[level]['e'].sum()
            loss += self.mu[level] * sum_e / states[level]['e'].shape[1]

        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    SEQUENCE_LENGTH = 20
    BATCH_SIZE = 32
    TEST_BATCH_SIZE = 32
    
    train_path, test_path = prepare_data('data/harry_potter.txt')

    data_train = HarryPotterDataset(train_path, SEQUENCE_LENGTH, BATCH_SIZE)
    data_test = HarryPotterDataset(test_path, SEQUENCE_LENGTH, TEST_BATCH_SIZE)

    vocab_size = data_train.vocab_size()

    a_hat_stack_sizes=[
        [16, 16, 16], 
        [32, 32, 32], 
        [64, 128, 128]]
    r_stack_sizes=[
        (32, 3),
        (64, 3),
        (128, 3),
    ]

    mu = torch.Tensor([1.0, 0.0, 0.0])

    precnetlm = PreCNetLM(
        vocabs_size=vocab_size,
        a_hat_stack_sizes=a_hat_stack_sizes,
        r_stack_sizes=r_stack_sizes,
        mu=torch.Tensor([1.0, 0.0, 0.0])
    )
    
    # predictions, states = precnetlm(torch.rand(100, 10), states = {})
    # predictions, states = precnetlm(torch.rand(100, 10), states = states)
    # loss = precnetlm.training_step(torch.rand(100, 20, 10), 0)
    # print(loss)

    trainer = pl.Trainer(
        max_epochs=1
    )

    trainer.fit(
        precnetlm, 
        DataLoader(data_train, batch_size=BATCH_SIZE), 
        DataLoader(data_test, batch_size=TEST_BATCH_SIZE),
    )

    vocab = data_train.vocab
    states = {}
    bootstrap = 'Harry potter and the '
    res = ""

    for c in bootstrap:
        x = vocab.voc2ind[c]
        x = F.one_hot(torch.LongTensor([x]), data_train.vocab_size())
        x = torch.unsqueeze(x, 0)
        predictions, states = precnetlm(x, states = states)

    for i in range(30):
        predictions = torch.unsqueeze(predictions, 0)
        predictions, states = precnetlm(predictions, states = states)
        c = predictions.argmax().item()
        res += vocab.ind2voc[c]
    
    print(bootstrap + res)

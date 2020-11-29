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

from harry_potter_data_util import *



class PreCNetLM(pl.LightningModule):
    def __init__(self, vocabs_size, a_hat_stack_sizes, r_stack_sizes, 
        mu, error_activation='leaky_relu', a_hat_activation='leaky_relu', r_unit_type='gru'):

        assert len(a_hat_stack_sizes) == len(r_stack_sizes)
        assert len(a_hat_stack_sizes) > 1
        assert r_unit_type in ['lstm', 'gru']
        assert a_hat_activation in ['relu', 'leaky_relu']
        assert error_activation in ['relu', 'leaky_relu']

        super(PreCNetLM, self).__init__()

        self.num_stacks = len(a_hat_stack_sizes)
        self.e_sizes = {}

        # a nn.ModuleDict of level -> (nn.ModuleDict of unit type -> nn.Module)
        self.units = {}
        for level in range(self.num_stacks):
            self.units[str(level)] = {}
            # append R unit
            if r_unit_type in ['lstm', 'gru']:
                hidden_size, num_layers = r_stack_sizes[level]

                if level == 0:
                    input_size = vocabs_size
                else:
                    input_size = r_stack_sizes[level - 1][0]
                input_size *= 2

                if r_unit_type == 'lstm':
                    self.units[str(level)]['r'] = nn.LSTM(
                        input_size, 
                        hidden_size, 
                        num_layers, 
                        batch_first=True,
                    )
                elif r_unit_type == 'gru':
                    self.units[str(level)]['r'] = nn.GRU(
                        input_size, 
                        hidden_size, 
                        num_layers, 
                        batch_first=True,
                    )

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
                    elif a_hat_activation == 'leaky_relu':
                        a_hat_unit[f'a_hat_{level}_activation_{i}'] = nn.LeakyReLU()
                    
            a_hat_unit = nn.Sequential(a_hat_unit)
            self.units[str(level)]['a_hat'] = a_hat_unit

            # append E unit sizes
            self.e_sizes[level] = hidden_sizes[-1] * 2
            
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
        self.r_unit_type = r_unit_type
        self.r_stack_sizes = r_stack_sizes

        # training hyperparams
        self.mu = mu
    
    def forward(self, x, states={}, mode='train', predict_next=None):

        """
        current: time step t - 1
        updated: time step t
        """
        # we want: (batch size, seq len, embedding size)
        assert (mode == 'predict' and predict_next) or (mode == 'train' and len(x.shape) == 3)

        batch_size = x.shape[0] if mode == 'train' else 1
        seq_length = x.shape[1] if mode == 'train' else predict_next

        predictions = []
        errors = {level: [] for level in range(self.num_stacks)}

        for seq_idx in range(seq_length):
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
                        e_current = torch.zeros(
                            batch_size, 
                            self.e_sizes[level],
                        ).to(self.device)

                    r_unit_input = e_current         
                else:
                    # get the E from upper layer (remember affline)
                    e_affline_unit = self.units[str(level + 1)]['e_affline']
                    e_top_down = states_updated[level + 1]['e']
                    r_unit_input = e_affline_unit(e_top_down)
                
                r_unit_input = torch.unsqueeze(r_unit_input, 1)

                if self.r_unit_type in ['lstm', 'gru']:
                    if 'r_internal' in current_state:
                        r, r_internal = r_unit(r_unit_input, current_state['r_internal'])
                    else:
                        r, r_internal = r_unit(r_unit_input) 

                r = torch.squeeze(r, 1)

                state_updated['r'] = r
                state_updated['r_internal'] = r_internal

                # calculate a_hat
                a_hat_unit = units['a_hat']
                a_hat = a_hat_unit(r)

                # use softmax on the lowest layer
                if level == 0:
                    a_hat = F.softmax(a_hat, dim=-1)

                state_updated['a_hat'] = a_hat

                # calculate e
                if level == 0:
                    if mode == 'predict':
                        actual = a_hat
                    elif mode == 'train':
                        actual = x[:, seq_idx, :]
                else:
                    if level - 1 in states and 'r' in states[level-1]:
                        actual = states[level - 1]['r']
                    else:
                        actual = torch.zeros(batch_size, r_stack_sizes[level-1][0]).to(self.device)
                prediction = a_hat

                if self.error_activation == 'relu':
                    e = F.relu(torch.cat((prediction - actual, actual - prediction), 1))
                elif self.error_activation == 'leaky_relu':
                    e = F.leaky_relu(torch.cat((prediction - actual, actual - prediction), 1))

                state_updated['e'] = e

                # save the results into a map
                states_updated[level] = state_updated
                
            # correction phase 
            for level in range(self.num_stacks):
                # calculate e
                if level > 0:
                    actual = states_updated[level - 1]['r']
                    prediction = states_updated[level]['a_hat']

                    if self.error_activation == 'relu':
                        e = F.relu(torch.cat((prediction - actual, actual - prediction), 1))
                    elif self.error_activation == 'leaky_relu':
                        e = F.leaky_relu(torch.cat((prediction - actual, actual - prediction), 1))

                    states_updated[level]['e'] = e

                # calculate r
                if level < self.num_stacks - 1:
                    r_unit = self.units[str(level)]['r']
                    r_unit_input = states_updated[level]['e']
                    r_unit_input = torch.unsqueeze(r_unit_input, 1)

                    if self.r_unit_type in ['lstm', 'gru']:
                        r, r_internal = r_unit(r_unit_input, states_updated[level]['r_internal'])
                    
                    r = torch.squeeze(r, 1)

                    states_updated[level]['r'] = r
                    states_updated[level]['r_internal'] = r_internal

            states = states_updated
            predictions.append(states_updated[0]['a_hat'])

            for level in range(self.num_stacks):
                errors[level].append(states_updated[level]['e'])
        
        # transform lists to tensors
        predictions = torch.stack(predictions, axis=1)
        errors = {level: torch.stack(error, axis=1) for level, error in errors.items()}

        return predictions, errors, states_updated

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        _, errors, states = self(batch)

        loss = 0.0
        for level in errors:
            sum_e = errors[level][:, 1:, :].mean()
            loss += self.mu[level] * sum_e

        self.log('Loss/train', loss, self.current_epoch)

        for level in states:
            for unit in ['a_hat', 'e', 'r']:
                self.log(f'State_norm/{level}/{unit}/train', states[level][unit].norm(), self.current_epoch)
        
        return loss

    def validation_step(self, batch, batch_idx):
        _, errors, _ = self(batch)

        loss = 0.0
        for level in errors:
            sum_e = errors[level][:, 1:, :].mean()
            loss += self.mu[level] * sum_e

        self.log('Loss/val', loss, self.current_epoch)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer


if __name__ == "__main__":
    SEQUENCE_LENGTH = 40
    BATCH_SIZE = 16
    TEST_BATCH_SIZE = 16
    
    train_path, test_path = prepare_data('data/harry_potter.txt')

    data_train = HarryPotterDataset(train_path, SEQUENCE_LENGTH, BATCH_SIZE)
    data_test = HarryPotterDataset(test_path, SEQUENCE_LENGTH, TEST_BATCH_SIZE)

    vocab_size = data_train.vocab_size()

    a_hat_stack_sizes=[
        [128, 128], 
        [128, 128], 
        [128, 128], 
    ]
    r_stack_sizes=[
        (128, 1),
        (128, 1),
        (128, 1),
    ]
    mu = torch.FloatTensor([1.0, 0.01, 0.01])

    precnetlm = PreCNetLM(
        vocabs_size=vocab_size,
        a_hat_stack_sizes=a_hat_stack_sizes,
        r_stack_sizes=r_stack_sizes,
        mu=mu
    )

    print(precnetlm)

    tb_logger = pl_loggers.TensorBoardLogger('lightning_logs/')

    trainer = pl.Trainer(
        logger=tb_logger,
        gradient_clip_val=0.25,
        # weights_summary='full',
        max_epochs=10,
        log_every_n_steps=25,
        # track_grad_norm=2,
        overfit_batches=0.01,
    )

    trainer.fit(
        precnetlm, 
        DataLoader(data_train, batch_size=BATCH_SIZE), 
        DataLoader(data_test, batch_size=TEST_BATCH_SIZE),
    )

    # prepare the prompt
    vocab = data_train.vocab
    bootstrap = 'Knock, knock '
    res = ""

    x = torch.LongTensor([vocab.voc2ind[c] for c in bootstrap])
    x = torch.stack([F.one_hot(xx, data_train.vocab_size()) for xx in x])
    x = x.type(torch.FloatTensor)
    x = torch.unsqueeze(x, 0)
    
    # go through the prompt
    _, _, states = precnetlm(x)

    # predict the next characters
    predictions, _, states = precnetlm(
        None, 
        states = states,
        mode='predict',
        predict_next=30
    )

    # decode the characters
    for t in range(predictions.shape[1]):
        c = predictions[0,t,:].argmax().item()
        res += vocab.ind2voc[c]
    
    print(bootstrap + res)

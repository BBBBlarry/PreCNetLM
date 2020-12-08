import math
import re
import tqdm
import pickle
import random

import torch


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, mode, vocab_size, sequence_length, batch_size, num_batches):
        super(SimpleDataset, self).__init__()
        self.mode = mode
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_batches = num_batches

    def __len__(self):
        # return the number of unique sequences you have, not the number of characters.
        return self.batch_size * self.num_batches
        
    def __getitem__(self, idx):
        # Return the data and label for a character sequence as described above.
        # The data and labels should be torch long tensors.
        # You should return a single entry for the batch using the idx to decide which chunk you are 
        # in and how far down in the chunk you are.
        if self.mode == 'repeat_last':
            data = [self.vocab_size - 1 for _ in range(idx, idx + self.sequence_length)]
        elif self.mode == 'repeat_any':
            data = [idx % self.vocab_size for _ in range(idx, idx + self.sequence_length)]
        elif self.mode == 'sequence':
            data = [i % self.vocab_size for i in range(idx, idx + self.sequence_length)]
        elif self.mode == 'sequence_skip':
            data = [i * 2 % self.vocab_size for i in range(idx, idx + self.sequence_length)]
        elif self.mode == 'sequence_double':
            data = [[i % self.vocab_size] * 2 for i in range(idx, idx + self.sequence_length // 2)]
            data = [ii for i in data for ii in i]
        elif self.mode == 'sequence_skip_double':
            data = [[i * 2 % self.vocab_size] * 2 for i in range(idx, idx + self.sequence_length // 2)]
            data = [ii for i in data for ii in i]
        elif self.mode == 'binary_count':
            count_to = idx % (self.vocab_size - 1) + 1
            data = [count_to]
            while len(data) < self.sequence_length:
                data += [1 for _ in range(count_to)]
                data += [0 for _ in range(count_to)]
            data = data[:self.sequence_length]
        else:
            raise Exception(f'No such mode: {self.mode}')

        data = torch.nn.functional.one_hot(torch.LongTensor(data), self.vocab_size)
        return data

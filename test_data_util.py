import math
import re
import tqdm
import pickle

import torch


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, vocab_size, sequence_length, batch_size, num_batches):
        super(TestDataset, self).__init__()

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
        data = [i for i in range(idx, idx + self.sequence_length)]
        data = torch.nn.functional.one_hot(torch.LongTensor(data), self.vocab_size)
        
        return data

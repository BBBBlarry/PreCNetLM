import math
import re
import tqdm
import pickle

import torch

DATA_PATH = './data/'

class HarryPotterDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, sequence_length, batch_size):
        super(HarryPotterDataset, self).__init__()

        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.vocab = Vocabulary(data_file)

        with open(data_file, 'rb') as data_pkl:
            dataset = pickle.load(data_pkl)  # torchvision.datasets
        # {'tokens': train_text, 'ind2voc': ind2voc, 'voc2ind':voc2ind}
        # TODO: Any preprocessing on the data to get it to the right shape.
        self.voc2ind = dataset['voc2ind']
        self.ind2voc = dataset['ind2voc']
        self.tokens = dataset['tokens']
        tok_cnt = 0
        tok_length = len(self.tokens)

        # if data 1 - 18; 1 - 20; 1 - 17;
        self.chunk_length = math.ceil(tok_length / batch_size) - 1
        self.chunk_length_total = math.floor(tok_length / batch_size)
        self.num_batches = math.ceil(self.chunk_length / sequence_length)

        # calculate the offset on each chunck
        last_chunk_length = tok_length - (self.chunk_length_total * (self.batch_size - 1))
        other_chunk_length = self.chunk_length_total
        if last_chunk_length == other_chunk_length:
          self.offset = 1
        else:
          self.offset = 0
    

    def __len__(self):
        # TODO return the number of unique sequences you have, not the number of characters.
        return self.batch_size * self.num_batches
        
    def __getitem__(self, idx):
        # Return the data and label for a character sequence as described above.
        # The data and labels should be torch long tensors.
        # You should return a single entry for the batch using the idx to decide which chunk you are 
        # in and how far down in the chunk you are.
        
        """ 
        Example:
        Batch size = 2
        sequence len = 4

        (1  2   3  4)  (5  6  7  8)  (9) 10  <- chunck idx
        (11 12 13 14) (15 16 17 18) (19) 20 
             ^
             |
        sequence idx
        """

        # which chunk we are in
        chunk_idx = idx % self.batch_size
        # within the chunk, which sequence we are in
        sequence_idx = idx // self.batch_size
        
        # data
        data_first_element_index = (chunk_idx * self.chunk_length_total) + (sequence_idx * self.sequence_length)
        data_last_element_index = min(data_first_element_index + self.sequence_length, (chunk_idx + 1) * self.chunk_length_total - self.offset)
        data = self.tokens[data_first_element_index:data_last_element_index]
        data = torch.nn.functional.one_hot(torch.LongTensor(data), self.vocab_size())
        
        return data

    def vocab_size(self):
        return len(self.vocab)

def prepare_data(data_path):
    with open(data_path) as f:
        # This reads all the data from the file, but does not do any processing on it.
        data = f.read()
    
    # TODO Add more preprocessing
    data = re.sub(r'\s+', ' ', data)
    # uniq_chars = set(data)
    # print(list(uniq_chars))

    # store the corresponding char -> idx
    voc2ind = {key: idx for idx, key in enumerate(list(set(data)))}
    
    processed_data = []
    
    # Compute voc2ind and transform the data into an integer representation of the tokens.
    for char in data:
      if char in voc2ind:
        processed_data.append(voc2ind[char]) # TODO: Fill this in
    print(len(processed_data))

    ind2voc = {val: key for key, val in voc2ind.items()}

    # Naive strategy: length 6227358, 9/10 train, 1/10 test
    fold = len(processed_data) // 10
    train_text = processed_data[:9*fold] # TODO Fill this in
    test_text = processed_data[9*fold:] # TODO Fill this in

    train_path = DATA_PATH + 'harry_potter_chars_train.pkl'
    test_path = DATA_PATH + 'harry_potter_chars_test.pkl'
    
    pickle.dump({'tokens': train_text, 'ind2voc': ind2voc, 'voc2ind':voc2ind}, open(train_path, 'wb'))
    pickle.dump({'tokens': test_text, 'ind2voc': ind2voc, 'voc2ind':voc2ind}, open(test_path, 'wb'))
    
    return train_path, test_path

class Vocabulary(object):
    def __init__(self, data_file):
        with open(data_file, 'rb') as data_file:
            dataset = pickle.load(data_file)
        self.ind2voc = dataset['ind2voc']
        self.voc2ind = dataset['voc2ind']

        # these are useful for word level generation
        if 'unk' in dataset:
          self.UNK = dataset['unk']
        if 'word_level' in dataset:
          self.sep = ' '
        else:
          self.sep = ''

    # Returns a string representation of the tokens.
    def array_to_words(self, arr):
        return self.sep.join([self.ind2voc[int(ind)] for ind in arr])

    # Returns a torch tensor representing each token in words.
    def words_to_array(self, words):
        res = []
        for word in words:
          if word not in self.voc2ind:
            word = self.UNK
          res.append(self.voc2ind[word])
        return torch.LongTensor(res)

    # Returns the size of the vocabulary.
    def __len__(self):
        return len(self.voc2ind)

import math
import re
import tqdm
import pickle
import json

import torch

DATA_PATH = './data/'

class TwitterDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, batch_size, min_sequence_length = 20):
        super(TwitterDataset, self).__init__()

        self.batch_size = batch_size
        self.vocab = Vocabulary(data_file)

        with open(data_file, 'rb') as data_pkl:
            dataset = pickle.load(data_pkl)
        # {'tokens': train_text, 'ind2voc': ind2voc, 'voc2ind':voc2ind}
        self.voc2ind = dataset['voc2ind']
        self.ind2voc = dataset['ind2voc']
        self.data = sorted(dataset['data'], key=len)
        self.data = [tweet for tweet in self.data if len(tweet) >= min_sequence_length]
        self.num_samples = len(self.data)
    
    def __len__(self):
        # TODO return the number of unique sequences you have, not the number of characters.
        return self.num_samples
        
    def __getitem__(self, idx):
        batch_start_idx = idx - (idx % self.batch_size)
        batch_end_idx = batch_start_idx + self.batch_size
        
        tweets_within_batch = [tweet for tweet in self.data[batch_start_idx:batch_end_idx]]
        cut_off_length = len(min(tweets_within_batch, key=len))

        tweet = self.data[idx][:cut_off_length]
        data = torch.nn.functional.one_hot(torch.LongTensor(tweet), self.vocab_size()).type(torch.FloatTensor)
        
        return data

    def vocab_size(self):
        return len(self.vocab)

def prepare_data(data_path):
    with open(data_path) as f:
        # This reads all the data from the file, but does not do any processing on it.
        tweets = json.load(f)
    
    data = [tweet['text'] for tweet in tweets]
    data = [re.sub(r'\s+', ' ', tweet) for tweet in data]
    data = [re.sub(r'\u200b', '', tweet) for tweet in data]
    vocab = set(" ".join(data))

    # store the corresponding char -> idx
    voc2ind = {key: idx for idx, key in enumerate(list(vocab))}
    
    processed_data = []
    
    # Compute voc2ind and transform the data into an integer representation of the tokens.
    for tweet in data:
      processed_data.append([])
      for char in tweet:
        if char in voc2ind:
          processed_data[-1].append(voc2ind[char]) # TODO: Fill this in

    ind2voc = {val: key for key, val in voc2ind.items()}

    # Naive strategy: length 6227358, 9/10 train, 1/10 test
    fold = len(processed_data) // 10
    train_text = processed_data[:9*fold] # TODO Fill this in
    test_text = processed_data[9*fold:] # TODO Fill this in

    train_path = DATA_PATH + 'twitter_chars_train.pkl'
    test_path = DATA_PATH + 'twitter_chars_test.pkl'
    
    pickle.dump({'data': train_text, 'ind2voc': ind2voc, 'voc2ind':voc2ind}, open(train_path, 'wb'))
    pickle.dump({'data': test_text, 'ind2voc': ind2voc, 'voc2ind':voc2ind}, open(test_path, 'wb'))
    
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

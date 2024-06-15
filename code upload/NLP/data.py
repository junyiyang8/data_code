import os
import torch

from collections import Counter
from torch.utils.data import Dataset, DataLoader


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids[:-1]

class Custom_Dataset(Dataset):
    def __init__(self, data, transform ,length=150):
        self.data = data
        self.length = length
        self.transform=transform

    def __getitem__(self, idx):
        idx=idx*self.length
        return self.transform(self.data[idx:idx+self.length]), self.data[idx+1:idx+self.length+1]

    def __len__(self):
        return (len(self.data)-1) // self.length

def get_dataloader(data_path, batch_size, slice_length, num_workers, transform):
    SAVE_PATH=os.path.join(data_path,'data.pt')
    if(os.path.exists(SAVE_PATH)):
        corpus=torch.load(SAVE_PATH)
    else:
        corpus=Corpus(data_path)
        torch.save(corpus,SAVE_PATH)
    train_data = Custom_Dataset(corpus.train, transform, slice_length)
    val_data = Custom_Dataset(corpus.valid,transform, slice_length)
    test_data = Custom_Dataset(corpus.test,transform, slice_length)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader=DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    final_test_data=corpus.test
    final_test_loader = [(transform(final_test_data[:-1].unsqueeze(0)), final_test_data[1:].unsqueeze(0))]
    return train_loader, val_loader, test_loader, final_test_loader
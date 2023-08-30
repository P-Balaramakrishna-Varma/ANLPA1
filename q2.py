import nltk
import string
import torch
import torchtext
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from tqdm import tqdm
from math import exp as exponential
import matplotlib.pyplot as plt


# read the entrire file into a string
def get_tokens_from_text_corpus(file_path):
    # read the text from the file as a string.
    dataset_file_path = file_path
    with open(dataset_file_path, 'r') as f:
        text = f.read()

    # word tokenizer.
    tokens = nltk.word_tokenize(text)

    # lower casing
    tokens = [word.lower() for word in tokens]

    # removing puncuations
    punctuations = string.punctuation + "\u201C" + "\u201D" + "\u2019" + "\u2018"
    tokens = [word for word in tokens if word not in punctuations]

    # handling words which have a period at the end.
    tokens = [word[:-1] if word[-1] == '.' else word for word in tokens]
    return tokens


## visually check if the tokenization is good.
def test_get_tokens_from_text_corpus():
    file_path = 'Auguste_Maquet.txt'
    tokens = get_tokens_from_text_corpus(file_path)

    tokens_file_path = 'tokens.txt'
    with open(tokens_file_path, 'w') as f:
        for token in tokens:
            f.write(token)
            f.write('\n')

# test_get_tokens_from_text_corpus()


# Custom dataset
class LSTMDataset(Dataset):
    def __init__(self, filename, len) :
        super().__init__()
        # Preprocessing the text corpus
        self.tokens = get_tokens_from_text_corpus(filename)[:1000]
        
        # Loading pretrained embedding
        self.pretrained_embedding = torchtext.vocab.GloVe(name='6B', dim=300)
        
        # Creating vocabulary
        self.vocab = torchtext.vocab.build_vocab_from_iterator([[token] for token in self.tokens], min_freq=2, specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])
        
        # Max sequence length
        self.seq_len = len
        
    def __len__(self):
        return len(self.tokens) - self.seq_len
    
    def __getitem__(self, idx):
        # getting pretrained embedding for previous 5 words
        X = self.pretrained_embedding.get_vecs_by_tokens(self.tokens[idx : idx + self.seq_len])
                
        # Target (6th word)
        y =  [self.vocab[token] for token in self.tokens[idx + 1 : idx + self.seq_len + 1]]
        y = torch.tensor(y)
        return X, y
 

class ReccurentLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=300, hidden_size=500, batch_first=True)
        self.hidden2 = nn.Linear(500, vocab_size)
        self.softmax = nn.Softmax(dim=2)
 
    def forward(self, x):
        # Lstm layer with 500 dim output
        x, hidden = self.lstm(x)
        
        # converts 500 dim vector into vocab_size dim vector
        x = self.hidden2(x)
        x = self.softmax(x)
        return x


def train_loop(dataloader, model, loss_fn, optimizer, device):
    model.train()
    for X, y in tqdm(dataloader):
        # data
        X, y = X.to(device), y.to(device)
        
        # forward pass
        pred = model(X)
        y = y.reshape(-1)
        pred = pred.reshape(-1, pred.shape[2])
        loss = loss_fn(pred, y)
        
        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

 

dataset = LSTMDataset('Auguste_Maquet.txt', 5)
dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
Model = ReccurentLanguageModel(len(dataset.vocab)).to('cuda')
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(Model.parameters(), lr=0.001)

train_loop(dataloader, Model, loss_fn, optimizer, 'cuda')
import nltk
import string
import torch
from torch.utils.data.dataset import ConcatDataset, Dataset
import torchtext
from torch.utils.data import Dataset, DataLoader
from collections import Counter, OrderedDict
import torch.nn as nn
from tqdm import tqdm


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
class NGramDataset(Dataset):
    def __init__(self, filename) :
        super().__init__()
        # Preprocessing the text corpus
        self.tokens = get_tokens_from_text_corpus(filename)
        
        # Loading pretrained embedding
        self.pretrained_embedding = torchtext.vocab.GloVe(name='6B', dim=300)
        
        # Creating vocabulary
        self.vocab = torchtext.vocab.build_vocab_from_iterator([[token] for token in self.tokens], min_freq=2, specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])
        
    def __len__(self):
        return len(self.tokens) - 5
    
    def __getitem__(self, idx):
        # getting pretrained embedding for previous 5 words
        X = self.pretrained_embedding.get_vecs_by_tokens(self.tokens[idx:idx+5])
        
        # concatination
        X = torch.flatten(X)
        
        # Target (6th word)
        y =  self.vocab[self.tokens[idx + 5]]
        return X, y
    
 
class NueralLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.hidden1 = nn.Linear(300 * 5, 300)
        self.activation = nn.ReLU()
        self.hidden2 = nn.Linear(300, vocab_size)
        self.softmax = nn.Softmax(dim=1)
 
    def forward(self, x):
        # converts input into 300 dim vector
        x = self.hidden1(x)
        x = self.activation(x)
        
        # converts 300 dim vector into vocab_size dim vector
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
        loss = loss_fn(pred, y)
        
        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
       
def test_loop(dataloader, model, loss_fun, device):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            # getting data
            X, y = X.to(device), y.to(device)
            
            # forward pass
            pred = model(X)
            
            # stats
            test_loss += loss_fun(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    test_loss /= len(dataloader)
    correct /= len(dataloader.dataset)
    return test_loss, correct



        
# dataset = NGramDataset('Auguste_Maquet.txt')
# print("vocab size ", len(dataset.vocab))
# dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
# Model = NueralLanguageModel(len(dataset.vocab)).to('cuda')
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(Model.parameters(), lr=0.1)

# print(test_loop(dataloader, Model, loss_fn, "cuda"))
# train_loop(dataloader, Model, loss_fn, optimizer, "cuda")
# print(test_loop(dataloader, Model, loss_fn, "cuda"))

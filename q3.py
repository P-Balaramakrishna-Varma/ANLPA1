import nltk
import string
import torch, math
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
class GPTDataset(Dataset):
    def __init__(self, filename, seq_len) :
        super().__init__()
        # Preprocessing the text corpus
        self.tokens = get_tokens_from_text_corpus(filename)
        self.tokens = self.tokens[:(int(len(self.tokens) / 2))]

        
        # Loading pretrained embedding
        self.pretrained_embedding = torchtext.vocab.GloVe(name='6B', dim=300)
        
        # Creating vocabulary
        self.vocab = torchtext.vocab.build_vocab_from_iterator([[token] for token in self.tokens], min_freq=2, specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])
        
        # Max sequence length
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.tokens) - self.seq_len
    
    def __getitem__(self, idx):
        # getting pretrained embedding for previous 5 words
        X = self.pretrained_embedding.get_vecs_by_tokens(self.tokens[idx : idx + self.seq_len])
                
        # Target (6th word)
        y =  [self.vocab[token] for token in self.tokens[idx + 1 : idx + self.seq_len + 1]]
        y = torch.tensor(y)
        return X, y


def gen_pe(max_length, d_model):
    n = 10000
    pe = torch.zeros(max_length*d_model).reshape(max_length, d_model) 

    for k in torch.arange(max_length):
        for i in torch.arange(d_model//2):
            theta = k / (n ** ((2*i)/d_model))       
            pe[k, 2*i] = math.sin(theta) 
            pe[k, 2*i+1] = math.cos(theta)

    return pe


def gen_mask(seq_len):
    mask = torch.zeros((seq_len, seq_len)) == 0
    for i in range(seq_len):
        mask[i, :i + 1] = False
    return mask

 
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, seq_len, device):
        super().__init__()
        self.PE = gen_pe(seq_len, 300).to(device)
        self.mask = gen_mask(seq_len).to(device)
        self.attention = nn.MultiheadAttention(embed_dim=300, num_heads=3, batch_first=True)
        self.hidden2 = nn.Linear(300, vocab_size)
        self.softmax = nn.Softmax(dim=2)

 
    def forward(self, x):
        # Decoder
        x = x + self.PE
        x, hidden = self.attention(x, x, x, attn_mask=self.mask)
        
        # converts 300 dim vector into vocab_size dim vector
        x = self.hidden2(x)
        # x = self.softmax(x)
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


def test_loop(dataloader, model, loss_fun, device):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            # getting data
            X, y = X.to(device), y.to(device)
            
            # forward pass
            pred = model(X)
            y = y.reshape(-1)
            pred = pred.reshape(-1, pred.shape[2])            
            
            # stats
            test_loss += loss_fun(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    test_loss /= len(dataloader)
    try:
        peprlexity = exponential(test_loss)
    except OverflowError:
        peprlexity = float('inf')
    correct /= (len(dataloader.dataset) * X.shape[1])
    return test_loss, correct, peprlexity
 
 
def plot_stats(stats):
    x = range(len(stats))
    loss = [stat[0] for stat in stats]
    accuracy = [stat[1] for stat in stats]
    perplexity = [stat[2] for stat in stats]
    
    plt.clf()
    plt.plot(x, loss, label='loss')
    plt.savefig('g_loss.png')
    
    plt.clf()
    plt.plot(x, accuracy, label='accuracy')
    plt.savefig('g_accuracy.png')
    
    plt.clf()
    plt.plot(x, perplexity, label='perplexity')
    plt.savefig('g_perplexity.png')
  

if __name__ == "__main__":        
    # hyperparameters
    device = 'cuda'
    batch_size = 1024
    epcohs = 5
    seq_len = 15
   
    # Data creation
    dataset = GPTDataset('Auguste_Maquet.txt', seq_len)
    train_data, valid_data, test_data = random_split(dataset, [0.8, 0.1, 0.1])
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # Model creation
    Model = GPTLanguageModel(len(dataset.vocab), seq_len, device).to(device)
    
    # Training
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(Model.parameters(), lr=0.001)
    stats = []
    print(test_loop(valid_dataloader, Model, loss_fn, device))
    for epoch in tqdm(range(epcohs)):
        train_loop(train_dataloader, Model, loss_fn, optimizer, device)
        stats.append(test_loop(valid_dataloader, Model, loss_fn, device))
        print(stats[-1])
    plot_stats(stats)
    print(stats)
    
    # Testing
    Results = test_loop(test_dataloader, Model, loss_fn, device)
    print(Results)
    
    
    # Saving the model
    torch.save(Model.state_dict(), 'd_model.pt')

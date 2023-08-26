import nltk
import string
import torch
from torch.utils.data.dataset import ConcatDataset, Dataset
import torchtext
from torch.utils.data import Dataset, DataLoader
from collections import Counter, OrderedDict


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
        self.vocab = torchtext.vocab.build_vocab_from_iterator(self.tokens)
        self.vocab.set_default_index(0)
        
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
    
 
 
 
 
 
 
 
   
# dataset = NGramDataset('Auguste_Maquet.txt')    
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
# for X, y in dataloader:
#     print(X, "\n\n", y)

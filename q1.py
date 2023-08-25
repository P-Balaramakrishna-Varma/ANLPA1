import nltk
import string

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
            
            

test_get_tokens_from_text_corpus()
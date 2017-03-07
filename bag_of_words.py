from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer


def tokenize_string(raw_str):
    raw_str_lower = raw_str.lower()
    tokens = word_tokenize(raw_str_lower[:1400])  # Restrict number of tokens for testing purposes
    return tokens


def tokenize_remove_punct(raw_str):
    raw_str_lower = raw_str.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(raw_str_lower[:1400])  # Restrict number of tokens for testing purposes
    return tokens


def filter_words(tokens):
    stop_words_set = set(stopwords.words('english'))
    filtered_tokens = filter(lambda token: token not in stop_words_set, tokens)
    return list(filtered_tokens)


def stem_porter(tokens):
    ps = PorterStemmer()
    stemmed_tokens = [ps.stem(token) for token in tokens]
    return stemmed_tokens


def stem_lancaster(tokens):
    ps = LancasterStemmer()
    stemmed_tokens = [ps.stem(token) for token in tokens]
    return stemmed_tokens


def remove_short_words(tokens, threshold=2):
    filtered_tokens = filter(lambda token: len(token) > threshold, tokens)
    return list(filtered_tokens)

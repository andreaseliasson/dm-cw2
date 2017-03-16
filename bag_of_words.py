import re
import nltk
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer


THRESHOLD = 100000


def tokenize_string(raw_str):
    raw_str_lower = raw_str.lower()
    print('string length')
    print(len(raw_str_lower))
    tokens = word_tokenize(raw_str_lower[:THRESHOLD])  # Restrict number of tokens for testing purposes
    return tokens


def tokenize_remove_punct(raw_str):
    raw_str_lower = raw_str.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    print('string length')
    print(len(raw_str_lower))
    tokens = tokenizer.tokenize(raw_str_lower[::])  # Restrict number of tokens for testing purposes
    return tokens


def tokenize_incl_numbers(raw_str):
    raw_str_lower = raw_str.lower()
    tokenizer = RegexpTokenizer(r'[A-Za-z0-9]\w+')
    print('string length')
    print(len(raw_str_lower))
    tokens = tokenizer.tokenize(raw_str_lower[::])  # Restrict number of tokens for testing purposes
    return tokens


def tokenize_and_stem(text):
    stemmer = SnowballStemmer("english")
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z0-9]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def filter_words(tokens):
    stop_words_set = set(stopwords.words('english'))
    filtered_tokens = filter(lambda token: token not in stop_words_set, tokens)
    return list(filtered_tokens)


def stem_porter(tokens):
    ps = PorterStemmer()
    print(tokens)
    stemmed_tokens = [ps.stem(token) for token in tokens]
    return stemmed_tokens


def stem_lancaster(tokens):
    ps = LancasterStemmer()
    stemmed_tokens = [ps.stem(token) for token in tokens]
    return stemmed_tokens


def stem_snowball(tokens):
    ps = SnowballStemmer('english')
    stemmed_tokens = [ps.stem(token) for token in tokens]
    return stemmed_tokens


def remove_short_words(tokens, threshold=3):
    filtered_tokens = filter(lambda token: len(token) > threshold, tokens)
    return list(filtered_tokens)


def create_bag_of_words(list_of_strings):
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=50)

    features = vectorizer.fit_transform(list_of_strings)
    features = features.toarray()
    vocab = vectorizer.get_feature_names()
    print('bag of words')
    print(vocab)
    print(features[0])
    print(features[1])

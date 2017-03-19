from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from bag_of_words import tokenize_and_stem


# The original tf_idf to compare others against
def create_tf_idf(list_of_strings):
    vectorizer = TfidfVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=50)

    feature_matrix = vectorizer.fit_transform(list_of_strings)
    feature_matrix = feature_matrix.toarray()
    vocab = vectorizer.get_feature_names()
    print('tf-idf weights')
    print(vocab)
    print(feature_matrix[0])
    print(feature_matrix[1])
    return feature_matrix, vectorizer


def create_tf_idf_v2(list_of_strings):
    vectorizer = TfidfVectorizer(analyzer='word',
                                 # max_df=0.8,
                                 max_features=200000,
                                 # min_df=0.2,
                                 tokenizer=None,
                                 # ngram_range=(1, 2)
                                 )

    feature_matrix = vectorizer.fit_transform(list_of_strings)
    feature_matrix = feature_matrix.toarray()
    print(feature_matrix.shape)
    vocab = vectorizer.get_feature_names()
    print('vocab')
    print(vocab)
    print(feature_matrix[0].argsort()[::-1])
    # print('tf-idf weights')
    # print(vocab)
    # print('feature matrix')
    # print(feature_matrix)
    # print(feature_matrix[0])
    # print(feature_matrix[1])
    return feature_matrix, vectorizer


def pair_wise_cosine_similarity(docs_matrix):
    # Using L2-normalized data
    # Function is equivalent to linear-kernel
    cos_similarity = cosine_similarity(docs_matrix)
    print('cosine similarities')
    print(cos_similarity)
    return cos_similarity


def get_k_most_sign_features_for_docs(vectorizer, k):
    indices = np.argsort(vectorizer.idf_)[::-1]
    features = vectorizer.get_feature_names()
    top_features = [features[i] for i in indices[:k]]
    print('k most significant terms')
    print(top_features)
    return top_features


def get_k_most_significant_terms_for_doc(weights_vector, k, vectorizer):
    print('k top doc terms')
    indices = weights_vector.argsort()[::-1]
    features = vectorizer.get_feature_names()
    top_features = [features[i] for i in indices[:k]]
    print(top_features)
    print('')
    return top_features

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
    return feature_matrix


def pair_wise_cosine_similarity(docs_matrix):
    # Using L2-normalized data
    # Function is equivalent to linear-kernel
    cos_similarity = cosine_similarity(docs_matrix)
    print('cosine similarities')
    print(cos_similarity)
    return cos_similarity

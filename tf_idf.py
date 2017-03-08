from sklearn.feature_extraction.text import TfidfVectorizer


def create_tf_idf(list_of_strings):
    vectorizer = TfidfVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=50)

    feature_vector = vectorizer.fit_transform(list_of_strings)
    feature_vector = feature_vector.toarray()
    vocab = vectorizer.get_feature_names()
    print('tf-idf weights')
    print(vocab)
    print(feature_vector[0])
    print(feature_vector[1])

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

n_top_words = 10
n_topics = 10
n_features = 100000


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


def get_topics_nmf(data_samples):
    tfidf_vectorizer = TfidfVectorizer(
                                       # max_df=0.95,
                                       # min_df=2,
                                       max_features=n_features,
                                       stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(data_samples)
    nmf = NMF(
              n_components=2,
              ).fit(tfidf)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print(tfidf_feature_names)
    print_top_words(nmf, tfidf_feature_names, n_top_words)


def get_topics_lda(data_samples):
    # tf_vectorizer = CountVectorizer(
    #                                 # max_df=0.95,
    #                                 # min_df=0.1,
    #                                 max_features=n_features,
    #                                 stop_words='english')
    # tf = tf_vectorizer.fit_transform(data_samples)
    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.8,
        min_df=0.2,
        max_features=n_features,
        stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(data_samples)

    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    lda.fit(tfidf)
    print("\nTopics in LDA model:")
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print_top_words(lda, tfidf_feature_names, n_top_words)


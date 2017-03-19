from pre_process import *
from bag_of_words import *
from helpers import tokens_to_string
from tf_idf import *
from clustering import *
from mds import *
from topic_extraction import *

if __name__ == "__main__":
    parse_html = False
    if parse_html:
        sub_dirs = get_sub_dir('../data/gap-html')
        text_docs = get_html_files_from_sub_dirs('../data/gap-html/', sub_dirs)
        raw_docs = parse_html_files(text_docs)
        raw_docs_2 = parse_html_files_2(text_docs)
        to_csv(raw_docs)
        to_csv_2(raw_docs_2)

    raw_docs_df = pd.read_csv('../data/raw_text_2.csv', usecols=['id', 'raw_text'])
    raw_text_list = raw_docs_df['raw_text'].values.tolist()
    print(raw_text_list[0][:10])
    print(len(raw_text_list[0]))
    print(len(raw_text_list))

    tokenized_docs = []
    stemmed_docs = []

    for doc in raw_text_list[::]:  # Restrict to just the first two docs for testing purposes
        tokenized_doc = tokenize_incl_numbers(doc)
        filter_tokenized_doc = filter_words(tokenized_doc)
        filter_tokenized_doc_non_short = remove_short_words(filter_tokenized_doc)
        tokenized_docs.append(filter_tokenized_doc_non_short)

        # Stemming
        stemmed_doc = stem_porter(filter_tokenized_doc_non_short)
        stemmed_docs.append(stemmed_doc)
    # print(stemmed_docs)

    testing_list_of_strings = ['This is a 88 string.',
                               'This is another string, yes',
                               'another doc not seen',
                               'completely different stuff',
                               'different then the rest almost yup',
                               'a soldier in 500 bc fought in the roman war and was also a spartan']

    # Create a list of docs where each doc is represented as a long string with eacg stemmed word separated by a space
    docs_as_strings = tokens_to_string(stemmed_docs)
    print("docs as strings")
    # print(len(docs_as_strings[0]))
    # print(docs_as_strings)

    # Bag of words weights matrix
    # bag_of_words = create_bag_of_words(docs_as_strings)

    # Tf-Idf weights matrix. We can alternate between the testing strings (testing_list_of_strings)
    # or the real docs (docs as strings)
    # tf_idf_weights, vectorizer = create_tf_idf(testing_list_of_strings)

    # Experiment with different params for computing tf idf raw_text_list[:3]
    tf_idf_weights, vectorizer = create_tf_idf_v2(docs_as_strings)
    print('tf idf weights')
    print(tf_idf_weights)
    # tf_idf_weights, vectorizer = create_tf_idf_v2(docs_as_strings)

    # Cosine similarities
    cos_similarity = pair_wise_cosine_similarity(tf_idf_weights)

    # set dist to pass into mds
    dist = 1 - pair_wise_cosine_similarity(tf_idf_weights)

    # Get top k significant terms for all documents
    # k_most_significant_terms_for_docs = get_k_most_sign_features_for_docs(vectorizer, 3)

    # Get top k significant terms for a single doc
    k_most_sign_term_for_doc = get_k_most_significant_terms_for_doc(tf_idf_weights[0], 2, vectorizer)

    # Perform agglomerative clustering
    clusters = compute_agglomerative_clustering(3, tf_idf_weights)

    # Perform k-means clustering
    # clusters = compute_k_means_clustering(tf_idf_weights, 3)

    # Get cluster indices
    cluster_indices = get_indices_of_clusters(clusters)

    # Get word count per cluster
    word_count_per_cluster = get_word_count_per_cluster(cluster_indices, vectorizer, tf_idf_weights)

    # Draw dendrogram
    draw_dendrogram(tf_idf_weights, dist)

    # Apply multi-dimensional scaling (MDS) and plot the result
    mds_pos = apply_mds(tf_idf_weights, 2, 'precomputed')
    # mds_pos = apply_mds(tf_idf_weights, 2)
    plot_mds(mds_pos)

    # Plot the points with their cluster labels
    plot_mds_with_cluster_labels(mds_pos, cluster_indices)

    # Extract topics
    # test_doc = [testing_list_of_strings[5]]
    # test_doc = [raw_text_list[0]]
    # get_topics_lda(raw_text_list)
    # get_topics_nmf(test_doc)

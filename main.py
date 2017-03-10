from pre_process import *
from bag_of_words import *
from helpers import tokens_to_string
from tf_idf import *
from clustering import *
from mds import *

if __name__ == "__main__":
    parse_html = False
    if parse_html:
        sub_dirs = get_sub_dir('../data/gap-html')
        text_docs = get_html_files_from_sub_dirs('../data/gap-html', sub_dirs)
        raw_docs = parse_html_files(text_docs)
        to_csv(raw_docs)

    raw_docs_df = pd.read_csv('../data/raw_text.csv', usecols=['id', 'raw_text'])
    raw_text_list = raw_docs_df['raw_text'].values.tolist()
    print(raw_text_list[0][:10])
    print(len(raw_text_list[0]))
    print(len(raw_text_list))

    tokenized_docs = []
    stemmed_docs = []

    for doc in raw_text_list[:3]:  # Restrict to just the first two docs for testing purposes
        tokenized_doc = tokenize_remove_punct(doc)
        filter_tokenized_doc = filter_words(tokenized_doc)
        filter_tokenized_doc_non_short = remove_short_words(filter_tokenized_doc)
        tokenized_docs.append(filter_tokenized_doc_non_short)

        # Stemming
        stemmed_doc = stem_porter(filter_tokenized_doc_non_short)
        stemmed_docs.append(stemmed_doc)
    print(stemmed_docs)

    testing_list_of_strings = ['This is a string', 'This is another string string string yes', 'another doc', 'different stuff']

    docs_as_strings = tokens_to_string(stemmed_docs)

    # Bag of words weights matrix
    bag_of_words = create_bag_of_words(docs_as_strings)

    # Tf-Idf weights matrix
    tf_idf_weights, vectorizer = create_tf_idf(testing_list_of_strings)

    # Cosine similarities
    cos_similarity = pair_wise_cosine_similarity(tf_idf_weights)

    # Get top k significant terms for all documents
    k_most_significant_terms_for_docs = get_k_most_sign_features_for_docs(vectorizer, 3)

    # Get top k significant terms for a single doc
    k_most_sign_term_for_doc = get_k_most_significant_terms_for_doc(tf_idf_weights[0], 3, vectorizer)

    # Perform agglomerative clustering
    clusters = compute_agglomerative_clustering(2, 'average', 'cosine', tf_idf_weights)

    # Get cluster indices
    cluster_indices = get_indices_of_clusters(clusters)

    # Get word count per cluster
    word_count_per_cluster = get_word_count_per_cluster(cluster_indices, vectorizer, tf_idf_weights)

    # Draw dendrogram
    # draw_dendrogram(tf_idf_weights)

    # Apply multi-dimensional scaling (MDS) and plot the result
    mds_pos = apply_mds(tf_idf_weights, 2, 'precomputed')
    plot_mds(mds_pos)

    # Plot the points with their cluster labels
    plot_mds_with_cluster_labels(mds_pos, cluster_indices)

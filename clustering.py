from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy
from tf_idf import *


def compute_agglomerative_clustering(n_clusters, linkage, affinity, weights_matrix):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity=affinity)
    model.fit(weights_matrix)
    print(model.labels_)
    return model.labels_


def get_indices_of_clusters(cluster_labels):
    cluster_indices = []
    n_clusters = set(cluster_labels)
    for cluster in n_clusters:
        cluster_indices.append([i for i, x in enumerate(cluster_labels) if x == cluster])
    print('cluster indices')
    print(cluster_indices)
    return cluster_indices


def get_word_count_per_cluster(cluster_indices, vectorizer, weights_matrix):
    cluster_word_count = []
    for cluster in cluster_indices:
        word_count = {}
        for index in cluster:
            # Get most significant terms for each doc in the cluster
            cluster_words = get_k_most_significant_terms_for_doc(weights_matrix[index], 20, vectorizer)

            # Count the occurrences of each word throughout all docs in the cluster
            for word in cluster_words:
                if word not in word_count:
                    word_count[word] = 1
                else:
                    word_count[word] += 1
        cluster_word_count.append(word_count)
    print('cluster word count')
    print(cluster_word_count)
    return cluster_word_count


def draw_dendrogram(weights_matrix, linkage='average', affinity='cosine'):
    print('dendrogram')
    Z = hierarchy.linkage(weights_matrix, linkage, affinity)
    plt.figure()
    plt.title('Hierarchical Clustering Dendrogram')
    plt.ylabel('document index')
    plt.xlabel('distance')
    hierarchy.dendrogram(Z, orientation='right')
    plt.show()

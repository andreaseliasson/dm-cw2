from sklearn.cluster import AgglomerativeClustering


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

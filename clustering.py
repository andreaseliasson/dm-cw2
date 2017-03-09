from sklearn.cluster import AgglomerativeClustering


def compute_agglomerative_clustering(n_clusters, linkage, affinity, weights_matrix):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity=affinity)
    model.fit(weights_matrix)
    print(model.labels_)
    return model.labels_

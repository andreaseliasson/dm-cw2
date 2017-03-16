from sklearn import manifold
from sklearn.metrics import euclidean_distances
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from colors import UNIQUE_COLORS


def apply_mds(feature_matrix, dimensions=2, dissimilarity_measure='euclidean', metric=True):
    # we can play around with different parameters here, e.g. metrics, seed, dissimilarity
    if dissimilarity_measure == 'precomputed':
        # similarities = euclidean_distances(feature_matrix)
        similarities = 1 - cosine_similarity(feature_matrix)
        print('distance similarities')
        print(similarities)
        mds = manifold.MDS(metric=metric, n_components=dimensions, max_iter=3000, eps=1e-9,
                           dissimilarity='precomputed')
        pos = mds.fit(similarities).embedding_
        print(pos)
    else:
        mds = manifold.MDS(metric=metric, n_components=dimensions, max_iter=3000, eps=1e-9,
                           dissimilarity='euclidean')
        pos = mds.fit(feature_matrix).embedding_
    return pos


# TO-DO: Plot using the cluster labels/indices
def plot_mds(pos):
    np_pos = np.array(pos)

    fig, ax = plt.subplots()
    ax.scatter(np_pos[:, 0], np_pos[:, 1])
    for i, n in enumerate(range(len(np_pos))):
        ax.annotate('doc ' + str(i), (np_pos[:, 0][i], np_pos[:, 1][i]))

    plt.title('Multidimensional Scaling')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


def plot_mds_with_cluster_labels(pos, cluster_indices):
    np_pos = np.array(pos)
    fig, ax = plt.subplots()
    distinct_colors = UNIQUE_COLORS
    for idx, cluster in enumerate(cluster_indices):
        x_cluster = []
        y_cluster = []
        for i in cluster:
            x_cluster.append(np_pos[:, 0][i])
            y_cluster.append(np_pos[:, 1][i])
        ax.scatter(x_cluster, y_cluster, color=distinct_colors[idx])

    # Add doc labels
    for i, n in enumerate(range(len(np_pos))):
        ax.annotate('doc ' + str(i), (np_pos[:, 0][i], np_pos[:, 1][i]))

    plt.title('Multidimensional Scaling with doc labels and cluster colors')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

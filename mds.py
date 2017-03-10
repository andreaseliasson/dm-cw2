from sklearn import manifold
from sklearn.metrics import euclidean_distances
from matplotlib import pyplot as plt
import numpy as np


def apply_mds(feature_matrix, dimensions=2, dissimilarity_measure='euclidean', metric=True):
    # we can play around with different parameters here, e.g. metrics, seed, dissimilarity
    if dissimilarity_measure == 'precomputed':
        similarities = euclidean_distances(feature_matrix)
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

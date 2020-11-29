import numpy as np
import sklearn.datasets
from .. import cluster

def generate_data(num_clusters=5, points_per_cluster=1000, isotropic=True):
    cluster_centers = []
    for i in range(num_clusters):
        cluster_centers.append(np.random.randint(0, 10, size=2))
    print('Cluster Centers: ', cluster_centers)

    i = 0
    data = []
    labels = []
    for c in cluster_centers:
        if isotropic:
            cov = np.eye(2)*0.5
        else:
            cov = sklearn.datasets.make_spd_matrix(n_dim=2)

        data.append(np.random.multivariate_normal(c, cov, size=points_per_cluster))
        labels.append(np.ones(points_per_cluster, dtype=int)*i)
        i += 1
    data = np.array(data).reshape(num_clusters*points_per_cluster,2)
    labels = np.array(labels).reshape(-1)
    return data, labels, np.array(cluster_centers)

def test_clustering():
    data, _, _ = generate_data()

    num_clusters = 5
    print('k_means with ' + str(num_clusters) + ' clusters.')
    k_labels, k_centers = cluster.k_means_clustering(data, num_clusters)
    assert len(k_labels) == num_clusters
    assert len(k_labels) == k_centers.shape[0]

    print('Optics.')
    optics_labels, optics_centers = cluster.optics_clustering(data)
    assert len(optics_labels) == optics_centers.shape[0]

    print('Birch subclusters.')
    birch_labels, birch_centers = cluster.birch_clustering(data, 'none')
    assert len(birch_labels) == birch_centers.shape[0]

    print('Birch agglomorative with ' + str(num_clusters) + ' clusters.')
    birch_labels_1, birch_centers_1 = cluster.birch_clustering(data, 'agglomorate', num_clusters=5)
    assert len(birch_labels_1) == birch_centers_1.shape[0]

    print('Birch k_means with ' + str(num_clusters) + ' clusters.')
    birch_labels_2, birch_centers_2 = cluster.birch_clustering(data, 'k_means', num_clusters=5)
    assert len(birch_labels_2) == birch_centers_2.shape[0]

    print('Birch optics.')
    birch_labels_3, birch_centers_3 = cluster.birch_clustering(data, 'optics')
    assert len(birch_labels_3) == birch_centers_3.shape[0]

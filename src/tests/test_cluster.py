import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
from .. import cluster


def generate_data(center_range=100, num_clusters=5, points_per_cluster=1000, isotropic=True):
    cluster_centers = []
    for i in range(num_clusters):
        cluster_centers.append(np.random.randint(-1*center_range, center_range+1, size=2))
    print('Cluster Centers: ', cluster_centers)

    i = 0
    data = []
    labels = []
    for c in cluster_centers:
        if isotropic:
            cov = np.eye(2)*np.ceil(center_range/10)
        else:
            cov = sklearn.datasets.make_spd_matrix(n_dim=2)*np.ceil(center_range/10)

        data.append(np.random.multivariate_normal(c, cov, size=points_per_cluster))
        labels.append(np.ones(points_per_cluster, dtype=int)*i)
        i += 1
    data = np.array(data).reshape(num_clusters*points_per_cluster, 2)
    labels = np.array(labels).reshape(-1)
    return data, labels, np.array(cluster_centers)


def plot_clusters(data, method, labels, centers, limit):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()
    ax.set_aspect('equal')
    print(limit)
    ax.set_xlim(-1*limit, limit)
    ax.set_ylim(-1*limit, limit)
    X = data[:, 0]
    Y = data[:, 1]

    num_clusters = np.max(labels) + 1
    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(1.*i/num_clusters) for i in range(num_clusters)]

    if len(X[labels == -1]):
        plt.scatter(X[labels == -1], Y[labels == -1], color='k')

    for i in range(num_clusters):
        if len(X[labels == i]):
            plt.scatter(X[labels == i], Y[labels == i], color=colors[i])

    if len(X[labels == -1]):
        outlier = np.mean(data[labels == -1, :], axis=0)
        plt.scatter(outlier[0], outlier[1], color='k', marker="*", label="Cluster -1")

    for i, center in enumerate(centers):
        plt.scatter(center[0], center[1], color=colors[i], marker="*", label="Cluster " + str(i))
    if num_clusters < 10:
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.title('Method: ' + str(method) + ', Num. Clusters: ' + str(num_clusters))
    plt.show(block=False)


def run_clustering(plot = False, center_range = 10, isotropic=True):
    data, true_labels, true_centers = generate_data(center_range=center_range, isotropic=isotropic)
    limit = np.max(np.abs(data)) * 1.1
    if plot:
        plot_clusters(data, 'True', true_labels, true_centers, limit)

    num_clusters = 5
    print('k_means with ' + str(num_clusters) + ' clusters.')
    labels, centers = cluster.k_means_clustering(data, num_clusters, num_iter=500, tol=1.e-8)
    if plot:
        plot_clusters(data, 'K-means', labels, centers, limit)
    assert max(labels) + 1 == centers.shape[0]
    assert len(labels) == data.shape[0]

    print('Optics.')
    labels, centers = cluster.optics_clustering(data, metric='cosine', min_samples=0.05)
    if plot:
        plot_clusters(data, 'Optics', labels, centers, limit)
    assert max(labels) + 1 == centers.shape[0]
    assert len(labels) == data.shape[0]

    print('Birch subclusters.')
    labels, centers = cluster.birch_clustering(data, 'none', branching_factor=100, threshold=0.75)
    if plot:
        plot_clusters(data, 'Birch Subclusters', labels, centers, limit)
    assert max(labels) + 1 == centers.shape[0]
    assert len(labels) == data.shape[0]

    print('Birch agglomorative with ' + str(num_clusters) + ' clusters.')
    labels, centers = cluster.birch_clustering(data, 'agglomorate', num_clusters, linkage='average', affinity='cosine')
    if plot:
        plot_clusters(data, 'Birch Agglomorate', labels, centers, limit)
    assert max(labels) + 1 == centers.shape[0]
    assert len(labels) == data.shape[0]

    print('Birch k-means with ' + str(num_clusters) + ' clusters.')
    labels, centers = cluster.birch_clustering(data, 'k_means', num_clusters, tol=1.e-6, num_iter=500)
    if plot:
        plot_clusters(data, 'Birch K-means', labels, centers, limit)
    assert max(labels) + 1 == centers.shape[0]
    assert len(labels) == data.shape[0]

    print('Birch optics.')
    labels, centers = cluster.birch_clustering(data, 'optics', min_samples=0.05)
    if plot:
        plot_clusters(data, 'Birch Optics', labels, centers, limit)
    assert max(labels) + 1 == centers.shape[0]
    assert len(labels) == data.shape[0]


def test_edge_cases():
    data, _, _ = generate_data(center_range=10, isotropic = False)
    _, _ = cluster.k_means_clustering(data, num_clusters=5, num_iter=3, tol=1.e-14)
    _, _ = cluster.birch_clustering(data, 'test')
    _, _ = cluster.birch_clustering(data, 'agglomorate', num_clusters=5, linkage='ward', affinity='cosine')


def test_clustering(plot=False):
    run_clustering(True, 10, True)
    run_clustering(plot, 100, False)

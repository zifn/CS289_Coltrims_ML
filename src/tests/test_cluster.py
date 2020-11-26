import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
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

def plot_clusters(X, Y, labels, centers):
    num_clusters = np.max(labels) + 1
    #colors = np.linspace(0, 1, num=num_clusters)
    for i in range(num_clusters):
        plt.scatter(X[labels==i], Y[labels==i])#, c=colors[i]*np.ones(np.sum(labels==i)))
        #plt.scatter(centers[i,0], centers[i,1], c=colors[i], marker="*")

    plt.title('Number of Clusters: ' + str(num_clusters))
    plt.show()

def test_clustering(plot=False):
    data, true_labels, true_centers = generate_data()
    if plot:
        plot_clusters(data[:,0], data[:,1], true_labels, true_centers)

    num_clusters = 5
    print('k_means with ' + str(num_clusters) + ' clusters.')
    k_labels, k_centers = cluster.k_means_clustering(data, num_clusters)
    if plot:
        plot_clusters(data[:,0], data[:,1], k_labels, k_centers)
    #print(k_centers)

    print('Optics.')
    optics_labels, optics_centers = cluster.optics_clustering(data)
    if plot:
        plot_clusters(data[:,0], data[:,1], optics_labels, optics_centers)
    #print(optics_centers)

    print('Birch subclusters.')
    birch_labels, birch_centers = cluster.birch_clustering(data, 'none')
    if plot:
        plot_clusters(data[:,0], data[:,1], birch_labels, birch_centers)
    #print(birch_centers)

    print('Birch agglomorative with ' + str(num_clusters) + ' clusters.')
    birch_labels_1, birch_centers_1 = cluster.birch_clustering(data, 'agglomorate', num_clusters=5)
    if plot:
        plot_clusters(data[:,0], data[:,1], birch_labels_1, birch_centers_1)
    #print(birch_centers_1)

    print('Birch k_means with ' + str(num_clusters) + ' clusters.')
    birch_labels_2, birch_centers_2 = cluster.birch_clustering(data, 'k_means', num_clusters=5)
    if plot:
        plot_clusters(data[:,0], data[:,1], birch_labels_2, birch_centers_2)
    #print(birch_centers_2)

    print('Birch optics.')
    birch_labels_3, birch_centers_3 = cluster.birch_clustering(data, 'optics')
    if plot:
        plot_clusters(data[:,0], data[:,1], birch_labels_3, birch_centers_3)
    #print(birch_centers_3)

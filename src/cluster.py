import sys
import sklearn.cluster
import numpy as np

# sk-learn user guide for clustering - https://scikit-learn.org/stable/modules/clustering.html

def k_means_clustering(data, num_clusters, num_iter=300):
    """
    Function to perform k_means clustering using sklearn package. The number of
    clusters is specified by 'num_clusters' parameter, so this is a hyperparameter
    chosen by the user.

    Parameters
    ------------
    data : numpy array
        numpy array of data to be clustered. For the canonical D2O example, each row
        will contain a single colision experiement, while the columns contain the
        (possibly featurized) data.
    num_clusters : int
        Number of clusters to produce from k_means; thus a hyperparameter that needs
        to be tuned by the user and evaluated by other, domain specific means (see
        angular momentum distribution fitting).
    num_iter : int, default=300
        Number of iterations for K-means algorithm. We will get a warning if we use all
        iterations. This is a sign that we may not have converged.

    Returns
    --------
    array
        Return list of clusters to which each data point is assigned. The number of
        entries in this list should be equal to the number of rows in our data
        matrix.
    array
        Returns list of cluster centers. The shape of this array will be
        num_clusters x data.shape[1] (the number of features). This should be
        equivalent to
    """

    kmeans = sklearn.cluster.KMeans(n_cluster=num_clusters, max_iter=num_iter, verbose=0)
    result = kmeans.fit(data)
    assert data.shape[0] == len(result.labels_), "Array of cluster labels is not same length as number of data points."
    if result.n_iter == num_iter:
        print('Warning: k-means may not have converged.')
    return result.labels_, result.cluster_centers

def optics_clustering(data, num_clusters):
    """
    Function to perform OPTICS (Ordering Points to Identify the Clustering
    structure) clustering using sklearn package. This method finds core samples
    of high density and expands clusters from them. This method benefits from
    not having to specify hyper-parameters such as number of clusters.We
    attempt to use all available processors.

    Parameters
    ------------
    data : numpy array
        numpy array of data to be clustered. For the canonical D2O example, each row
        will contain a single colision experiement, while the columns contain the
        (possibly featurized) data.

    Returns
    --------
    array
        Return list of clusters to which each data point is assigned. The number of
        entries in this list should be equal to the number of rows in our data
        matrix.
    array
        Returns list of cluster centers. The shape of this array will be
        num_clusters x data.shape[1] (the number of features).
    """

    num_points = data.shape[0]
    optics = sklearn.cluster.Optics(min_samples=int(np.sqrt(num_points)))
    result = optics.fit(data)
    assert data.shape[0] == len(result.labels_), "Array of cluster labels is not same length as number of data points."

    num_outliers = np.sum(np.all(result.labels_ == -1))
    num_clusters = np.max(result.labels_)
    cluster_centers = []
    for i in range(num_clusters):
        cluster_centers.append(np.mean(data[result.labels_==i,:], axis=0))

    print('Number of points not assigned cluster: ', num_outliers)
    return result.labels_, cluster_centers

def birch_clustering(data, clustering_type, num_clusters):
    """
    Function to perform Birch clustering using sklearn package. This
    online-learning method constructs a tree data structure. The final clusters
    are given by several different options: returning the subclusters from the
    leaves, agglomorative clustering, or a sklearn.cluster estimator.

    Parameters
    ------------
    data : numpy array
        numpy array of data to be clustered. For the canonical D2O example, each row
        will contain a single colision experiement, while the columns contain the
        (possibly featurized) data.
    clustering_type : string
        Choices: 'none', 'agglomorate', 'k-means', 'optics'
        Indicates which method we use at the end of Birch clustering to convert the
        subclusters into our final desired clusters.
    num_clusters : int
        Number of final clusters. Parameter matters in 'Agglomorate' or 'k-means'
        clustering_type. Not used otherwise.

    Returns
    --------
    array
        Return list of clusters to which each data point is assigned. The number of
        entries in this list should be equal to the number of rows in our data
        matrix.
    array
        Returns list of cluster centers. The shape of this array will be
        num_clusters x data.shape[1] (the number of features).
    """

    if clustering_type == 'none':
        birch = sklearn.cluster.Birch(n_clusters=None)
    elif clustering_type == 'agglomorate':
        birch = sklearn.cluster.Birch(n_clusters=num_clusters)
    elif clustering_type == 'k-means':
        kmeans = sklearn.cluster.Kmeans(n_clusters=num_clusters)
        birch = sklearn.cluster.Birch(n_clusters=kmeans)
    elif clustering_type == 'optics':
        num_points = data.shape[0]
        optics = sklearn.cluster.Optics(min_samples=int(np.sqrt(num_points)))
        birch = sklearn.cluster.Birch(n_clusters=optics)
    else:
        print("Unrecognized 'clustering_type' parameter: ", clustering_type)
        sys.exit(0)

    labels = birch.fit_predict(data)
    assert data.shape[0] == len(labels), "Array of cluster labels is not same length as number of data points."

    num_outliers = np.sum(np.all(labels == -1))
    num_clusters = np.max(labels)
    cluster_centers = []
    for i in range(num_clusters):
        cluster_centers.append(np.mean(data[labels==i,:], axis=0))

    print('Number of points not assigned cluster: ', num_outliers)
    return labels, cluster_centers

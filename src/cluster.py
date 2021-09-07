import warnings
import sklearn.cluster
import numpy as np

# sk-learn user guide for clustering - https://scikit-learn.org/stable/modules/clustering.html


def k_means_clustering(data, num_clusters, num_iter=300, num_init=10, tol=1.e-4):
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
    num_init : int, default=10
        Number of times the K-means algorithm will be run; best output in terms of
        inertial will be used as final results.
    tol : float, default=1.e-4
        Relative threshold for convergence for Frobenius norm of the difference in
        cluster centers.

    Returns
    --------
    array
        Return list of clusters to which each data point is assigned. The number of
        entries in this list should be equal to the number of rows in our data
        matrix.
    array
        Returns list of cluster centers. The shape of this array will be
        num_clusters x data.shape[1] (the number of features). This is calculated by
        averaging the elements in each cluster.
    """

    kmeans = sklearn.cluster.KMeans(n_clusters=num_clusters, max_iter=num_iter, verbose=0,
                                    n_init=num_init, tol=tol)
    result = kmeans.fit(data)
    assert data.shape[0] == len(result.labels_), "Array of cluster labels is not same length as number of data points."
    if result.n_iter_ == num_iter:
        warnings.warn('Warning: k-means may not have converged.')

    labels = np.array(result.labels_)

    num_outliers = np.sum(result.labels_ == -1)
    num_clusters = np.max(result.labels_) + 1
    cluster_centers = []
    for i in range(num_clusters):
        cluster_centers.append(np.mean(data[labels == i, :], axis=0))

    print('Number of clusters found: ', len(np.unique(labels)))
    print('Number of points not assigned cluster: ', num_outliers)

    return np.array(result.labels_), np.array(cluster_centers)


def optics_clustering(data, min_samples=5, max_eps=np.inf, metric_power=2,
                      metric='minkowski'):
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
    min_samples: int > 1 or float between 0 and 1, default=5
        Minimum number of samples in a neighborhood for a point to be considered a
        core point. Also used when extracting clusters using the 'Xi' method as
        the minimum number of points in a sample.
    max_epx: float, default=np.inf
        Maximum distance between two points to be considered in the neighborhood
        of one another. Smaller values lead to faster runtimes.
    metric: str or callable, default='minkowski'
        Metric to be used for distance computation. Metric can be a callable
        function (see sklearn documentation for details). Of interest to us are
        'minkowski' (L_p) and 'cosine' (for angles).
    metric_power: int, default=2
        Parameter for Minkowski metric, p -> L_P norm.


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

    optics = sklearn.cluster.OPTICS(min_samples=min_samples, max_eps=max_eps, metric=metric,
                                   p=metric_power)
    result = optics.fit(data)
    assert data.shape[0] == len(result.labels_), "Array of cluster labels is not same length as number of data points."

    num_outliers = np.sum(result.labels_ == -1)
    num_clusters = np.max(result.labels_) + 1
    cluster_centers = []
    for i in range(num_clusters):
        cluster_centers.append(np.mean(data[result.labels_ == i, :], axis=0))

    print('Number of clusters found: ', num_clusters)
    print('Number of points not assigned cluster: ', num_outliers)
    return np.array(result.labels_), np.array(cluster_centers)


def birch_clustering(data, clustering_type, num_clusters=None, threshold=0.5, branching_factor=50,
                     num_iter=300, num_init=10, tol=1.e-4,
                     min_samples=5, max_eps=np.inf, metric_power=2, metric='minkowski',
                     affinity='euclidean', linkage='ward'):
    """
    Function to perform Birch clustering using sklearn package. This
    online-learning method constructs a tree data structure. The final clusters
    are given by several different options: returning the subclusters from the
    leaves, agglomorative clustering, or a sklearn.cluster estimator.

    The first line of arguments is for Birch, the second line for K-means, the
    third line for OPTICS, and the last line for agglomorative clustering. See
    K-means and Optics functions (really see the sklearn documentation) for
    descriptions of these variables.

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

    BIRCH PARAMETERS
    threshold : float, default=0.5
        Radius of the subcluster obtained by merging a new sample and the closest
        subcluster must be smaller than threshold, if not a new subcluster is formed.
    branching_factor : int, default=50
        Maximum number of subclusters in each node.
    num_clusters : int
        Number of final clusters. Parameter matters in 'Agglomorate' or 'k-means'
        clustering_type. Not used otherwise.

    K-MEANS PARAMETERS
    num_iter : int, default=300
        Number of iterations for K-means algorithm. We will get a warning if we use all
        iterations. This is a sign that we may not have converged.
    num_init : int, default=10
        Number of times the K-means algorithm will be run; best output in terms of
        inertial will be used as final results.
    tol : float, default=1.e-4
        Relative threshold for convergence for Frobenius norm of the difference in
        cluster centers.

    OPTICS PARAMETERS
    min_samples: int > 1 or float between 0 and 1, default=5
        Minimum number of samples in a neighborhood for a point to be considered a
        core point. Also used when extracting clusters using the 'Xi' method as
        the minimum number of points in a sample.
    max_epx: float, default=np.inf
        Maximum distance between two points to be considered in the neighborhood
        of one another. Smaller values lead to faster runtimes.
    metric: str or callable, default='minkowski'
        Metric to be used for distance computation. Metric can be a callable
        function (see sklearn documentation for details). Of interest to us are
        'minkowski' (L_p) and 'cosine' (for angles).
    metric_power: int, default=2
        Parameter for Minkowski metric, p -> L_P norm.

    AGGLOMORATE PARAMETERS
    affinity: str, default='euclidean'
        Metric used to compute the linkage. Of interest to us are 'euclidean' and
        'cosine' for geometry and angle reasons. If linkage='ward', we need to use
        'euclidean'
    linkage: str, default='ward'
        Linkage criterion to be used. Algorithm will merge the pairs of clusters that
        minimizes this criterion. Choices are 'ward', 'average', 'complete', 'single.'

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
        birch = sklearn.cluster.Birch(threshold=threshold,
                                      branching_factor=branching_factor, n_clusters=None)
    elif clustering_type == 'agglomorate':
        if linkage == 'ward' and affinity != 'euclidean':
            warnings.warn("Affinity='euclidean' needed for linkage='ward,' so default to affinity='euclidean.'")
            affinity = 'euclidean'
        agglomorate = sklearn.cluster.AgglomerativeClustering(n_clusters=num_clusters,
                                                              affinity=affinity, linkage=linkage)
        birch = sklearn.cluster.Birch(threshold=threshold,
                                      branching_factor=branching_factor, n_clusters=agglomorate)
    elif clustering_type == 'k_means':
        kmeans = sklearn.cluster.KMeans(n_clusters=num_clusters, max_iter=num_iter,
                                        verbose=0, n_init=num_init, tol=tol)
        birch = sklearn.cluster.Birch(threshold=threshold,
                                      branching_factor=branching_factor, n_clusters=kmeans)
    elif clustering_type == 'optics':
        optics = sklearn.cluster.OPTICS(min_samples=min_samples, max_eps=max_eps,
                                        metric=metric, p=metric_power)
        birch = sklearn.cluster.Birch(threshold=threshold,
                                      branching_factor=branching_factor, n_clusters=optics)
    else:
        warnings.warn("Unrecognized 'clustering_type' parameter: " + clustering_type
                      + ". Defaulting to 'none' (agglomorate with no options).")
        birch = sklearn.cluster.Birch(threshold=threshold,
                                      branching_factor=branching_factor, n_clusters=None)

    labels = birch.fit_predict(data)
    assert data.shape[0] == len(labels), "Array of cluster labels is not same length as number of data points."

    num_outliers = np.sum(labels == -1)
    num_clusters = np.max(labels) + 1
    cluster_centers = []
    for i in range(num_clusters):
        cluster_centers.append(np.mean(data[labels == i, :], axis=0))

    print('Number of clusters found: ', num_clusters)
    print('Number of points not assigned cluster: ', num_outliers)
    return np.array(labels), np.array(cluster_centers)

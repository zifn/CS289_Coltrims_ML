import numpy as np
import scipy as sp
import sklearn.cluster

def k_means(data, num_clusters)
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

    Returns
    --------
    array
            Return list of clusters to which each data point is assigned. The number of
            entries in this list should be equal to the number of rows in our data
            matrix.
    array
            Returns list of cluster centers. The shape of this array will be
            num_clusters x data.shape[1] (the number of features).
    float
            inertia - Sum of squared distances of samples to their closest cluster point
    """

    kmeans = sklearn.cluster.KMeans(n_cluster=num_clusters, verbose=0)
    result = kmeans.fit(data)
    assert(data.shape[0] == len(result.labels_), 'Array of cluster labels is not
           same length as number of data points.')
    if result.n_iter == 300:
        print('Warning: k-means may not have converged.')
    return result.labels_, result.cluster_centers

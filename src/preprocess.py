import scipy as sp
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.model_selection
import numpy as np


def generate_feature_matrix(data, degree=2):

    """
    Function to generate a featurized data matrix from a data matrix where each
    row is a scattering event. We use polynomial feature with arbitrary degree
    as well as dot product of different ion/electron momenta.

    Parameters
    ------------
    data : np.array (pandas dataframe not currently supported)
        Array containing a row for each scattering event. In our prototypical
        example of D2O scattering, we have 15 columns (px, py, pz) for each of
        5 scattering products: ion 1, ion 2, neutral, electron 1, electron 2.
    degree : int
        Degree of the featurization desired. For $n$ features, a degree $d$
        featurization will have $(d+n)C(d) = (d+n)C(n)$ features. With $n$
        fixed, this scales as $d^n$. We will include a constant feature of one.

    Returns
    --------
    np.array (pandas dataframe not currently supported)
        Returns featurized matrix where each row contains the featurized data.
        The new matrix will have shape:
            (data.shape[0], (data.shape[1] + degree)C(degree))
    """

    featurizer = sklearn.preprocessing.PolynomialFeatures(degree=degree,
                                                          interaction_only=False,
                                                          include_bias=True)
    # interaction_only - use distinct input features only
    # include_bias - include a bias column of all ones.

    feature_data = featurizer.fit_transform(data)

    # Ensure we have the right number of rows.
    assert feature_data.shape[0] == data.shape[0]
    # Ensure we have the predicted number of columns.
    assert feature_data.shape[1] == sp.special.comb(data.shape[1] + degree,
                                                    degree)

    return feature_data


def standardize_data(data):

    """
    Standardize features by removing the mean and scaling to unit variance.
    This method DOES NOT whiten the data, so the covariance matrix will not be
    an identity matrix.

    Mean is subtracted from each column (feature), and standard deviation is
    set to 1 by dividing by the std of the demeaned data. Centering and scaling
    happen independently on each feature. Such data preprocessing is helpful
    for clustering approaches that use L1 or L2 regularizers of data
    (interested in spatial relationships between data points). We can think
    about this in terms of k-means; we want each cluster to be a standard
    gaussian for easy identification.

    Parameters
    ------------
    data : np.array (pandas dataframe not currently supported)
        Array containing a row for each scattering event. This array can contain
        either the raw or featurized data. The original data will NOT be altered.

    Returns
    --------
    np.array (pandas dataframe not currently supported)
        Returns the standardized data matrix for use in clustering.
    """

    standardizer = sklearn.preprocessing.StandardScaler(copy=True,
                                                        with_mean=True,
                                                        with_std=True)
    # copy - create copy of data rather than do in-place scaling.
    # with_mean - center data before scaling.
    # with_std - scale data to unit variance (unit std)

    standard_data = standardizer.fit_transform(data)
    assert np.all(standard_data.shape == data.shape), 'Standardized data and original data do not have same shape.'

    return standard_data


def whiten_data(data):

    """
    Whiten the data, i.e. demean the columns and transform data so that the
    covariance matrix is an identity. We are thus losing information (the
    relative variance scales of the columns), but this may help when trying to
    cluster using geometric methods.

    Parameters
    ------------
    data : np.array (pandas dataframe not currently supported)
        Array containing a row for each scattering event. This array can contain
        either the raw or featurized data. The original data will NOT be altered.

    Returns
    --------
    np.array (pandas dataframe not currently supported)
        Returns the standardized data matrix for use in clustering.
    """

    whitener = sklearn.decomposition.PCA(n_components=None,
                                         whiten=True)
    # n_components - how many dimensions of data to keep; None keeps all.
    # whiten - perform whitening on data

    white_data = whitener.fit_transform(data)
    assert np.all(white_data.shape == data.shape), 'Standardized data and original data do not have same shape.'

    return white_data


def perform_PCA(data, components=2):

    """
    Perform principal component analysis on the data to identify the two
    directions with maximal variance. This will be used in visualization to
    identify trends in the data.

    Parameters
    ------------
    data : np.array (pandas dataframe not currently supported)
        Array containing a row for each scattering event. This array can contain
        either the raw or featurized data. The original data will NOT be altered.
    components : int
        Number of components to keep when performing PCA.

    Returns
    --------
    np.array (pandas dataframe not currently supported)
        Returns the rotated data matrix with two columns for use in visualization.
    """
    assert data.shape[1] >= components, 'Not enough columns in data to reduce via PCA.'
    reducer = sklearn.decomposition.PCA(n_components=components, whiten=False)
    # n_components - how many dimensions of data to keep; None keeps all.
    # whiten - perform whitening on data

    PCA_data = reducer.fit_transform(data)
    assert np.all(PCA_data.shape == data[:,:components].shape), 'Standardized data and original data do not have same shape.'

    return PCA_data


def data_split(data, fraction=2/3, random_state=0):

    """
    Split data into testing and training sets depending on fraction of split and
    random seed. Data is shuffled.

    Parameters
    ------------
    data : np.array (pandas dataframe not currently supported)
        Array containing a row for each scattering event. This array can contain
        either the raw or featurized data. The original data will NOT be altered.
    fraction : float
        Fraction of data to be used for training data.
    random_state : int
        Seed for random number generator used to shuffle the data.
    Returns
    --------
    np.array (pandas dataframe not currently supported)
        Returns the data matrix for training.
    np.array (pandas dataframe not currently supported)
        Returns the data matrix for testing.
    """

    assert fraction <= 1
    splitter = sklearn.model_selection.train_test_split
    data_train, data_test = splitter(data, train_size=fraction, random_state=random_state)
    return data_train, data_test


def molecular_frame(data, check_norms=False):
    """
    Convert data into the molecular frame by rotating data so that the hydrogens
    (ions) define a plane and thus the x, y, and z directions. Ions (hydrogens)
    must be in columns 0 and 1.

    This function only works with D2O or Water where we expect 15 columns,
    (px, py, pz) for each of the 5 scattering constituents - ion1, ion2, neutral,
    electron1, electron2.

    Parameters
    ------------
    data : np.array (pandas dataframe not currently supported)
        Array containing a row for each scattering event. This array can contain
        either the raw or featurized data. The original data will NOT be altered.
    check_norm : bool
        Flag for if we should check the norms, only used in testing.
    Returns
    --------
    np.array (pandas dataframe not currently supported)
        Returns the data matrix in a molecular frame.
    """
    print('here)')
    print(data.shape)
    assert data.shape[1] == 15

    molecular_data = []
    ion1 = data[:, 0:3]
    ion2 = data[:, 3:6]
    neutral = data[:, 6:9]
    e1 = data[:, 9:12]
    e2 = data[:, 12:]

    norm_ion1 = ion1 / np.linalg.norm(ion1, axis=1)[:, None]
    norm_ion2 = ion2 / np.linalg.norm(ion2, axis=1)[:, None]

    if check_norms:
        assert np.allclose(np.linalg.norm(norm_ion1, axis=1), np.ones(data.shape[0]))
        assert np.allclose(np.linalg.norm(norm_ion2, axis=1), np.ones(data.shape[0]))

    y_axis = np.cross(norm_ion1, norm_ion2)
    y_axis = y_axis / np.linalg.norm(y_axis, axis=1)[:, None]
    z_axis = norm_ion1 + norm_ion2
    z_axis = z_axis / np.linalg.norm(z_axis, axis=1)[:, None]
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis, axis=1)[:, None]
    if check_norms:
        assert np.allclose(np.linalg.norm(x_axis, axis=1), np.ones(data.shape[0]))
        assert np.allclose(np.linalg.norm(y_axis, axis=1), np.ones(data.shape[0]))
        assert np.allclose(np.linalg.norm(z_axis, axis=1), np.ones(data.shape[0]))

    assert np.allclose(np.cross(x_axis, y_axis), z_axis)
    assert y_axis.shape == z_axis.shape == x_axis.shape == ion1.shape

    print(np.einsum('ij,ij->i', ion1, x_axis).reshape(data.shape[0],1).shape)
    molecular_data = np.hstack((np.einsum('ij,ij->i', ion1, x_axis).reshape(data.shape[0],1),
                                np.einsum('ij,ij->i', ion1, y_axis).reshape(data.shape[0],1),
                                np.einsum('ij,ij->i', ion1, z_axis).reshape(data.shape[0],1),
                                np.einsum('ij,ij->i', ion2, x_axis).reshape(data.shape[0],1),
                                np.einsum('ij,ij->i', ion2, y_axis).reshape(data.shape[0],1),
                                np.einsum('ij,ij->i', ion2, z_axis).reshape(data.shape[0],1),
                                np.einsum('ij,ij->i', neutral, x_axis).reshape(data.shape[0],1),
                                np.einsum('ij,ij->i', neutral, y_axis).reshape(data.shape[0],1),
                                np.einsum('ij,ij->i', neutral, z_axis).reshape(data.shape[0],1),
                                np.einsum('ij,ij->i', e1, x_axis).reshape(data.shape[0],1),
                                np.einsum('ij,ij->i', e1, y_axis).reshape(data.shape[0],1),
                                np.einsum('ij,ij->i', e1, z_axis).reshape(data.shape[0],1),
                                np.einsum('ij,ij->i', e2, x_axis).reshape(data.shape[0],1),
                                np.einsum('ij,ij->i', e2, y_axis).reshape(data.shape[0],1),
                                np.einsum('ij,ij->i', e2, z_axis).reshape(data.shape[0],1)))

    return molecular_data

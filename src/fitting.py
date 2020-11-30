from scipy.special import sph_harm
import numpy as np

def cart_to_spherical(M_xyz):
    """
    Converts every pair of 3 columns of an input X numpy matrix from cartesian
    to spherical coordinates.

    Parameters
    ------------
    M_sph: ndarray (n, 3*m)
        Expected order is X, Y, Z for each set of three columns in a row

    Returns
    --------
    M_sph: ndarray (n, 3*m)
        Returned order is R, Theta, Phi for each set of three columns in a row
    """
    assert M_xyz.shape[1]%3 == 0

    M_sph = np.zeros(M_xyz.shape)
    for i in range(M_xyz.shape[1]//3):
        X = M_xyz[:,i*3]
        Y = M_xyz[:,i*3 + 1]
        Z = M_xyz[:,i*3 + 2]

        R = np.sqrt(X**2 + Y**2 + Z**2) # R
        M_sph[:,i*3] = R
        M_sph[:,i*3 + 1] = np.arctan2(Y, X) # theta
        M_sph[:,i*3 + 2] = np.arccos(Z/R) # phi
    return M_sph

def spherical_to_cart(M_sph):
    """
    Converts every pair of 3 columns of an input X numpy matrix from spherical
    to cartesian scoordinates.

    Parameters
    ------------
    M_sph: ndarray (n, 3*m)
        Expected order is R, Theta, Phi for each set of three columns in a row

    Returns
    --------
    M_xyz: ndarray (n, 3*m)
        Returned order is X, Y, Z for each set of three columns in a row
    """
    assert M_sph.shape[1]%3 == 0

    M_xyz = np.zeros(M_sph.shape)
    for i in range(M_sph.shape[1]//3):
        R = M_sph[:,i*3]
        Theta = M_sph[:,i*3 + 1]
        Phi = M_sph[:,i*3 + 2]

        M_xyz[:,i*3] = R*np.sin(Phi)*np.cos(Theta) # X
        M_xyz[:,i*3 + 1] = R*np.sin(Phi)*np.sin(Theta) # Y
        M_xyz[:,i*3 + 2] = R*np.cos(Phi) # Z
    return M_xyz

def Y_lm_features(theta, phi, L_max, only_even_Ls=False):
    """
    Make Y_lm feature matrix

    Parameters
    ------------
    theta: 1D numpy array (n,)
        array of theta angles ranging from 0 to 2*pi. Used to generate Y_lm
        features
    phi: 1D numpy array (n,)
        array of phi angles ranging from 0 to pi. Used to generate Y_lm features.
        Must be the same size as theta
    L_max: int
        maximum L value used to generate a truncated array of Y_lms
    only_even_Ls: bool
        if in the molecular frame, only use even L values to generate features

    Returns
    --------
    Y_lm_features, lm_order: ndarray, list of tuples
        Y_lm_features is the n by m real special harmonic feature matrix for
        each l,m pair allowed by L_max. lm_order is a list of tuples where
        each tuple is the associated (L, M) pair for each column in the Y_lm
        feature matrix.
    """
    Y_lm_feat = []
    lm_order = []
    for L in range(0, L_max + 1):
        if only_even_Ls and L%2 == 1:
            continue
        for M in range(0, L + 1):
            temp = np.real(sph_harm(M, L, theta, phi))
            Y_lm_feat.append(temp)
            lm_order.append((L, M))
    Y_lm_feat = np.array(Y_lm_feat).T
    return Y_lm_feat, lm_order

def Y_lms_distribution(theta, phi, L_max, B_lms, only_even_Ls=False):
    """
    Computes the probability density given a set of thetas and phis to be
    sampled, an L_max to generate features and the associated array of coeficients.

    Parameters
    ------------
    theta: 1D numpy array
        array of theta angles ranging from 0 to 2*pi. Used to generate Y_lm features
    phi: 1D numpy array
        array of phi angles ranging from 0 to pi. Used to generate Y_lm features.
        Must be the same size as theta
    L_max: int
        maximum L value used to generate a truncated array of Y_lms
    B_lms: 1d numpy array
        1d array of coeficients associated with a given real Y_lm
    only_even_Ls: bool
        if in the molecular frame, only use even L values to generate features

    Returns
    --------
    float
        probability density scaled by the B_00 coeficient
    """
    feats, _ = Y_lm_features(theta, phi, L_max, only_even_Ls)
    product = feats @ B_lms
    return abs(product)/B_lms[0]

def fit_Y_lms_binning_least_squares(M_xyz, L_max, numb_bins, only_even_Ls=False):
    """
    Least Squares Fitting to get B_lm coeficients

    Parameters
    ------------
    M_xyz: n by 3 numpy array
        observered data points representing the distribution in cartesian coordiantes
    L_max: int
        Maximum value of L quantum number determing the number of terms to be used in the fits
    numb_bins: int
        number of bins in theta and phi axes

    Returns
    --------
    B_lms: 1d array
        optimized B_lm coeficients used for the probability distribution
    lm_order: list of tuples
        first index of each tuple is L and the second is M
    """
    assert M_xyz.shape[1] == 3
    assert numb_bins > 0 and int(numb_bins) == numb_bins
    M_sph = cart_to_spherical(M_xyz)

    angular_hist, theta_hist, phi_hist = np.histogram2d(M_sph[:, 1], M_sph[:, 2], numb_bins,
                                                                                    range=[[0, 2*np.pi], [0, np.pi]],
                                                                                    density=True)
    #estimate bin by center value
    theta_hist = (theta_hist[1:] + theta_hist[:-1])/2
    phi_hist = (phi_hist[1:] + phi_hist[:-1])/2
    theta_hist, phi_hist = np.meshgrid(theta_hist, phi_hist)

    # vectorize 2d histogram and input angles
    angular_hist_flat = angular_hist.flatten()
    theta_hist_flat = theta_hist.flatten()
    phi_hist_flat = phi_hist.flatten()

    # make ylm feature matrix
    ylm_features, lm_order = Y_lm_features(theta_hist_flat, phi_hist_flat, L_max, only_even_Ls)

    # solve for coeficients
    B_lms = np.linalg.solve( ylm_features.T @ ylm_features, ylm_features.T @ angular_hist_flat)

    return B_lms, lm_order

def validation_cross_entropy(data_val_xyz, labels, model_params, L_max, only_even_Ls=False):
    """
    computes the cross entropy using the shpericial harmonic distribution and labeled data
    from clustering

    Parameters
    ------------
    data_val_xyz: array Nx3
        Validation data used to compute the cross entropy of the clasification
    labels: array of ints
        Each entry varries from 0 to (number classes - 1). Used to reference the model params.
    params: list of arrays
        Each entry is a list of arrays representing the B_lms for an individual model
    L_max: int
        maximum L value used to generate a truncated array of Y_lms
    only_even_Ls: bool
        if in the molecular frame, only use even L values to generate features

    Returns
    --------
    cross_entropy: float
        the cross entropy given labeled validation data
    """
    assert data_val_xyz.shape[1] == 3
    data_val_sph = cart_to_spherical(data_val_xyz)

    unique_labels = np.unique(labels)
    assert  set(list(unique_labels)).issubset(set(range(len(model_params))))
    assert unique_labels.shape[0] == len(model_params) or unique_labels.shape[0] == len(model_params) + 1
    unique_labels = list(range(len(model_params)))

    # make qs
    qs = []
    for label in unique_labels:
        qs.append(Y_lms_distribution(data_val_sph[:, 1], data_val_sph[:, 2],
                                                    L_max, model_params[label], only_even_Ls))
    qs = np.array(qs).T
    qs /= qs.sum(axis=1)[:, None] # normalization of probability density to probabilities of each class

    #compute cross-entropy
    cross_entropy = 0
    for label in unique_labels:
        class_qs = qs[labels == label]
        cross_entropy += -sum(np.log(class_qs[:, label]))*class_qs.shape[0]

    return cross_entropy

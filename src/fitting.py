from itertools import chain
import numpy as np
from scipy.special import sph_harm, assoc_laguerre

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
        feature matrix. n is the number of (theta, phi) pairs
        while m is the number of (L,M) pairs, $\sum_{L=0}^{L_{max}} 2L + 1$.
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
        for M in range(-L, 0):
            temp = np.imag(sph_harm(abs(M), L, theta, phi))
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
    assert unique_labels.shape[0] == len(model_params) or unique_labels.shape[0] == len(model_params) + 1, \
                f"unique_labels.shape = {unique_labels.shape} and len(model_params) = {len(model_params)}"
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

def Psi_nlm_features(r, theta, phi, N_max, only_even_Ls=False):
    """
    Make Psi_nlm feature matrix. Each feature is generated by
    Psi_nlm(r, theta, phi) = R_{nl}(r) Y_{lm}(theta, phi), where R_nl is the radial
    wave function for Hydrogen and Y_lm is the spherical harmonics. Y_lm's form a
    basis for functions on a sphere while the R_nl's form a basis for functions of
    one variable between 0 and infinity. The formula for R_nl's is equation (7.40)
    of http://umich.edu/~ners311/CourseLibrary/Lecture07.pdf and is reproduced below:
    ..math::
        R_{n\ell}(r) = \sqrt{x^3 \frac{(n-\ell-1)!}{2n(n+\ell)!}} (xr)^{\ell} \exp(-rx/2) L^{2\ell + 1}_{n-\ell-1}(xr),

    where $x = \frac{2Z}{n a_{\mu}}$, $Z$ is the proton number (1 for hydrogen),
    n in the principle quantum number, \ell is the angular quantum number, and
    $a_{\mu}$ is the Bohr radius (set to 1 for our purposes).

    Parameters
    ------------
    r: 1D numpy array (n,)
        array of radial magnitudes ranging from 0 to max(r) (varies depending on
        input data). Used to generate Psi_nlm features. Must be the same size as
        theta and phi
    theta: 1D numpy array (n,)
        array of theta angles ranging from 0 to 2*pi. Used to generate Psi_nlm
        features. Must be the same size as r and phi.
    phi: 1D numpy array (n,)
        array of phi angles ranging from 0 to pi. Used to generate Psi_nlm features.
        Must be the same size as theta and r.
    n_max: int
        maximum n (principal quantum number) value used to generate a truncated
        array of Psi_nlm's. L will range from 0 <-> N-1, and for each L, M will
        range from -L <-> L.
    only_even_Ls: bool
        if in the molecular frame, only use even L values to generate features

    Returns
    --------
    Psi_nlm_features, nlm_order: ndarray, list of tuples
        Psi_nlm_features is the n by m real special harmonic feature matrix for
        each N,L,M pair allowed by N_max. nlm_order is a list of tuples where
        each tuple is the associated (N, L, M) pair for each column in the Psi_nlm
        feature matrix. n is the number of (r, theta, phi) pairs (input points)
        while m is the number of (N, L,M) pairs,
        $\sum_{N=1}^{N_{max}}\sum_{L=0}^{N-1} 2L + 1$.
    """
    assert len(r) == len(theta) == len(phi)

    Psi_nlm_feat = []
    nlm_order = []
    R_nl_feat = []
    nl_order = []
    # Generate R_nl feature
    for N in range(1, N_max + 1):   # Loop between 1 and N
        for L in range(0, N):
            if only_even_Ls and L%2 == 1:
                continue
            radial_part = np.sqrt((2/N)**3 * np.math.factorial(N-L-1)/(2*N*np.math.factorial(N+L))) * \
                    assoc_laguerre(2*r/N, N-L-1, 2*L+1)*np.exp(-r/N)*(2*r/N)**L
            R_nl_feat.append(radial_part)
            nl_order.append((N,L))
    R_nl_feat = np.array(R_nl_feat).T
    assert R_nl_feat.shape[0] == len(r)

    nl_counter = 0      # How many NL pairs have we used thus far.
    for n_prime in range(1, N_max+1):
        Y_lm_feat, lm_order = Y_lm_features(theta, phi, n_prime-1, only_even_Ls)
        lm_counter = 0  # How many LM pairs have we used thus far.
        for l_prime in range(0, n_prime):
            if only_even_Ls and l_prime%2 == 1:
                continue

            r_feat = R_nl_feat[:,nl_counter]
            for m_prime in chain(range(0, l_prime + 1), range(-l_prime,0)):
                y_feat = Y_lm_feat[:, lm_counter]
                Psi_nlm_feat.append(r_feat * y_feat)
                assert (l_prime, m_prime) == lm_order[lm_counter] and (n_prime, l_prime) == nl_order[nl_counter]
                nlm_order.append((n_prime, l_prime, n_prime))
                lm_counter += 1

            nl_counter += 1
    Psi_nlm_feat = np.array(Psi_nlm_feat).T
    assert Psi_nlm_feat.shape[0] == len(r)
    return Psi_nlm_feat, nlm_order

def Psi_nlms_distribution(r, theta, phi, n_max, B_nlms, only_even_Ls=False):
    """
    Computes the probability density given a set of rs, thetas, and phis to be
    sampled, an N_max to generate features, and the associated array of coeficients.

    Parameters
    ------------
    r: 1D numpy array (n,)
        array of radial magnitudes ranging from 0 to max(r) (varies depending on
        input data). Used to generate Psi_nlm features. Must be the same size as
        theta and phi
    theta: 1D numpy array (n,)
        array of theta angles ranging from 0 to 2*pi. Used to generate Psi_nlm
        features. Must be the same size as r and phi.
    phi: 1D numpy array (n,)
        array of phi angles ranging from 0 to pi. Used to generate Psi_nlm features.
        Must be the same size as theta and r.
    N_max: int
        maximum N value used to generate a truncated array of Psi_nlm's
    B_nlms: 1d numpy array
        1d array of coeficients associated with a given real Psi_nlm's
    only_even_Ls: bool
        if in the molecular frame, only use even L values to generate features

    Returns
    --------
    float
        probability density scaled by the B_000 coeficient
    """
    feats, _ = Psi_nlm_features(r, theta, phi, n_max, only_even_Ls)
    product = feats @ B_nlms
    return abs(product)/B_nlms[0]

def fit_Psi_nlms_binning_least_squares(M_xyz, N_max, numb_bins, only_even_Ls=False):
    """
    Least Squares Fitting to get B_nlm coeficients

    Parameters
    ------------
    M_xyz: n by 3 numpy array
        observered data points representing the distribution in cartesian coordiantes
    N_max: int
        Maximum value of N quantum number determing the number of terms to be used in the fits
    numb_bins: int
        number of bins in r, theta, and phi axes

    Returns
    --------
    B_nlms: 1d array
        optimized B_nlm coeficients used for the probability distribution
    nlm_order: list of tuples
        first index of each tuple is N, second is L, final is M.
    """
    assert M_xyz.shape[1] == 3
    assert numb_bins > 0 and int(numb_bins) == numb_bins
    M_sph = cart_to_spherical(M_xyz)

    # Range in r is from 0 to max(r), so the size of each bin depends on the inputed data.
    spherical_hist, bin_edges = np.histogramdd((M_sph[:,0], M_sph[:, 1], M_sph[:, 2]), numb_bins,
                                                    range=[None, [0, 2*np.pi], [0, np.pi]],
                                                    density=True)

    r_hist = bin_edges[0]
    theta_hist = bin_edges[1]
    phi_hist = bin_edges[2]

    #estimate bin by center value
    r_hist = (r_hist[1:] + r_hist[:-1])/2
    theta_hist = (theta_hist[1:] + theta_hist[:-1])/2
    phi_hist = (phi_hist[1:] + phi_hist[:-1])/2
    r_hist, theta_hist, phi_hist = np.meshgrid(r_hist, theta_hist, phi_hist)

    # vectorize 3d histogram and input radii and angles
    spherical_hist_flat = spherical_hist.flatten()
    r_hist_flat = r_hist.flatten()
    theta_hist_flat = theta_hist.flatten()
    phi_hist_flat = phi_hist.flatten()

    # make ylm feature matrix
    nlm_features, nlm_order = Psi_nlm_features(r_hist_flat, theta_hist_flat, phi_hist_flat, N_max, only_even_Ls)

    # solve for coeficients
    B_nlms = np.linalg.solve( nlm_features.T @ nlm_features, nlm_features.T @ spherical_hist_flat)

    return B_nlms, nlm_order

def validation_cross_entropy_nlm(data_val_xyz, labels, model_params, N_max, only_even_Ls=False):
    """
    computes the cross entropy using the sphericial harmonic distribution and labeled data
    from clustering

    Parameters
    ------------
    data_val_xyz: array Nx3
        Validation data used to compute the cross entropy of the clasification
    labels: array of ints
        Each entry varries from 0 to (number classes - 1). Used to reference the model params.
    params: list of arrays
        Each entry is a list of arrays representing the B_lms for an individual model
    N_max: int
        maximum N value used to generate a truncated array of Y_lms
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
    assert unique_labels.shape[0] == len(model_params) or unique_labels.shape[0] == len(model_params) + 1, \
                f"unique_labels.shape = {unique_labels.shape} and len(model_params) = {len(model_params)}"
    unique_labels = list(range(len(model_params)))

    # make qs
    qs = []
    for label in unique_labels:
        qs.append(Psi_nlms_distribution(data_val_sph[:,0], data_val_sph[:, 1], data_val_sph[:, 2],
                                                    N_max, model_params[label], only_even_Ls))
    qs = np.array(qs).T
    qs /= qs.sum(axis=1)[:, None] # normalization of probability density to probabilities of each class

    #compute cross-entropy
    cross_entropy = 0
    for label in unique_labels:
        class_qs = qs[labels == label]
        cross_entropy += -sum(np.log(class_qs[:, label]))*class_qs.shape[0]

    return cross_entropy

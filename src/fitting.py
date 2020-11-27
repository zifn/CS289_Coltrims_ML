from scipy.special import shp_harm
import numpy as np

def cart_to_spherical(M_xyz):
    """
    Converts every pair of 3 columns of an input X numpy matrix from cartesian 
    to spherical coordinates.
    """
    assert(M_xyz.shape[1]%3 == 0)

    M_sph = np.zeros(M_xyz.shape)
    for i in range(M_xyz.shape[1]//3):
        X = M_xyz[:,i*3]
        Y = M_xyz[:,i*3 + 1]
        Z = M_xyz[:,i*3 + 2]
        
        M_sph[:,i*3] = np.sqrt(X**2 + Y**2 + Z**2) # R
        M_sph[:,i*3 + 1] = np.arctan2(Y, X) # theta
        M_sph[:,i*3 + 2] = np.arccos(Z/R) # phi
    
    return M_sph
    
def spherical_to_cart(M_sph):
    """
    Converts every pair of 3 columns of an input X numpy matrix from cartesian 
    to spherical coordinates.
    """
    assert(M_sph.shape[1]%3 == 0)

    M_sph = np.zeros(M_sph.shape)
    for i in range(M_sph.shape[1]//3):
        R = M_sph[:,i*3]
        Theta = M_sph[:,i*3 + 1]
        Phi = M_sph[:,i*3 + 2]
        
        M_xyz[:,i*3] = R*np.sin(Phi)*np.cos(Theta) # X
        M_xyz[:,i*3 + 1] = R*np.sin(Phi)*np.sin(Theta) # Y
        M_xyz[:,i*3 + 2] = R*np.cos(Phi) # Z
    
    return M_xyz

def sum_Y_lms(theta, phi, L_max, B_lms):
    """
    Sum of Y_lms for spherical harmonics
    """
    assert(B_lms.shape[0] == L_max and B_lms.shape[1] == L_max)
    accum = np.zeros(len(theta))
    for L in range(0, L_max):
        for M in range(0, L):
            accum += B_lms[L, M]*np.real(sph_harm(M, L, theta, phi))
    return accum

def Y_lms_distribution(theta, phi, L_max, B_lms):
    return sum_Y_lms(theta, phi, L_max, B_lms)/B_lms[0, 0]

def SDA_Y_lms_distribution(M_xyz, L_max, eta=0.001, epochs=3, batch_fraction=0.01):
    """
    Stochastic gradient assent update for probability distribution based on spherical harmonics
    
    Parameters
    ------------
    M_xyz: n by 3 numpy array
        observered data points representing the distribution in cartesian coordiantes 
    L_max: int
        Maximum value of L quantum number determing the number of terms to be used in the fits
    eta: float > 0
        learning rate/step size
    epochs: int
        number of times to cycle throught the data
    batch_fraction: 1 >= float > 0
        fraction of data points in each batch used in the SGD/SGA update
    
    Returns
    --------
    B_lms: 2D array
        optimized B_lm coeficients used for the probability distribution
    """
    assert(M_xyz.shape[1] == 3)
    assert(batch_fraction <= 1 and batch_fraction > 0)
    assert(eta > 0)
    
    rng = np.random.default_rng()
    # set up initial parameters and parameters that are called often
    N = M_xyz.shape[0]
    N_f = int(N*batch_fraction)
    B_lms = np.zeros([L_max, L_max])
    B_lms[0, 0] = 1
    # convert to spherical
    M_sph = cart_to_spherical(M_xyz)
    
    for i in range(epochs):
        # set up new array of indices and choose a subset for the next batch
        indices = np.arange(0, N, 1)
        batch_ind = rng.choice(indices, size=N_f, replace=False)
        indices = np.delete(indices, batch_ind)
        for j in range(int(1/batch_fraction)):
            # compute the update
            M_sph_batch = M_sph[batch_ind, :]
            F = sum_Y_lms(M_sph_batch[:,1], M_sph_batch[:,2], L_max, B_lms)
            for L in range(0, L_max):
                for M in range(0, L):
                    if L == 0 and M == 0:
                        B_lms[0, 0] -= eta*M_sph_batch.shape[0]/B_lms[0, 0]
                    f = np.real(sph_harm(M, L, M_sph_batch[:,1], M_sph_batch[:,2]))
                    B_lms[L, M] += eta*sum(f/F)
            # choose next subset of indices for the next batch
            batch_ind = rng.choice(indices, size=N_f, replace=False)
            indices = np.delete(indices, batch_ind)
    return B_lms
    


















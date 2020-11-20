from scipy.special import shp_harm
import numpy as np

def cart_to_spherical(M_xyz):
    """
    Converts every pair of 3 columns of an input X numpy matrix from cartesian 
    to spherical coordinates.
    """
    assert(M_xyz.shape[1]%3 == 0)

    M_sph = np.zeros(M_xyz.shape)
    for i in M_xyz.shape[1]//3:
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
    for i in M_sph.shape[1]//3:
        R = M_sph[:,i*3]
        Theta = M_sph[:,i*3 + 1]
        Phi = M_sph[:,i*3 + 2]
        
        M_xyz[:,i*3] = R*np.sin(Phi)*np.cos(Theta) # X
        M_xyz[:,i*3 + 1] = R*np.sin(Phi)*np.sin(Theta) # Y
        M_xyz[:,i*3 + 2] = R*np.cos(Phi) # Z
    
    return M_xyz

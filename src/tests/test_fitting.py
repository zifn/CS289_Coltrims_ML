from itertools import product

import numpy as np

from .. import fitting

def test_cart_spherical_conversions():
    """
    Checks if cart_to_spherical() returns expected values given unit vectors along
    cardinal directions and if cart_to_spherical() and spherical_to_cart() are inverses
    if each other
    """
    rng = np.random.default_rng()
    
    M_xyz_exp = np.array([[1, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, 1]])

    M_sph_exp = np.array([[1, 0, np.pi/2],
                                      [1, np.pi/2, np.pi/2],
                                      [1, 0, 0]])
    
    M_sph_trans = fitting.cart_to_spherical(M_xyz_exp)
    assert np.allclose(M_sph_exp, M_sph_trans)
    
    M_xyz_trans = fitting.spherical_to_cart(M_sph_trans)
    assert np.allclose(M_xyz_exp, M_xyz_trans)
    
    M_xyz_rand = rng.standard_normal([200, 3])
    M_xyz_rand_trans = fitting.spherical_to_cart(fitting.cart_to_spherical(M_xyz_rand))
    assert np.allclose(M_xyz_rand, M_xyz_rand_trans)
    
    M_xyz_rand = rng.standard_normal([200, 6])
    M_xyz_rand_trans = fitting.spherical_to_cart(fitting.cart_to_spherical(M_xyz_rand))
    assert np.allclose(M_xyz_rand, M_xyz_rand_trans)
    
def test_SDA_for_Y_lm_distribution():
    """
    Checks if SDA is able to find an equivalent probability distribution to a source distribution
    """
    rng = np.random.default_rng()
    theta = np.linspace(0, 2*np.pi, 21)
    phi = np.linspace(0, np.pi, 21)
    angles = list(product(theta, phi))
    angles_array = np.array(angles)
    
    # all L values
    for L_max, evens in [(3, False), (2, True)]:
        for trial in range(3):
            # make some random B_lms
            sample_B_lms = np.zeros(2*[L_max + 1])
            if False:
                for L in range(L_max + 1):
                    if  evens and L%2 == 1:
                        pass
                    for M in range(L):
                        sample_B_lms[L, M] += rng.normal()
                        sample_B_lms[0, 0] += abs(sample_B_lms[L, M])
            sample_B_lms[0, 0] += 1
            
            # get probabilities from B_lms
            angles_probs = fitting.Y_lms_distribution(angles_array[:,0], angles_array[:,1], L_max, sample_B_lms, evens)
            angles_probs /= sum(angles_probs)

            # get samples
            angle_indicies = np.arange(0, angles_array.shape[0], 1)
            samples_indicies = rng.choice(angle_indicies, size=100000, replace=True, p=angles_probs)
            samples_angles = angles_array[samples_indicies]
            samples_sph = np.hstack((np.ones([samples_angles.shape[0], 1]), samples_angles))
            samples_xyz = fitting.spherical_to_cart(samples_sph)
            
            # run SDA and get probability estimates
            print("running SDA trial = ", trial)
            estimate_B_lms = fitting.SDA_Y_lms_distribution(samples_xyz, L_max, 0.001, 10, 1, evens)
            est_probs = fitting.Y_lms_distribution(angles_array[:,0], angles_array[:,1], L_max, estimate_B_lms, evens)
            est_probs /= sum(est_probs)
            print(f"estimate_B_lms = {estimate_B_lms}")
            print(f"sample_B_lms {sample_B_lms}")
            assert np.allclose(est_probs, angles_probs)
        
        
        
        
        
        
        
        
        
        
        
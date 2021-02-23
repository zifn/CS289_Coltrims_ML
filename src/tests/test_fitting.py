from itertools import product, chain
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

def test_fit_Y_lms_va_binning():
    """
    Checks if least squares va binning is able to find an equivalent probability distribution to a source distribution
    """
    rng = np.random.default_rng()
    theta = np.linspace(0, 2*np.pi, 201)
    phi = np.linspace(0, np.pi, 201)
    angles = list(product(theta, phi))
    angles_array = np.array(angles)

    # all L values
    for L_max, evens, rand_dist in [(1, False, False), (2, True, False), (3, False, True), (2, True, True)]:
        for trial in range(1):
            # make some random B_lms
            sample_B_lms = []

            for L in range(0, L_max + 1):
                if  evens and L%2 == 1:
                    continue
                for _ in chain(range(L+1), range(-L,0)):
                    sample_B_lms.append(rng.normal())
            sample_B_lms[0] = 3
            if not rand_dist:
                sample_B_lms = len(sample_B_lms)*[0]
                sample_B_lms[0] = 1
            sample_B_lms = np.array(sample_B_lms)

            # get probabilities from B_lms
            angles_probs = fitting.Y_lms_distribution(angles_array[:,0], angles_array[:,1], L_max, sample_B_lms, evens)
            angles_probs = abs(angles_probs)
            angles_probs /= sum(angles_probs)

            # get samples
            angle_indicies = np.arange(0, angles_array.shape[0], 1)
            samples_indicies = rng.choice(angle_indicies, size=10000, replace=True, p=angles_probs)
            samples_angles = angles_array[samples_indicies]
            samples_sph = np.hstack((np.ones([samples_angles.shape[0], 1]), samples_angles))
            samples_xyz = fitting.spherical_to_cart(samples_sph)

            # run least squares and get probability estimates
            print("------------------\nrunning least squares trial = ", trial)
            estimate_B_lms, _ = fitting.fit_Y_lms_binning_least_squares(samples_xyz, L_max, 100, evens)
            est_probs = fitting.Y_lms_distribution(angles_array[:,0], angles_array[:,1], L_max, estimate_B_lms, evens)
            est_probs /= sum(est_probs)
            print(f"\nnormed estimate_B_lms:\n{estimate_B_lms/estimate_B_lms[0]}")
            print(f"normed sample_B_lms:\n{sample_B_lms/sample_B_lms[0]}")

            print("MSE of probs after opt = ", np.mean((est_probs - angles_probs)**2))
            assert np.all(est_probs >= 0)
            assert np.mean((est_probs - angles_probs)**2) < 10**-8

def test_fit_R_nlms_va_binning():
    rng = np.random.default_rng()
    r = np.linspace(0.1,5,5)
    theta = np.linspace(0, 2*np.pi, 51)
    phi = np.linspace(0, np.pi, 51)
    vectors = list(product(r, theta, phi))
    vectors_array = np.array(vectors)

    # all L values
    for N_max, evens, rand_dist in [(1, False, False), (2, True, False), (3, False, True), (2, True, True)]:
        for trial in range(1):
            # make some random B_lms
            sample_B_nlms = []
            for n in range(1, N_max+1):
                for L in range(0, n):
                    if  evens and L%2 == 1:
                        continue
                    for _ in chain(range(L+1), range(-L,0)):
                        sample_B_nlms.append(rng.normal())
            sample_B_nlms[0] = 3
            if not rand_dist: # Not random distribution. Instead use nlm = 100
                sample_B_nlms = len(sample_B_nlms)*[0]
                sample_B_nlms[0] = 1
            sample_B_nlms = np.array(sample_B_nlms)

            # get probabilities from B_nlms
            vectors_probs = fitting.Psi_nlms_distribution(vectors_array[:,0], vectors_array[:,1], vectors_array[:,2],
                                                          N_max, sample_B_nlms, evens)
            vectors_probs = abs(vectors_probs)
            vectors_probs /= sum(vectors_probs)

            # get samples
            vectors_indicies = np.arange(0, vectors_array.shape[0], 1)
            samples_indicies = rng.choice(vectors_indicies, size=10000, replace=True, p=vectors_probs)
            samples_vectors = vectors_array[samples_indicies]
            #samples_sph = np.hstack((np.ones([samples_angles.shape[0], 1]), samples_angles))
            samples_xyz = fitting.spherical_to_cart(samples_vectors)

            # run least squares and get probability estimates
            print("------------------\nrunning least squares trial = ", trial)
            estimate_B_nlms, _ = fitting.fit_Psi_nlms_binning_least_squares(samples_xyz, N_max, 100, evens)
            est_probs = fitting.Psi_nlms_distribution(vectors_array[:,0], vectors_array[:,1], vectors_array[:,2],
                                                      N_max, estimate_B_nlms, evens)
            est_probs /= sum(est_probs)
            print(f"\nnormed estimate_B_nlms:\n{estimate_B_nlms/estimate_B_nlms[0]}")
            print(f"normed sample_B_nlms:\n{sample_B_nlms/sample_B_nlms[0]}")

            print("MSE of probs after opt = ", np.mean((est_probs - vectors_probs)**2))
            assert np.all(est_probs >= 0)
            assert np.mean((est_probs - vectors_probs)**2) < 10**-8

def test_validation_cross_entropy():
    """
    simple test of cross entropy function
    """
    # make uniform B_lms for sphere
    sample_B_lms = []
    N_valid = 100
    L_max = 3

    for L in range(0, L_max + 1):
        for _ in chain(range(L+1), range(-L,0)):
            sample_B_lms.append(0)
    sample_B_lms[0] = 1
    sample_B_lms = np.array(sample_B_lms)

    # sample points around sphere
    rng = np.random.default_rng()
    theta = np.linspace(0, 2*np.pi, 2001)
    phi = np.linspace(0, np.pi, 2001)

    samples = np.array([np.ones(N_valid),
                                    rng.choice(theta, size=N_valid, replace=True),
                                    rng.choice(phi, size=N_valid, replace=True)]).T
    samples = fitting.spherical_to_cart(samples)
    labels = np.zeros(N_valid)
    model_params = [sample_B_lms]

    #fit
    entropy = fitting.validation_cross_entropy(samples, labels, model_params, L_max, only_even_Ls=False)
    assert np.isclose(entropy, 0)

    #check no labels edge case
    entropy = fitting.validation_cross_entropy(samples, [], model_params, L_max, only_even_Ls=False)
    assert np.isclose(entropy, np.inf), f"expected {np.inf} but recievied {entropy}"

def test_validation_cross_entropy_nlm():
    """
    simple test of cross entropy function
    """
    # make uniform B_lms
    sample_B_nlms = []
    N_valid = 100
    N_max = 3
    for n in range(1, N_max+1):
        for L in range(0, n):
            for _ in chain(range(L+1), range(-L,0)):
                sample_B_nlms.append(0)
    sample_B_nlms[0] = 1
    sample_B_nlms = np.array(sample_B_nlms)

    # sample points
    rng = np.random.default_rng()
    r = np.linspace(0, 5, 2002)     # Max radius is 5.
    r = r[1:]                       # Do not allow radius of 0.
    theta = np.linspace(0, 2*np.pi, 2001)
    phi = np.linspace(0, np.pi, 2001)

    samples = np.array([rng.choice(r, size=N_valid, replace=True),
                                    rng.choice(theta, size=N_valid, replace=True),
                                    rng.choice(phi, size=N_valid, replace=True)]).T
    samples = fitting.spherical_to_cart(samples)
    labels = np.zeros(N_valid)
    model_params = [sample_B_nlms]

    entropy = fitting.validation_cross_entropy_nlm(samples, labels, model_params, N_max, only_even_Ls=False)
    assert np.isclose(entropy, 0)

    #check no labels edge case
    entropy = fitting.validation_cross_entropy_nlm(samples, [], model_params, N_max, only_even_Ls=False)
    assert np.isclose(entropy, np.inf), f"expected {np.inf} but recievied {entropy}"

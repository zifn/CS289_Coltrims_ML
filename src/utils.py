from itertools import product

import numpy as np
import pandas as pd

ELECTRON_MASS = 510.99895000e3 # eV
ATOMIC_MASS = 931.49410242e6 # eV

ELECTRON_MASS /= ATOMIC_MASS # Use atomic mass as units.

DEUTERON_MASS = 2.01410177811 # a.u.
OXYGEN_MASS = 15.999 # a.u.

def kinetic_energy(px, py, pz, mass):
    """
    Function to compute kinetic energy release from the COLTRIMS data. Takes the
    xyz momentums and the mass of the particle and returns the nonrelativistic
    kinetic energy. If px, py, pz, or mass are array-like, it must be possible
    to broadcast these shapes together. Units should be consistent.

    Parameters
    ------------
    px : float or array-like
        The x momentum of the particle.
    py : float or array-like
        The y momentum of the particle.
    pz : float or array-like
        The z momentum of the particle.
    mass : float or array-like
        The mass or masses of the particle(s).

    Returns
    --------
    float or array-like
        Returns a float, array, or dataframe, depending on the input data type,
        that contains the kinetic energies of the particles computed from the
        momentums and mass.
    """
    return 0.5*(px**2 + py**2 + pz**2)/mass

def ejection_angle(px1, py1, pz1, px2, py2, pz2, cos=False):
    """
    Function to compute the ejection angle between two ions from the COLTRIMS
    data. Takes the xyz momentums of each ion and returns the angle theta
    between the two momentum vectors. If the momentums are array-like, it must
    be possible to broadcast these shapes together.

    Parameters
    ------------
    px1 : float or array-like
        The x momentum of the particle.
    py1 : float or array-like
        The y momentum of the particle.
    pz1 : float or array-like
        The z momentum of the particle.
    px2 : float or array-like
        The x momentum of the particle.
    py2 : float or array-like
        The y momentum of the particle.
    pz2 : float or array-like
        The z momentum of the particle.
    cos : boolean
        Whether to return the cos of the ejection angle or the ejection angle.
        Defaults to True.

    Returns
    --------
    float or array-like
        Returns a float, array, or dataframe, depending on the input data type,
        that contains the kinetic energies of the particles computed from the
        momentums and mass.
    """
    normalization = np.sqrt(
        (px1**2 + py1**2 + pz1**2) * (px2**2 + py2**2 + pz2**2)
    )

    ca = (px1 * px2 + py1 * py2 + pz1 * pz2)/normalization

    return ca if cos else np.arccos(ca)

def extract_data(dataset):
    """
    Utility function to extract the momentums for each particle in the COLTRIMS
    dataset from an entry in the dataset. Takes a numpy array or dataset and
    returns a tuple of numpy arrays corresponding to each particle.

    Parameters
    ------------
    dataset : array or dataframe
        A numpy array or dataframe containing the COLTRIMS data

    Returns
    --------
    tuple of arrays
        Returns a tuple of numpy arrays corresponding to the xyz momentums of
        each particle.
    """
    if isinstance(dataset, pd.DataFrame):
        dataset = dataset.to_numpy()

    ion1 = dataset[:, :3]
    ion2 = dataset[:, 3:6]
    neutral = dataset[:, 6:9]
    e1 = dataset[:, 9:12]
    e2 = dataset[:, 12:]

    return ion1, ion2, neutral, e1, e2

def generate_synthetic_data(k, num_particles, points_per_cluster):
    """
    Function to generate a random synthetic dataset for running the clustering
    and fitting algorithms. Takes the number of clusters, number of particles,
    and the points per cluster, and returns a dataset with shape
    (k*points_per_cluster data points, 3*num_particles).

    Parameters
    ------------
    k : int
        The number of clusters to generate.
    num_particles: int
        The number of particles to include in the data. 3*num_particles values
        will be generated for each entry of the dataset, corresponding to the
        x, y, z momentums of the particles.
    points_per_cluster: int
        The number of points to add to each cluster.

    Returns
    --------
    array-like
        Returns a dataset with k*points_per_cluster entries.
    """
    means = np.array([((phi + 0.5)*np.pi/2, (theta + 0.5)*np.pi/2)
                                 for phi, theta in product(list(range(4)), list(range(2)))])

    cluster_angles = means[np.random.randint(0, 8, size=(k, num_particles))]
    cluster_radii = np.random.uniform(-10, 10, size=(k, num_particles))

    dataset = []
    dataset_labels = []
    for i in range(k):
        phis = np.random.normal(cluster_angles[i, :, 0],
                                              scale=np.pi/16,
                                              size=(points_per_cluster, num_particles))
        thetas = np.random.normal(cluster_angles[i, :, 1],
                                                 scale=np.pi/16,
                                                 size=(points_per_cluster, num_particles))
        r = np.random.normal(cluster_radii[i],
                                         scale=0.3,
                                         size=(points_per_cluster, num_particles))

        x = r*np.cos(phis)*np.sin(thetas)
        y = r*np.sin(phis)*np.sin(thetas)
        z = r*np.cos(thetas)

        ps = np.hstack([x, y, z])
        ps = ps[:, np.array([(i, num_particles + i, 2*num_particles + i) for i in range(num_particles)]).flatten()]
        dataset_labels += [i]*points_per_cluster
        dataset.append(ps)

    dataset = np.vstack(dataset)
    return dataset, dataset_labels

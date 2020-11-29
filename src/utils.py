import numpy as np
import pandas as pd

ELECTRON_MASS = 510.99895000e3 # eV
ATOMIC_MASS = 931.49410242e6 # eV

DEUTERON_MASS = 2.01410177811 * ATOMIC_MASS
OXYGEN_MASS = 15.999 * ATOMIC_MASS

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

def ejection_angle(px1, py1, pz1, px2, py2, pz2, cos=True):
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

    return ca if cos else np.arcos(ca)

def extract_data(dataset):
    if isinstance(dataset, pd.DataFrame):
        dataset = dataset.to_numpy()

    ion1 = dataset[:, :3]
    ion2 = dataset[:, 3:6]
    neutral = dataset[:, 6:9]
    e1 = dataset[:, 9:12]
    e2 = dataset[:, 12:]

    return ion1, ion2, neutral, e1, e2

import numpy as np
import pandas as pd

from .. import utils

def test_kinetic_energy():
    """
    Checks if kinetic_energy() returns expected values.
    """
    p1s = (1, 0, 0)
    m1 = 0.5

    KE1 = utils.kinetic_energy(*p1s, m1)
    assert np.isclose(KE1, 1)

    p2s = (0, 0, 2)
    m2 = 1

    KE2 = utils.kinetic_energy(*p2s, m2)
    assert np.isclose(KE2, 2)

def test_ejection_angle():
    """
    Checks if ejection_angle() returns expected values.
    """

    p1s = (1, 0, 0)
    p2s = (0, 2, 0)
    p3s = (0, 1, 0)
    p4s = (0, 0, -1)
    p5s = (0, 0, 1)

    a12 = utils.ejection_angle(*p1s, *p2s)
    assert np.isclose(a12, np.pi/2)

    ca12 = utils.ejection_angle(*p1s, *p2s, cos=True)
    assert np.isclose(ca12, 0)

    a23 = utils.ejection_angle(*p2s, *p3s)
    assert np.isclose(a23, 0)

    ca23 = utils.ejection_angle(*p2s, *p3s, cos=True)
    assert np.isclose(ca23, 1)

    a45 = utils.ejection_angle(*p4s, *p5s)
    assert np.isclose(a45, np.pi)

    ca45 = utils.ejection_angle(*p4s, *p5s, cos=True)
    assert np.isclose(ca45, -1)

def test_extract_data():
    """
    Checks if extract_data() returns the correct values.
    """

    data = pd.DataFrame(np.arange(15).reshape(1, 15))

    ion1, ion2, neutral, e1, e2 = utils.extract_data(data)

    assert np.array_equal(ion1, np.array([[0, 1, 2]]))
    assert np.array_equal(ion2, np.array([[3, 4, 5]]))
    assert np.array_equal(neutral, np.array([[6, 7, 8]]))
    assert np.array_equal(e1, np.array([[9, 10, 11]]))
    assert np.array_equal(e2, np.array([[12, 13, 14]]))

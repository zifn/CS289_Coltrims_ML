import os
import shutil

import numpy as np

from .. import parsing
from .. import utils

def test_read_write_momentum():
    temp_path = "temp.dat"
    rng = np.random.default_rng()
    values = rng.integers(0, 200, [200, 15])
    headers = ["Px_ion_1", "Py_ion_1", "Pz_ion_1",
                        "Px_ion_2", "Py_ion_2", "Pz_ion_2",
                        "Px_neutral", "Py_neutral", "Pz_neutral",
                        "Px_elec_1", "Py_elec_1", "Pz_elec_1",
                        "Px_elec_2", "Py_elec_2", "Pz_elec_2"]

    for write_headers in [False, True]:
        for return_np in [True, False]:
            print(f"write_headers = {write_headers}, return_np = {return_np}")
            parsing.write_momentum(temp_path, values, write_headers)
            read_values = parsing.read_momentum(temp_path, return_np, write_headers)
            if not return_np:
                read_headers = read_values.columns.to_list()
                assert read_headers == headers
                parsing.write_momentum(temp_path, read_values, write_headers)
                read_values = parsing.read_momentum(temp_path, True, write_headers)
            print("values.shape = ", values.shape)
            print("read_values.shape = ", read_values.shape)
            assert np.array_equal(values, read_values)
            os.remove(temp_path)

def test_read_write_clusters():
    # generate some fake data
    data, labels = utils.generate_synthetic_data(5, 5, 100)

    dir_root = os.path.join(os.getcwd(), "temp")

    #write data
    dir_1 = parsing.save_clusters(labels, data, 0, 0, 0.001, dir_root, "synthetic_data")
    dir_2 = parsing.save_clusters(labels, data, 0, 0, 0.001, dir_root, "synthetic_data")
    dir_3 = parsing.save_clusters(labels, data, 0, 0, np.inf, dir_root, "synthetic_data")

    assert dir_1 != dir_2

    #read_data
    data1, labels1 = parsing.read_clusters(dir_1)
    data2, labels2 = parsing.read_clusters(dir_2)
    data3, labels3 = parsing.read_clusters(dir_3)

    assert np.array_equal(data1, data2)
    assert np.array_equal(labels1, labels2)
    assert np.array_equal(data1, data3)
    assert np.array_equal(labels1, labels3)
    assert len(labels) == len(labels1)
    assert data.shape == data1.shape

    #clean up
    shutil.rmtree(dir_root)

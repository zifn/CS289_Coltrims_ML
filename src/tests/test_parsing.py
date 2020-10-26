import os

import numpy as np
import pandas as pd

from .. import parsing

def test_read_write():
    temp_path = "temp.dat"
    rng = np.random.default_rng()
    #values = rng.standard_normal([2, 15])
    values = rng.integers(0, 9, [2, 15])
    headers = ["Px_ion_1", "Py_ion_1", "Pz_ion_1",
                        "Px_ion_2", "Py_ion_2", "Pz_ion_2",
                        "Px_neutral", "Py_neutral", "Pz_neutral",
                        "Px_elec_1", "Py_elec_1", "Pz_elec_1",
                        "Px_elec_2", "Py_elec_2", "Pz_elec_2"]
    
    for write_headers in [False, True]:
        for return_np in [True, False]:
            print(write_headers, return_np)
            parsing.write_momentum(temp_path, values, write_headers)
            read_values = parsing.read_momentum(temp_path, return_np, write_headers)
            if not return_np:
                if write_headers:
                    read_headers = read_values.columns.to_list()
                    assert(read_headers == headers)
                parsing.write_momentum(temp_path, values, write_headers)
                read_values = parsing.read_momentum(temp_path, True, write_headers)
            print(values.shape)
            print(read_values.shape)
            assert(np.array_equal(values, read_values))
            os.remove(temp_path)

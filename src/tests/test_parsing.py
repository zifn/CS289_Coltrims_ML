import os

import numpy as np
import pandas as pd

from .. import parsing

def test_read_write():
    temp_path = "temp.dat"
    rng = np.random.default_rng()
    values = rng.standard_normal([200, 18])
    headers = ["Px_ion_1", "Py_ion_1", "Pz_ion_1",
                        "Px_ion_2", "Py_ion_2", "Pz_ion_2",
                        "Px_neutral", "Py_neutral", "Pz_neutral",
                        "Px_elec_1", "Py_elec_1", "Pz_elec_1",
                        "Px_elec_2", "Py_elec_2", "Pz_elec_2"]
    
    for write_headers in [True, False]:
        for return_np in [True, False]:
            parsing.write_momentum(temp_path, values, write_headers)
            read_values = parsing.read_momentum(temp_path, return_np, write_headers)
            if not return_np:
                if write_headers:
                    read_headers = read_values.columns.to_list()
                    assert(read_headers == headers)
                read_values = read_values.to_numpy()
            assert(np.array_equal(values, read_values))
            os.remove(temp_path)

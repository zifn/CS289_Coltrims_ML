import pandas as pd
import numpy as np
import yaml

def read_momentum(input_file_path, return_numpy=True, has_headers=False):
    """
    Function to read input momentum data from a COLTRIMS dataset. Expects the
    input file to be a ' ' delimited file with n rows and 15 columns representing the
    momenta of 5 particles detected in coicidence.
    
    Parameters
    ------------
    input_file_path : str or path-like
        path to file with momentum information to read
    return_numpy : bool
        If true the function returns a numpy array else it returns a pandas dataframe
    has_headers : bool
        asks if the input file has headers or not. If the input file has a header on the 
        first line, the header names will be over written to the following names in 
        order:
            ["Px_ion_1", "Py_ion_1", "Pz_ion_1",
                "Px_ion_2", "Py_ion_2", "Pz_ion_2",
                "Px_neutral", "Py_neutral", "Pz_neutral",
                "Px_elec_1", "Py_elec_1", "Pz_elec_1",
                "Px_elec_2", "Py_elec_2", "Pz_elec_2"]        
    
    Returns
    --------
    array or dataframe
            Returns an n by 18 array or dataframe depending on the input parameters.
            If a pandas dataframe is returned the headers will be
            ["Px_ion_1", "Py_ion_1", "Pz_ion_1",
            "Px_ion_2", "Py_ion_2", "Pz_ion_2",
            "Px_neutral", "Py_neutral", "Pz_neutral",
            "Px_elec_1", "Py_elec_1", "Pz_elec_1",
            "Px_elec_2", "Py_elec_2", "Pz_elec_2"]
    """
    headers = ["Px_ion_1", "Py_ion_1", "Pz_ion_1",
                    "Px_ion_2", "Py_ion_2", "Pz_ion_2",
                    "Px_neutral", "Py_neutral", "Pz_neutral",
                    "Px_elec_1", "Py_elec_1", "Pz_elec_1",
                    "Px_elec_2", "Py_elec_2", "Pz_elec_2"]
    if has_headers:
        momentum_df = pd.read_csv(input_file_path,
                                                    sep=" ",
                                                    header=0,
                                                    names=headers)
    else:
        momentum_df = pd.read_csv(input_file_path,
                                                    sep=" ",
                                                    header=None,
                                                    names=headers)
    if return_numpy:
        return momentum_df.to_numpy()
    else:
        return momentum_df

def write_momentum(output_file_path, momentum, write_headers=False):
    """
    Saves momenta data in either numpy array or dataframe format as a ' ' delimited
    file. Expects an n by 18 shaped input
    
    Parameters
    ------------
    output_file_path: str or path-like
        file location to save the input data
    momentum : array or dataframe
        data to save. Should be an n by 15 shaped numpy array or dataframe.
    write_headers : bool
        If true will write the following list as a header to the top of the file:
        ["Px_ion_1", "Py_ion_1", "Pz_ion_1",
            "Px_ion_2", "Py_ion_2", "Pz_ion_2",
            "Px_neutral", "Py_neutral", "Pz_neutral",
            "Px_elec_1", "Py_elec_1", "Pz_elec_1",
            "Px_elec_2", "Py_elec_2", "Pz_elec_2"]
    
    """
    headers = ["Px_ion_1", "Py_ion_1", "Pz_ion_1",
                    "Px_ion_2", "Py_ion_2", "Pz_ion_2",
                    "Px_neutral", "Py_neutral", "Pz_neutral",
                    "Px_elec_1", "Py_elec_1", "Pz_elec_1",
                    "Px_elec_2", "Py_elec_2", "Pz_elec_2"]
    
    if type(momentum) == type(np.array([])):
        if not write_headers:
            np.savetxt(output_file_path, momentum, delimiter=' ')
        else:
            header_txt = ''
            for header in headers:
                header_txt += header + " "
            np.savetxt(output_file_path, momentum, delimiter=' ', header=header_txt[:-1], comments='')
    else: # assume momentum is a pandas dataframe
        if not write_headers:
            momentum.to_csv(output_file_path, sep=" ", header=headers, index=False)
        else:
            momentum.to_csv(output_file_path, sep=" ", header=False, index=False)
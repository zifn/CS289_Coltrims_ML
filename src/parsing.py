import os

import pandas as pd
import numpy as np

def read_momentum(input_file_path, return_numpy=True, has_headers=False,
                                    headers=("Px_ion_1", "Py_ion_1", "Pz_ion_1",
                                                    "Px_ion_2", "Py_ion_2", "Pz_ion_2",
                                                    "Px_neutral", "Py_neutral", "Pz_neutral",
                                                    "Px_elec_1", "Py_elec_1", "Pz_elec_1",
                                                    "Px_elec_2", "Py_elec_2", "Pz_elec_2")):
    """
    Function to read input momentum data from a COLTRIMS dataset. Expects the
    input file to be a ' ' delimited file representing the momenta of each particle
    detected in coicidence.

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
    headers : list of strings
        a list of header names for the intermidiate or output dataframe

    Returns
    --------
    array or dataframe
            Returns an array or dataframe depending on the input parameters.
            If a pandas dataframe is returned the headers will be those supplied
            to the function.
    """
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

def write_momentum(output_file_path, momentum, write_headers=False,
                                    headers=("Px_ion_1", "Py_ion_1", "Pz_ion_1",
                                                    "Px_ion_2", "Py_ion_2", "Pz_ion_2",
                                                    "Px_neutral", "Py_neutral", "Pz_neutral",
                                                    "Px_elec_1", "Py_elec_1", "Pz_elec_1",
                                                    "Px_elec_2", "Py_elec_2", "Pz_elec_2")):
    """
    Saves momenta data in either numpy array or dataframe format as a ' ' delimited
    file.

    Parameters
    ------------
    output_file_path: str or path-like
        file location to save the input data
    momentum : array or dataframe
        data to save. Should be an n by 15 shaped numpy array or dataframe.
    write_headers : bool
        If true will write the following list as a header to the top of the file
    headers : list of strings
        a list of header names for the saved data file
    """

    if isinstance(momentum, np.ndarray):
        if not write_headers:
            np.savetxt(output_file_path, momentum, delimiter=' ')
        else:
            header_txt = ''
            for header in headers:
                header_txt += header + " "
            np.savetxt(output_file_path, momentum, delimiter=' ', header=header_txt[:-1], comments='')
    else: # assume momentum is a pandas dataframe
        if write_headers:
            momentum.to_csv(output_file_path, sep=" ", header=headers, index=False)
        else:
            momentum.to_csv(output_file_path, sep=" ", header=False, index=False)

def save_clusters(labels, data, L_max, bins, entropy, method="kmeans-molecular-frame"):
    k = len(np.unique(labels))
    root_dir = "privileged"
    if not os.path.isdir(root_dir):
        os.mkdir(root_dir)

    dir_name = f"{method}_with_{k}_clusters_{L_max}_{bins}_{int(entropy)}"
    dir_name = os.path.join(root_dir, dir_name)

    while os.path.isdir(dir_name):
        dir_name += "_"
    os.mkdir(dir_name)
    
    for label in np.unique(labels):
        cluster_data = data[labels == label]
        file_name = f"cluster_{label}_of_{k}.dat"
        file_path = os.path.join(dir_name, file_name)
        write_momentum(file_path, cluster_data, write_headers=True)

    return dir_name

def read_clusters(directory, has_headers=False):
    # Cluster directories are named as follows
    # f'{clustering-method}_with_{k}_clusters_{L_max}_{bins}_{entropy}'
    cluster_metadata = os.path.basename(directory.strip('_/\\')).split('_')
    method, _, k, _, L_max, bins, entropy = cluster_metadata
    
    cluster_files = os.listdir(directory)

    data = []
    labels = []
    for filename in cluster_files:
        if not filename.endswith('.dat'):
            continue
        idx = int(filename.split('_')[1])
        cluster_data = read_momentum(os.path.join(directory, filename), has_headers=has_headers)

        N = cluster_data.shape[0]

        data.append(cluster_data)
        labels.append(idx*np.ones(N))

    return np.vstack(data), np.concatenate(labels)
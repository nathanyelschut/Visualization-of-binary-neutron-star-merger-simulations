"""Script for converting raw data files produced by an Einstein Toolkit simulation to time sliced files.
"""

import multiprocessing
import os
import re
import time

import h5py
import pandas as pd

def convert_to_time_sliced(source_file_path_format, save_folder, file_number):
    """Write contents of a single raw data file produced by an Einstein Toolkit simulation to time sliced files.

    Args:
        source_file_path_format (str): file path and file name up to the excluding the file number and extension.
            example: /home/Bvec[0].xyz.file_
        save_folder (str): path to where the time sliced files should be stored.
        file_number (int): number of the raw file to convert to time sliced files.
    """

    source_file_path = rf"{source_file_path_format}{file_number}.h5"
    
    with h5py.File(source_file_path, 'r') as source_file:
        # Get metadata
        keys = source_file.keys()
        keys_string = ' '.join(keys)
        pattern = r"(?:([a-zA-z]+::[a-zA-Z]+)\[(\d)\] it=([\d]+) tl=([\d]+) rl=([0-4]) c=(\d+))"
        metadata = re.findall(pattern, keys_string)

        var_name = metadata[0][0]
        direction = metadata[0][1]
        tl = metadata[0][3]

        columns = ['Var_name', 'Direction', 'it', 'tl', 'rl', 'c']
        df = pd.DataFrame(metadata, columns=columns)[[columns[2], columns[4], columns[5]]]
        df = df.apply(pd.to_numeric)
        df = df.sort_values(['it', 'rl', 'c'],ascending = [True, True, True], ignore_index=True)

        it_list = df['it'].drop_duplicates().to_list()
        rl_list = df['rl'].drop_duplicates().to_list()
        c_list = df['c'].drop_duplicates().to_list()

        # First iterate through components, then reference levels and lastly iterations
        for it in it_list:
            for rl in rl_list:
                for c in c_list:
                    # Copy dataset and its attributes
                    key = f'{var_name}[{direction}] it={it} tl={tl} rl={rl} c={c}'
                    dataset = source_file[key]
                    attributes = list(dataset.attrs.items())

                    file_name = os.path.join(fr"{save_folder}", fr"{var_name.split('::')[-1]}[{direction}].xyz_it={it}.h5")

                    # Try to write to the new time-sliced file until successful
                    completed = False
                    while completed is False:
                        try:
                            with h5py.File(file_name, 'r+') as file:
                                try:
                                    file.create_dataset(key, data=dataset)
                                    for attribute in attributes:
                                        file[key].attrs.create(attribute[0], attribute[1])
                                except ValueError:
                                    for attribute in attributes:
                                        file[key].attrs.create(attribute[0], attribute[1])
                                
                                completed = True

                        except FileNotFoundError:
                            with h5py.File(file_name, 'w') as file:
                                try:
                                    file.create_dataset(key, data=dataset)
                                    for attribute in attributes:
                                        file[key].attrs.create(attribute[0], attribute[1])
                                except ValueError:
                                    for attribute in attributes:
                                        file[key].attrs.create(attribute[0], attribute[1])
                                
                                completed = True
                        
                        except BlockingIOError:
                            completed = False
                            time.sleep(5)

def time_slicer(source_file_path_format, n_files, save_folder, n_processes):
    """Function that calls convert_to_time_sliced with multiprocessing.

    Args:
        source_file_path_format (str): file path and file name up to the excluding the file number and extension.
            example: /home/Bvec[0].xyz.file_
        n_files (int): number of raw data files.
        save_folder (str): path to where the time sliced files should be stored.
        n_processes (int): number of computing cores to use for parallelization.
    """

    for i in range(0, int(n_files), n_processes):
        arguments = []
        for j in range(i, i+n_processes):
            if j < int(n_files):
                argument = (source_file_path_format, save_folder, j)
                arguments.append(argument)

        with multiprocessing.Pool(n_processes) as pool:
            res = pool.starmap(convert_to_time_sliced, arguments)
            pool.close()
            pool.join()

# Example use:
# source_file_path_format = r"C:\Users\Nathanyel\Documents\BachelorProject\Data\Bvec[2]2.xyz.file_"
# file_range = range(120)
# save_folder = r"C:\Users\Nathanyel\Documents\BachelorProject\Data"

# time_slicer(source_file_path_format, file_range, save_folder)

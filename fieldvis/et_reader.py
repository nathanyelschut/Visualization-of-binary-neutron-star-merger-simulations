"""FieldVis module containing tools for processing simulation data made with the Einstein Toolkit.

Data files should already be time sliced before using this module. The module provides a built-in data
reader for other FieldVis modules to convert the time-sliced data files to np.ndarrays.
"""

import re
import h5py
import numpy as np

def ET_file_parser(file):
    """Reads file and finds the variable name, iteration, time level, reference levels
    and components available in the file using regex.

    Args:
        file (h5py.File): h5py file object containing the data.

    Returns:
        str, int, int, list, list: variable name, iteration, time level, reference levels
            and components
    """

    # Get metadata: variable name, it, tl
    # Run regex twice, once with match, second time with findall
    keys = list(file.keys())
    keys_string = ' '.join(keys)
    pattern = r"(?:([a-zA-Z0-9:\[\]]+) it=(\d+) tl=(\d+) rl=[0-9]+ c=\d+)"
    metadata = re.match(pattern, keys_string)

    var_name = metadata.groups()[0]
    it = metadata.groups()[1]
    tl = metadata.groups()[2]
    
    # Get metadata: rl, c
    pattern = r"(?:[a-zA-Z0-9:\[\]]+ it=\d+ tl=\d+ rl=([0-9]+) c=(\d+))"
    rl_c = re.findall(pattern, keys_string)
    rl_list, c_list = zip(*rl_c)
    rl_list, c_list = list(map(int, list(set(rl_list)))), list(map(int, list(set(c_list))))
    rl_list.sort()
    c_list.sort()

    return var_name, it, tl, rl_list, c_list

def ET_get_grid_info(var_name, it, tl, rl, c_list, file):
    """Finds attributes of the total grid of a certain reference level. Arguments can be found
    using the ET_file_parser function.

    Args:
        var_name (str): variable name.
        it (int): iteration number.
        tl (int): time level.
        rl (int): reference level.
        c_list (list): list of all component numbers.
        file (h5py.File): h5py file object containing the data.

    Returns:
        list, list, list, list, list: grid spacing, number of ghostzones, dimensions, minimum and maximum
            coordinates in all three directions.
    """

    # Get attributes: delta, nghostzones
    read_dataset = file[f"{var_name} it={it} tl={tl} rl={rl} c={c_list[0]}"]
    delta = read_dataset.attrs['delta']
    nghost = read_dataset.attrs['cctk_nghostzones']

    # Get coordinates information
    x_coordinates = []
    y_coordinates = []
    z_coordinates = []
    
    for c in c_list:
        read_dataset = file[f"{var_name} it={it} tl={tl} rl={rl} c={c}"]
        origin = read_dataset.attrs['origin']
        x_coordinates.append(origin[0] + nghost[0]*delta[0])
        y_coordinates.append(origin[1] + nghost[1]*delta[1])
        z_coordinates.append(origin[2] + nghost[2]*delta[2])
    
    # Calculate maximum and minimum coordinate
    dimensions = []
    min_coords = []
    max_coords = []

    for count, coordinates in enumerate([x_coordinates, y_coordinates, z_coordinates]):
        max_c = c_list[coordinates.index(max(coordinates))]
        shape = np.shape(file[f"{var_name} it={it} tl={tl} rl={rl} c={max_c}"])
        shape = (shape[2], shape[1], shape[0])

        max_dim = shape[count]
        max_ = max(coordinates) + (max_dim - (2 * nghost[count])) * delta[count]
        
        min_ = min(coordinates)
        
        range_ = max_ - min_
        dimension = round(range_ / delta[count])
        dimensions.append(dimension)
        min_coords.append(min_)
        max_coords.append(max_)

    return delta, nghost, dimensions, min_coords, max_coords

def ET_to_numpy(rl, time_sliced_file_names):
    """Converts one (scalar) or three (vector) time sliced data files with data made in the Einstein Toolkit 
    to a list containing np.ndarrays.

    Args:
        rl (int): reference level.
        time_sliced_file_names (list): list containing the relevant file names for a single iteration of data.

    Raises:
        ValueError: raised when the provided reference level is not available.

    Returns:
        list: list containing the np.ndarrays with the field data. Either a list of len 1 for scalar data or a 
            list of len 3 for vector data.
    """

    files = [h5py.File(file_name) for file_name in time_sliced_file_names]
    datasets = []

    # Get data information
    var_name, it, tl, rl_list, c_list = ET_file_parser(files[0])
    _, nghost, dimensions, _, _ = ET_get_grid_info(var_name, it, tl, rl, c_list, files[0])
    
    gx, gy, gz = nghost
    
    if '[' in var_name[-3:] and ']' in var_name[-3:]:
        var_name = var_name[:-3]

    if rl not in rl_list:
        raise ValueError(f"Reference level {rl} is not present in the data set. Available reference levels: {rl_list}")

    # Easy access to data components
    if len(files) == 3:
        get_data = lambda file, c: np.array(file[f"{var_name}[{files.index(file)}] it={it} tl={tl} rl={rl} c={c}"])[gz:-gz, gy:-gy, gx:-gx]
    
    elif len(files) == 1:
        get_data = lambda file, c: np.array(file[f"{var_name} it={it} tl={tl} rl={rl} c={c}"])[gz:-gz, gy:-gy, gx:-gx]

    # Join components
    max_x, max_y, max_z = dimensions
    max_c = max(c_list)

    for file in files:
        stack = get_data(file, 0)
        x_i, y_i, z_i, square, cube, c = (0, 0, 0, 0, 0, 1)

        while y_i != max_y:
            while x_i != max_x:
                while z_i != max_z:
                    stack = np.concatenate((stack, get_data(file, c)), axis=0)
                    z_i = np.shape(stack)[0]
                    c += 1

                if isinstance(square, np.ndarray):
                    square = np.concatenate((square, stack), axis=2)
                    x_i = np.shape(square)[2]
                else:
                    square = stack

                z_i = 0
                if c <= max_c:
                    stack = get_data(file, c)
                c += 1

            if isinstance(cube, np.ndarray):
                cube = np.concatenate((cube, square), axis=1)
                y_i = np.shape(cube)[1]
            else:
                cube = square

            x_i = 0
            square = 0

        datasets.append(np.swapaxes(cube, 0, 2))

    return datasets

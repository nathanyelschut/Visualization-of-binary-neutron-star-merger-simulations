"""FieldVis module for processing field data once it has been processed to a numpy array.

Functionalities include finding a selection of filenames, creating pyvista grids and PolyData
sets containing field data and streamlines and creating custom opacity mappings.
"""

import os
import re

import matplotlib.colors as colors
import numpy as np
import pyvista as pv
from matplotlib import cm


def find_it_files(variable_name, it, path_to_files):
    """Finds all files containing the given variable name and iteration.
    The pattern for filenames should be: 
    (variable name)(possible vector component indicator)(random text)it=(iteration number)(extension)

    Args:
        variable_name (str): string indicating the variable name in a filename.
        it (int): iteration number of the simulation.
        path_to_files (str): string providing the relative or absolute path to the data files.

    Returns:
        list: list of filenames found based on the variable name and iteration.
    """  

    files_str = ''
    for _, _, all_files in os.walk(path_to_files):
        files_str = ' '.join(all_files)

    pattern = re.compile(fr"({variable_name}(?:\[[0-2]\])*\.[^=\[\]]+it={it}(?:\.[a-zA-Z0-9\-]+)+)")
    files = pattern.findall(files_str)

    files = [os.path.join(path_to_files, file) for file in files]
    files.sort()

    return files

def find_iterations(variable_name, path_to_files, return_files=False):
    """Finds all iterations available for files containing a given variable name.
    The pattern for filenames should be: 
    (variable name)(possible vector component indicator)(random text)it=(iteration number)(extension)

    Args:
        variable_name (str): string indicating the variable name in a filename.
        path_to_files (str): string providing the relative or absolute path to the data files.
        return_files (bool, optional): indicates whether a list of the full filenames is returned. Defaults to False.

    Returns:
        list, list (if return_files is True): first list contains all available iteration numbers. Second list contains
            the full filenames.
    """    

    files_str = ''
    for _, _, all_files in os.walk(path_to_files):
        files_str = ' '.join(all_files)

    if return_files:
        pattern = re.compile(fr"({variable_name}(?:\[[0-2]\])*\.[^=\[\]]+it=(\d+)(?:\.[a-zA-Z0-9\-]+)+)")

        def keys(text):
            key1 = re.search(r"\[([0-2])\]", text)
            key2 = re.search(r"it=(\d+)", text)

            if key1 is None:
                return [int(key2.group(1))]
            else:
                return [int(key2.group(1)), int(key1.group(1))]
        
        output = pattern.findall(files_str)
        files, it_list = zip(*output)
        
        files = [os.path.join(path_to_files, file) for file in files]
        files.sort(key=keys)

        it_list = list(map(int, list(set(it_list))))
        it_list.sort()
        
        return it_list, files

    else:
        pattern = re.compile(fr"{variable_name}(?:\[[0-2]\])*\.[^=\[\]]+it=(\d+)(?:\.[a-zA-Z0-9\-]+)+")
        
        it_list = pattern.findall(files_str)
        it_list = list(map(int, list(set(it_list))))
        it_list.sort()

        return it_list

def create_pyvista_grid(field_data, name='field_data', spacing=[1, 1, 1], origin=[0, 0, 0]):
    """Creates a pyvista UniformGrid containing the field data that was passed.
    Field data can be either vector data or scalar data.

    Args:
        field_data (list or np.ndarray): either a list containing all components of the
            vector field data, each as an np.ndarray or an np.ndarray containing the scalar
            field data.
        name (str, optional): name of the field data. This name will show 
            in the pyvista scalar bar. Defaults to 'field_data'.
        spacing (list, optional): grid spacing. Defaults to [1, 1, 1].
        origin (list, optional): lowest grid coordinates. Defaults to [0, 0, 0].

    Raises:
        ValueError: raised when field_data is not provided in the right format.

    Returns:
        pyvista.core.grid.UniformGrid: pyvista grid containing the fielddata.
    """

    if isinstance(field_data, np.ndarray):
        grid = pv.UniformGrid(dims=np.shape(field_data), spacing=spacing, origin=origin)
        grid[name] = field_data.flatten(order="F")
    
    elif isinstance(field_data, list):
        for data in field_data:
            if not isinstance(data, np.ndarray):
                raise ValueError("Data should be passed as an 3d numpy array or as a list of length 3 containing 3d numpy arrays for vector data.")
        
        grid = pv.UniformGrid(dims=np.shape(field_data[0]), spacing=spacing, origin=origin)
        grid[name] = np.column_stack((field_data[0].flatten(order="F"), field_data[1].flatten(order="F"), field_data[2].flatten(order="F")))

    else:
        raise ValueError("Data should be passed as an 3d numpy array or as a list of length 3 containing 3d numpy arrays for vector data.")

    return grid

def get_streamlines(
    field_data,
    name='field_data',
    spacing=[1, 1, 1],
    origin=[0, 0, 0],
    n_points=100,
    source_radius=20,
    radius=0.1,
    source_center=None,
    mirror_z=False,
    source=None,
    return_source=False
    ):

    """Calculates streamlines for a given pyvista UniformGrid containing vector field data. Log scale
    option for streamlines can be applied later in the add_mesh function.

    Args:
        field_data (list): list containing all components of the vector field data, each as an np.ndarray.
        name (str, optional): name of the field data. This name will show in the pyvista scalar bar. 
            Defaults to 'field_data'.
        spacing (list, optional): grid spacing. Defaults to [1, 1, 1].
        origin (list, optional): lowest grid coordinates. Defaults to [0, 0, 0].
        n_points (int, optional): number of seeding points for generating streamlines. Defaults to 100.
        source_radius (int, optional): radius of the sphere containing the seeding points. Defaults to 20.
        radius (float, optional): radius of the tubes showing the streamlines. Defaults to 0.1.
        source_center (list, optional): center of the sphere containing the seeding points. 
            Defaults to the center of the grid.
        mirror_z (bool, optional): indicates whether the streamlines are mirrored in the xy-plane. 
            Defaults to False.
        source (pyvista.DataSet, optional): seeding points for streamlines, will override the
            sphere source. Defaults to None.
        return_source (bool, optional): inidicates whether the seeding points from a sphere source are returned.
            Not available when providing a custom source. Defaults to False.

    Raises:
        ValueError: raised when return_source is True and a custom source is provided.
        TypeError: raised when field data is passed in an incorrect format.

    Returns:
        pyvista.PolyData, pyvista.PolyData (when return_source is True): pyvista PolyData containing the 
            streamlines as tubes and PolyData containing the seeding points.
    """

    if source is not None and return_source:
        raise ValueError("Return_source is not available when providing a source.")

    for data in field_data:
        if not isinstance(data, np.ndarray):
            raise TypeError("Data should be passed as a list of length 3 containing 3d numpy arrays.")
    
    if source_center is None:
        source_center = [(origin[i] + (origin[i] + spacing[i]*np.shape(field_data)[i+1])) / 2 for i in range(3)]

    grid = create_pyvista_grid(field_data, name=name, spacing=spacing, origin=origin)
    
    if return_source:
        streamlines, src = grid.streamlines(name, return_source=True, n_points=n_points, source_radius=source_radius, source_center=source_center)
        streamlines = streamlines.tube(radius=radius)

    elif source is not None:
        streamlines = grid.streamlines_from_source(source, name).tube(radius=radius)

    else:
        streamlines = grid.streamlines(name, n_points=n_points, source_radius=source_radius, source_center=source_center).tube(radius=radius)
    
    if mirror_z:
        transform_matrix = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, -1, 0],
                            [0, 0, 0, 1]])
        streamlines_lower = streamlines.transform(transform_matrix, inplace=False)
        streamlines = streamlines.merge(streamlines_lower, merge_points=True)
    
    if return_source:
        return streamlines, src
    
    else:
        return streamlines

def get_volume(
    field_data,
    name='field_data',
    spacing=[1, 1, 1],
    origin=[0, 0, 0],
    log_scale=False,
    mirror_z=False
    ):

    """Wrapper around create_pyvista_grid with more options like log_scale and mirror_z.

    Args:
        field_data (np.ndarray): np.ndarray containing the scalarfield data.
        name (str, optional): name of the field data. This name will show in the pyvista scalar bar.
            Defaults to 'field_data'.
        spacing (list, optional): grid spacing. Defaults to [1, 1, 1].
        origin (list, optional): lowest grid coordinates. Defaults to [0, 0, 0].
        log_scale (bool, optional): indicates whether the log10 should be taken of the scalar data. 
            Defaults to False.
        mirror_z (bool, optional): indicates whether the volume is mirrored in the xy-plane. 
            Defaults to False.

    Raises:
        TypeError: returned when field data is passed in an incorrect format.

    Returns:
        pyvista.core.grid.UniformGrid: grid containing scalar field data.
    """    

    if not isinstance(field_data, np.ndarray):
        raise TypeError("Data should be passed as a 3d numpy array.")

    # Getting axis and scalar field field
    if log_scale:
        field_data = np.log10(field_data)

    grid = create_pyvista_grid(field_data, name=name, spacing=spacing, origin=origin)

    # Rerun grid creation with updated origin, grid needs to be recreated as a UniformGrid is required
    if mirror_z:
        transform_matrix = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, -1, 0],
                            [0, 0, 0, 1]])
        grid_lower = grid.transform(transform_matrix, inplace=False)
        grid = grid.merge(grid_lower, merge_points=True)

        new_origin = origin
        new_origin[2] = grid.bounds[4]

        field_data_flipped = np.flip(field_data, axis=2)
        field_data = np.concatenate((field_data_flipped, field_data), 2)

        grid = create_pyvista_grid(field_data, name=name, spacing=spacing, origin=new_origin)

    return grid

def get_plot_object(field_data, **kwargs):
    """Wrapper around get_streamlines and get_volume that picks the right function
    based on the provided field data. Any remaining kwargs will be returned.

    Args:
        field_data (list or np.ndarray): either a list containing all components of the
            vector field data, each as an np.ndarray or an np.ndarray containing the scalar
            field data.

    Kwargs:
        name (str, optional): name of the field data. This name will show in the pyvista scalar bar.
            Defaults to 'field_data'.
        spacing (list, optional): grid spacing. Defaults to [1, 1, 1].
        origin (list, optional): lowest grid coordinates. Defaults to [0, 0, 0].
        mirror_z (bool, optional): indicates whether the volume or streamlines are mirrored in the xy-plane. 
            Defaults to False.
        log_scale (bool, optional): indicates whether the log10 should be taken of the scalar data. Only applied
            to scalar field data as log_scale argument is not supported by the pyvista function add_volume.
            Defaults to False.
        n_points (int, optional): number of seeding points for generating streamlines. Defaults to 100.
        source_radius (int, optional): radius of the sphere containing the seeding points. Defaults to 20.
        radius (float, optional): radius of the tubes showing the streamlines. Defaults to 0.1.
        source_center (list, optional): center of the sphere containing the seeding points. 
            Defaults to the center of the grid.
        source (pyvista.DataSet, optional): seeding points for streamlines, will override the
            sphere source. Defaults to None.
        return_source (bool, optional): inidicates whether the seeding points from a sphere source are returned.
            Not available when providing a custom source. Defaults to False.

    Raises:
        ValueError: raised if neither scalar data or three-dimensional vector data is passed.

    Returns:
        pyvista.PolyData, pyvista.PolyData (when return_source is True), dict: if field data is vector data,
            pyvista PolyData containing the streamlines as tubes, PolyData containing the seeding points and
            dictionary containing remaining keyword arguments.
        
        pyvista.core.grid.UniformGrid, dict: if field data is scalar data, grid containing scalar field data
            and dictionary containing remaining keyword arguments.
    """

    name = kwargs.pop('name', 'field_data')
    spacing = kwargs.pop('spacing', [1, 1, 1])
    origin = kwargs.pop('origin', [0, 0, 0])
    mirror_z = kwargs.pop('mirror_z', False)

    if len(field_data) == 3:
        n_points = kwargs.pop('n_points', 100)
        source_radius = kwargs.pop('source_radius', 20)
        radius = kwargs.pop('radius', 0.1)
        source_center = kwargs.pop('source_center', None)
        source = kwargs.pop('source', None)
        return_source = kwargs.pop('return_source', False)

        returns = get_streamlines(
            field_data,
            name=name,
            spacing=spacing,
            origin=origin,
            n_points=n_points,
            source_radius=source_radius,
            radius=radius,
            source_center=source_center,
            mirror_z=mirror_z,
            source=source,
            return_source=return_source
            )

        if return_source:
            mesh, src = returns
            return mesh, src, kwargs
        
        else:
            mesh = returns
            return mesh, kwargs
    
    elif len(field_data) == 1:
        log_scale = kwargs.pop('log_scale', False)

        mesh = get_volume(
            field_data[0],
            name=name,
            spacing=spacing,
            origin=origin,
            log_scale=log_scale,
            mirror_z=mirror_z
            )

        kwargs['scalars'] = name

        return mesh, kwargs

    else:
        raise ValueError(f"Field data has incorrect size. Size should be either 3 (for vector fields) or 1 (for scalar fields). Not {len(field_data)}.")

def custom_opacity(n_colors, n_star_colors, star_color, opacity_value, cmap):
    """Function for creating constant opacity mappings, with the ability for the
    higher end of the mapping to have no transparency to highlight this part of the
    volume rendering. A matching colormap can be created which modifies an existing
    colormap to include a single color for the part of the opacity mapping which has no
    transparency.

    Args:
        n_colors (int): total length of the opacity mapping and colormap.
        n_star_colors (int): length of the part of the opacity mapping and colormap with
            no transparency.
        star_color (np.array, len 4): rgba color for the high end of the colormap.
        opacity_value (float): transparency value.
        cmap (str): original colormap.

    Returns:
        list, matplotlib.colors.ListedColormap: list containing transparency values and
            a new colormap.
    """

    opacity = np.full(n_colors, opacity_value*n_colors, dtype='float')
    opacity[0] = 0
    if n_star_colors == 0:
        pass

    else:
        opacity[-n_star_colors:] = n_colors
        
    cmap = cm.get_cmap(cmap, n_colors-n_star_colors)

    newcolors = cmap(np.linspace(0, 1, n_colors-n_star_colors))
    for _ in range(n_star_colors):
        newcolors = np.concatenate((newcolors, [star_color]), axis=0)
    new_cmap = colors.ListedColormap(newcolors)

    return opacity, new_cmap

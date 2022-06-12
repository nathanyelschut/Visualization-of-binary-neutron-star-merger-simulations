"""FieldVis module for plotting or animating pyvista objects created with the fieldvis.field_dp module. 

Plotting options include two-dimensional slice plots or three-dimensional plots of the entire scene.
"""

import multiprocessing
import os
import platform
import shutil
import types
from math import sqrt

from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from fieldvis import field_dp

def plot_slice(field_data, slice_axis, slice_coordinate, cmap='jet', log_scale=False, clim=None, fig_scale=5, name='slice_plot.png', origin=None, spacing=None):
    """Shows a slice of a three-dimensional scalar field at a given coordinate on a given axis.
    Can be used to find a suitable data range for making volume renderings.

    Args:
        field_data (np.ndarray): three-dimensional numpy array containing scalar field data.
        slice_axis (str): string indicating on which axis to slice. 'x', 'y', 'z' as possible options.
        slice_coordinate (float or int): position on the sliced axis, where the slice will happen.
        cmap (str, optional): matplotlib colormap. Defaults to 'jet'.
        log_scale (bool, optional): indicates whether or not the log10 of the scalar field data should be taken. 
            Defaults to False.
        clim (list or tuple len 2, optional): clips scalar bar to within the range provided. Defaults to None.
        fig_scale (int, optional): scale of the figure. Defaults to 5.
        name (str, optional): name for a screenshot taken of the plot. Defaults to 'slice_plot.png'.
        origin (list or tuple len 3, optional): lowest coordinates corresponding to the field data. Defaults to [0, 0, 0].
        spacing (_type_, optional): grid spacing corresponding to the field data. Defaults to [1, 1, 1].

    Returns:
        np.ndarray, float: two-dimensional array containing the sliced data and the closest available coordinate on the grid.
    """    

    if origin is None or not isinstance(origin, (list, tuple)) or len(origin) != 3:
        origin = [0, 0, 0]

    if spacing is None or not isinstance(spacing, (list, tuple)) or len(spacing) != 3:
        spacing = [1, 1, 1]

    # Get axis
    axes_dimensions = [shape + 1 for shape in np.shape(field_data)]
    max_coords = [origin[i] + spacing[i] * axes_dimensions[i] for i in range(3)]

    x, y, z = [np.arange(origin[i], max_coords[i] + spacing[i], spacing[i]) for i in range(3)]

    # Create figure
    fig = plt.figure()
    ax = plt.axes()

    # Create plot settings depending on the slice axis.
    if slice_axis == 'x':
        aspect_ratio = len(y) / len(z)

        closest_coordinate = min(x, key=lambda x:abs(x-slice_coordinate))
        slice_index = list(x).index(closest_coordinate)

        sliced_data = field_data[slice_index,:,:]

        ax.set_xlabel('y-axis')
        ax.set_ylabel('z-axis')
        ax.set_title(f'x={closest_coordinate}')

        extent = [min(y), max(y), min(z), max(z)]

    elif slice_axis == 'y':
        aspect_ratio = len(x) / len(z)

        closest_coordinate = min(y, key=lambda x:abs(x-slice_coordinate))
        slice_index = list(y).index(closest_coordinate)

        sliced_data = field_data[:,slice_index,:]

        ax.set_xlabel('x-axis')
        ax.set_ylabel('z-axis')
        ax.set_title(f'y={closest_coordinate}')

        extent = [min(x), max(x), min(z), max(z)]

    elif slice_axis == 'z':
        aspect_ratio = len(x) / len(y)

        closest_coordinate = min(z, key=lambda x:abs(x-slice_coordinate))
        slice_index = list(z).index(closest_coordinate)

        sliced_data = field_data[:,:,slice_index]

        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        ax.set_title(f'z={closest_coordinate}')

        extent = [min(x), max(x), min(y), max(y)]

    # Create final plot
    fig.set_size_inches(aspect_ratio*fig_scale, fig_scale)

    if log_scale:
        im = ax.imshow(sliced_data, norm=colors.LogNorm(vmin=np.min(sliced_data), vmax=np.max(sliced_data)), origin='lower', cmap=cmap, extent=extent)
    else:
        im = ax.imshow(sliced_data, origin='lower', cmap=cmap, extent=extent)

    plt.colorbar(im)
    if clim is not None:
        im.set_clim(vmin=clim[0], vmax=clim[1])

    plt.savefig(name)
    
    return sliced_data, closest_coordinate

def plotter(plot_objects, add_plot_objects_kwargs, plotter_settings={}):
    """Creates a rendered image of either streamline data in the form of a
    pyvista.core.pointset.PolyData object or scalar field data in the form of a 
    pyvista.core.grid.UniformGrid.

    Args:
        plot_objects (list): list of streamline or scalar field data to plot.
        add_plot_objects_kwargs (list): list of dictionaries containing plot options 
            for the given plot object.
        plotter_settings (dict, optional): dictionary containing general plot settings. 
            Defaults to {}.

    Add_plot_object_kwargs for streamline plots:
        All possible keyword arguments for the pyvista add_mesh function.
        See https://docs.pyvista.org/api/plotting/_autosummary/pyvista.Plotter.add_mesh.html.
        Some keyword arguments might not be applicable to the streamlines.

    Add_plot_object_kwargs for volume plots:
        All possible keyword arguments for the pyvista add_volume function.
        See https://docs.pyvista.org/api/plotting/_autosummary/pyvista.Plotter.add_volume.html.
        Some keyword arguments might not be applicable to the volume plot.

    Plotter_settings (documentation from https://docs.pyvista.org/api/plotting/_autosummary/pyvista.Plotter.html):
        off_screen (bool): Renders off screen when True. Useful for automated screenshots.
        notebook (bool): When True, the resulting plot is placed inline a jupyter notebook.
            Assumes a jupyter console is active. Automatically enables off_screen. Defaults to False.
        window_size (list): Window size in pixels. Defaults to [1024, 768].
        multi_samples (int):The number of multi-samples used to mitigate aliasing. 4 is a
            good default but 8 will have better results with a potential impact on performance.
            Defaults to 4.
        line_smoothing (bool): If True, enable line smoothing. Defaults to True
        polygon_smoothing (bool): If True, enables polygon smoothing. Defaults to False
        lighting (str): What lighting to set up for the plotter. Accepted options:
            * 'light_kit': a vtk Light Kit composed of 5 lights.
            * 'three lights': illumination using 3 lights.
            * 'none': no light sources at instantiation.
            Defaults to 'light_kit'.
        transparent_background (bool): Enables transparent backgrounds in screenshots when True. 
            Defaults to False
        screenshot (bool or str): When a string is passed a screenshot of the initial plot state 
            is saved with the given string as file name. Defaults to False
        jupyter_backend (str): Jupyter notebook plotting backend to use. One of the following:
            * 'none' : Do not display in the notebook.
            * 'pythreejs' : Show a pythreejs widget
            * 'static' : Display a static figure.
            * 'ipygany' : Show a ipygany widget
            * 'panel' : Show a panel widget.
            Defaults to 'none'.
        return_cpos (bool): Return the last camera position from the render window when enabled.
            Deafualts to False.
        headless_display (bool): indicates whether a headless display will be made using Xvfb. Only
            available on Linux. Useful for remote visualizations on data processing services. 
            Defaults to False.
            
        show_grid (bool): Gridlines and axes labels are shown when True. Defaults to False.

        Additional keyword arguments include pyvista Plotter() attributes.
        See https://docs.pyvista.org/api/plotting/_autosummary/pyvista.Plotter.html Attributes.

    Returns:
        list len 3 (if return_cpos is True): list containing 3 tuples of len 3 indicating the camera position
            focal point and up vector respectively.
    """

    # Get keyword arguments
    plotter_init_kwargs = dict(
        off_screen=plotter_settings.pop('off_screen', False),
        notebook=plotter_settings.pop('notebook', False),
        window_size=plotter_settings.pop('window_size', [1024, 768]),
        multi_samples=plotter_settings.pop('multi_samples', 4),
        line_smoothing=plotter_settings.pop('line_smoothing', True),
        polygon_smoothing=plotter_settings.pop('polygon_smoothing', False),
        lighting=plotter_settings.pop('lighting', 'light_kit')
    )

    transparent_background = plotter_settings.pop('transparent_background', False)
    screenshot = plotter_settings.pop('screenshot', False)
    jupyter_backend = plotter_settings.pop('jupyter_backend', 'none')
    return_cpos = plotter_settings.pop('return_cpos', False)
    show_grid = plotter_settings.pop('show_grid', False)
    headless_display = plotter_settings.pop('headless_display', False)
    
    # Setup headless display when plotting off-screen. Needed for using remote servers.
    if platform.system() == 'Linux' and headless_display:
        from xvfbwrapper import Xvfb
        vdisplay = Xvfb()
        vdisplay.start()

    # Initialize plotter
    pv.set_jupyter_backend(jupyter_backend)
    if transparent_background:
        pv.rcParams['transparent_background'] = True

    p = pv.Plotter(**plotter_init_kwargs)
    
    for plot_object, add_plot_object_kwargs in list(zip(plot_objects, add_plot_objects_kwargs)):
        # Range settings
        if 'clim' in add_plot_object_kwargs:
            clim = add_plot_object_kwargs.pop('clim')
            field = plot_object.active_scalars
            
            if isinstance(clim, (list, tuple)):
                if clim[0] is None:
                    clim[0] = np.nanmin(field.ravel())
                if clim[1] is None:
                    clim[1] = np.nanmax(field.ravel())
            add_plot_object_kwargs['clim'] = clim

        if isinstance(plot_object, pv.core.pointset.PolyData):
            p.add_mesh(mesh=plot_object, **add_plot_object_kwargs)

        elif isinstance(plot_object, pv.core.grid.UniformGrid):
            p.add_volume(volume=plot_object, **add_plot_object_kwargs)

    # Additional plotter settings
    for key, value in plotter_settings.items():
        setattr(p, key, value)
        setattr(p.camera, key, value)

    if show_grid:
        p.show_grid()

    if plotter_init_kwargs['off_screen'] and isinstance(screenshot, str):
        p.screenshot(screenshot)

    if return_cpos:
        cam = p.show(screenshot=screenshot, return_cpos=return_cpos)
        if platform.system() == 'Linux' and headless_display:
            vdisplay.stop()
        return cam

    else:
        p.show(screenshot=screenshot, return_cpos=return_cpos)
        if platform.system() == 'Linux' and headless_display:
            vdisplay.stop()
        return

def get_meshes(data_readers, data_index, plot_objects_kwargs):
    """Function which uses provided data readers to obtain pyvista object using
    the get_plot_object function from field_dp.


    Args:
        data_readers (list): list of functions which have an integer as an argument which
            specifies the index of the dataset from a collection of iterations to be visualized.
            The function should return an np.ndarray representing the requested data.
        data_index (int): index of the data to be visualized. Example: if there are 3 datasets available
            representing different iterations of a simulation, then indexes can be 0, 1 or 2.
        plot_objects_kwargs (list): list of dictionaries containing plot options for the given plot object.

    Returns:
        list, list: list of pyvista objects that are to be visualized and a list of dictionaries with
            the remaining plot settings for each plot object.
    """
    
    meshes = []
    new_plot_objects_kwargs = []

    for data_reader, plot_object_kwargs in zip(data_readers, plot_objects_kwargs):
        data = data_reader(data_index)
        mesh, remaining_kwargs = field_dp.get_plot_object(data, **plot_object_kwargs)

        meshes.append(mesh)
        new_plot_objects_kwargs.append(remaining_kwargs)

    return meshes, new_plot_objects_kwargs

def create_frame(data_readers, data_index, plot_objects_kwargs, plotter_kwargs):
    """Helper function for multiprocessing. Passes arguments to the plotter function
    in the right format.

    Args:
        data_readers (list): list of functions which have an integer as an argument which
            specifies the index of the dataset from a collection of iterations to be visualized.
            The function should return an np.ndarray representing the requested data.
        data_index (int): index of the data to be visualized. Example: if there are 3 datasets available
            representing different iterations of a simulation, then indexes can be 0, 1 or 2.
        plot_objects_kwargs (list): list of dictionaries containing plot options for the given plot object.
        plotter_kwargs (dict): dictionary containing general plot settings. 
    """

    meshes, plot_objects_kwargs = get_meshes(data_readers, data_index, plot_objects_kwargs)

    plotter(meshes, plot_objects_kwargs, plotter_kwargs)

def animator(data_readers, plot_objects_kwargs, n_datasets, save_path='', n_processes=1, **kwargs):
    """Function for creating animations of a provided collection of datasets. The function creates the frames
    for the animation, but does not put them together in a movie. For this use the following ffmpeg command:
        ffmpeg -y -vcodec png -framerate 11 -i frame_%04d.png -pix_fmt yuv420p  -vcodec libx264 -crf 22 -threads 0 -preset slow name.mp4
    Framerate can be adjusted by changing the '-framerate 11' from 11 to any framerate desired.

    Args:
        data_readers (list): list of functions which have an integer as an argument which
            specifies the index of the dataset from a collection of iterations to be visualized.
            The function should return an np.ndarray representing the requested data.
        plot_objects_kwargs (list): list of dictionaries containing plot options for the given plot object.
        n_datasets (int): number of frames worth of datasets available.
        save_path (str, optional): path to where the animation frames will be stored. Defaults to 'animation_i'.
        n_processes (int, optional): number of computing cores to use for generating the frames in parallel. Defaults to 1.
    
    Keyword arguments provided are for general plot settings. For possible plot object and general plot settings 
    see the documentation of field_plot.plotter.

    Keyword arguments specific to animator include:
        scene_view_args (dict): dictionary with settings for creating an intro shot of the plot scene. Arguments include
            * cam_speed (int): distance units / second for movement of the camera between two selected angles.
            * cam_positions (list of tuples len 3): list containing the camera positions where the camera should pause. List
                should include at least two positions.
            * pause_time (int or float): time in seconds that the camera should pause at each provided camera position.
        framerate (int): frames per second. Relevant for creating intro shots of the plot scene and should be the same as the
            framerate provided to the ffmpeg command. Defaults to 11.
    """

    # Check arguments
    check_animator_args(data_readers, plot_objects_kwargs, n_datasets)
    save_path = check_save_path(save_path)

    kwargs['off_screen'] = True
    frame = 0

    # Create intro frames
    if 'scene_view_args' in kwargs:
        meshes, new_plot_objects_kwargs = get_meshes(data_readers, 0, plot_objects_kwargs)

        scene_view_args = kwargs.pop('scene_view_args')
        frame, position = create_scene_view_frames(scene_view_args, meshes, new_plot_objects_kwargs, save_path, n_processes, **kwargs)
        kwargs['position'] = position

    # Generate sources for fieldlines
    for data_reader, plot_object_kwargs in zip(data_readers, plot_objects_kwargs):
        test_data = data_reader(0)

        if len(test_data) == 3:
            plot_object_kwargs['return_source'] = True
            _, src, _ = field_dp.get_plot_object(test_data, **plot_object_kwargs)
            plot_object_kwargs['return_source'] = False
            plot_object_kwargs['source'] = src

    # Collect arguments for producing each frame
    frame_settings = []
    for data_index in range(n_datasets):
        kwargs['screenshot'] = os.path.join(save_path, 'frames', f'frame_{frame:04d}.png')
        frame += 1
        
        settings = (data_readers, data_index, plot_objects_kwargs, kwargs.copy())
        frame_settings.append(settings)
    
    # Produce frames
    for i in range(0, len(frame_settings), n_processes):
        if i+n_processes <= len(frame_settings):
            arguments = frame_settings[i:i+n_processes]
        else:
            arguments = frame_settings[i:]
        
        with multiprocessing.Pool(n_processes) as pool:
            pool.starmap(create_frame, arguments)
            pool.close()
            pool.join()

def check_animator_args(data_readers, plot_objects_kwargs, n_datasets):
    """Checks arguments passed to animator and raises errors if they are not
    correctly formatted.

    Args:
        data_readers (list): list of functions which have an integer as an argument which
            specifies the index of the dataset from a collection of iterations to be visualized.
            The function should return an np.ndarray representing the requested data.
        plot_objects_kwargs (list): list of dictionaries containing plot options for the given plot object.
        n_datasets (int): number of frames worth of datasets available.
    """

    if not isinstance(data_readers, (list, tuple)):
        raise TypeError(f"Incorrect type {type(data_readers)}. data_readers should be passed as a list or tuple of functions!")
    
    if not isinstance(plot_objects_kwargs, (list, tuple)):
        raise TypeError(f"Incorrect type {type(plot_objects_kwargs)}. plot_objects_kwargs should be passed as a list or tuple of generators!")

    if not isinstance(n_datasets, int):
        raise TypeError(f"Incorrect type {type(n_datasets)}. n_datasets should be passed as an integer")

    for data_reader in data_readers:
        if not isinstance(data_reader, types.FunctionType):
            raise TypeError(f"Incorrect type {type(data_reader)}. data_reader should be a function object!")
    
    for plot_object_kwargs in plot_objects_kwargs:
        if not isinstance(plot_object_kwargs, dict):
            raise TypeError(f"Incorrect type {type(plot_object_kwargs)}. plot_object_kwargs should be a dictionary object!")
    
    return

def check_save_path(save_path):
    """Checks if the save path provided to animator already exists and creates the directory
    if not. If no path is provided the default path animation_i will be created.

    Args:
        save_path (str, optional): path to where the animation frames will be stored.

    Returns:
        str: path to where the animation frames will be stored.
    """

    if save_path == '':
        i = 1
        directory_exists = True
        while directory_exists:
            save_path = f'animation_{i}'
            directory_exists = os.path.isdir(save_path)
            i+=1
        os.mkdir(save_path)
        os.mkdir(os.path.join(save_path, 'frames'))

    else:
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        if not os.path.isdir(os.path.join(save_path, 'frames')):
            os.mkdir(os.path.join(save_path, 'frames'))
    
    return save_path

def create_scene_view_frames(scene_view_args, meshes, plot_objects_kwargs, save_path, n_processes, **kwargs):
    """Generates the frames for a possible intro shot for the animation produced by animator.

    Args:
        scene_view_args (dict): dictionary with settings for creating an intro shot of the plot scene. Arguments include
            * cam_speed (int): distance units / second for movement of the camera between two selected angles.
            * cam_positions (list of tuples len 3): list containing the camera positions where the camera should pause. List
                should include at least two positions.
            * pause_time (int or float): time in seconds that the camera should pause at each provided camera position.
        meshes (list): list of streamline or scalar field data to plot.
        plot_objects_kwargs (list): list of dictionaries containing plot options for the given plot object.
        save_path (str): path to where the animation frames will be stored.
        n_processes (int): number of computing cores to use for generating the frames in parallel.

    Raises:
        ValueError: raised when 0 or 1 camera position(s) have been passed.

    Returns:
        int, tuple len 3: frame number of the next frame after the intro shot and the last camera position.
    """
    
    if 'cam_positions' in scene_view_args:
        cam_positions = scene_view_args.pop('cam_positions')

        if len(cam_positions) < 3:
            raise ValueError("At least 2 camera positions should be passed to scene_view_args.")

    else:
        raise ValueError("No camera positions have been passed to scene_view_args")

    pause_time = scene_view_args.pop('pause_time', 3)
    cam_speed = scene_view_args.pop('cam_speed', 5)
    framerate = kwargs.pop('framerate', 11)
    frame = 0
    scene_view_frame_settings = []
    
    for position in range(len(cam_positions) - 1):
        # Generate frames for cam point
        kwargs['position'] = cam_positions[position]
        copy_name = os.path.join(save_path, 'frames', f'frame_{frame:04d}.png')
        kwargs['screenshot'] = copy_name
        plotter(meshes, plot_objects_kwargs, kwargs.copy())
        frame += 1

        for i in range(0, (pause_time*framerate)-1):
            shutil.copy(copy_name, os.path.join(save_path, 'frames', f'frame_{frame:04d}.png'))
            frame += 1

        # Move from point to point
        distance = sqrt(sum((px - qx) ** 2.0 for px, qx in zip(cam_positions[position], cam_positions[position+1])))
        move_frames = round((distance / cam_speed) * framerate)
        x_coordinates = np.linspace(cam_positions[position][0], cam_positions[position+1][0], move_frames)
        y_coordinates = np.linspace(cam_positions[position][1], cam_positions[position+1][1], move_frames)
        
        # Create arch in z coordinate
        b = abs(cam_positions[position][2] - cam_positions[position+1][2])
        t = np.arange(0, move_frames, 1)
        if cam_positions[position][2] >= cam_positions[position+1][2]:
            z_coordinates = b * np.sqrt(1 - (t / (move_frames-1))**2) + cam_positions[position+1][2]
        
        elif cam_positions[position][2] < cam_positions[position+1][2]:
            t_reversed = np.flip(t)
            z_coordinates = b * np.sqrt(1 - (t_reversed / (move_frames-1))**2) + cam_positions[position][2]

        # Collect frame generation settings
        for i in range(move_frames):
            kwargs['position'] = (x_coordinates[i], y_coordinates[i], z_coordinates[i])
            kwargs['screenshot'] = os.path.join(save_path, 'frames', f'frame_{frame:04d}')
            settings = (meshes, plot_objects_kwargs, kwargs.copy())
            scene_view_frame_settings.append(settings)
            frame += 1

    # Last cam point pause
    kwargs['position'] = cam_positions[-1]
    copy_name = os.path.join(save_path, 'frames', f'frame_{frame:04d}.png')
    kwargs['screenshot'] = copy_name
    plotter(meshes, plot_objects_kwargs, kwargs.copy())
    frame += 1

    for i in range(0, (pause_time*framerate)-1):
        shutil.copy(copy_name, os.path.join(save_path, 'frames', f'frame_{frame:04d}.png'))
        frame += 1

    # Create frames
    for i in range(0, len(scene_view_frame_settings), n_processes):
        if i+n_processes <= len(scene_view_frame_settings):
            arguments = scene_view_frame_settings[i:i+n_processes]
        else:
            arguments = scene_view_frame_settings[i:]
        
        with multiprocessing.Pool(n_processes) as pool:
            pool.starmap(plotter, arguments)
            pool.close()
            pool.join()

    return frame, cam_positions[-1]

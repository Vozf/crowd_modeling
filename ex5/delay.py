import numpy as np
import holoviews as hv


def load_data(filename):
    """
    Loads data from a file.

    Args:
        filename (str): Path to the file.

    Returns:
        numpy.ndarray: Loaded data.
    """
    return np.loadtxt(filename)


def add_time_column(data):
    """
    Adds a time column to the data.

    Args:
        data (numpy.ndarray): Input data.

    Returns:
        numpy.ndarray: Data with an added time column.
    """
    time_column = np.arange(data.shape[0])[..., None]
    return np.hstack((time_column, data))


def plot(to_plot):
    """
    Plots the given HoloViews object.

    Args:
        to_plot: HoloViews object to plot.
    """
    hv.renderer('matplotlib').show(to_plot)


def plot_delayed(data, delay):
    """
    Plots the delayed version of the data against the original data.

    Args:
        data (numpy.ndarray): Input data.
        delay (int): Delay value.

    Returns:
        None
    """
    delayed = np.roll(data, delay, axis=0)
    plot(hv.Scatter((data, delayed)).opts(title=f'{delay=}'))


def plot_delayed_lorenz(x, delay, linewidth=0.1):
    """
    Plots the delayed version of the Lorenz attractor coordinate against the original coordinate.

    Args:
        x (numpy.ndarray): Coordinate values of the Lorenz attractor.
        delay (int): Delay value.
        linewidth (float): Line width for plotting.

    Returns:
        None
    """
    delayed = np.roll(x, delay, axis=0)
    delayedx2 = np.roll(x, delay * 2, axis=0)
    title = f'{delay=}'
    path = hv.Path3D((x, delayed, delayedx2)).opts(title=title).opts(hv.opts.Path3D(linewidth=linewidth))
    plot(path)


# Define the Lorenz system
def lorenz(start_point, sigma=10, beta=2.667, ro=28):
    """
    Computes the derivative of the Lorenz system at a given point.

    Args:
        start_point (tuple): The initial point (x, y, z) in the Lorenz system.
        sigma (float): Lorenz parameter.
        beta (float): Lorenz parameter.
        ro (float): Lorenz parameter.

    Returns:
        numpy.ndarray: The derivative of the Lorenz system at the given point.
    """
    x, y, z = start_point
    return np.array([sigma * (y - x), x * (ro - z) - y, x * y - beta * z])


# Define a function to draw the Lorenz attractor
def draw_lorenz(x=10., y=10.0, z=10., sigma=10, beta=8 / 3, ro=28, time=100, linewidth=.5, dt=0.01):
    """
    Draws the Lorenz attractor.

    Args:
        x (float): Initial x-coordinate.
        y (float): Initial y-coordinate.
        z (float): Initial z-coordinate.
        sigma (float): Lorenz parameter.
        beta (float): Lorenz parameter.
        ro (float): Lorenz parameter.
        time (float): Total time duration.
        linewidth (float): Line width for plotting.
        dt (float): Time step size.

    Returns:
        numpy.ndarray: Array representing the points of the Lorenz attractor.
    """
    line = []
    start_point = point = x, y, z
    timesteps = np.linspace(0, time, int(time / dt))
    for _ in timesteps:
        line.append(point)
        dpoint = lorenz(point, sigma, beta, ro)
        point = point + (dpoint * dt)

    line = np.array(line)
    title = f'{start_point=}\nfinal_point={line[-1]}\n{sigma=}, {beta=}, {ro=}'
    path = hv.Path3D(line).opts(title=title).opts(hv.opts.Path3D(linewidth=linewidth))

    plot(path)
    return line

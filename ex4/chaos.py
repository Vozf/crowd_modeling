import numpy as np


def logistic_map(x, r):
    """
    Computes the next value in the logistic map sequence.

    Args:
        x (float): The current value in the sequence.
        r (float): The growth rate parameter.

    Returns:
        float: The next value in the logistic map sequence.
    """
    return r * x * (1 - x)


def generate_bifurcation_diagram(x0=0.5, num_iterations=100, transient=100):
    """
    Generates the bifurcation diagram for the logistic map.

    Args:
        x0 (float): The initial value for the logistic map.
        num_iterations (int): The number of iterations to perform for each value of r.
        transient (int): The number of transient iterations to discard.

    Returns:
        tuple: A tuple containing two arrays: bifurcation_values and population_values.
            bifurcation_values: An array of bifurcation parameter values (r).
            population_values: An array of population values corresponding to each r value.
    """
    r_values = np.linspace(0., 4.0, 1000)
    population_values = []
    bifurcation_values = []

    for r in r_values:
        x = x0
        for _ in range(transient):
            x = logistic_map(x, r)

        for _ in range(num_iterations):
            x = logistic_map(x, r)
            population_values.append(x)
            bifurcation_values.append(r)

    return bifurcation_values, population_values


def lorenz(start_point, sigma=10, beta=2.667, ro=28):
    """
    Computes the derivative of the Lorenz system at a given point.

    Args:
        start_point (tuple): The initial point (x, y, z) in the Lorenz system.
        sigma (float): lorenz parameter
        beta (float): lorenz parameter
        ro (float): lorenz parameter

    Returns:
        numpy.ndarray: The derivative of the Lorenz system at the given point.
    """
    x, y, z = start_point
    return np.array([sigma * (y - x), x * (ro - z) - y, x * y - beta * z])

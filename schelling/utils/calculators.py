import numpy as np
from scipy.ndimage import convolve
from schelling.objects import Grid


def calculate_satisfaction(grid: Grid, kernel: np.ndarray) -> np.ndarray:
    """Calculate the satisfaction of each agent.

    Args:
        grid (Grid): The grid object.
        kernel (np.ndarray): The kernel to use for the convolution.

    Returns:
        np.ndarray: The satisfaction of each agent.
    """
    # Calculate the number of neighbors of each kind.
    color_1_neighbors = convolve(grid.grid == -1, kernel, mode="wrap")
    color_2_neighbors = convolve(grid.grid == 1, kernel, mode="wrap")

    # Calculate the satisfaction of each agent.
    satisfaction = np.zeros(grid.nr_agents)
    for i, agent in enumerate(grid.agents):
        kind = agent.kind
        location = agent.location
        if kind == -1:
            satisfaction[i] = color_1_neighbors[location] / grid.threshold
        else:
            satisfaction[i] = color_2_neighbors[location] / grid.threshold
    return satisfaction
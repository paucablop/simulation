import matplotlib.pyplot as plt
import numpy as np


def plot_agents(grid: np.ndarray) -> None:
    plt.imshow(grid, cmap="bwr", vmin=-1, vmax=1)
    plt.show()


def plot_satisfaction(grid: np.ndarray, kernel: np.ndarray) -> None:
    satisfaction = np.abs(np.convolve(grid, kernel, mode="same"))
    plt.imshow(satisfaction, cmap="bwr", vmin=-1, vmax=1)
    plt.show()

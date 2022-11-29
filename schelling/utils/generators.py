import numpy as np

from schelling.objects import Agent, Grid


def generate_grid(size: int, threshold: float, occupancy: float, color_ratio: float) -> Grid:
    return Grid(size, threshold, occupancy, color_ratio)

def generate_agents_for_grid(grid: Grid) -> list[Agent]:
    agents = np.empty(grid.nr_agents, dtype=Agent)
    for i in range(grid.nr_agents):
        kind = -1 if i < grid.nr_agents * grid.color_ratio else 1
        index = i
        location = grid.agent_locations[i]
        satisfaction = 0
        agents[i] = Agent(kind, index, location, satisfaction)
    return agents


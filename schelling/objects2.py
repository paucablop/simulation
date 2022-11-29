from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve


@dataclass
class Agent:
    kind: int
    index: int
    location: int
    satisfaction: float


@dataclass
class Grid:
    size: int
    threshold: float
    vacancy_ratio: float
    kind_ratio: float

    def __post_init__(self):
        self.nr_slots = self.size * self.size
        self.nr_vacancies = int(self.nr_slots * self.vacancy_ratio)
        self.nr_agents = self.nr_slots - self.nr_vacancies
        self.nr_kind_1 = int(self.nr_agents * self.kind_ratio)
        self.nr_kind_2 = self.nr_agents - self.nr_kind_1

    def place_agents(self):
        self.agents = np.empty(self.nr_agents, dtype=Agent)
        locations = set(np.random.choice(self.nr_slots, self.nr_agents, replace=False))

        for i in range(self.nr_agents):
            kind = -1 if i < self.nr_kind_1 else 1
            index = i
            location = locations[i]
            satisfaction = 0
            self.agents[i] = Agent(kind, index, location, satisfaction)

        self.agent_locations = set(locations)
        self.vacant_locations = set(range(self.nr_slots)) - self.agent_locations

    def update_grid(self, agents):
        self.grid = np.zeros((self.size, self.size), dtype=float)
        for agent in agents:
            row = agent.location // self.size
            col = agent.location % self.size
            self.grid[row, col] = agent.kind

    def calculate_satisfaction(self):
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        kind_1_grid = np.where(self.grid == -1, 1, 0)
        kind_2_grid = np.where(self.grid == 1, 1, 0)
        kind_1_satisfaction = convolve(kind_1_grid, kernel, mode="reflect", cval=0)
        kind_2_satisfaction = convolve(kind_2_grid, kernel, mode="reflect", cval=0)
        for agent in self.agents:
            row = agent.location // self.size
            col = agent.location % self.size
            if agent.kind == -1:
                agent.satisfaction = (
                    kind_1_satisfaction[row, col] / kind_2_satisfaction[row, col]
                )
            else:
                agent.satisfaction = (
                    kind_2_satisfaction[row, col] / kind_1_satisfaction[row, col]
                )

        satisfaction = convolve(
            self.grid,
            np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]),
            mode="reflect",
            cval=1,
        )
        for agent in self.agents:
            row = agent.location // self.size
            col = agent.location % self.size
            agent.satisfaction = np.abs(satisfaction[row, col] / 80)

        self.overall_satisfaction = np.mean(
            [agent.satisfaction for agent in self.agents]
        )

    def move_agents(self):
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        for agent in self.agents:
            if agent.satisfaction < self.threshold:
                current_location = agent.location
                new_location = np.random.choice(list(self.vacant_locations))
                agent.location = new_location
                
                new_satisfaction = self.point_convolution(self.grid, new_location, kernel)

                self.agent_locations.remove(agent.location)
                self.vacant_locations.add(agent.location)
                self.agent_locations.add(new_location)
                self.vacant_locations.remove(new_location)
                agent.location = new_location

    def point_convolution(self,
        matrix: np.ndarray, location: tuple[int, int], kernel: np.ndarray
    ) -> np.ndarray:
        """Convolve a kernel with a matrix at a given location."""
        row, col = location
        kernel_size = kernel.shape[0]
        kernel_radius = kernel_size // 2
        row_start = row - kernel_radius
        row_end = row + kernel_radius + 1
        col_start = col - kernel_radius
        col_end = col + kernel_radius + 1
        return convolve(
            matrix[row_start:row_end, col_start:col_end], kernel, mode="reflect", cval=0
        )


    def plot_satisfaction(self):
        satisfaction = convolve(
            self.grid,
            np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]),
            mode="reflect",
            cval=0,
        )
        plt.imshow(np.abs(satisfaction), cmap="Reds", alpha=1.0)
        plt.show()

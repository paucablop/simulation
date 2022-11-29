import numpy as np

from schelling.objects.agent import Agent


class Grid:
    def __init__(self, size: int, threshold: float, occupancy: float, color_ratio: float):
        self.size = size
        self.threshold = threshold
        self.occupancy = occupancy
        self.color_ratio = color_ratio
        self.nr_slots = self._set_nr_slots()
        self.nr_empty = self._set_nr_empty()
        self.nr_agents = self._set_nr_agents()
        self.nr_color_1 = self._set_nr_color_1()
        self.nr_color_2 = self._set_nr_color_2()
        self.agent_locations = self._set_agent_locations()
        self.empty_locations = self._set_empty_locations()
        self.agents = self._generate_agents()
        self.matrix = self._generate_matrix()

    def _set_nr_slots(self) -> int:
        return self.size * self.size

    def _set_nr_empty(self) -> int:
        return int(self.nr_slots * (1 - self.occupancy))
    
    def _set_nr_agents(self) -> int:
        return self.nr_slots - self.nr_empty
    
    def _set_nr_color_1(self) -> int:
        return int(self.nr_agents * self.color_ratio)
    
    def _set_nr_color_2(self) -> int:
        return self.nr_agents - self.nr_color_1

    def _set_agent_locations(self) -> set[int]:
        return np.random.choice(self.nr_slots, self.nr_agents, replace=False)

    def _set_empty_locations(self) -> set[int]:
        return np.array(list(set(range(self.nr_slots)) - set(self.agent_locations)))

    def _generate_agents(self) -> list[Agent]:
        agents = np.empty(self.nr_agents, dtype=Agent)
        for i in range(self.nr_agents):
            color = -1 if i < self.nr_agents * self.color_ratio else 1
            index = i
            location = self.agent_locations[i]
            satisfaction = 0
            agents[i] = Agent(color, index, location, satisfaction)
        return agents

    def _generate_matrix(self) -> np.ndarray:
        matrix = np.zeros((self.size, self.size), dtype=int)
        for agent in self.agents:
            row = agent.location // self.size
            col = agent.location % self.size
            matrix[row, col] = agent.color
        return matrix

    def _calculate_satisfaction_of_agent(self, agent: Agent) -> float:      
        # Get coordinates of agent in matrix
        row = agent.location // self.size
        column = agent.location % self.size

        # Get neighborhood matrix
        start_row = row - 1 if row > 0 else self.size - 1
        end_row = row + 2 if row < self.size - 1 else 1
        start_column = column - 1 if column > 0 else self.size - 1
        end_column = column + 2 if column < self.size - 1 else 1
        neighborhood = self.matrix[start_row:end_row, start_column:end_column]

        # Get neighbors of same color
        neighbors_color_1 = np.sum(np.where(neighborhood == -1, 1, 0))
        neighbors_color_2 = np.sum(np.where(neighborhood == 1, 1, 0))

        total_neighbors = neighbors_color_1 + neighbors_color_2
        if total_neighbors == 0:
            return 0.0

        # Calculate satisfaction
        if agent.color == -1:
            return neighbors_color_1 / total_neighbors
        else:
            return neighbors_color_2 / total_neighbors

    def calculate_satisfactions(self) -> None:
        for agent in self.agents:
            agent.satisfaction = self._calculate_satisfaction_of_agent(agent)

    def move_agent(self, agent: Agent) -> None:
        # Get empty and agent locations
        empty_locations = set(self.empty_locations)
        agent_locations = set(self.agent_locations)

        for virtual_location in empty_locations:
            # Virtual agent
            virtual_agent = Agent(agent.color, agent.index, virtual_location, 0)
            virtual_agent.satisfaction = self._calculate_satisfaction_of_agent(virtual_agent)

            if virtual_agent.move(self.threshold):
                # Update agent_locations
                agent_locations.remove(agent.location)
                agent_locations.add(virtual_location)

                # Update empty_locations
                empty_locations.remove(virtual_location)
                empty_locations.add(agent.location)

                # Update matrix
                row = agent.location // self.size
                column = agent.location % self.size
                self.matrix[row, column] = 0

                row = virtual_location // self.size
                column = virtual_location % self.size
                self.matrix[row, column] = agent.color

                # Update agent
                agent.location = virtual_location

                # Update agent locations
                self.agent_locations = agent_locations

                # Update empty locations
                self.empty_locations = empty_locations

                break
    
    def move_agents(self) -> None:
        for agent in self.agents:
            if agent.move(self.threshold):
                self.move_agent(agent)






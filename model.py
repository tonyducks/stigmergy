from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import numpy as np
import random

from agents import TransportAgent, HeavyObject, Obstacle


class TransportModel(Model):
    """
    Mesa Model for stigmergy-based cooperative transport.
    """
    def __init__(
        self,
        width=50,
        height=50,
        initial_agents=20,
        initial_objects=10,
        obstacle_fraction=0.1,
        pheromone_decay=0.01,
        pheromone_deposit=1.0,
        required_carriers=2,
        max_steps=2000
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.num_agents = initial_agents
        self.num_objects = initial_objects
        self.obstacle_fraction = obstacle_fraction
        self.pheromone_decay = pheromone_decay
        self.pheromone_deposit = pheromone_deposit
        self.required_carriers = required_carriers
        self.max_steps = max_steps

        # Step counter and running flag
        self.step_count = 0
        self.running = True

        # 1. Create grid and scheduler
        self.grid = MultiGrid(self.width, self.height, torus=False)
        self.schedule = RandomActivation(self)

        # 2. Create a 2D numpy array for pheromone
        self.pheromone = np.zeros((self.width, self.height), dtype=float)

        # 3. Place obstacles, heavy objects, and transport agents
        self._place_obstacles()
        self._place_heavy_objects()
        self._place_agents()

        # 4. DataCollector setup
        self.datacollector = DataCollector(
            model_reporters={
                "CompletedObjects": self.count_completed_objects,
                "AbandonedObjects": self.count_abandoned_objects,
                "AvgTimeToComplete": self.average_time_to_completion,
                "AvgRedundancy": self.average_redundancy,
            },
            agent_reporters={
                "Carrying": lambda a: getattr(a, "carrying_object_id", None) is not None,
                "LastDeposit": lambda a: getattr(a, "last_deposit", 0),
            }
        )

    def _place_obstacles(self):
        total_cells = self.width * self.height
        num_obstacles = int(total_cells * self.obstacle_fraction)
        all_positions = [(x, y) for x in range(self.width) for y in range(self.height)]
        obstacle_positions = random.sample(all_positions, num_obstacles)

        for pos in obstacle_positions:
            obs = Obstacle(self.next_id(), pos, self)
            self.grid.place_agent(obs, pos)
            self.schedule.add(obs)

    def _place_heavy_objects(self):
        # Find empty cells (no obstacles or other agents)
        empty_cells = [
            (x, y)
            for (contents, x, y) in self.grid.coord_iter()
            if len(self.grid.get_cell_list_contents((x, y))) == 0
        ]
        chosen = random.sample(empty_cells, self.num_objects)

        for pos in chosen:
            obj = HeavyObject(
                self.next_id(), pos, self, required_carriers=self.required_carriers
            )
            self.grid.place_agent(obj, pos)
            self.schedule.add(obj)

    def _place_agents(self):
        # Find empty cells again
        empty_cells = [
            (x, y)
            for (contents, x, y) in self.grid.coord_iter()
            if len(self.grid.get_cell_list_contents((x, y))) == 0
        ]
        agent_positions = random.sample(empty_cells, self.num_agents)

        for pos in agent_positions:
            a = TransportAgent(self.next_id(), pos, self)
            self.grid.place_agent(a, pos)
            self.schedule.add(a)

    def step(self):
        # 1. Activate all agents in random order
        self.schedule.step()

        # 2. Evaporate pheromone globally
        self.pheromone *= (1 - self.pheromone_decay)
        # Threshold tiny values to zero (optional)
        self.pheromone[self.pheromone < 1e-6] = 0

        # 3. Collect data
        self.datacollector.collect(self)

        # 4. Increment step counter
        self.step_count += 1

        # 5. Termination criteria
        if (
            self.step_count >= self.max_steps
            or self.count_completed_objects() == self.num_objects
        ):
            self.running = False

    def count_completed_objects(self):
        count = sum(
            1
            for a in self.schedule.agents
            if isinstance(a, HeavyObject) and a.completed
        )
        return count

    def count_abandoned_objects(self):
        count = sum(
            1
            for a in self.schedule.agents
            if isinstance(a, HeavyObject) and not a.discovered
        )
        return count

    def average_time_to_completion(self):
        times = []
        for a in self.schedule.agents:
            if isinstance(a, HeavyObject) and a.completed:
                duration = a.completion_time - a.discovery_time
                times.append(duration)
        return float(np.mean(times)) if times else 0.0

    def average_redundancy(self):
        redundancies = []
        for a in self.schedule.agents:
            if isinstance(a, HeavyObject) and a.completed:
                redundancies.append(a.max_carriers_used - a.required_carriers)
        return float(np.mean(redundancies)) if redundancies else 0.0

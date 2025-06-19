from mesa import Agent
import random


class Obstacle(Agent):
    """
    Static obstacle: occupies a grid cell and never moves.
    """
    def __init__(self, unique_id, pos, model):
        super().__init__(unique_id, model)
        self.pos = pos

    def step(self):
        pass  # Obstacles do not act


class HeavyObject(Agent):
    """
    A heavy object that requires multiple TransportAgents to move.
    Once enough agents latch onto it, it will move (with its carriers)
    toward a fixed dropoff_cell.
    """
    def __init__(self, unique_id, pos, model, required_carriers=2):
        super().__init__(unique_id, model)
        self.pos = pos
        self.required_carriers = required_carriers
        self.current_carriers = set()   # IDs of TransportAgents that have latched
        self.discovered = False
        self.completed = False
        self.discovery_time = None
        self.completion_time = None
        self.max_carriers_used = 0
        # Dropoff cell is fixed at (0, 0) by default; model can override if desired.
        self.dropoff_cell = (0, 0)

    def step(self):
        if self.completed:
            return

        # If discovered but discovery_time not yet recorded, record it
        if self.discovered and self.discovery_time is None:
            self.discovery_time = self.model.step_count

        # Check how many latched agents are actually still on this cell
        cell_contents = self.model.grid.get_cell_list_contents([self.pos])
        latched_agents = [
            a for a in cell_contents
            if isinstance(a, TransportAgent) and a.unique_id in self.current_carriers
        ]

        # Update max_carriers_used for redundancy tracking
        if len(latched_agents) > self.max_carriers_used:
            self.max_carriers_used = len(latched_agents)

        # If not enough carriers yet, wait for more
        if len(latched_agents) < self.required_carriers:
            return

        # Determine a move toward dropoff_cell using Manhattan reduction
        x, y = self.pos
        dx = self.dropoff_cell[0] - x
        dy = self.dropoff_cell[1] - y

        possible_moves = []
        if dx < 0:
            possible_moves.append((x - 1, y))
        elif dx > 0:
            possible_moves.append((x + 1, y))
        if dy < 0:
            possible_moves.append((x, y - 1))
        elif dy > 0:
            possible_moves.append((x, y + 1))

        # Filter out invalid moves (off-grid or obstacle)
        valid_moves = []
        for nx, ny in possible_moves:
            if not (0 <= nx < self.model.width and 0 <= ny < self.model.height):
                continue
            contents = self.model.grid.get_cell_list_contents([(nx, ny)])
            if any(isinstance(c, Obstacle) for c in contents):
                continue
            valid_moves.append((nx, ny))

        if not valid_moves:
            # Cannot move this turn (blocked). Wait.
            return

        new_pos = random.choice(valid_moves)

        # Move the heavy object
        self.model.grid.move_agent(self, new_pos)
        self.pos = new_pos

        # Move all latched agents with it
        for agent in latched_agents:
            self.model.grid.move_agent(agent, new_pos)
            agent.pos = new_pos

        # Check if arrived at dropoff
        if new_pos == self.dropoff_cell:
            self.completed = True
            self.completion_time = self.model.step_count
            # Record final redundancy count
            self.max_carriers_used = max(self.max_carriers_used, len(latched_agents))
            # Detach carriers
            for agent in latched_agents:
                agent.latching = False
                agent.carrying_object_id = None
            self.current_carriers.clear()


class TransportAgent(Agent):
    """
    Mobile agent that searches for HeavyObject, drops pheromone,
    and latches onto objects when found. When enough agents latch,
    the HeavyObject.step() handles movement.
    """
    def __init__(self, unique_id, pos, model):
        super().__init__(unique_id, model)
        self.pos = pos
        self.latching = False
        self.carrying_object_id = None
        self.last_deposit = 0

    def step(self):
        # If currently latched (and object is being moved), do nothing—
        # HeavyObject.step() will reposition the agent.
        if self.latching and self.carrying_object_id is not None:
            return

        # Check for HeavyObject at current cell
        cell_contents = self.model.grid.get_cell_list_contents([self.pos])
        heavy_objs = [
            obj for obj in cell_contents
            if isinstance(obj, HeavyObject) and not obj.completed
        ]
        if heavy_objs:
            heavy = heavy_objs[0]
            heavy.discovered = True

            # If not yet latched and fewer than required_carriers, latch on
            if heavy.current_carriers is None:
                heavy.current_carriers = set()

            if len(heavy.current_carriers) < heavy.required_carriers:
                self.latching = True
                self.carrying_object_id = heavy.unique_id
                heavy.current_carriers.add(self.unique_id)
            # If already enough carriers, do nothing (HeavyObject will move)
            return

        # Otherwise, not on any HeavyObject → “Search” mode
        self.leave_pheromone()
        self.move_by_pheromone_or_random()

    def leave_pheromone(self):
        x, y = self.pos
        self.model.pheromone[x][y] += self.model.pheromone_deposit
        self.last_deposit = self.model.pheromone_deposit

    def move_by_pheromone_or_random(self):
        # Get Moore neighbors (8-connected)
        neighbors = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        free_neighbors = []
        highest_pher = -1.0
        best_cells = []

        for n in neighbors:
            # Skip any neighbor that has an Obstacle
            contents = self.model.grid.get_cell_list_contents([n])
            if any(isinstance(a, Obstacle) for a in contents):
                continue
            # Read pheromone at neighbor
            pher_val = self.model.pheromone[n[0]][n[1]]
            if pher_val > highest_pher:
                highest_pher = pher_val
                best_cells = [n]
            elif pher_val == highest_pher:
                best_cells.append(n)
            free_neighbors.append(n)

        # If pheromone gradient exists, follow it
        if highest_pher > 0 and best_cells:
            new_pos = random.choice(best_cells)
        else:
            # Random walk among free neighbors
            if not free_neighbors:
                return  # No move possible
            new_pos = random.choice(free_neighbors)

        self.model.grid.move_agent(self, new_pos)
        self.pos = new_pos

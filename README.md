# stigmergy
How to run

    Install dependencies

pip install -r requirements.txt

Run a single simulation

    python run.py

        This will execute one instance of TransportModel until completion (either all heavy objects are delivered or max_steps is reached).

        At the end, it prints the last few rows of the model‐level DataFrame and the first few rows of the batch results.

    Run batch experiments

        The same run.py script also performs a parameter sweep using Mesa’s BatchRunner. After the single run output, you’ll see a summary of the first few rows from the batch DataFrame.

        Adjust param_dict or iterations as needed to explore pheromone decay, agent count, required carriers, obstacle density, etc.

Notes & Possible Extensions

    Dropoff Cell

        Currently, every HeavyObject has dropoff_cell = (0, 0). You can pass a dropoff_cell argument from TransportModel to each HeavyObject’s constructor if you’d like a different target location (or even multiple dropoff zones).

    Neighborhood Choice

        The code uses moore=True (8-neighbor) in TransportAgent.move_by_pheromone_or_random(). If you want strictly four‐directional movement, change this to moore=False.

    Pheromone Diffusion (Optional)

        Right now, only evaporation is implemented. If you want pheromone to spread, insert something like the following into TransportModel.step() just before or after evaporation:

from scipy.ndimage import convolve
kernel = np.array([
    [0.05, 0.10, 0.05],
    [0.10, 0.40, 0.10],
    [0.05, 0.10, 0.05]
])
self.pheromone = convolve(self.pheromone, kernel, mode='constant', cval=0.0)

Then follow with the decay line:

        self.pheromone *= (1 - self.pheromone_decay)

    Visualization

        If you want a live browser‐based visualization, you can add a visualization/ directory with a server.py that registers a CanvasGrid portrayal for each agent type and charts for the data. Mesa’s documentation has examples on how to hook the datacollector into live charts.

    Adjustable Parameters

        Feel free to tune pheromone_deposit, pheromone_decay, required_carriers, initial_agents, initial_objects, obstacle_fraction, and max_steps to see how they affect success rate, average completion time, redundancy, and abandonment.

With this code in place, you have a fully functioning Mesa simulation of stigmergy‐based cooperative transport. You can build on it, add visualization, or expand the agent logic as needed to evaluate different hypotheses about decentralized coordination.

from mesa.batchrunner import BatchRunner
from model import TransportModel

if __name__ == "__main__":
    # --------------------------------------------
    # 1) Single-run example to verify functionality
    # --------------------------------------------
    single_model = TransportModel(
        width=50,
        height=50,
        initial_agents=20,
        initial_objects=10,
        obstacle_fraction=0.1,
        pheromone_decay=0.02,
        pheromone_deposit=1.0,
        required_carriers=2,
        max_steps=2000
    )

    while single_model.running:
        single_model.step()

    model_data = single_model.datacollector.get_model_vars_dataframe()
    agent_data = single_model.datacollector.get_agent_vars_dataframe()
    print("=== Single Run Results ===")
    print(model_data.tail())
    # Optionally save to CSV:
    # model_data.to_csv("single_run_model.csv", index=False)
    # agent_data.to_csv("single_run_agent.csv", index=False)

    # --------------------------------------------
    # 2) Batch-run for parameter sweep
    # --------------------------------------------
    param_dict = {
        "initial_agents": [10, 20, 30],
        "pheromone_decay": [0.01, 0.02, 0.05],
        "required_carriers": [2, 3, 4],
        "obstacle_fraction": [0.05, 0.10, 0.20],
    }

    batch_run = BatchRunner(
        TransportModel,
        variable_parameters=param_dict,
        iterations=5,      # Five runs per parameter combination
        max_steps=2000,
        model_reporters={
            "CompletedObjects": lambda m: m.count_completed_objects(),
            "AvgTimeToComplete": lambda m: m.average_time_to_completion(),
            "AvgRedundancy": lambda m: m.average_redundancy(),
            "AbandonedObjects": lambda m: m.count_abandoned_objects(),
        },
    )

    batch_run.run_all()
    batch_df = batch_run.get_model_vars_dataframe()
    print("\n=== Batch Run Summary ===")
    print(batch_df.head(20))
    # Optionally save batch results:
    # batch_df.to_csv("batch_results.csv", index=False)

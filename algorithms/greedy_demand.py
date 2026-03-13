import os
import sys
import numpy as np

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from custom_environment.StationPlacementEnv import StationPlacement
import custom_environment.helpers as H


def run_greedy_demand(location="DongDa"):
    # Base directory and paths
    base_dir = os.path.join(project_root, "custom_environment", "data")
    graph_file = os.path.join(base_dir, "Graph", location, location + ".graphml")
    node_file = os.path.join(base_dir, "Graph", location, "nodes_extended_" + location + ".txt")
    plan_file = os.path.join(base_dir, "Graph", location, "existingplan_" + location + ".pkl")

    # Env for testing
    env = StationPlacement(graph_file, node_file, plan_file, location=location)
    obs, _ = env.reset(seed=1)

    total_reward = 0
    terminated = False
    truncated = False

    print(f"Starting Greedy Demand for {location}...")

    while not (terminated or truncated):
        # Action selection: Greedy by Demand
        # action 1 in StationPlacementEnv is choose_node_bydemand for free nodes
        # action 3 is choose_node_bydemand for occupied nodes

        station_list = [s[0][0] for s in env.plan_instance.plan]
        free_list = [node for node in env.node_list if node[0] not in station_list]

        if free_list:
            action = 1  # New station by demand
        else:
            action = 3  # Expand existing station by demand

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

    final_score = env.best_score
    num_stations = len(env.plan_instance.plan)
    print(
        f"Greedy Demand Finished: Total Reward = {total_reward:.2f}, Final Score = {final_score:.2f}, Stations = {num_stations}")

    # Save result
    results_dir = f"Results/greedy/{location}/"
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "demand_result.txt"), "w") as f:
        f.write(f"Reward: {total_reward}\nScore: {final_score}\nStations: {num_stations}")


if __name__ == '__main__':
    run_greedy_demand()

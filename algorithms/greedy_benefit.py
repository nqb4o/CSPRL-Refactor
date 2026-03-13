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


def run_greedy_benefit(location="DongDa"):
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

    print(f"Starting Greedy Benefit for {location}...")

    while not (terminated or truncated):
        # Action selection: Greedy by Benefit
        # action 0 in StationPlacementEnv is choose_node_new_benefit for free nodes
        # action 2 is choose_node_new_benefit for occupied nodes

        # We check if we should add to existing or build new
        # For simplicity in this greedy script, we follow the logic:
        # If we have budget, try to build or expand.
        # In the environment, action 0 is build new station by benefit.
        # Action 2 is add charger to existing station by benefit.

        # We can implement a simple heuristic: 
        # Alternating between building new and adding chargers, or just picking action 0 if budget allows.
        # However, a pure "Greedy by Benefit" should probably just always pick the best action according to benefit.

        # Let's try action 0 (build new) as long as we have free nodes, 
        # otherwise action 2 (expand existing).

        station_list = [s[0][0] for s in env.plan_instance.plan]
        free_list = [node for node in env.node_list if node[0] not in station_list]

        if free_list:
            action = 0  # New station by benefit
        else:
            action = 2  # Expand existing station by benefit

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

    final_score = env.best_score
    num_stations = len(env.plan_instance.plan)
    print(
        f"Greedy Benefit Finished: Total Reward = {total_reward:.2f}, Final Score = {final_score:.2f}, Stations = {num_stations}")

    # Save result
    results_dir = f"Results/greedy/{location}/"
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "benefit_result.txt"), "w") as f:
        f.write(f"Reward: {total_reward}\nScore: {final_score}\nStations: {num_stations}")


if __name__ == '__main__':
    run_greedy_benefit()

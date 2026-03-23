import os
import sys
import pickle
from math import ceil

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import osmnx as ox
from stable_baselines3 import DQN

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import custom_environment.helpers as H
from custom_environment.StationPlacementEnv import StationPlacement
from algorithms.ga.ga_utils import GAPolicy, unflatten_weights


def prepare_existing_plan(my_plan, my_node_list):
    my_cost_dict = {}
    my_node_dict = {}
    for my_node in my_node_list:
        my_node_dict[my_node[0]] = {}  # prepare node_dict
        my_node[1]["charging station"] = None
        my_node[1]["distance"] = None

    for index in range(len(my_plan)):
        my_plan[index] = H.s_dictionnary(my_plan[index], my_node_list)

    my_node_list, _, _ = H.station_seeking(my_plan, my_node_list, my_node_dict, my_cost_dict)

    for index in range(len(my_plan)):
        my_plan[index] = H.s_dictionnary(my_plan[index], my_node_list)

    return my_node_list, my_plan


def travel_metric(my_node_list):
    """Calculates the mean travel time in minutes."""
    big_travel_list = []
    for my_node in my_node_list:
        travel = my_node[1]["distance"] / H.VELOCITY * 60
        times = ceil(10 * H.weak_demand(my_node))
        for _ in range(times):
            big_travel_list.append(travel)
    
    travel_mean = max(big_travel_list) if big_travel_list else 0
    return travel_mean


def waiting_metric(my_plan):
    """Calculates the mean waiting time in minutes."""
    big_waiting_list = []
    for my_station in my_plan:
        times = ceil(my_station[2]["D_s"])
        for _ in range(times):
            big_waiting_list.append(my_station[2]["W_s"] * 60)
            
    wait_mean = max(big_waiting_list) if big_waiting_list else 0
    return wait_mean


def eci_test(
    my_plan,
    my_node_list,
    my_norm_benefit,
    my_norm_charging,
    my_norm_waiting,
    my_norm_travel,
    my_norm_fairness,
    grid_penalty=None,
):
    score, benefit, cost, fairness, charg_time, wait_time, cost_travel = H.norm_score(
        my_plan,
        my_node_list,
        my_norm_benefit,
        my_norm_charging,
        my_norm_waiting,
        my_norm_travel,
        my_norm_fairness,
        grid_penalty,
    )
    return score


def run_episode(agent, env, agent_type="rl"):
    obs, _ = env.reset(seed=1)
    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        if agent_type == "rl":
            action, _ = agent.predict(obs, deterministic=True)
        elif agent_type == "ga":
            action = agent.select_action(obs)
        elif agent_type in ["greedy_benefit", "greedy_demand"]:
            station_list = [s[0][0] for s in env.plan_instance.plan]
            free_list = [node for node in env.node_list if node[0] not in station_list]
            if agent_type == "greedy_benefit":
                action = 0 if free_list else 2
            else:
                action = 1 if free_list else 3

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

    best_node_list, best_plan = env.render()

    # Calculate all metrics
    norm_benefit = env.plan_instance.norm_benefit
    norm_charg = env.plan_instance.norm_charg
    norm_wait = env.plan_instance.norm_wait
    norm_travel = env.plan_instance.norm_travel
    norm_fairness = env.plan_instance.norm_fairness

    grid_penalty = None
    if env.grid_adapter:
        station_nodes = [(s[0], s[2]["capability"]) for s in best_plan]
        grid_penalty, cap_penalty, _, _ = env.grid_adapter.calculate_grid_penalty(station_nodes)

    score, benefit, cost, fairness, charg_time, wait_time, cost_travel = H.norm_score(
        best_plan,
        best_node_list,
        norm_benefit,
        norm_charg,
        norm_wait,
        norm_travel,
        norm_fairness,
        grid_penalty,
    )

    if env.grid_adapter and cap_penalty < 0:
        score -= 100

    travel_max = travel_metric(best_node_list)
    wait_max = waiting_metric(best_plan)

    # Budget calculation (consistent with run_metrics.py)
    basic_cost = sum(s[2]["fee"] for s in env.plan_instance.extend_existing_plan)
    total_inst_cost = (sum(station[2]["fee"] for station in best_plan) - basic_cost) / H.BUDGET

    metrics = {
        "reward": total_reward,
        "score": score * 100,
        "benefit": benefit * 100,
        "cost": cost * 100,
        "fairness": fairness * 100,
        "charg_time": charg_time * 100,
        "wait_time": wait_time * 100,
        "cost_travel": cost_travel * 100,
        "travel_max": travel_max,
        "wait_max": wait_max,
        "num_stations": len(best_plan),
        "used_budget": total_inst_cost * 100,
    }

    return metrics, best_plan


# ── Visualization helpers (from visualise.py) ──────────────────────────


def nodesize(station_list, my_graph, my_plan):
    ns = []
    for node in my_graph.nodes():
        if node not in station_list:
            ns.append(2)
        else:
            i = station_list.index(node)
            station = my_plan[i]
            try:
                capacity = station[2]["capability"]
            except (KeyError, IndexError):
                capacity = np.sum(H.CHARGING_POWER * station[1])
                
            if capacity < 100:
                ns.append(6)
            elif 100 <= capacity < 200:
                ns.append(11)
            elif 200 <= capacity < 300:
                ns.append(16)
            else:
                ns.append(21)
    return ns


def visualise_stations(my_graph, my_plan, my_filepath, title=None):
    """Create plot of the charging station distribution."""
    station_list = [station[0][0] for station in my_plan]
    colours = ["r", "grey"]
    nc = [colours[0] if node in station_list else colours[1] for node in my_graph.nodes()]
    labels = ["Charging station", "Normal road junction"]
    
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", lw=0, markerfacecolor=colours[0], markersize=7),
        Line2D([0], [0], marker="o", color="w", lw=0, markerfacecolor=colours[1], markersize=4),
    ]
    ns = nodesize(station_list, my_graph, my_plan)
    
    fig, ax = ox.plot_graph(
        my_graph,
        node_color=nc,
        save=False,
        node_size=ns,
        edge_linewidth=0.2,
        edge_alpha=0.8,
        show=False,
        close=False,
    )
    ax.legend(legend_elements, labels, loc=2, prop={"size": 10})
    
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", color="white")
        
    os.makedirs(os.path.dirname(my_filepath), exist_ok=True)
    plt.savefig(my_filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Map saved: {my_filepath}")


# ── Main comparison ────────────────────────────────────────────────────


def compare(location="DongDa"):
    # Base directory and paths
    base_dir = os.path.join(project_root, "custom_environment", "data")
    graph_file = os.path.join(base_dir, "Graph", location, f"{location}.graphml")
    node_file = os.path.join(base_dir, "Graph", location, f"nodes_extended_{location}.txt")
    plan_file = os.path.join(base_dir, "Graph", location, f"existingplan_{location}.pkl")

    use_gnn = False  # Set to True to evaluate the GNN model
    obs_type = "gnn" if use_gnn else "mlp"
    
    # Env for testing
    env = StationPlacement(graph_file, node_file, plan_file, location=location, obs_type=obs_type)

    # 0. Baseline calculation (as in run_metrics.py)
    with open(node_file, "r") as file:
        baseline_node_list = eval(file.readline())
        
    with open(plan_file, "rb") as f:
        baseline_plan = pickle.load(f)
        
    baseline_node_list, baseline_plan = prepare_existing_plan(baseline_plan, baseline_node_list)
    
    (
        b_norm_benefit,
        b_norm_cost,
        b_norm_fairness,
        b_norm_charging,
        b_norm_waiting,
        b_norm_travel,
    ) = H.existing_score(baseline_plan, baseline_node_list)
    
    baseline_grid_penalty = None
    if env.grid_adapter:
        station_nodes = [(s[0], s[2]["capability"]) for s in baseline_plan]
        baseline_grid_penalty, baseline_cap_penalty, _, _ = env.grid_adapter.calculate_grid_penalty(station_nodes)
    
    norm_score_baseline = eci_test(
        baseline_plan,
        baseline_node_list,
        b_norm_benefit,
        b_norm_charging,
        b_norm_waiting,
        b_norm_travel,
        b_norm_fairness,
        baseline_grid_penalty,
    )
    if env.grid_adapter and baseline_cap_penalty < 0:
        norm_score_baseline -= 100
    print(f"Baseline (Existing Plan) Norm Score: {norm_score_baseline:.6f}")

    # Load graph for visualization
    G = ox.load_graphml(graph_file)

    # 1. Load RL (DQN) Model
    step = 66400
    rl_log_dir = os.path.join("Results", "tmp", location, obs_type)
    best_rl_model = None
    if os.path.exists(rl_log_dir):
        # Allow loading either the GNN or MLP model
        prefix = "best_model_gnn_" if use_gnn else "best_model_"
        best_rl_model = os.path.join(rl_log_dir, f"{prefix}{location}_{step}.zip")

    if best_rl_model and os.path.exists(best_rl_model):
        print(f"Loading RL model from {best_rl_model}")
        if use_gnn:
            from custom_environment.gnn_extractor import GNNFeaturesExtractor
            custom_objects = {"GNNFeaturesExtractor": GNNFeaturesExtractor}
            rl_agent = DQN.load(best_rl_model, custom_objects=custom_objects)
        else:
            rl_agent = DQN.load(best_rl_model)
    else:
        print("Warning: RL model not found.")
        rl_agent = None

    # 2. Load GA Model (Skipped if using GNN)
    ga_agent = None
    if not use_gnn:
        ga_model_path = os.path.join("Results", "ga", location, f"best_ga_model_{location}.pt")
        if os.path.exists(ga_model_path):
            print(f"Loading GA model from {ga_model_path}")
            chromosome = torch.load(ga_model_path, weights_only=False)
            input_dim = env.observation_space.shape[0]
            output_dim = env.action_space.n
            ga_agent = GAPolicy(input_dim, output_dim, hidden_dim=256)
            unflatten_weights(ga_agent, chromosome)
        else:
            print("Warning: GA model not found.")
    else:
        print("Note: GA comparison skipped (Incompatible with GNN observation format).")

    # Run comparisons
    results = {}
    plans = {}

    if rl_agent:
        print("Running RL evaluation...")
        metrics, plan = run_episode(rl_agent, env, "rl")
        results["RL"] = metrics
        plans["RL"] = plan

    if ga_agent:
        print("Running GA evaluation...")
        metrics, plan = run_episode(ga_agent, env, "ga")
        results["GA"] = metrics
        plans["GA"] = plan

    print("Running Greedy Benefit evaluation...")
    metrics, plan = run_episode(None, env, "greedy_benefit")
    results["G-Benefit"] = metrics
    plans["G-Benefit"] = plan

    print("Running Greedy Demand evaluation...")
    metrics, plan = run_episode(None, env, "greedy_demand")
    results["G-Demand"] = metrics
    plans["G-Demand"] = plan

    # Display results
    print("\n--- Comparison Results ---")
    for alg, m in results.items():
        # Relative score calculation (consistent with run_metrics.py)
        # raw_score / norm_score * 100
        # since m['score'] is already multiplied by 100 in run_episode, we just divide by norm_score_baseline
        rel_score = m["score"] / (norm_score_baseline + 1e-9)

        print(f"{alg}:")
        print(f"  Score (raw): {m['score']:.2f}")
        print(f"  Score (relative to baseline): {rel_score:.2f}%")
        print(f"  Benefit: {m['benefit']:.2f}%")
        print(f"  Fairness: {m['fairness']:.2f}%")
        print(
            f"  Waiting time: {m['wait_time']:.2f}%, "
            f"Travel time: {m['cost_travel']:.2f}%, "
            f"Charging time: {m['charg_time']:.2f}%"
        )
        print(f"  Max Travel time: {m['travel_max']:.2f} min, Max Waiting time: {m['wait_max']:.2f} min")
        print(f"  Used budget: {m['used_budget']:.2f}%")
        print(f"  Nodes covered (Stations): {m['num_stations']}")
        print(f"  Total Reward: {m['reward']:.2f}\n")

    # ── Bar/Line chart ──
    if results:
        algorithms = list(results.keys())
        rewards = [m["reward"] for m in results.values()]
        scores = [m["score"] for m in results.values()]

        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = "tab:blue"
        ax1.set_xlabel("Algorithm")
        ax1.set_ylabel("Total Reward", color=color)
        ax1.bar(algorithms, rewards, color=color, alpha=0.6, label="Reward")
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()
        color = "tab:red"
        ax2.set_ylabel("Final Score", color=color)
        ax2.plot(
            algorithms, scores, color=color, marker="o", label="Score", linewidth=2, markersize=8
        )
        ax2.tick_params(axis="y", labelcolor=color)

        plt.title(f"RL vs GA vs Greedy Performance comparison ({location})")
        fig.tight_layout()
        chart_path = os.path.join("Results", f"comparison_{location}.png")
        plt.savefig(chart_path, dpi=150)
        plt.close(fig)
        print(f"Comparison chart saved to {chart_path}")

    # ── Station maps ──
    print("\n--- Generating Station Maps ---")
    map_dir = os.path.join("Results", "maps", location)
    for alg_name, plan in plans.items():
        safe_name = alg_name.replace("-", "_")
        filepath = os.path.join(map_dir, f"map_{safe_name}_{location}.png")
        m = results[alg_name]
        title = f"{alg_name} — Score: {m['score']:.2f}% — Stations: {m['num_stations']}"
        visualise_stations(G, plan, filepath, title=title)

    print(f"\nAll maps saved to {map_dir}{os.sep}")


if __name__ == "__main__":
    compare()


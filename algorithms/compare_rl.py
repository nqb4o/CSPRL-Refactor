import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
import osmnx as ox
from matplotlib.lines import Line2D

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from custom_environment.StationPlacementEnv import StationPlacement
from algorithms.ga.ga_utils import GAPolicy, unflatten_weights
import custom_environment.helpers as H


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
        elif agent_type == "greedy_benefit":
            station_list = [s[0][0] for s in env.plan_instance.plan]
            free_list = [node for node in env.node_list if node[0] not in station_list]
            action = 0 if free_list else 2
        elif agent_type == "greedy_demand":
            station_list = [s[0][0] for s in env.plan_instance.plan]
            free_list = [node for node in env.node_list if node[0] not in station_list]
            action = 1 if free_list else 3

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

    final_score = env.best_score
    num_stations = len(env.plan_instance.plan)
    best_plan = env.best_plan if env.best_plan else env.plan_instance.plan
    return total_reward, final_score, num_stations, best_plan


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
    colours = ['r', 'grey']
    nc = [colours[0] if node in station_list else colours[1] for node in my_graph.nodes()]
    labels = ['Charging station', 'Normal road junction']
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', lw=0, markerfacecolor=colours[0], markersize=7),
        Line2D([0], [0], marker='o', color='w', lw=0, markerfacecolor=colours[1], markersize=4),
    ]
    ns = nodesize(station_list, my_graph, my_plan)
    fig, ax = ox.plot_graph(my_graph, node_color=nc, save=False, node_size=ns,
                            edge_linewidth=0.2, edge_alpha=0.8, show=False, close=False)
    ax.legend(legend_elements, labels, loc=2, prop={"size": 10})
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', color='white')
    os.makedirs(os.path.dirname(my_filepath), exist_ok=True)
    plt.savefig(my_filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Map saved: {my_filepath}")


# ── Main comparison ────────────────────────────────────────────────────
def compare(location="DongDa"):
    # Base directory and paths
    base_dir = os.path.join(project_root, "custom_environment", "data")
    graph_file = os.path.join(base_dir, "Graph", location, location + ".graphml")
    node_file = os.path.join(base_dir, "Graph", location, "nodes_extended_" + location + ".txt")
    plan_file = os.path.join(base_dir, "Graph", location, "existingplan_" + location + ".pkl")

    # Env for testing
    env = StationPlacement(graph_file, node_file, plan_file, location=location)

    # Load graph for visualization
    G = ox.load_graphml(graph_file)

    # 1. Load RL (DQN) Model
    rl_log_dir = f"Results/tmp/{location}/"
    best_rl_model = None
    if os.path.exists(rl_log_dir):
        models = [f for f in os.listdir(rl_log_dir) if f.startswith(f"best_model_{location}_") and f.endswith(".zip")]
        if models:
            models.sort()
            best_rl_model = os.path.join(rl_log_dir, models[-1])

    if best_rl_model:
        print(f"Loading RL model from {best_rl_model}")
        rl_agent = DQN.load(best_rl_model)
    else:
        print("Warning: RL model not found.")
        rl_agent = None

    # 2. Load GA Model
    ga_model_path = f"Results/ga/{location}/best_ga_model_{location}.pt"
    if os.path.exists(ga_model_path):
        print(f"Loading GA model from {ga_model_path}")
        chromosome = torch.load(ga_model_path, weights_only=False)
        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.n
        ga_agent = GAPolicy(input_dim, output_dim)
        unflatten_weights(ga_agent, chromosome)
    else:
        print("Warning: GA model not found.")
        ga_agent = None

    # Run comparisons
    results = {}
    plans = {}

    if rl_agent:
        print("Running RL evaluation...")
        reward, score, stations, plan = run_episode(rl_agent, env, "rl")
        results['RL'] = (reward, score, stations)
        plans['RL'] = plan

    if ga_agent:
        print("Running GA evaluation...")
        reward, score, stations, plan = run_episode(ga_agent, env, "ga")
        results['GA'] = (reward, score, stations)
        plans['GA'] = plan

    print("Running Greedy Benefit evaluation...")
    reward, score, stations, plan = run_episode(None, env, "greedy_benefit")
    results['G-Benefit'] = (reward, score, stations)
    plans['G-Benefit'] = plan

    print("Running Greedy Demand evaluation...")
    reward, score, stations, plan = run_episode(None, env, "greedy_demand")
    results['G-Demand'] = (reward, score, stations)
    plans['G-Demand'] = plan

    # Display results
    print("\n--- Comparison Results ---")
    for alg, (reward, score, stations) in results.items():
        print(f"{alg}: Total Reward = {reward:.2f}, Final Score = {score:.2f}, Stations = {stations}")

    # ── Bar/Line chart ──
    if results:
        algorithms = list(results.keys())
        rewards = [r[0] for r in results.values()]
        scores = [r[1] for r in results.values()]

        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:blue'
        ax1.set_xlabel('Algorithm')
        ax1.set_ylabel('Total Reward', color=color)
        ax1.bar(algorithms, rewards, color=color, alpha=0.6, label='Reward')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Final Score', color=color)
        ax2.plot(algorithms, scores, color=color, marker='o', label='Score', linewidth=2, markersize=8)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title(f'RL vs GA vs Greedy Performance comparison ({location})')
        fig.tight_layout()
        chart_path = f"Results/comparison_{location}.png"
        plt.savefig(chart_path, dpi=150)
        plt.close(fig)
        print(f"Comparison chart saved to {chart_path}")

    # ── Station maps ──
    print("\n--- Generating Station Maps ---")
    map_dir = f"Results/maps/{location}"
    for alg_name, plan in plans.items():
        safe_name = alg_name.replace("-", "_")
        filepath = os.path.join(map_dir, f"map_{safe_name}_{location}.png")
        score = results[alg_name][1]
        title = f"{alg_name} — Score: {score:.2f} — Stations: {results[alg_name][2]}"
        visualise_stations(G, plan, filepath, title=title)

    print(f"\nAll maps saved to {map_dir}/")


if __name__ == '__main__':
    compare()

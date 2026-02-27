import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools
import json

# Add current directory to path
sys.path.append(os.getcwd())

from custom_environment.StationPlacementEnv import StationPlacement
from custom_environment.power_grid.csprl_adapter import CSPRLGridAdapter
import custom_environment.helpers as H

def greedy_policy(env):
    """Simple greedy policy to evaluate config quality."""
    obs, info = env.reset()
    terminated = False
    total_reward = 0
    
    while not terminated:
        action = np.random.choice([0, 1, 2, 3]) 
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
            
    return total_reward, env.render()[1] # return reward and final plan

def evaluate_advanced_config(params):
    """Evaluates a config with extended physical and reward parameters."""
    
    # Paths for DongDa
    location = "DongDa"
    graph_file = f"custom_environment/data/Graph/{location}/{location}.graphml"
    node_file = f"custom_environment/data/Graph/{location}/nodes_extended_{location}.txt"
    plan_file = f"custom_environment/data/Graph/{location}/existingplan_{location}.pkl"
    
    # 1. Temporarily Patch Constants in helpers.py
    original_lambda = H.my_lambda
    original_alpha = H.alpha
    H.my_lambda = params.get('lambda', 0.5)
    H.alpha = params.get('alpha', 0.4)
    
    # Path for grid adapter
    grid_folder = "custom_environment/data/hanoi_citywide"
    
    # 2. Initialize Env with Grid Adapter
    env = StationPlacement(
        my_graph_file=graph_file,
        my_node_file=node_file,
        my_plan_file=plan_file,
        location=location
    )
    adapter = CSPRLGridAdapter(
        grid_data_folder=grid_folder,
        ev_station_power_mw=0.35
    )
    # Patch bus_limit in loader
    adapter.loader.bus_limit = params.get('bus_limit', 0.8)
    env.grid_adapter = adapter
    
    results = []
    # Run 3 times to average random heuristic
    for _ in range(3):
        reward, plan = greedy_policy(env)
        
        # Physical Validation from Pandapower
        net = adapter.loader.net
        min_vm_pu = net.res_bus.vm_pu.min()
        max_line_loading = net.res_line.loading_percent.max()
        max_trafo_loading = net.res_trafo.loading_percent.max()
        
        # Calculate Reality Penalties
        voltage_penalty = max(0, 0.95 - min_vm_pu) * 1000
        loading_penalty = max(0, max_line_loading - 100) * 5
        
        reality_score = reward - (voltage_penalty + loading_penalty)
        
        results.append({
            'reward': reward,
            'min_vm_pu': min_vm_pu,
            'max_line_loading': max_line_loading,
            'max_trafo_loading': max_trafo_loading,
            'reality_score': reality_score,
            'num_stations': len(plan)
        })
    
    # Restore original constants
    H.my_lambda = original_lambda
    H.alpha = original_alpha
    
    # Average results
    avg_res = {k: np.mean([r[k] for r in results]) for k in results[0].keys()}
    return avg_res

def run_advanced_search():
    # Defining weights that sum to 1
    weight_pairs = [(round(w, 1), round(1-w, 1)) for w in np.linspace(0.1, 0.9, 9)]
    
    other_params = {
        'lambda': [0.3, 0.5, 0.7],
        'alpha': [0.2, 0.4, 0.6], 
        'bus_limit': [0.7, 0.9]
    }
    
    # Create combinations
    combinations = []
    keys = other_params.keys()
    other_combos = [dict(zip(keys, v)) for v in itertools.product(*other_params.values())]
    
    for dist_w, cap_w in weight_pairs:
        for combo in other_combos:
            config = combo.copy()
            config['penalty_distance_weight'] = dist_w
            config['penalty_capacity_weight'] = cap_w
            combinations.append(config)
    
    print(f"Starting Advanced Search with {len(combinations)} combinations (Constraint: dist+cap=1)...")
    all_results = []
    
    for config in tqdm(combinations):
        metrics = evaluate_advanced_config(config)
        config.update(metrics)
        all_results.append(config)
        
    df = pd.DataFrame(all_results)
    os.makedirs("Results", exist_ok=True)
    df.to_csv("Results/advanced_optimization_results.csv", index=False)
    
    best = df.loc[df['reality_score'].idxmax()]
    print("\n--- OPTIMAL VALUES FOUND ---")
    print(best)

if __name__ == "__main__":
    run_advanced_search()

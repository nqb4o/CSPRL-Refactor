import os
import sys
import copy
import numpy as np

# Append the directory so we can import helpers
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import custom_environment.helpers as H

def test_parameters(scaling_factor, distance_decay_factor, node_list, iterations=30):
    def custom_dynamic_demand(my_node, my_plan):
        power_factor = 0
        base_demand = H.weak_demand(my_node)
        for station in my_plan:
            s_pos, s_x, s_dict = station[0], station[1], station[2]
            s_r = s_dict["radius"]
            s_cap = s_dict["capability"]
            distance = H.haversine(s_pos, my_node)
            if distance < s_r:
                power_factor += s_cap * np.exp(-distance_decay_factor * distance)
        power_factor *= -scaling_factor
        new_demand = base_demand * np.exp(power_factor)
        return new_demand
    
    # Store old function
    old_dynamic_demand = H.dynamic_demand
    H.dynamic_demand = custom_dynamic_demand
    
    my_plan = []
    my_node_list = copy.deepcopy(node_list)
    free_nodes = [node for node in my_node_list]
    
    config_lookup = H.get_lookup(os.path.join(current_dir, "custom_environment", "data", "processed", "config_lookup.json"))
    
    for i in range(iterations):
        if not free_nodes:
            break
            
        chosen_node = H.choose_node_bydemand(free_nodes, my_plan)
        free_nodes.remove(chosen_node)
        
        best_config = H.initial_solution(config_lookup, my_node_list, chosen_node)
        
        station = [chosen_node, best_config.copy(), {"radius": 1.0, "capability": 0.0}]
        station = H.charging_capability(station)
        station = H.influence_radius(station)
        
        my_plan.append(station)

    # Calculate coverage for each node so fairness works
    for node in my_node_list:
        station_counts = 0
        for my_station in my_plan:
            s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
            radius_s = s_dict["radius"]
            distance = H.haversine(s_pos, node)
            if distance <= radius_s:
                station_counts += 1
        node[1]['n_stations'] = station_counts

    # Evaluate the plan precisely to set distances and cost metrics
    my_node_dict = {node[0]: {} for node in my_node_list}
    my_cost_dict = {node[0]: {} for node in my_node_list}
    
    # Initialize node attributes
    for node in my_node_list:
        node[1]["charging station"] = None
        node[1]["distance"] = 0.0
    
    # Initialize station details
    for i in range(len(my_plan)):
        my_plan[i] = H.s_dictionnary(my_plan[i], my_node_list)

    for _ in range(2):
        my_node_list, my_node_dict, my_cost_dict = H.station_seeking(
            my_plan, my_node_list, my_node_dict, my_cost_dict
        )
        for i in range(len(my_plan)):
            my_plan[i] = H.total_number_EVs(my_plan[i], my_node_list)
            my_plan[i] = H.avg_waiting(my_plan[i])

    fairness = H.social_fairness(my_node_list)
    benefit = H.social_benefit(my_plan, my_node_list)
    cost = H.social_cost(my_plan, my_node_list)
    
    # Restore old function
    H.dynamic_demand = old_dynamic_demand
    return fairness, benefit, cost

if __name__ == "__main__":
    location = "BaDinh"
    base_dir = "custom_environment/data"
    node_file = os.path.join(base_dir, "Graph", location, "nodes_extended_" + location + ".txt")
    
    with open(node_file, "r") as file:
        node_list = eval(file.readline())
        
    print(f"Total nodes in environment: {len(node_list)}")
    
    scaling_factors = np.linspace(0, 1, 20)
    decay_factors = np.linspace(0, 1, 20)
    
    results = []
    
    for sf in scaling_factors:
        for df in decay_factors:
            # We redirect stdout locally to suppress the print(std) in fairness function
            import sys, io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            fairness, benefit, cost = test_parameters(sf, df, node_list, iterations=40)
            
            sys.stdout = old_stdout
            results.append((sf, df, fairness, benefit, cost))
            
    # Calculate min and max for normalization
    f_vals = [r[2] for r in results]
    b_vals = [r[3] for r in results]
    c_vals = [r[4] for r in results]
    
    min_f, max_f = min(f_vals), max(f_vals)
    min_b, max_b = min(b_vals), max(b_vals)
    min_c, max_c = min(c_vals), max(c_vals)
    
    def normalize(val, min_val, max_val):
        return (val - min_val) / (max_val - min_val) if max_val > min_val else 0

    scored_results = []
    for sf, df, f, b, c in results:
        norm_f = normalize(f, min_f, max_f)
        norm_b = normalize(b, min_b, max_b)
        # Cost is lower is better, so we invert it for the score
        norm_c = 1.0 - normalize(c, min_c, max_c)
        
        # Combined score with equal weights
        combined_score = (norm_f + norm_b + norm_c) / 3.0
        scored_results.append((sf, df, f, b, c, combined_score))
            
    # Sort results
    scored_results.sort(key=lambda x: x[5], reverse=True)
    
    print("\nTop 5 optimal parameters balancing fairness, benefit, and cost:")
    for i in range(5):
        best = scored_results[i]
        print(f"Rank {i+1} -> Scaling Factor: {best[0]:.2f}, Distance Decay Factor: {best[1]:.2f}")
        print(f"         Fairness: {best[2]:.5f}, Benefit: {best[3]:.5f}, Cost: {best[4]:.5f}")
        print(f"         Combined Score: {best[5]:.5f}\n")

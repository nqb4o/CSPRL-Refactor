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

    fairness = H.social_fairness(my_node_list)
    
    # Restore old function
    H.dynamic_demand = old_dynamic_demand
    return fairness

if __name__ == "__main__":
    location = "DongDa"
    base_dir = "custom_environment/data"
    node_file = os.path.join(base_dir, "Graph", location, "nodes_extended_" + location + ".txt")
    
    with open(node_file, "r") as file:
        node_list = eval(file.readline())
        
    print(f"Total nodes in environment: {len(node_list)}")
    
    scaling_factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    decay_factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    results = []
    
    for sf in scaling_factors:
        for df in decay_factors:
            # We redirect stdout locally to suppress the print(std) in fairness function
            import sys, io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            fairness = test_parameters(sf, df, node_list, iterations=40)
            
            sys.stdout = old_stdout
            results.append((sf, df, fairness))
            
    # Sort results
    results.sort(key=lambda x: x[2], reverse=True)
    
    print("\nTop 5 optimal parameters for MAXIMUM fairness (least clustering):")
    for i in range(5):
        best = results[i]
        print(f"Rank {i+1} -> Scaling Factor: {best[0]}, Distance Decay Factor: {best[1]}, Fairness: {best[2]:.5f}")

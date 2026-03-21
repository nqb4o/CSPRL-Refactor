import os
import ast
import pickle
import pandas as pd


# Path to file
location = "ThanhXuan"
base_dir = os.path.join("custom_environment", "data")
existing_plan_file = os.path.join(base_dir, "QGIS", "existing_plan", "existing_plan_filtered.csv")
node_file = os.path.join(base_dir, "Graph", f"{location}", f"nodes_extended_{location}.txt")
graph_file = os.path.join(base_dir, "Graph", f"{location}", f"{location}.graphml")

# Load data
with open(node_file, "r") as file:
    node_list = eval(file.readline())
detail_stations_df = pd.read_csv(existing_plan_file)

# Prepare existing plan
existing_plan = []
# use set as temporal solution for multiple stations on a node
used_nodes = set()  # This will keep track of nodes already taken

for row in detail_stations_df.iterrows():
    # extract station config and position
    s_x = ast.literal_eval(row[1]['connector_array'])
    s_lat, s_lon = row[1]['y'], row[1]['x']

    s_pos = None
    for node in node_list:
        # Assuming node[0] is the unique ID of the node
        node_id = node[0]
        n_lat, n_lon = node[1]['y'], node[1]['x']

        # Check if coordinates match
        if s_lat == n_lat and s_lon == n_lon:
            # NEW: Check if this node is already in our 'used' set
            if node_id not in used_nodes:
                s_pos = node
                used_nodes.add(node_id)  # Mark this node as 'taken'

            # Since we found the matching node, stop looking through node_list
            break

    if s_pos is not None:
        existing_plan.append([s_pos, s_x, {}])

# save to file
pickle.dump(existing_plan, open(f"custom_environment/data/Graph/{location}/existingplan_" + location + ".pkl", "wb"))
print(f"Successfully saved {len(existing_plan)} existing plans.")
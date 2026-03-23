from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import pickle
import osmnx as ox
from custom_environment.StationPlacementEnv import StationPlacement
from visualise import visualise_stations
import seaborn as sns
import matplotlib.pyplot as plt
import os

"""
Generate a charging plan based on the model.
"""
# Prepare the environment
base_dir = "custom_environment/data"
location = "DongDa"

graph_file = os.path.join(base_dir, "Graph", location, location + ".graphml")
node_file = os.path.join(base_dir, "Graph", location, "nodes_extended_" + location + ".txt")
plan_file = os.path.join(base_dir, "Graph", location, "existingplan_" + location + ".pkl")

use_gnn = True  # Set to True if evaluating a GNN model
obs_type = "gnn" if use_gnn else "mlp"
env = StationPlacement(graph_file, node_file, plan_file, location=location, obs_type=obs_type)

# Updated to match the log directory used in train.py (Results/tmp/gnn/)
log_dir = f"Results/tmp/{location}/{obs_type}/"

"""
Start evaluating
"""
print("Evaluation for best model")
os.makedirs(log_dir, exist_ok=True)
env = Monitor(env, log_dir)  # new environment for evaluation
G = ox.load_graphml(graph_file)

step = 66400
if use_gnn:
    from custom_environment.gnn_extractor import GNNFeaturesExtractor
    custom_objects = {"GNNFeaturesExtractor": GNNFeaturesExtractor}
    model = DQN.load(log_dir + "best_model_gnn_" + location + f"_{step}.zip", env=env, custom_objects=custom_objects)
else:
    model = DQN.load(log_dir + "best_model_" + location + f"_{step}.zip", env=env)

obs, _ = env.reset()
done = False
best_plan, best_node_list = None, None
action_history = []
total_reward = 0
while not done:
    action, _states = model.predict(obs, deterministic=True)
    action_history.append(action.item())
    
    # Gymnasium steps return 5 variables
    obs, reward, done, truncated, info = env.step(action)
    env.render() # print out the evaluation

    # In StationPlacementEnv, done acts as terminated
    if done or truncated:
        best_node_list, best_plan = env.render()
        break

sns.countplot(x=action_history)
plt.title('Frequency of Chosen Actions')
plt.show()

output_dir = os.path.join("Results", "optimal_plan", location)
os.makedirs(output_dir, exist_ok=True)

pickle.dump(best_plan, open(os.path.join(output_dir, f"plan_RL_{step}.pkl"), "wb"))
with open(os.path.join(output_dir, f"nodes_RL_{step}.txt"), 'w') as file:
    file.write(str(best_node_list))

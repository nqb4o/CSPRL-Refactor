from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
import torch
import random
from custom_environment.StationPlacementEnv import StationPlacement

"""
Trai the model by reinforcement learning.
"""

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Code from Stable Baselines3,
    https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/monitor_training.ipynb
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param my_log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, my_log_dir: str, my_modelname: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = my_log_dir
        self.modelname = my_modelname
        self.save_path = os.path.join(self.log_dir, self.modelname)
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                my_mean_reward = np.mean(y[-3:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                        self.best_mean_reward, my_mean_reward))

                if my_mean_reward > self.best_mean_reward:
                    self.best_mean_reward = my_mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("New best mean reward: {:.2f}".format(self.best_mean_reward))
                        # we want to make sure that the best models are not overwritten
                        new_name = self.modelname + str(self.num_timesteps)
                        if self.save_path is not None:
                            os.makedirs(self.save_path, exist_ok=True)
                        self.save_path = os.path.join(self.log_dir, new_name)
                        print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)
                else:
                    if self.verbose > 0:
                        new_name = self.modelname + str(self.num_timesteps)
                        if self.save_path is not None:
                            os.makedirs(self.save_path, exist_ok=True)
                        self.save_path = os.path.join(self.log_dir, new_name)
                        print("Saving model on frequency to {}".format(self.save_path))
                    self.model.save(self.save_path)

        return True


if __name__ == '__main__':
    #pip uninstall -r requirements.txt -ypip uninstall -r requirements.txt -ypip uninstall -r requirements.txt -y set a seed for reproducibility
    os.environ['PYTHONASHSEED'] = '0'
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    seed = 1
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    random.seed(seed)
    # Instantiate the env
    location = "DongDa"  # take a location of your choice
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_environment", "data")
    graph_file = os.path.join(base_dir, "Graph", location, location + ".graphml")
    node_file = os.path.join(base_dir, "Graph", location, "nodes_extended_" + location + ".txt")
    plan_file = os.path.join(base_dir, "Graph", location, "existingplan_" + location + ".pkl")

    env = StationPlacement(graph_file, node_file, plan_file, location=location)
    log_dir = f"Results/tmp6/{location}/"
    modelname = "best_model_" + location + "_"

    """
    Define and train the agent 
    """
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, os.path.join(log_dir, "monitor.csv"))
    policy_kwargs = dict(net_arch=[256, 256]) # hidden layers
    model = DQN("MlpPolicy", env, verbose=1, batch_size=128, buffer_size=10000, learning_rate=1e-5,
                exploration_initial_eps=1, exploration_final_eps=0.05, exploration_fraction=0.2, policy_kwargs=policy_kwargs,
                device='cuda' if torch.cuda.is_available() else 'cpu', seed=seed)
    callback = SaveOnBestTrainingRewardCallback(check_freq=400, my_log_dir=log_dir, my_modelname=modelname)
    model.learn(total_timesteps=200000, log_interval=10 ** 4, callback=callback)
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.env_checker import check_env
import pickle
from random import choice
import custom_environment.helpers as H
import sys
import os

# Add parent directory to path to allow importing power_grid
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from custom_environment.power_grid.csprl_adapter import create_adapter_for_location

"""
Custom environment
"""

class FeatureScaler:
    """
    Scale features separately based on realistic ranges.
    All outputs are in [-1, 1] to match Box observation space.
    """
    def __init__(self):
        # Estimated realistic ranges (adjust based on your actual dataset!)
        self.lon_min, self.lon_max = 105.7, 106.0      # Hanoi/DongDa approximate
        self.lat_min, self.lat_max = 20.95, 21.05
        self.pop_min, self.pop_max = 2.646, 468.3
        self.demand_max = 1.0                          # assuming already normalized [0,1]
        self.land_price_max = 214.245                  # triệu VND/m²
        self.private_cs_max = 1.0
        self.charger_max = float(H.K)
        self.budget_max = float(H.BUDGET)

    def scale_lon(self, v):
        return 2 * (np.clip(v, self.lon_min, self.lon_max) - self.lon_min) / (self.lon_max - self.lon_min + 1e-9) - 1

    def scale_lat(self, v):
        return 2 * (np.clip(v, self.lat_min, self.lat_max) - self.lat_min) / (self.lat_max - self.lat_min + 1e-9) - 1

    def scale_pop(self, v):
        return 2 * (np.clip(v, self.pop_min, self.pop_max) - self.pop_min) / (self.pop_max - self.pop_min + 1e-9) - 1

    def scale_demand(self, v):
        return 2 * np.clip(v / (self.demand_max + 1e-9), 0, 1) - 1

    def scale_land_price(self, v):
        return 2 * np.clip(v / (self.land_price_max + 1e-9), 0, 1) - 1

    def scale_private_cs(self, v):
        return 2 * np.clip(v / (self.private_cs_max + 1e-9), 0, 1) - 1

    def scale_charger_count(self, v):
        return 2 * (np.clip(v, 0, self.charger_max) / (self.charger_max + 1e-9)) - 1

    def scale_budget(self, v):
        return 2 * (np.clip(v, 0, self.budget_max) / (self.budget_max + 1e-9)) - 1


class Plan:
    def __init__(self, my_node_list, my_node_dict, my_cost_dict, my_plan_file):
        with (open(my_plan_file, "rb")) as f:
            self.plan = pickle.load(f)
        self.plan = [H.s_dictionnary(my_station, my_node_list) for my_station in self.plan]
        my_node_list, _, _ = H.station_seeking(self.plan, my_node_list, my_node_dict, my_cost_dict)
        # update the dictionnary
        self.plan = [H.s_dictionnary(my_station, my_node_list) for my_station in self.plan]
        self.norm_benefit, self.norm_cost, self.norm_charg, self.norm_wait, self.norm_travel = \
            H.existing_score(self.plan, my_node_list)
        self.existing_plan = self.plan.copy()
        self.existing_plan = [s[0] for s in self.existing_plan]

    def __repr__(self):
        return "The charging plan is {}".format(self.plan)

    def add_plan(self, my_station):
        self.plan.append(my_station)

    def remove_plan(self, my_station):
        self.plan.remove(my_station)

    def steal_column(self, stolen_station, my_budget):
        """
        steal a charger from the station, give budget back and check which charger type has been stolen
        """
        my_budget += stolen_station[2]["fee"]
        station_index = self.plan.index(stolen_station)
        config_index = None

        # we choose the most expensive charging column
        N_types = len(H.CHARGING_POWER)
        for i in reversed(range(N_types)):
            if stolen_station[1][i] > 0:
                self.plan[station_index][1][i] -= 1
                config_index = i
                break # if found the largest port, move to the next step

        if sum(stolen_station[1]) == 0:
            # this means we remove the entire stations as it only has one charger
            self.remove_plan(stolen_station)
        else:
            # the station remains, we only steal one charging column
            H.installment_fee(stolen_station)
            my_budget -= stolen_station[2]["fee"]
            my_budget -= H.get_relocate_cost(config_index) # add relocate cost
        return my_budget, config_index


class Station:
    def __init__(self):
        self.s_pos = None # station positioon
        self.s_x = None # station config
        self.s_dict = {} # this use to store additional station data (fee, )
        self.station = [self.s_pos, self.s_x, self.s_dict]

    def __repr__(self):
        return "This station is {}".format(self.station)

    def add_position(self, my_node):
        self.station[0] = my_node

    def add_chargers(self, my_config):
        self.station[1] = my_config

    def establish_dictionnary(self, node_list):
        self.station = H.s_dictionnary(self.station, node_list)


class StationPlacement(gym.Env):
    """Custom Environment that follows gym interface"""
    node_dict = {}
    cost_dict = {}

    def __init__(self, my_graph_file, my_node_file, my_plan_file, location="DongDa"):
        super(StationPlacement, self).__init__()

        # Initialize Grid Adapter with Fallback
        try:
            # Adapter automatically finds data in CSPRL/power_grid/data
            self.grid_adapter = create_adapter_for_location(location)
        except Exception as e:
            print(f"Warning: Could not initialize Power Grid Adapter ({ascii(e)}). Grid constraints will be ignored.")
            self.grid_adapter = None

        _graph, self.node_list = H.prepare_graph(my_graph_file, my_node_file)

        self.node_list = [self._init(my_node) for my_node in self.node_list]

        self.plan_file = my_plan_file
        self.game_over = None
        self.budget = None
        self.plan_instance = None
        self.plan_length = None
        self.row_length = 5
        self.best_score = None
        self.best_plan = None
        self.best_node_list = None
        self.schritt = None
        self.config_dict = None
        self.previous_score = None
        self.feature_scaler = FeatureScaler()
        # new action space including all charger types
        self.action_space = spaces.Discrete(5)
        shape = (self.row_length + 1) * len(self.node_list) + 1
        self.observation_space = spaces.Box(low=-1, high=1, shape=(shape,), dtype=np.float16)

    def reset(self, seed=None, options=None):
        """
        Reset the state of the environment to an initial state
        """
        # Handle the seed for Gymnasium compatibility
        if seed is not None:
            np.random.seed(seed)

        self.budget = H.BUDGET
        self.game_over = False
        self.plan_instance = Plan(self.node_list, StationPlacement.node_dict, StationPlacement.cost_dict,
                                  self.plan_file)
        self.best_score, _, _, _, _, _ = H.norm_score(self.plan_instance.plan, self.node_list,
                                                       self.plan_instance.norm_benefit, self.plan_instance.norm_charg,
                                                       self.plan_instance.norm_wait, self.plan_instance.norm_travel)

        # Extend node features with grid data (if available)
        if self.grid_adapter:
            station_nodes = [(s[0], s[2]["capability"]) for s in self.plan_instance.plan]
            self.node_list = self.grid_adapter.extend_node_features(self.node_list, station_nodes)

        self.previous_score = self.best_score
        # Add grid penalty to initial best_score to match evaluation logic
        if self.grid_adapter:
            station_nodes = [(s[0], s[2]["capability"]) for s in self.plan_instance.plan]
            grid_penalty, grid_utilization, grid_distance = self.grid_adapter.calculate_grid_penalty(station_nodes)
            self.best_score += grid_penalty

        self.best_score = max(self.best_score, -25)
        self.plan_length = len(self.plan_instance.existing_plan)
        self.schritt = 0
        self.best_plan = []
        self.best_node_list = []
        self.best_node_list = []
        # Use absolute path for config lookup
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed", "config_lookup.json")
        self.config_dict = H.get_lookup(config_path)
        H.coverage(self.node_list, self.plan_instance.plan)
        obs = self.establish_observation()

        # Return obs AND an empty info dict (Required by new SB3/Gymnasium)
        return obs, {}

    def _init(self, my_node):
        StationPlacement.node_dict[my_node[0]] = {}  # prepare node_dict
        StationPlacement.cost_dict[my_node[0]] = {}
        my_node[1]["charging station"] = None
        my_node[1]["distance"] = None
        return my_node

    def establish_observation(self):
        """
        Build observation matrix
        """
        row_length = self.row_length + 1
        width = row_length * len(self.node_list) + 1
        obs = np.zeros(width, dtype=np.float32)

        for j, node in enumerate(self.node_list):
            i = j * row_length
            # obs[i + 0] = self.feature_scaler.scale_lon(node[1]['x'])
            # obs[i + 1] = self.feature_scaler.scale_lat(node[1]['y'])
            obs[i + 0] = H.dynamic_demand(node, self.plan_instance.plan)
            obs[i + 1] = self.feature_scaler.scale_land_price(node[1]['land_price'])
            # obs[i + 2] = self.feature_scaler.scale_private_cs(node[1]['private_cs'])z
            obs[i + 2] = 2 * (np.clip(node[1].get('grid_distance_km', 3.0), 0, 3.0) / 3.0) - 1
            obs[i + 3] = 2 * (np.clip(node[1].get('grid_available_mw', 0.0), 0, 10.0) / 10.0) - 1
            obs[i + 4] = 2 * (np.clip(node[1].get('covered', 0.0), 0, 10.0) / 10.0) - 1

            for station in self.plan_instance.plan:
                if station[0][0] == node[0]:
                    # for e in range(len(H.CHARGING_POWER)):
                        # obs[i + self.row_length + e] = self.feature_scaler.scale_charger_count(station[1][e])
                    obs[i + self.row_length + 1] = station[2]["capability"]
                    break

        obs[-1] = self.feature_scaler.scale_budget(self.budget)
        return obs

    def budget_adjustment(self, my_station):
        inst_cost = my_station[2]["fee"]
        if self.budget - inst_cost > 0:
            # if we have enough money, we build the station
            self.budget -= inst_cost
        else:
            self.game_over = True

    def budget_adjustment_small(self, chosen_node, config_index):
        single_charger_budget = H.evs_parking_area * chosen_node[1]['land_price'] + H.INSTALL_FEE[config_index]
        if self.budget - single_charger_budget > 0:
            # if we have enough money, we build the charger
            self.budget -= single_charger_budget
        else:
            self.game_over = True

    def prepare_score(self):
        """
        We have to make a loop to reorganise the station assignment
        """
        for j in range(2):
            self.node_list, _, _ = H.station_seeking(self.plan_instance.plan, self.node_list,
                                                      StationPlacement.node_dict,
                                                      StationPlacement.cost_dict)
            for i in range(len(self.plan_instance.plan)):
                self.plan_instance.plan[i] = H.total_number_EVs(self.plan_instance.plan[i], self.node_list)
                self.plan_instance.plan[i] = H.W_s(self.plan_instance.plan[i])
            j += 1

    def step(self, my_action):
        """
        Perform a step in the episode
        """
        for station in self.plan_instance.plan:
            # if there is an empty station, delete it
            if np.sum(station[1]) <= 0:
                self.plan_instance.remove_plan(station)

        chosen_node, free_list_zero, config_index, action = self._control_action(my_action)
        if chosen_node in free_list_zero:
            # build new station
            default_config = H.initial_solution(self.config_dict, self.node_list, chosen_node)
            station_instance = Station()
            station_instance.add_position(chosen_node)
            station_instance.add_chargers(default_config.copy())
            station_instance.establish_dictionnary(self.node_list)
            # Step: Control budget
            self.budget_adjustment(station_instance.station)
            if not self.game_over:
                self.plan_instance.add_plan(station_instance.station) # this must be the reason why the budget exceed 100%
        else:
            # add column to existing CS
            station_index = None
            for station in self.plan_instance.plan:
                if station[0][0] == chosen_node[0]:
                    station_index = self.plan_instance.plan.index(station)
                    break
            # Step: Control budget
            self.budget_adjustment_small(chosen_node, config_index)
            if not self.game_over:
                self.plan_instance.plan[station_index][1][config_index] += 1

        # update station capacity
        for station in self.plan_instance.plan:
            H.s_dictionnary(station, self.node_list)

        # Extend node features with grid data (if available)
        if self.grid_adapter:
            station_nodes = [(s[0], s[2]["capability"]) for s in self.plan_instance.plan]
            self.node_list = self.grid_adapter.extend_node_features(self.node_list, station_nodes)

        # Step: calculate reward
        reward = self.evaluation()
        H.coverage(self.node_list, self.plan_instance.plan)
        obs = self.establish_observation()
        # episode end conditions
        if len(self.plan_instance.plan) == len(self.node_list):
            self.game_over = True
        self.schritt += 1
        if self.schritt >= len(self.node_list) / 2:
            self.game_over = True
        # if self.game_over:
        #     print("Best score {}.".format(self.best_score))
        # Return gymnasium format: (obs, reward, terminated, truncated, info)
        # best_node_list, best_plan = self.render()
        return obs, reward, self.game_over, False, {}

    def station_config_check(self, my_station):
        """
        no more than K chargers are allowed at the station
        """
        capacity = True
        if sum(my_station[1]) >= H.K:
            capacity = False
        return capacity

    def _control_action(self, chosen_action):
        """
        we have three possibilities here: either build a new station, add a charger to an exisiting station or move a
        charger from an exisiting station to a station in need
        """
        my_action = chosen_action
        config_index = None
        full_station_list = [s[0][0] for s in self.plan_instance.plan if self.station_config_check(s)
                             is False]  # these are the stations with exactly K chargers
        station_list = [s[0][0] for s in self.plan_instance.plan]  # all charging stations
        occupied_list = [node for node in self.node_list if node[0] not in full_station_list and node[0] in
                         station_list]  # nodes with non-full stations
        free_list = [node for node in self.node_list if node[0] not in station_list]  # nodes without stations
        if 0 <= my_action <= 1:
            # build
            if my_action == 0:
                chosen_node = H.choose_node_new_benefit(free_list, self.node_list)
            else:
                chosen_node = H.choose_node_bydemand(free_list, self.plan_instance.plan)
        elif 2 <= my_action <= 3:
            # add column to existing station
            config_index = 3
            if len(occupied_list) == 0:
                chosen_node = choice(free_list)
            else:
                if my_action == 2:
                    chosen_node = H.choose_node_new_benefit(occupied_list, self.node_list)
                else:
                    chosen_node = H.choose_node_bydemand(occupied_list, self.plan_instance.plan, add=True)
        else:
            # move station
            steal_plan = [s for s in self.plan_instance.plan if s[0] not in self.plan_instance.existing_plan]
            # we can not steal from the existing charging plan
            stolen_station = H.anti_choose_node_bybenefit(self.node_list, steal_plan)
            if stolen_station is None:
                # only necessary if we take this action in the very beginning
                chosen_node = choice(free_list)
            else:
                self.budget, config_index = self.plan_instance.steal_column(stolen_station, self.budget)
                chosen_node = H.support_stations(self.plan_instance.plan, free_list)
        return chosen_node, free_list, config_index, my_action

    def evaluation(self):
        """
        Calculate the reward
        """
        reward = 0
        self.prepare_score()
        new_score, benefit, cost, charg_time, wait_time, cost_travel = H.norm_score(self.plan_instance.plan, self.node_list,
                                                 self.plan_instance.norm_benefit, self.plan_instance.norm_charg,
                                                 self.plan_instance.norm_wait, self.plan_instance.norm_travel)


        # Add Grid Penalty (if adapter is active)
        if self.grid_adapter:
            station_nodes = [(s[0], s[2]["capability"]) for s in self.plan_instance.plan]
            grid_penalty, grid_utilization, grid_distance = self.grid_adapter.calculate_grid_penalty(station_nodes)
            new_score += grid_penalty

        # Compare against the score from the PREVIOUS step, not the all-time best
        step_improvement = new_score - self.previous_score
        reward += step_improvement
        # Update previous score for the next step
        self.previous_score = new_score

        new_score = max(new_score, -25)  # if negative score
        if new_score - self.best_score > 0:
            # reward += (new_score - self.best_score)
            # avoid jojo learning
            self.best_score = new_score
            self.best_plan = self.plan_instance.plan.copy()
            self.best_node_list = self.node_list.copy()
        return reward

    def render(self, mode='human', close=False):
        """
        Render the environment to the screen
        """
        print(f'Score is: {self.best_score}')
        print(f'Number of stations in charging plan: {len(self.plan_instance.plan)}')
        return self.best_node_list, self.best_plan


if __name__ == '__main__':
    location = "DongDa"
    graph_file = os.path.join(current_dir, "data", "Graph", location, location + ".graphml")
    node_file = os.path.join(current_dir, "data", "Graph", location, "nodes_extended_" + location + ".txt")
    plan_file = os.path.join(current_dir, "data", "Graph", location, "existingplan_" + location + ".pkl")
    env = StationPlacement(graph_file, node_file, plan_file, location=location)
    # It will check your custom environment and output additional warnings if needed
    check_env(env)

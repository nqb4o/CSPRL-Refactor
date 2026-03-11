import custom_environment.helpers as H
import pickle
from math import ceil
import numpy as np
import osmnx as ox
import os

"""
Calculate evaluation metrics for the created charging plan.
"""

def travel_metric(my_node_list):
    """ this gives the mean travel time in minutes """
    big_travel_list = []
    for my_node in my_node_list:
        travel = my_node[1]['distance'] / H.VELOCITY * 60
        times = ceil(10 * H.weak_demand(my_node))
        for time in range(times):
            big_travel_list.append(travel)
    travel_mean = max(big_travel_list)
    return travel_mean


def waiting_metric(my_plan):
    """ mean waiting time in minutes """
    big_waiting_list = []
    for my_station in my_plan:
        times = ceil(my_station[2]["D_s"])
        for time in range(times):
            big_waiting_list.append(my_station[2]["W_s"] * 60)
    wait_mean = max(big_waiting_list)
    return wait_mean


def eci_test(my_plan, my_node_list, my_norm_benefit, my_norm_charging, my_norm_waiting,
             my_norm_travel, my_norm_fairness):
    score, benefit, cost, fairness, charg_time, wait_time, cost_travel = H.norm_score(my_plan, my_node_list, my_norm_benefit,
                                                                             my_norm_charging, my_norm_waiting,
                                                                             my_norm_travel, my_norm_fairness)
    return score


def test(my_plan, my_node_list, my_basic_cost, my_norm_benefit, my_norm_charging, my_norm_waiting,
         my_norm_travel, my_norm_score, my_norm_fairness):
    """
    prints results of the evaulation metrics
    """
    travel_max = travel_metric(my_node_list)
    wait_max = waiting_metric(my_plan)
    score, benefit, cost, fairness, charg_time, wait_time, cost_travel = H.norm_score(my_plan, my_node_list, my_norm_benefit,
                                                                             my_norm_charging, my_norm_waiting,
                                                                             my_norm_travel, my_norm_fairness)
    # test if solution satisfies all constraints
    H.constraint_check(my_plan, my_node_list, my_basic_cost)
    total_inst_cost = (sum([my_station[2]["fee"] for my_station in my_plan]) - my_basic_cost) / H.BUDGET
    score = score / my_norm_score * 100
    print("The score is {}".format(score))
    print("Benefit: {}".format(benefit * 100))
    print("Fairness: {}".format(fairness * 100))
    print("Waiting time: {}, Travel time: {}, Charging time: {}".format(wait_time * 100, cost_travel * 100,
                                                                        charg_time * 100))
    print(travel_max, wait_max)
    print("Used budget: {} \n".format(total_inst_cost * 100))


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


def perform_test(my_node_file, my_basic_cost, my_result_file, my_norm_benefit, my_norm_charging,
                 my_norm_waiting, my_norm_travel, my_norm_fairness, my_norm_score):
    with open(my_node_file, "r") as file:
        my_node_list = eval(file.readline())
    with (open(my_result_file, "rb")) as f:
        my_plan = pickle.load(f)
    print("Number of charging stations: {}".format(len(my_plan)))
    test(my_plan, my_node_list, my_basic_cost, my_norm_benefit, my_norm_charging, my_norm_waiting,
         my_norm_travel, my_norm_score, my_norm_fairness)


if __name__ == '__main__':
    location = "DongDa"
    base_dir = "custom_environment/data"
    graph_file = os.path.join(base_dir, "Graph", location, location + ".graphml")
    node_file = os.path.join(base_dir, "Graph", location, "nodes_extended_" + location + ".txt")
    plan_file = os.path.join(base_dir, "Graph", location, "existingplan_" + location + ".pkl")

    """
    Test existing charging stations.
    """
    graph = ox.load_graphml(graph_file)
    with open(node_file, "r") as file:
        node_list = eval(file.readline())
    with (open(plan_file, "rb")) as f:
        plan = pickle.load(f)
    print("Number of already existing charging stations: {}".format(len(plan)))

    node_list, plan = prepare_existing_plan(plan, node_list)
    basic_cost = sum([station[2]["fee"] for station in plan])
    norm_benefit, norm_cost, norm_fairness, norm_charging, norm_waiting, norm_travel = H.existing_score(plan, node_list)
    norm_score = eci_test(plan, node_list, norm_benefit, norm_charging, norm_waiting, norm_travel, norm_fairness)
    test(plan, node_list, basic_cost, norm_benefit, norm_charging, norm_waiting, norm_travel, norm_score, norm_fairness)
    pickle.dump(plan, open("Results/" + "debug/" + location + f"/existing_plan.pkl", "wb"))
    print("Reinforcement Learning")
    step = 61600
    node_file = "Results/" + "optimal_plan/" + location + f"/nodes_RL_{step}.txt"
    result_file = "Results/" + "optimal_plan/" + location + f"/plan_RL_{step}.pkl"
    perform_test(node_file, basic_cost, result_file, norm_benefit, norm_charging, norm_waiting, norm_travel, norm_fairness, norm_score)

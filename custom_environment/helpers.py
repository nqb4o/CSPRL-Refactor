import json
import osmnx as ox
import numpy as np
import networkx as nx
from math import sin, cos, sqrt, atan2, radians, ceil

"""
Utility model and help functions.
"""


def prepare_graph(my_graph_file, my_node_file):
    """
    loads graph and nodes prepared in load_graph.py
    """
    my_graph = ox.load_graphml(my_graph_file)
    with open(my_node_file, "r") as file:
        my_node_list = eval(file.readline())
    return my_graph, my_node_list


def cost_single(my_node, my_station, my_node_dict, my_cost_dict):
    """
    calculate the social cost for one station
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    node_id, station_id = my_node[0], s_pos[0]
    # check if distance has to be calculated
    if station_id in my_node_dict[node_id]:
        distance = my_node_dict[node_id][station_id]
    else:
        distance = calculate_distance(s_pos, my_node)
        my_node_dict[node_id][station_id] = distance
    # check if cost has to be calculated
    if node_id not in my_cost_dict:
        my_cost_dict[node_id] = {}
    station_signature = (
        tuple(np.asarray(s_x).tolist()),
        float(s_dict.get("W_s", 0.0)),
        float(s_dict.get("service rate", 0.0)),
    )
    cached_entry = my_cost_dict[node_id].get(station_id)
    if isinstance(cached_entry, dict) and cached_entry.get("state") == station_signature:
        node_cost = cached_entry["cost"]
    else:
        cost_travel = alpha * distance / VELOCITY
        cost_boring = (1 - alpha) * (s_dict["W_s"] + 1 / (s_dict["service rate"] + eps))
        node_cost = weak_demand(my_node) * (cost_travel + cost_boring)
        my_cost_dict[node_id][station_id] = {
            "state": station_signature,
            "cost": node_cost
        }
    return node_cost, my_node_dict, my_cost_dict


def station_seeking(my_plan, my_node_list, my_node_dict, my_cost_dict):
    """
    output station assignment: Each node gets assigned the charging station with minimal social cost
    """
    for node in my_node_list:
        cost_list = []
        for station in my_plan:
            node_cost, my_node_dict, my_cost_dict = cost_single(node, station, my_node_dict, my_cost_dict)
            cost_list.append(node_cost)
        costminindex = int(np.argmin(cost_list))
        chosen_station = my_plan[costminindex]
        s_pos = chosen_station[0]
        node[1]["charging station"] = s_pos[0]
        node[1]["distance"] = my_node_dict[node[0]][s_pos[0]]
    return my_node_list, my_node_dict, my_cost_dict


def calculate_distance(s_pos, my_node):
    """
    Calculates distance between two nodes using the precomputed distance matrix.
    Falls back to haversine if matrix lookup fails.
    """
    try:
        # s_pos[0] and my_node[0] là the OSM node IDs
        u = s_pos[0]
        v = my_node[0]
        distance = nx.shortest_path_length(graph, u, v, weight='length')
        return distance / 1000.0
    except (KeyError, IndexError):
        # Fallback if node not found in matrix
        # print(f"Matrix lookup failed for {u} -> {v}, falling back to Haversine")
        pass
    except Exception as e:
        # print(f"Distance loader error: {e}")
        pass
    # if not available, use haversine instead
    return haversine(s_pos, my_node)


################################################################################################
def installment_fee(my_station):
    """
    returns cost to install the respective chargers at that position
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    n_chargers = np.sum(s_x)
    charger_cost = np.sum(INSTALL_FEE * s_x)
    fee = evs_parking_area * n_chargers * s_pos[1]['land_price'] + charger_cost
    s_dict["fee"] = fee  # [fee] = €t
    return my_station


def charging_capability(my_station):
    """
    returns the summed up charging capability of the CS
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    total_capacity = np.sum(CHARGING_POWER * s_x)
    s_dict["capability"] = total_capacity / 1000.0  # [capability] = MW
    return my_station


def weak_demand(my_node):
    return my_node[1]["demand"] * (1 - 0.1 * my_node[1]["private_cs"])

def dynamic_demand(my_node, my_plan, scaling_factor=0.4, distance_decay_factor=0.5):
    power_factor = 0
    base_demand = weak_demand(my_node)
    for station in my_plan:
        s_pos, s_x, s_dict = station[0], station[1], station[2]
        s_r = s_dict["radius"]
        s_cap = s_dict["capability"]
        distance = haversine(s_pos, my_node)
        if distance < s_r:
            power_factor += s_cap * np.exp(-distance_decay_factor * distance)
    power_factor *= -scaling_factor
    new_demand = base_demand * np.exp(power_factor)

    return new_demand

def influence_radius(my_station):
    """
    gives the radius of the nodes whose charging demand the CS could satisfy
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    total_capacity = s_dict["capability"]
    radius_s = RADIUS_MAX * 1 / (1 + np.exp(-total_capacity / (100 * capacity_unit)))
    s_dict["radius"] = radius_s  # [radius] = km
    return my_station


def haversine(s_pos, my_node):
    """
    yields the approximate distance of two GPS points, middle computational cost
    """
    lon1, lat1 = s_pos[1]['x'], s_pos[1]['y']
    R_earth = 6372800  # approximate radius of earth. [R_earth] = m
    lon2, lat2 = my_node[1]['x'], my_node[1]['y']
    dlon = radians(lon2 - lon1)
    dlat = radians(lat2 - lat1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R_earth * c  # [distance] = m
    if distance < 0.1:  # to avoid ZeroDivisionError
        distance = 0.1
    return distance / 1000.0


def station_coverage(my_station, my_node_list):
    """yields the number of nodes within a station influential radius (raw count - use for other purposes)"""
    node_counts = 0
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    radius_s = s_dict["radius"]
    for node in my_node_list:
        distance = haversine(s_pos, node)
        if distance < radius_s:
            node_counts += 1
    return node_counts


def station_coverage_diminishing(my_station, my_node_list):
    """
    Yields the benefit of nodes within a station's influential radius.
    More nodes covered = more benefit (no diminishing returns here).
    This is different from node_coverage which has diminishing returns
    due to redundancy (multiple stations covering one node).
    For fair comparison with node_coverage, we normalize by total possible nodes.
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    radius_s = s_dict["radius"]

    # Count nodes within radius
    covered_nodes = 0
    for node in my_node_list:
        distance = haversine(s_pos, node)
        if distance < radius_s:
            covered_nodes += 1

    # Normalize to 0-1 range based on total nodes, then scale to be comparable to node_coverage
    # (which typically ranges from ~1-4 with diminishing returns)
    normalized_coverage = (covered_nodes / len(my_node_list)) * 10  # scale factor to match node_coverage range

    return normalized_coverage


def node_coverage(my_plan, my_node):
    """
    yields the number of station nodes which cover a given node
    """
    station_counts, diminishing_benefit = 0, 0
    priv_CS = my_node[1]["private_cs"]
    for my_station in my_plan:
        s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
        radius_s = s_dict["radius"]
        distance = haversine(s_pos, my_node)
        if distance <= radius_s:
            station_counts += 1

    for ith in range(station_counts):
        diminishing_benefit += 1 / (
                    ith + 1)  # diminishing return, as more stations cover node v, the higher the benefit
    single_benefit = diminishing_benefit * (1 - 0.1 * priv_CS)
    return single_benefit


def total_number_EVs(my_station, my_node_list):
    """
    yields total number of EVs coming to S in a unit time interval for charging
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    # D_s = sum([1 / my_node[1]["distance"] * weak_demand(my_node) if my_node[1]["charging station"] == s_pos[0]
    #            else 0 for my_node in my_node_list])
    D_s = sum([ceil(ev_per_capita * my_node[1]['pop']) if my_node[1]["charging station"] == s_pos[0]
               else 0 for my_node in my_node_list])
    s_dict["D_s"] = D_s  # dimensionless
    return my_station


def service_rate(my_station):
    """
    returns how many cars can be served within one hour
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    s_dict["service rate"] = s_dict["capability"] / BATTERY  # [service rate] = 1/h
    return my_station


def W_s(my_station, max_wait_multiplier=100):
    """
    Returns the expected waiting time, capped to prevent 'infinite' spikes.
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]

    # tau_s is the average service time
    tau_s = 1 / (s_dict["service rate"] + 1e-6)
    rho_s = s_dict["D_s"] * tau_s * time_unit

    # CLIP RHO: We cap rho at 0.99 (or similar) so the denominator never hits zero.
    # This prevents the 'massive' 10**6 jump.
    rho_s_capped = min(rho_s, 0.999)

    # Calculation (M/M/1 formula)
    my_W_s = (rho_s_capped * tau_s) / (2 * (1 - rho_s_capped))

    # Optional: Hard cap the final wait time to a multiple of service time
    # e.g., waiting 100x longer than the service time is effectively 'infinite'
    max_allowed = tau_s * max_wait_multiplier
    s_dict["W_s"] = min(my_W_s, max_allowed)
    return my_station


def s_dictionnary(my_station, my_node_list):
    """
    returns the dictionnary for the station
    """
    my_station = installment_fee(my_station)
    my_station = charging_capability(my_station)
    my_station = influence_radius(my_station)
    my_station = total_number_EVs(my_station, my_node_list)
    my_station = service_rate(my_station)
    my_station = W_s(my_station)
    return my_station


# SCORE over the plan #####################################################################
def social_benefit(my_plan, my_node_list):
    """
    Returns the social benefit of the charging plan.
    Combines two balanced components with fair weighting:
    1. Node coverage: how many stations cover each node (with diminishing returns)
    2. Station coverage: how many nodes each station covers (with diminishing returns)
    Both components use diminishing returns to encourage balanced, distributed placement.
    """
    if not my_plan:
        return 0

    # Component 1: Node perspective - how well are nodes covered by stations
    # (how many charging stations can each node access)
    node_coverage_total = 0
    for my_node in my_node_list:
        node_coverage_total += node_coverage(my_plan, my_node)
    node_coverage_avg = node_coverage_total / len(my_node_list)

    # Component 2: Station perspective - how efficiently do stations cover nodes
    # (with diminishing returns to encourage balanced coverage)
    station_coverage_total = 0
    for station in my_plan:
        station_coverage_total += station_coverage_diminishing(station, my_node_list)
    station_coverage_avg = station_coverage_total / len(my_plan)

    # Balance both components equally
    # This ensures neither metric dominates the benefit calculation
    my_benefit = (node_coverage_avg + station_coverage_avg) / 2
    return my_benefit


def travel_cost(my_node_list):
    """ yields the estimated travel time of all vehicles """
    my_cost_travel = sum([my_node[1]["distance"] * weak_demand(my_node) / VELOCITY for my_node in my_node_list])
    return my_cost_travel


def charging_time(my_plan):
    """
    yields the total charging time given the capability of the CS of the charging plan
    """
    # my_charg_time = sum([my_station[2]["D_s"] / my_station[2]["service rate"] for my_station in my_plan])
    my_charg_time = 0
    for my_station in my_plan:
        my_charg_time += (my_station[2]["D_s"] / (my_station[2]["service rate"] + 1e-6))
    return my_charg_time / time_unit


def waiting_time(my_plan):
    """
    returns the average total waiting time of the charging plan
    """
    my_wait_time = sum([my_station[2]["D_s"] * my_station[2]["W_s"] for my_station in my_plan])
    return my_wait_time / time_unit


def social_cost(my_plan, my_node_list):
    """
    returns the social cost, i.e. the negative side of the charging plan
    """
    cost_travel = travel_cost(my_node_list)  # dimensionless
    charg_time = charging_time(my_plan)  # dimensionless
    wait_time = waiting_time(my_plan)  # dimensionless
    cost_boring = charg_time + wait_time  # dimensionless
    my_social_cost = alpha * cost_travel + (1 - alpha) * cost_boring
    return my_social_cost


def existing_score(my_existing_plan, my_node_list):
    """
    computes the score of the existing infrastructure
    """
    my_benefit = social_benefit(my_existing_plan, my_node_list)
    travel_time = travel_cost(my_node_list)  # dimensionless
    charg_time = charging_time(my_existing_plan)  # dimensionless
    wait_time = waiting_time(my_existing_plan)
    cost_boring = charg_time + wait_time  # dimensionless
    my_cost = alpha * travel_time + (1 - alpha) * cost_boring
    return my_benefit, my_cost, charg_time, wait_time, travel_time


def norm_score(my_plan, my_node_list, norm_benefit, norm_charg, norm_wait, norm_travel):
    """
    same as score, but normalised.
    """
    my_score = -my_inf
    if not my_plan:
        return my_score
    benefit = social_benefit(my_plan, my_node_list) / norm_benefit
    cost_travel = travel_cost(my_node_list) / norm_travel # dimensionless
    charg_time = charging_time(my_plan) / norm_charg # dimensionless
    wait_time = waiting_time(my_plan) / norm_wait # dimensionless
    cost = (alpha * cost_travel + (1 - alpha) * (charg_time + wait_time)) / 3
    my_score = my_lambda * benefit - (1 - my_lambda) * cost
    return my_score, benefit, cost, charg_time, wait_time, cost_travel


def score(my_plan, my_node_list):
    """
    returns the final result, i.e., the social score
    """
    my_score = -my_inf
    benefit = 0
    cost = 0
    if not my_plan:
        return my_score, benefit, cost
    benefit = social_benefit(my_plan, my_node_list)  # dimensionless
    cost = social_cost(my_plan, my_node_list)
    my_score = my_lambda * benefit - (1 - my_lambda) * cost
    return my_score, benefit, cost

def get_relocate_cost(station_config_index):
    move_cost = RELOCATION_FACTOR * INSTALL_FEE[station_config_index]
    return move_cost

# Constraints checks ############################################################################
def station_capacity_check(my_plan):
    """
    check if number of stations exceed capacity
    """
    for my_station in my_plan:
        s_x = my_station[1]
        if sum(s_x) > K:
            print("Error: More chargers at the station than admitted: {} chargers".format(sum(s_x)))


def installment_cost_check(my_plan, my_basic_cost):
    """
    check if instalment costs exceed budget
    """
    total_inst_cost = sum([my_station[2]["fee"] for my_station in my_plan]) - my_basic_cost
    if total_inst_cost > BUDGET:
        print("Error: Maximal BUDGET for installation costs exceeded.")


def control_charg_decision(my_plan, my_node_list):
    for my_node in my_node_list:
        station_sum = sum([1 for my_station in my_plan if my_node[1]["charging station"] == my_station[0]])
        if station_sum > 1:
            print("Error: More than one station is assigned to a node.")


def waiting_time_check(my_plan):
    """
    check that wiating time is bounded
    """
    for my_station in my_plan:
        s_dict = my_station[2]
        if s_dict["W_s"] == my_inf:
            print("Error: Waiting time goes to infinity.")


def constraint_check(my_plan, my_node_list, basic_cost):
    """
    test if solution satisfies all constraints
    """
    installment_cost_check(my_plan, basic_cost)
    control_charg_decision(my_plan, my_node_list)
    station_capacity_check(my_plan)
    waiting_time_check(my_plan)


def get_lookup(path):
    with open(path, 'r') as f:
        lookup = json.load(f)
    return lookup


def initial_solution(my_config_dict, my_node_list, s_pos):
    """
    get the initial solution for the charging configuration
    """
    W = 0  # minimum capacity constraint
    radius = 50
    # search for all nodes within station radius
    for my_node in my_node_list:
        if haversine(s_pos, my_node) <= radius:
            W += weak_demand(my_node)
    W = ceil(W) * BATTERY
    key_list = sorted(list(my_config_dict.keys()))
    for key in key_list:
        if int(key) > W:  # convert str to int
            break
    best_config = my_config_dict[key]
    return best_config


def coverage(my_node_list, my_plan):
    """
    see which nodes are covered by the charging plan
    """
    for my_node in my_node_list:
        cover = node_coverage(my_plan, my_node)
        my_node[1]["covered"] = cover


def choose_node_new_benefit(free_list, all_node_list, R_search=10):
    """
    pick location with highest potential based on Potential/Coverage.
    """
    potential_scores = []
    epsilon = 0.001
    for candidate_node in free_list:
        local_demand = 0
        for node in all_node_list:
            dist = haversine(candidate_node, node)
            if dist <= R_search:
                local_demand += weak_demand(node)
        
        current_coverage = candidate_node[1].get("covered", 0)
        score = local_demand / (current_coverage + epsilon)
        
        potential_scores.append(score)
    best_index = np.argmax(potential_scores)
    
    return free_list[best_index]


def choose_node_bydemand(free_list, my_plan, add=False):
    """
    pick location with highest weakened demand
    """
    chosen_node = None
    if add:
        # choose the node with the highest waiting time
        priority_list = [station[2]["D_s"] * station[2]["W_s"] + station[2]["D_s"] / (station[2]["service rate"] + eps)
                         for station in my_plan]
        max_station_index = np.argmax(priority_list)
        max_station = my_plan[max_station_index]
        chosen_node = max_station[0]

    else:
        demand_list = [dynamic_demand(my_node, my_plan) for my_node in free_list]
        chosen_index = demand_list.index(max(demand_list))
        chosen_node = free_list[chosen_index]
    return chosen_node


def anti_choose_node_bybenefit(my_node_list, my_plan):
    """
    choose station with the least coverage
    """
    plan_list = [station[0][0] for station in my_plan]
    my_occupied_list = [node for node in my_node_list if node[0] in plan_list]
    if not my_occupied_list:
        return None
    upbound_list = [node[1]["upper_bound"] for node in my_occupied_list]
    pos_minindex = upbound_list.index(min(upbound_list))
    remove_node = my_occupied_list[pos_minindex]
    plan_index = plan_list.index(remove_node[0])
    remove_station = my_plan[plan_index]
    return remove_station


def _support_stations(station):
    charg_time = station[2]["D_s"] / (station[2]["service rate"] + 1e-6)  # not sure why this need
    wait_time = station[2]["D_s"] * station[2]["W_s"]
    neediness = (wait_time + charg_time)
    return neediness


def support_stations(my_plan, free_list):
    """
    choose a station which needs support due to highest waiting + charging time
    """
    cost_list = [_support_stations(station) for station in my_plan]
    if not cost_list:
        chosen_node = choose_node_bydemand(free_list)
    else:
        index = cost_list.index(max(cost_list))
        station_sos = my_plan[index]
        if sum(station_sos[1]) < K:
            chosen_node = station_sos[0]
        else:
            # look for nearest node that could support the station
            dis_list = [haversine(station_sos[0], my_node) for my_node in free_list]
            min_index = dis_list.index(min(dis_list))
            chosen_node = free_list[min_index]
    return chosen_node


# Load graph file for travel distance computing
location = "DongDa"
graph_file = f"custom_environment/data/Graph/{location}/{location}.graphml"
graph = nx.read_graphml(graph_file)

# Parameters ########################################################
alpha = 0.4
my_lambda = 0.5
eps = 1e-9
ev_per_capita = 0.022
evs_parking_area = 15  # meter square

K = 100  # maximal number of chargers at a station
RADIUS_MAX = 1  # [radius_max] = km
# INSTALL_FEE = np.array([300, 750, 28000])  # fee per installing a charger of type 1, 2 or 3. [fee] = $
# CHARGING_POWER = np.array([7, 22, 50])  # [power] = kW, rounded
CHARGING_POWER = np.array([3, 7, 11, 20, 22, 30, 60, 80, 120, 150, 180, 250])
INSTALL_FEE = np.array([5, 11, 12, 100, 12, 143, 278, 397, 416, 676, 956, 3272])
BATTERY = 85  # battery capacity, [BATTERY] = kWh
RELOCATION_FACTOR = 0.2  # Assumption: Moving costs 20% of a new one

BUDGET = 900000

time_unit = 1  # [time_unit] = h, introduced for getting the units correctly
capacity_unit = 1  # [cap_unit] = kW, introduced for getting the units correctly
VELOCITY = 40  # km/h

my_inf = 10 ** 6
my_dis_inf = 10 ** 7

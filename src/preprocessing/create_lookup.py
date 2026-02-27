import json
import os
import sys

# Ensure we can import modules from project root (d:/CSPRL-Refactor)
current_dir = os.path.dirname(os.path.abspath(__file__)) # src/preprocessing
src_dir = os.path.dirname(current_dir) # src
project_root = os.path.dirname(src_dir) # root

if project_root not in sys.path:
    sys.path.append(project_root)

import custom_environment.helpers as ef


def generate_lookup():
    """
    Generates the config_lookup.json file using Dynamic Programming.
    Finds the cheapest configuration (combination of chargers) for each total power capacity.
    Max number of chargers is ef.K.
    """
    print("Starting configuration generation with DP (create_lookup.py)...")

    powers = ef.CHARGING_POWER
    fees = ef.INSTALL_FEE
    K = ef.K
    num_types = len(powers)

    print(f"Constraints: K={K}, Num Types={num_types}")
    print(f"Powers: {powers}")
    print(f"Fees: {fees}")

    # dp[k] will store a dictionary: capacity -> {'cost': cost, 'config': [count_0, count_1, ...]}
    # representing reachable states using exactly k chargers.
    dp = [{} for _ in range(K + 1)]

    # Base case: 0 chargers
    dp[0][0] = {'cost': 0, 'config': [0] * num_types}

    # Iterate 1 to K chargers
    for k in range(K):
        # print(f"Processing layer {k} -> {k+1}...")
        for cap, data in dp[k].items():
            curr_cost = data['cost']
            curr_config = data['config']

            for i in range(num_types):
                # Add one charger of type i
                new_cap = cap + powers[i]
                new_cost = curr_cost + fees[i]

                # Update if new state is better (cheaper) for this capacity
                # Note: For the same k+1 and new_cap, we only keep the cheapest one.
                if new_cap not in dp[k + 1] or new_cost < dp[k + 1][new_cap]['cost']:
                    new_config = list(curr_config)
                    new_config[i] += 1
                    dp[k + 1][new_cap] = {
                        'cost': new_cost,
                        'config': new_config
                    }

    # Consolidate all levels into a single best_dict
    # capacity -> {'cost': min_cost, 'config': best_config}
    best_dict = {}
    for k in range(K + 1):
        for cap, data in dp[k].items():
            if cap not in best_dict or data['cost'] < best_dict[cap]['cost']:
                best_dict[cap] = data

    print(f"Found {len(best_dict)} unique capacities.")

    # Post-processing: Monotonicity optimization
    # If a higher capacity is cheaper than a lower capacity, the lower capacity should use the higher capacity's config.
    sorted_caps = sorted(best_dict.keys(), reverse=True)  # Descending

    min_cost_so_far = float('inf')
    best_config_so_far = None

    final_lookup = {}

    for cap in sorted_caps:
        cost = best_dict[cap]['cost']
        if cost < min_cost_so_far:
            min_cost_so_far = cost
            best_config_so_far = best_dict[cap]['config']

        final_lookup[str(cap)] = best_config_so_far  # Convert key to string for JSON

    print("Post-processing complete.")

    # Sort keys for cleaner JSON
    sorted_final = {str(k): final_lookup[str(k)] for k in sorted(map(int, final_lookup.keys()))}

    # Save to custom_environment/data/processed/config_lookup.json
    output_path = os.path.join(project_root, "custom_environment", "data", "processed", "config_lookup.json")

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(sorted_final, f, indent=4)

    print(f"Saved configuration to {output_path}")


if __name__ == "__main__":
    generate_lookup()
import pickle
import pandas as pd
import os
import custom_environment.helpers as H
import numpy as np

# Configuration
LOCATION = "DongDa"
STEP = 119200
RESULT_FILE = f"Results/optimal_plan/{LOCATION}/plan_RL_{STEP}.pkl"
OUTPUT_CSV = f"Results/optimal_plan/{LOCATION}/station_config_{STEP}.csv"


def export_station_config():
    if not os.path.exists(RESULT_FILE):
        print(f"Error: Result file {RESULT_FILE} not found.")
        return

    print(f"Loading plan from {RESULT_FILE}...")
    with open(RESULT_FILE, "rb") as f:
        plan = pickle.load(f)

    print(f"Found {len(plan)} stations in the plan.")

    station_data = []

    for idx, station in enumerate(plan):
        # station structure: [node_info, charger_config, station_dict]
        # node_info: [node_id, attributes_dict]
        # charger_config: array of counts for each charger type
        # station_dict: {"fee": ..., "capability": ..., "D_s": ...}

        node_id = station[0][0]
        lat = station[0][1]['y']
        lon = station[0][1]['x']

        # Charger configuration
        # ef.CHARGING_POWER has the power for each type
        charger_counts = station[1]

        # Create a readable string for config
        config_str = []
        for type_idx, count in enumerate(charger_counts):
            if count > 0:
                power = H.CHARGING_POWER[type_idx]
                config_str.append(f"{count}x{power}kW")

        # Prepare row data
        row = {
            "Station_ID": idx + 1,
            "Node_ID": node_id,
            "Latitude": lat,
            "Longitude": lon,
            "Total_Capacity_kW": station[2].get("capability", 0),
            "Install_Fee": station[2].get("fee", 0),
            "Service_Rate_per_h": round(station[2].get("service rate", 0), 2),
            "Expected_Wait_Time_h": round(station[2].get("W_s", 0), 4),
            "Number of EVs": station[2].get("D_s", 0),
            "Charging Time": station[2].get("D_s", 0) / station[2].get("service rate", 0),
            "Charger_Config": ", ".join(config_str),
            "Total_Chargers": sum(charger_counts)
        }

        # Add individual columns for each charger type for detailed analysis
        for type_idx, count in enumerate(charger_counts):
            power = H.CHARGING_POWER[type_idx]
            row[f"Qty_{power}kW"] = count

        station_data.append(row)

    # Save to CSV
    df = pd.DataFrame(station_data)

    # Reorder columns to put main info first
    cols = ["Station_ID", "Node_ID", "Latitude", "Longitude", "Total_Capacity_kW",
            "Total_Chargers", "Charger_Config", "Install_Fee", "Charging Time",
            "Service_Rate_per_h", "Expected_Wait_Time_h", "Number of EVs"]
    # Append dynamic columns
    dt_cols = [c for c in df.columns if c.startswith("Qty_")]
    final_cols = cols + dt_cols

    df = df[final_cols]

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print("-" * 50)
    print(f"Successfully exported configuration to:\n{os.path.abspath(OUTPUT_CSV)}")
    print("-" * 50)
    print(df[["Station_ID", "Total_Capacity_kW", "Charger_Config", "Number of EVs", "Expected_Wait_Time_h", "Charging Time"]].to_string())
    print("Total charg time:", df["Charging Time"].sum())

if __name__ == "__main__":
    export_station_config()

"""
Hanoi Power Grid Loader

Module để load dữ liệu CSV vào pandapower network
và thực hiện các phân tích lưới điện.
"""

import pandas as pd
import os
from typing import Optional, Dict, List, Tuple, Any

try:
    import pandapower as pp

    PANDAPOWER_AVAILABLE = True
except ImportError:
    PANDAPOWER_AVAILABLE = False


class GridLoader:
    """
    Loader để đọc dữ liệu CSV và tạo pandapower network.

    Sử dụng:
        loader = GridLoader("power_grid/data/hanoi_grid_data")
        net = loader.create_network()
        results = loader.run_power_flow()
    """

    def __init__(self, data_folder: str, bus_limit: float):
        """
        Khởi tạo loader.

        Args:
            data_folder: Thư mục chứa các file CSV
        """
        self.data_folder = data_folder
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.net = None
        self.bus_limit = bus_limit
        self._load_csv_files()

    def _load_csv_files(self) -> None:
        """Load tất cả file CSV từ thư mục."""
        required_files = ["bus.csv", "line.csv", "trafo.csv", "load.csv"]
        optional_files = ["ext_grid.csv", "metadata.csv"]

        for filename in required_files + optional_files:
            filepath = os.path.join(self.data_folder, filename)
            if os.path.exists(filepath):
                name = filename.replace(".csv", "")
                self.dataframes[name] = pd.read_csv(filepath, index_col=0)
                print(f"  [OK] Loaded: {filename} ({len(self.dataframes[name])} records)")
            elif filename in required_files:
                raise FileNotFoundError(f"Thiếu file bắt buộc: {filepath}")

    def create_network(self, name: str = "Hanoi Grid") -> Any:
        """
        Tạo pandapower network từ dữ liệu đã load.

        Args:
            name: Tên của network

        Returns:
            pandapower network object
        """
        if not PANDAPOWER_AVAILABLE:
            raise ImportError("pandapower chưa được cài đặt")

        # Tạo empty network
        self.net = pp.create_empty_network(name=name)

        # 1. Thêm buses
        bus_df = self.dataframes["bus"]
        bus_mapping = {}  # map CSV index -> pandapower index

        for idx, row in bus_df.iterrows():
            pp_idx = pp.create_bus(
                self.net,
                vn_kv=row["vn_kv"],
                name=row["name"],
                geodata=(row["x"], row["y"]) if "x" in row and "y" in row else None,
            )
            bus_mapping[idx] = pp_idx

        # 2. Thêm external grid (slack bus)
        if "ext_grid" in self.dataframes:
            ext_df = self.dataframes["ext_grid"]
            for _, row in ext_df.iterrows():
                pp.create_ext_grid(
                    self.net,
                    bus=bus_mapping[row["bus"]],
                    vm_pu=row.get("vm_pu", 1.0),
                    va_degree=row.get("va_degree", 0.0),
                    name=row.get("name", "External Grid"),
                )
        else:
            # Mặc định: external grid tại bus đầu tiên
            pp.create_ext_grid(self.net, bus=0, vm_pu=1.0, name="EVN_GRID")

        # 3. Thêm transformers
        trafo_df = self.dataframes["trafo"]
        for _, row in trafo_df.iterrows():
            # Kiểm tra std_type có trong thư viện không
            std_type = row.get("std_type", "63 MVA 110/20 kV")

            if std_type in self.net.std_types["trafo"]:
                pp.create_transformer(
                    self.net,
                    hv_bus=bus_mapping[row["hv_bus"]],
                    lv_bus=bus_mapping[row["lv_bus"]],
                    std_type=std_type,
                    name=row["name"],
                )
            else:
                # Tạo transformer từ thông số
                pp.create_transformer_from_parameters(
                    self.net,
                    hv_bus=bus_mapping[row["hv_bus"]],
                    lv_bus=bus_mapping[row["lv_bus"]],
                    sn_mva=row.get("sn_mva", 63.0),
                    vn_hv_kv=row.get("vn_hv_kv", 110.0),
                    vn_lv_kv=row.get("vn_lv_kv", 23.0),
                    vkr_percent=0.5,  # Điện trở ngắn mạch
                    vk_percent=12.0,  # Điện kháng ngắn mạch
                    pfe_kw=row.get("pfe_kw", 35.0),
                    i0_percent=row.get("i0_percent", 0.04),
                    name=row["name"],
                )

        # 4. Thêm lines
        line_df = self.dataframes["line"]
        for _, row in line_df.iterrows():
            std_type = row.get("std_type", "NAYY 4x240 SE")

            if std_type in self.net.std_types["line"]:
                pp.create_line(
                    self.net,
                    from_bus=bus_mapping[row["from_bus"]],
                    to_bus=bus_mapping[row["to_bus"]],
                    length_km=row["length_km"],
                    std_type=std_type,
                    name=row["name"],
                    max_i_ka=row.get("max_i_ka", 0.42),
                )
            else:
                # Tạo line từ thông số (cáp Cu/XLPE 240mm2)
                pp.create_line_from_parameters(
                    self.net,
                    from_bus=bus_mapping[row["from_bus"]],
                    to_bus=bus_mapping[row["to_bus"]],
                    length_km=row["length_km"],
                    r_ohm_per_km=0.0754,  # Cu 240mm2
                    x_ohm_per_km=0.089,
                    c_nf_per_km=0.0,
                    max_i_ka=row.get("max_i_ka", 0.42),
                    name=row["name"],
                )

        # 5. Thêm loads
        load_df = self.dataframes["load"]
        for _, row in load_df.iterrows():
            pp.create_load(
                self.net,
                bus=bus_mapping[row["bus"]],
                p_mw=row["p_mw"],
                q_mvar=row["q_mvar"],
                name=row["name"],
                scaling=row.get("scaling", 1.0),
            )

        print(f"\n[OK] Created pandapower network: {name}")
        print(f"   Buses: {len(self.net.bus)}, Lines: {len(self.net.line)}")
        print(f"   Trafos: {len(self.net.trafo)}, Loads: {len(self.net.load)}")

        return self.net

    def run_power_flow(self, algorithm: str = "nr") -> Dict[str, pd.DataFrame]:
        """
        Chạy power flow analysis.

        Args:
            algorithm: Thuật toán ('nr' = Newton-Raphson, 'bfsw' = Backward/Forward Sweep)

        Returns:
            Dictionary chứa kết quả (res_bus, res_line, res_trafo)
        """
        if self.net is None:
            self.create_network()

        try:
            pp.runpp(self.net, algorithm=algorithm, max_iteration=50)
            print("\n[OK] Power flow analysis completed!")

            return {
                "res_bus": self.net.res_bus.copy(),
                "res_line": self.net.res_line.copy(),
                "res_trafo": self.net.res_trafo.copy(),
                "res_load": self.net.res_load.copy(),
            }
        except Exception as e:
            print(f"\n[ERROR] Power flow failed: {e}")
            return {}

    def check_constraints(
            self,
            v_min: float = 0.95,
            v_max: float = 1.05,
            line_loading_max: float = 80.0,
            trafo_loading_max: float = 80.0,
    ) -> Dict[str, List[Dict]]:
        """
        Kiểm tra vi phạm ràng buộc lưới điện.

        Args:
            v_min: Điện áp tối thiểu (pu)
            v_max: Điện áp tối đa (pu)
            line_loading_max: % tải đường dây tối đa
            trafo_loading_max: % tải MBA tối đa

        Returns:
            Dictionary chứa danh sách vi phạm
        """
        if self.net is None or "res_bus" not in dir(self.net):
            self.run_power_flow()

        violations = {
            "voltage": [],
            "line_loading": [],
            "trafo_loading": [],
        }

        # Kiểm tra điện áp
        for idx, row in self.net.res_bus.iterrows():
            vm_pu = row["vm_pu"]
            if vm_pu < v_min or vm_pu > v_max:
                violations["voltage"].append({
                    "bus": idx,
                    "name": self.net.bus.at[idx, "name"],
                    "vm_pu": vm_pu,
                    "violation": "LOW" if vm_pu < v_min else "HIGH",
                })

        # Kiểm tra tải đường dây
        for idx, row in self.net.res_line.iterrows():
            loading = row["loading_percent"]
            if loading > line_loading_max:
                violations["line_loading"].append({
                    "line": idx,
                    "name": self.net.line.at[idx, "name"],
                    "loading_percent": loading,
                })

        # Kiểm tra tải MBA
        for idx, row in self.net.res_trafo.iterrows():
            loading = row["loading_percent"]
            if loading > trafo_loading_max:
                violations["trafo_loading"].append({
                    "trafo": idx,
                    "name": self.net.trafo.at[idx, "name"],
                    "loading_percent": loading,
                })

        # In kết quả
        total_violations = sum(len(v) for v in violations.values())
        if total_violations == 0:
            print("[OK] No constraint violations!")
        else:
            print(f"[WARN] Found {total_violations} violations:")
            for vtype, vlist in violations.items():
                if vlist:
                    print(f"   - {vtype}: {len(vlist)} violations")

        return violations

    def get_available_capacity(self, bus_idx: int) -> Dict[str, float]:
        """
        Tính công suất còn lại có thể sử dụng tại một bus.

        Args:
            bus_idx: Index của bus

        Returns:
            Dictionary với available_mw, current_load_mw, max_capacity_mw
        """
        if self.net is None:
            self.create_network()

        # Chạy power flow nếu chưa có
        if not hasattr(self.net, 'res_bus') or self.net.res_bus.empty:
            self.run_power_flow()

        # Tổng phụ tải hiện tại tại bus
        current_loads = self.net.load[self.net.load["bus"] == bus_idx]
        current_load_mw = current_loads["p_mw"].sum()

        # Ước tính công suất tối đa dựa trên đường dây kết nối
        # (Đơn giản: lấy max_i_ka của đường dây nhỏ nhất * điện áp)
        connected_lines = self.net.line[
            (self.net.line["from_bus"] == bus_idx) |
            (self.net.line["to_bus"] == bus_idx)
            ]

        if len(connected_lines) > 0:
            min_max_i_ka = connected_lines["max_i_ka"].min()
            vn_kv = self.net.bus.at[bus_idx, "vn_kv"]
            # S = sqrt(3) * V * I
            max_line_mva = 1.732 * vn_kv * min_max_i_ka
        else:
            max_line_mva = 9999.0  # Limitless if no lines (unlikely)

        # Check for connected transformers (feeding into this bus)
        # Typically transformers connect HV bus to LV bus.
        # If this is an LV bus, we care about trafo connecting to HV.
        connected_trafos = self.net.trafo[
            (self.net.trafo["lv_bus"] == bus_idx) |
            (self.net.trafo["hv_bus"] == bus_idx)
            ]

        if len(connected_trafos) > 0:
            # Sum of capacities if parallel, but usually we just take the one feeding it
            # For simplicity, assume redundancy or parallel operation sum
            # BUT safe bet: max_trafo_mva = sum(sn_mva)
            max_trafo_mva = connected_trafos["sn_mva"].sum()
        else:
            max_trafo_mva = 9999.0  # No trafo limit found

        max_capacity_mva = min(max_line_mva, max_trafo_mva)

        # Correction for "10.0" default if both limitless?
        if max_capacity_mva > 9000:
            max_capacity_mva = 10.0  # Default fallback if floating bus

        available_mw = max_capacity_mva * self.bus_limit - current_load_mw  # 80% loading limit

        return {
            "available_mw": max(0, available_mw),
            "current_load_mw": current_load_mw,
            "max_capacity_mva": max_capacity_mva,
            "bus_voltage_kv": self.net.bus.at[bus_idx, "vn_kv"],
        }

    def get_summary(self) -> Dict[str, Any]:
        """Lấy thông tin tổng quan về network."""
        if self.net is None:
            return {}

        summary = {
            "name": self.net.name,
            "n_buses": len(self.net.bus),
            "n_lines": len(self.net.line),
            "n_trafos": len(self.net.trafo),
            "n_loads": len(self.net.load),
            "total_load_mw": self.net.load["p_mw"].sum(),
            "total_load_mvar": self.net.load["q_mvar"].sum(),
        }

        if hasattr(self.net, 'res_bus') and not self.net.res_bus.empty:
            summary["min_voltage_pu"] = self.net.res_bus["vm_pu"].min()
            summary["max_voltage_pu"] = self.net.res_bus["vm_pu"].max()
            summary["max_line_loading"] = self.net.res_line["loading_percent"].max()
            summary["max_trafo_loading"] = self.net.res_trafo["loading_percent"].max()

        return summary

    def find_nearest_bus(
            self,
            lat: float,
            lon: float,
            voltage_kv: float = 22.0,
            prefer_available: bool = False
    ) -> Dict[str, Any]:
        """
        Tìm bus gần nhất với tọa độ GPS cho trước.

        Dùng cho RL agent khi chọn vị trí đặt trạm sạc EV.

        Args:
            lat: Vĩ độ (latitude)
            lon: Kinh độ (longitude)
            voltage_kv: Cấp điện áp của bus cần tìm (mặc định 22kV cho trạm sạc)
            prefer_available: Nếu True, ưu tiên chọn bus còn công suất ngay cả khi xa hơn một chút.

        Returns:
            Dictionary với bus_idx, distance_km, bus_name, available_capacity
        """
        if self.net is None:
            self.create_network()

        import numpy as np

        # Lọc các bus theo cấp điện áp
        if voltage_kv:
            candidate_buses = self.net.bus[self.net.bus["vn_kv"] == voltage_kv]
        else:
            candidate_buses = self.net.bus

        if len(candidate_buses) == 0:
            return {"error": f"Không tìm thấy bus {voltage_kv}kV nào"}

        # Tính khoảng cách đến từng bus (Haversine simplified)
        best_score = float('inf')
        nearest_idx = None
        min_dist = float('inf')

        for idx, row in candidate_buses.iterrows():
            # geodata stored as (x=lon, y=lat)
            bus_lon = row.get("geodata", (None, None))[0] if "geodata" in row else None
            bus_lat = row.get("geodata", (None, None))[1] if "geodata" in row else None

            # Fallback: check if coordinates are in separate columns
            if bus_lon is None and "x" in self.dataframes["bus"].columns:
                bus_df_row = self.dataframes["bus"].loc[idx]
                bus_lon = bus_df_row.get("x")
                bus_lat = bus_df_row.get("y")

            if bus_lon is None or bus_lat is None:
                continue

            # Approximate distance in km (Euclidean on lat/lon * 111km/degree)
            dist = np.sqrt((lat - bus_lat) ** 2 + (lon - bus_lon) ** 2) * 111
            
            # Simple scoring: distance + penalty if no capacity
            score = dist
            if prefer_available:
                capacity_info = self.get_available_capacity(idx)
                if capacity_info["available_mw"] <= 0:
                    # Add a large distance penalty (e.g., 5km) to overloaded buses
                    # This makes the agent prefer a bus with capacity within 5km over an overloaded one nearby
                    score += 5.0 

            if score < best_score:
                best_score = score
                min_dist = dist
                nearest_idx = idx

        if nearest_idx is None:
            return {"error": "Không tìm thấy bus nào có tọa độ"}

        # Lấy thông tin capacity cho bus tốt nhất tìm được
        capacity_info = self.get_available_capacity(nearest_idx)

        return {
            "bus_idx": nearest_idx,
            "bus_name": self.net.bus.at[nearest_idx, "name"],
            "distance_km": round(min_dist, 3),
            "voltage_kv": voltage_kv,
            **capacity_info,
        }


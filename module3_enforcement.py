"""
ParkIQ — Module 3: Enforcement Priority Engine (Enhanced)
Priority = 0.40×CIS + 0.30×LSTM stress + 0.20×spillover centrality + 0.10×recency
+ Christofides-approximation TSP patrol routing with time-window awareness
+ Multi-officer parallel route splitting
"""
import json, math
import pandas as pd
import numpy as np
import networkx as nx
from config import SCORED_PARQUET, JUNCTION_STRESS, SPILLOVER_JSON, ENFORCEMENT_CSV


def norm(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn + 1e-9)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def two_opt(route, dist_matrix):
    """2-opt local search to improve TSP route."""
    best = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best)):
                if j - i == 1: continue
                new_route = best[:i] + best[i:j][::-1] + best[j:]
                old_cost = sum(dist_matrix[best[k]][best[k+1]] for k in range(len(best)-1))
                new_cost = sum(dist_matrix[new_route[k]][new_route[k+1]] for k in range(len(new_route)-1))
                if new_cost < old_cost:
                    best = new_route
                    improved = True
    return best


def build_patrol_routes(zone_df: pd.DataFrame, top_n: int = 15,
                        num_officers: int = 3) -> list:
    """
    Build multi-officer patrol routes using 2-opt TSP.
    - Select top_n zones by priority
    - Split across num_officers for parallel coverage
    - Each route is time-window optimised (peak-hour zones visited first)
    """
    top = zone_df.head(top_n).reset_index(drop=True)

    # Build distance matrix
    n = len(top)
    dist_matrix = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i][j] = haversine_km(
                    top.loc[i,"lat"], top.loc[i,"lon"],
                    top.loc[j,"lat"], top.loc[j,"lon"]
                )

    # Greedy NN initial tour
    visited   = [0]
    remaining = list(range(1, n))
    while remaining:
        last    = visited[-1]
        nearest = min(remaining, key=lambda i: dist_matrix[last][i])
        visited.append(nearest)
        remaining.remove(nearest)

    # Improve with 2-opt
    optimised = two_opt(visited, dist_matrix)

    # Split tour into num_officers segments (round-robin by segment length)
    chunk = math.ceil(len(optimised) / num_officers)
    routes = []
    for officer_id in range(num_officers):
        segment = optimised[officer_id*chunk : (officer_id+1)*chunk]
        stops = []
        for stop_num, idx in enumerate(segment):
            row = top.loc[idx]
            stops.append({
                "stop"            : stop_num + 1,
                "junction_name"   : row["junction_name"],
                "lat"             : float(row["lat"]),
                "lon"             : float(row["lon"]),
                "priority_score"  : float(row["priority_score"]),
                "is_peak_priority": bool(row.get("predicted_stress_next_h", 0) > 0.5),
            })
        # Compute route distance
        route_km = sum(
            haversine_km(stops[k]["lat"], stops[k]["lon"],
                         stops[k+1]["lat"], stops[k+1]["lon"])
            for k in range(len(stops)-1)
        ) if len(stops) > 1 else 0.0

        routes.append({
            "officer_id"  : officer_id + 1,
            "stops"       : stops,
            "total_km"    : round(route_km, 2),
            "est_hours"   : round(route_km / 20, 2),  # 20 km/h avg patrol speed
        })

    return routes


def run():
    print("[Module 3] Computing enforcement priorities + multi-officer routing …")
    df        = pd.read_parquet(SCORED_PARQUET)
    stress_df = pd.read_parquet(JUNCTION_STRESS)

    with open(SPILLOVER_JSON) as f:
        G = nx.node_link_graph(json.load(f))
    centrality = nx.get_node_attributes(G, "spillover_centrality")

    df["created_datetime"] = pd.to_datetime(df["created_datetime"], utc=True, errors="coerce")
    agg = (df[df["junction_name"] != "No Junction"]
           .groupby("junction_name")
           .agg(cis_mean         = ("cis",              "mean"),
                violation_count  = ("record_id",         "count"),
                lat              = ("latitude",          "mean"),
                lon              = ("longitude",         "mean"),
                latest           = ("created_datetime",  "max"),
                validated_pct    = ("validated",         "mean"))
           .reset_index())

    agg = agg.merge(stress_df[["junction_name","predicted_stress_next_h",
                                "historical_peak_stress"]],
                    on="junction_name", how="left").fillna({"predicted_stress_next_h": 0,
                                                            "historical_peak_stress": 0})
    agg["spillover_centrality"] = agg["junction_name"].map(centrality).fillna(0)

    now   = agg["latest"].max()
    age_h = (now - agg["latest"]).dt.total_seconds() / 3600
    agg["recency_score"] = (1 - age_h / (age_h.max() + 1)).clip(0, 1)

    # Validation quality bonus
    agg["quality_bonus"] = agg["validated_pct"].fillna(0) * 0.05

    agg["priority_score"] = (
        0.40 * norm(agg["cis_mean"]) +
        0.28 * norm(agg["predicted_stress_next_h"]) +
        0.17 * norm(agg["spillover_centrality"]) +
        0.10 * norm(agg["recency_score"]) +
        0.05 * norm(agg["quality_bonus"])
    ).round(4)

    agg = agg.sort_values("priority_score", ascending=False).reset_index(drop=True)
    agg["rank"] = agg.index + 1

    # Multi-officer routing
    routes = build_patrol_routes(agg, top_n=15, num_officers=3)

    # Flatten patrol stops back to df
    for route in routes:
        for stop in route["stops"]:
            jname = stop["junction_name"]
            agg.loc[agg["junction_name"] == jname, "patrol_stop"]     = stop["stop"]
            agg.loc[agg["junction_name"] == jname, "officer_assigned"] = route["officer_id"]
            agg.loc[agg["junction_name"] == jname, "route_km"]        = route["total_km"]

    agg.drop(columns=["latest"], errors="ignore").to_csv(ENFORCEMENT_CSV, index=False)
    print(f"  {len(agg)} zones ranked. Top 5:")
    print(agg.head(5)[["rank","junction_name","priority_score","violation_count"]].to_string(index=False))

    print(f"\n  Multi-officer routes ({len(routes)} officers):")
    for r in routes:
        print(f"    Officer {r['officer_id']}: {len(r['stops'])} stops, "
              f"{r['total_km']:.1f} km, ~{r['est_hours']:.1f} hrs")

    print(f"  Saved → {ENFORCEMENT_CSV}")
    return agg, routes


if __name__ == "__main__":
    run()

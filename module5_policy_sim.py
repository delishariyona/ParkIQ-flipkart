"""
ParkIQ — Module 5: Policy Simulation Lab
What-if: restrict zone X by Y% → how much does total CIS (congestion) drop?
"""
import json
import pandas as pd
from config import ATTRIBUTION_PARQUET, ENFORCEMENT_CSV, POLICY_REPORT


def simulate(df: pd.DataFrame, junction_name: str, restriction_pct: float) -> dict:
    mask = df["junction_name"] == junction_name
    n    = int(mask.sum())
    if n == 0:
        return {"error": f"Junction not found: {junction_name}"}

    base   = float(df["cis"].sum())
    df_sim = df.copy()
    df_sim.loc[mask, "cis"] *= (1 - restriction_pct)
    new    = float(df_sim["cis"].sum())

    return {
        "junction"             : junction_name,
        "restriction_pct"      : restriction_pct,
        "violations_affected"  : n,
        "baseline_total_cis"   : round(base, 2),
        "simulated_total_cis"  : round(new, 2),
        "cis_reduction"        : round(base - new, 2),
        "congestion_delta_pct" : round((base - new) / base * 100, 3),
    }


def run():
    print("[Module 5] Running policy simulations …")
    df       = pd.read_parquet(ATTRIBUTION_PARQUET)
    zone_df  = pd.read_csv(ENFORCEMENT_CSV)
    top5     = zone_df.head(5)["junction_name"].tolist()

    report = {"simulations": []}
    for jname in top5:
        for pct in [0.25, 0.5, 0.75, 1.0]:
            result = simulate(df, jname, pct)
            report["simulations"].append(result)
            print(f"  {jname[:45]:45s} {int(pct*100):3d}% → Δcongestion={result.get('congestion_delta_pct','?')}%")

    with open(POLICY_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved → {POLICY_REPORT}")
    return report


if __name__ == "__main__":
    run()

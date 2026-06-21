"""
ParkIQ — Module 1: Junction Stress Predictor (fast GRU + fallback)
Trains a lightweight GRU per junction on hourly violation counts.
Falls back to exponential smoothing for junctions with sparse data.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from config import SCORED_PARQUET, JUNCTION_STRESS

SEQ_LEN = 8  # reduced look-back for speed

class StressGRU(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.gru = nn.GRU(1, hidden, 1, batch_first=True)
        self.fc  = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        return torch.sigmoid(self.fc(out[:, -1, :]))

def make_sequences(series, seq_len=SEQ_LEN):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def exp_smooth_predict(series, alpha=0.3):
    s = series[0]
    for v in series[1:]:
        s = alpha * v + (1 - alpha) * s
    return float(np.clip(s, 0, 1))

def train_gru(series, epochs=15):
    if len(series) < SEQ_LEN + 2:
        return None
    X, y = make_sequences(series)
    ds = TensorDataset(torch.tensor(X).unsqueeze(-1), torch.tensor(y).unsqueeze(-1))
    dl = DataLoader(ds, batch_size=min(64, len(ds)), shuffle=True)
    model = StressGRU()
    opt   = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    return model

def predict_next(model, series):
    model.eval()
    x = torch.tensor(series[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    with torch.no_grad():
        return float(model(x).item())

def run():
    print("[Module 1] Training GRU junction stress predictors …")
    df = pd.read_parquet(SCORED_PARQUET)
    df["created_datetime"] = pd.to_datetime(df["created_datetime"], utc=True, errors="coerce")
    df["hour_bucket"] = df["created_datetime"].dt.floor("1h")

    agg = (df.groupby(["junction_name","hour_bucket"])
             .size().reset_index(name="count"))

    results = []
    junctions = [j for j in agg["junction_name"].unique() if j != "No Junction"]
    print(f"  Processing {len(junctions)} junctions …")

    for jname in junctions:
        series = (agg[agg["junction_name"]==jname]
                  .sort_values("hour_bucket")["count"]
                  .values.astype(float))
        norm = series / series.max() if series.max() > 0 else series

        model = train_gru(norm)
        if model:
            pred = predict_next(model, norm)
        else:
            pred = exp_smooth_predict(norm) if len(norm) else 0.0

        results.append({
            "junction_name"           : jname,
            "predicted_stress_next_h" : round(pred, 4),
            "historical_mean_stress"  : round(float(np.mean(norm)), 4),
            "historical_peak_stress"  : round(float(np.max(norm)), 4),
            "total_violations"        : int(series.sum()),
            "data_hours"              : len(series),
        })

    stress_df = (pd.DataFrame(results)
                   .sort_values("predicted_stress_next_h", ascending=False)
                   .reset_index(drop=True))
    stress_df.to_parquet(JUNCTION_STRESS, index=False)
    print("  Top 5 stressed junctions:")
    print(stress_df.head(5)[["junction_name","predicted_stress_next_h","total_violations"]].to_string(index=False))
    print(f"  Saved → {JUNCTION_STRESS}")
    return stress_df

if __name__ == "__main__":
    run()

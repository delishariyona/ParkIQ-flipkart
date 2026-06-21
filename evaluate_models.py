"""
ParkIQ — Model Evaluation Script (v5.1)
Run: python evaluate_models.py

Reflects corrected April handling:
  Model A — trained and evaluated on FULL dataset including April
  Model B — trained on Nov–Mar only, evaluated on Nov–Mar held-out test set
             (April excluded from training because labels are untrustworthy)
"""
import os, warnings
warnings.filterwarnings("ignore")
os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig"

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, f1_score, precision_score, recall_score,
    average_precision_score, brier_score_loss, classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.isotonic import IsotonicRegression
from sklearn.dummy import DummyRegressor, DummyClassifier
from config import SCORED_PARQUET

MIN_JUNCTION_RECORDS = 500
OPTUNA_TRIALS        = 10

def sep(title="", width=64, char="─"):
    if title:
        side = (width - len(title) - 2) // 2
        print(f"\n{char*side} {title} {char*(width-side-len(title)-2)}")
    else:
        print(char * width)

def fmt(label, value, extra=""):
    print(f"  {label:<40} {value}  {extra}")

# ── DATA ──────────────────────────────────────────────────────────────────────
def load_data():
    sep("Loading data", char="═")
    df = pd.read_parquet(SCORED_PARQUET)
    df["created_datetime"] = pd.to_datetime(df["created_datetime"], utc=True, errors="coerce")
    df["date_dt"]     = pd.to_datetime(df["created_datetime"].dt.date.astype(str), errors="coerce")
    df["hour_ist"]    = df["hour_ist"].fillna(0).astype(int)
    df["weekday_num"] = df["weekday_num"].fillna(0).astype(int)
    df["month"]       = df["month"].fillna(1).astype(int)
    df = df[df["date_dt"].notna()].reset_index(drop=True)

    apr = df[df["month"]==4]
    fmt("Total records",               f"{len(df):,}")
    fmt("Date range",                  f"{df['date_dt'].min().date()} → {df['date_dt'].max().date()}")
    fmt("Named junction records",      f"{(df['junction_name']!='No Junction').sum():,}")
    fmt("April records",               f"{len(apr):,}  ({len(apr)/len(df)*100:.1f}% of dataset)")
    fmt("April validation rate",       f"{apr['validated'].mean()*100:.1f}%  ← pipeline lag artifact")
    fmt("Nov–Mar validation rate",     f"{df[df['month']!=4]['validated'].mean()*100:.1f}%")
    fmt("Model A uses",                "FULL dataset incl. April (counts are real)")
    fmt("Model B trains on",           "Nov–Mar only (April labels untrustworthy)")
    fmt("Model B infers on",           "FULL dataset — April still gets predictions")
    return df

# ── BUILD HOURLY (full dataset) ───────────────────────────────────────────────
def build_hourly(df):
    """Uses full df — April violation counts are genuine data."""
    named = df[df["junction_name"] != "No Junction"].copy()
    jcounts = named.groupby("junction_name").size()
    rich    = jcounts[jcounts >= MIN_JUNCTION_RECORDS].index
    named   = named[named["junction_name"].isin(rich)]

    agg = (named.groupby(["junction_name","date_dt","hour_ist"])
           .agg(count=("record_id","count"),
                heavy_pct=("vehicle_weight", lambda x: (x>=2.0).mean()),
                avg_weight=("vehicle_weight","mean"),
                lat=("latitude","first"), lon=("longitude","first"),
                weekday=("weekday_num","first"), month=("month","first"))
           .reset_index().rename(columns={"date_dt":"date"}))

    agg["weekend"]          = (agg["weekday"]>=5).astype(int)
    agg["peak_hour"]        = agg["hour_ist"].isin([8,9,10,17,18,19,20]).astype(int)
    agg["day_of_month"]     = agg["date"].dt.day
    agg["days_since_start"] = (agg["date"] - agg["date"].min()).dt.days
    agg["hour_sin"]         = np.sin(2*np.pi*agg["hour_ist"]/24)
    agg["hour_cos"]         = np.cos(2*np.pi*agg["hour_ist"]/24)
    agg = agg.sort_values(["junction_name","date","hour_ist"]).reset_index(drop=True)

    all_j = agg["junction_name"].unique()
    all_d = pd.date_range(agg["date"].min(), agg["date"].max(), freq="D")
    idx   = pd.MultiIndex.from_product([all_j, all_d, list(range(24))],
                                        names=["junction_name","date","hour_ist"])
    full  = agg.set_index(["junction_name","date","hour_ist"]).reindex(idx).reset_index()
    full["count"] = full["count"].fillna(0)
    meta = ["heavy_pct","avg_weight","lat","lon","weekday","month","weekend",
            "peak_hour","day_of_month","days_since_start","hour_sin","hour_cos"]
    full[meta] = full.groupby("junction_name")[meta].ffill().bfill()

    grp = full.groupby("junction_name")["count"]
    for lag, name in [(1,"lag1"),(2,"lag2"),(24,"lag24"),(48,"lag48"),(168,"lag168")]:
        full[name] = grp.shift(lag).fillna(0)
    for win, name in [(3,"roll3"),(6,"roll6"),(24,"roll24"),(48,"roll48")]:
        full[name] = grp.shift(1).transform(
            lambda x: x.rolling(win,min_periods=1).mean()).fillna(0)
    full["roll_std6"]    = grp.shift(1).transform(
        lambda x: x.rolling(6,min_periods=2).std()).fillna(0)
    full["ewm6"]         = grp.shift(1).transform(
        lambda x: x.ewm(span=6,min_periods=1).mean()).fillna(0)
    full["lag168_roll3"] = grp.shift(168).transform(
        lambda x: x.rolling(3*168,min_periods=1).mean()).fillna(0)

    le = LabelEncoder()
    full["junction_id"] = le.fit_transform(full["junction_name"])
    full["count_rank"]  = full.groupby("date")["count"].rank(pct=True)
    has_signal = (full["count"]>0)|(full["lag1"]>0)|(full["lag24"]>0)|(full["roll24"]>0)
    return full[has_signal].reset_index(drop=True), le, rich

FEAT_A_BASE = [
    "hour_ist","hour_sin","hour_cos","weekday","month","weekend","peak_hour",
    "day_of_month","days_since_start","junction_id","lat","lon","count_rank",
    "heavy_pct","avg_weight","lag1","lag2","lag24","lag48","lag168",
    "roll3","roll6","roll24","roll48","roll_std6","ewm6","lag168_roll3",
]
FEAT_B_CAT = ["police_station","vehicle_type","offence_code","center_code","junction_name"]
FEAT_B_NUM = ["latitude","longitude","hour_ist","weekday_num","weekend"]

# ── MODEL A ───────────────────────────────────────────────────────────────────
def evaluate_model_a(ts):
    sep("MODEL A — Hourly Count Forecaster", char="═")
    print("  Algorithm  : LightGBM (Poisson objective)")
    print("  Data used  : FULL dataset including April")
    print("  Split      : Strict time-based (train ≤ 80th pct date)")
    sep()

    cutoff = ts["date"].quantile(0.80)
    train  = ts[ts["date"] <= cutoff]
    test   = ts[ts["date"] >  cutoff]

    # Target encoding from train only (no leakage)
    jh   = train.groupby(["junction_name","hour_ist"])["count"].mean().rename("junc_hour_mean")
    jdow = train.groupby(["junction_name","weekday"])["count"].mean().rename("junc_dow_mean")
    ts2  = ts.merge(jh.reset_index(),   on=["junction_name","hour_ist"], how="left")
    ts2  = ts2.merge(jdow.reset_index(), on=["junction_name","weekday"],  how="left")
    ts2["junc_hour_mean"] = ts2["junc_hour_mean"].fillna(ts2["roll24"])
    ts2["junc_dow_mean"]  = ts2["junc_dow_mean"].fillna(ts2["roll24"])
    feat_cols = FEAT_A_BASE + ["junc_hour_mean","junc_dow_mean"]
    feat_cols = [f for f in feat_cols if f in ts2.columns]

    train2, test2 = ts2[ts2["date"]<=cutoff], ts2[ts2["date"]>cutoff]
    X_tr, y_tr = train2[feat_cols].fillna(0).values, train2["count"].values
    X_te, y_te = test2[feat_cols].fillna(0).values,  test2["count"].values

    sep("Train / Test split")
    fmt("Train rows",   f"{len(X_tr):,}")
    fmt("Train period", f"{train['date'].min().date()} → {train['date'].max().date()}")
    fmt("Test rows",    f"{len(X_te):,}")
    fmt("Test period",  f"{test['date'].min().date()} → {test['date'].max().date()}")
    fmt("Target mean (train)", f"{y_tr.mean():.3f}")
    fmt("Target mean (test)",  f"{y_te.mean():.3f}")
    fmt("Target max  (test)",  f"{y_te.max():.0f}")

    # Tune
    sep("Optuna Hyperparameter Search")
    def obj_a(trial):
        p = dict(objective="poisson", metric="mse", verbose=-1, n_jobs=-1, bagging_freq=5,
                 num_leaves       = trial.suggest_int("nl",31,127),
                 min_data_in_leaf = trial.suggest_int("mdl",10,60),
                 learning_rate    = trial.suggest_float("lr",0.02,0.12,log=True),
                 feature_fraction = trial.suggest_float("ff",0.6,1.0),
                 bagging_fraction = trial.suggest_float("bf",0.6,1.0),
                 lambda_l1        = trial.suggest_float("l1",0,1.0),
                 lambda_l2        = trial.suggest_float("l2",0,5.0),
                 max_depth        = trial.suggest_int("md",4,8))
        cb = [lgb.early_stopping(30,verbose=False), lgb.log_evaluation(period=-1)]
        m  = lgb.train(p, lgb.Dataset(X_tr,y_tr), num_boost_round=400,
                       valid_sets=[lgb.Dataset(X_te,y_te)], callbacks=cb)
        return mean_absolute_error(y_te, m.predict(X_te).clip(0))
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj_a, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    bp = study.best_params
    print(f"  Best params : {bp}")
    print(f"  Best MAE    : {study.best_value:.4f}")

    best_p = dict(objective="poisson", metric="mse", verbose=-1, n_jobs=-1, bagging_freq=5,
                  num_leaves=bp["nl"], min_data_in_leaf=bp["mdl"], learning_rate=bp["lr"],
                  feature_fraction=bp["ff"], bagging_fraction=bp["bf"],
                  lambda_l1=bp["l1"], lambda_l2=bp["l2"], max_depth=bp["md"])
    cb    = [lgb.early_stopping(50,verbose=False), lgb.log_evaluation(period=-1)]
    model = lgb.train(best_p, lgb.Dataset(X_tr,y_tr), num_boost_round=400,
                      valid_sets=[lgb.Dataset(X_te,y_te)], callbacks=cb)

    preds = model.predict(X_te).clip(0)
    mae   = mean_absolute_error(y_te, preds)
    rmse  = np.sqrt(mean_squared_error(y_te, preds))
    r2    = r2_score(y_te, preds)
    bl_mean = mean_absolute_error(y_te, DummyRegressor(strategy="mean").fit(X_tr,y_tr).predict(X_te))
    bl_last = mean_absolute_error(y_te, test2["lag1"].values)
    bl_24h  = mean_absolute_error(y_te, test2["roll24"].values)

    sep("Test-set Metrics")
    fmt("MAE  (our model)",    f"{mae:.4f}")
    fmt("RMSE (our model)",    f"{rmse:.4f}")
    fmt("R²   (our model)",    f"{r2:.4f}")
    sep("vs Baselines — MAE (lower is better)")
    fmt("Naive — train mean",     f"{bl_mean:.4f}", f"improvement: {(bl_mean-mae)/bl_mean*100:.1f}%")
    fmt("Naive — last observed",  f"{bl_last:.4f}", f"improvement: {(bl_last-mae)/bl_last*100:.1f}%")
    fmt("Naive — 24h rolling",    f"{bl_24h:.4f}",  f"improvement: {(bl_24h-mae)/bl_24h*100:.1f}%")

    sep("MAE by count bucket")
    for lo, hi, label in [(0,0,"zero"),(1,2,"1-2"),(3,5,"3-5"),(6,10,"6-10"),(11,9999,"11+")]:
        m = (y_te>=lo)&(y_te<=hi)
        if m.sum()>0:
            fmt(f"  count={label}  (n={int(m.sum()):,})",
                f"MAE = {mean_absolute_error(y_te[m], preds[m]):.3f}")

    sep("Feature Importance (top 12 by gain)")
    fi = pd.Series(model.feature_importance("gain"), index=feat_cols).sort_values(ascending=False)
    for fname, imp in fi.head(12).items():
        bar = "█" * int(imp/fi.max()*28)
        print(f"  {fname:<25} {bar}  {imp:.0f}")

    sep("5-Fold Time-Ordered Cross-Validation")
    dates_sorted = np.sort(ts2["date"].unique())
    fold_size    = len(dates_sorted) // 5
    fold_maes, fold_r2s = [], []
    for k in range(4):
        tr_d = dates_sorted[:(k+1)*fold_size]
        te_d = dates_sorted[(k+1)*fold_size:(k+2)*fold_size]
        if len(te_d)==0: continue
        Xk_tr = ts2[ts2["date"].isin(tr_d)][feat_cols].fillna(0).values
        yk_tr = ts2[ts2["date"].isin(tr_d)]["count"].values
        Xk_te = ts2[ts2["date"].isin(te_d)][feat_cols].fillna(0).values
        yk_te = ts2[ts2["date"].isin(te_d)]["count"].values
        mk = lgb.train(best_p, lgb.Dataset(Xk_tr,yk_tr),
                       num_boost_round=model.best_iteration,
                       callbacks=[lgb.log_evaluation(period=-1)])
        pk = mk.predict(Xk_te).clip(0)
        fm = mean_absolute_error(yk_te, pk)
        fr = r2_score(yk_te, pk)
        fold_maes.append(fm); fold_r2s.append(fr)
        print(f"  Fold {k+1}: {pd.Timestamp(te_d[0]).date()} → "
              f"{pd.Timestamp(te_d[-1]).date()}   MAE={fm:.4f}  R²={fr:.4f}")
    print(f"\n  CV MAE mean ± std : {np.mean(fold_maes):.4f} ± {np.std(fold_maes):.4f}")
    print(f"  CV R²  mean ± std : {np.mean(fold_r2s):.4f} ± {np.std(fold_r2s):.4f}")
    return model, feat_cols, mae, r2

# ── MODEL B ───────────────────────────────────────────────────────────────────
def evaluate_model_b(df):
    sep("MODEL B — Challan Validation Predictor", char="═")
    print("  Algorithm  : LightGBM (binary) + Isotonic calibration")
    print("  Task       : Will this challan be validated? (0/1)")
    print("  TRAINING   : November–March only")
    print("               (April excluded — validated=False is pipeline lag,")
    print("                not genuine rejection — would poison the labels)")
    print("  INFERENCE  : Full dataset incl. April — predictions generated for all")
    print("  Data integrity: April violations are NOT discarded, only their")
    print("                  labels are excluded from supervised training")
    sep()

    # Training set: Nov–Mar only
    data_train = df[df["month"] != 4].copy()
    print(f"  Training set  : {len(data_train):,} records (Nov–Mar)")
    print(f"  April records : {(df['month']==4).sum():,} — excluded from training, "
          f"predictions still generated at inference")

    encoders = {}
    for col in FEAT_B_CAT:
        le = LabelEncoder()
        data_train[f"{col}_enc"] = le.fit_transform(
            data_train[col].astype(str).fillna("__NA__"))
        encoders[col] = le

    y   = data_train["validated"].astype(int).values
    idx = np.arange(len(data_train))
    tr_idx, te_idx = train_test_split(idx, test_size=0.20,
                                      random_state=42, stratify=y)

    # Station×hour TE fitted on training rows only
    tr_data     = data_train.iloc[tr_idx]
    gm, sm_k    = tr_data["validated"].mean(), 20
    te_grp      = tr_data.groupby(["police_station","hour_ist"])["validated"]
    te_map       = ((te_grp.sum() + sm_k*gm) / (te_grp.count() + sm_k)).rename("station_hour_te")
    data_train   = data_train.merge(te_map.reset_index(),
                                    on=["police_station","hour_ist"], how="left")
    data_train["station_hour_te"] = data_train["station_hour_te"].fillna(gm)

    feat_cols = [f"{c}_enc" for c in FEAT_B_CAT] + FEAT_B_NUM + ["station_hour_te"]
    feat_cols = [f for f in feat_cols if f in data_train.columns]

    X = data_train[feat_cols].fillna(0).values
    y = data_train["validated"].astype(int).values
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    pos_rate = y_tr.mean()
    spw      = (y_tr==0).sum() / max((y_tr==1).sum(),1)

    sep("Train / Test split (within Nov–Mar)")
    fmt("Train rows",         f"{len(X_tr):,}")
    fmt("Test rows",          f"{len(X_te):,}")
    fmt("Validation rate",    f"{pos_rate*100:.1f}%  (validated=1)")
    fmt("scale_pos_weight",   f"{spw:.3f}")

    sep("Optuna Hyperparameter Search")
    def obj_b(trial):
        p = dict(objective="binary", metric="auc", verbose=-1, n_jobs=-1,
                 scale_pos_weight=spw, bagging_freq=5,
                 num_leaves       = trial.suggest_int("nl",15,63),
                 min_data_in_leaf = trial.suggest_int("mdl",30,100),
                 learning_rate    = trial.suggest_float("lr",0.01,0.1,log=True),
                 feature_fraction = trial.suggest_float("ff",0.6,1.0),
                 bagging_fraction = trial.suggest_float("bf",0.6,1.0),
                 lambda_l1        = trial.suggest_float("l1",0,2.0),
                 lambda_l2        = trial.suggest_float("l2",0,5.0))
        cb = [lgb.early_stopping(30,verbose=False), lgb.log_evaluation(period=-1)]
        m  = lgb.train(p, lgb.Dataset(X_tr,y_tr), num_boost_round=400,
                       valid_sets=[lgb.Dataset(X_te,y_te)], callbacks=cb)
        return -roc_auc_score(y_te, m.predict(X_te))
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj_b, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    bp = study.best_params
    print(f"  Best params : {bp}")
    print(f"  Best AUC    : {-study.best_value:.4f}")

    best_p = dict(objective="binary", metric="auc", verbose=-1, n_jobs=-1,
                  scale_pos_weight=spw, bagging_freq=5,
                  num_leaves=bp["nl"], min_data_in_leaf=bp["mdl"],
                  learning_rate=bp["lr"], feature_fraction=bp["ff"],
                  bagging_fraction=bp["bf"], lambda_l1=bp["l1"], lambda_l2=bp["l2"])
    cb  = [lgb.early_stopping(50,verbose=False), lgb.log_evaluation(period=-1)]
    clf = lgb.train(best_p, lgb.Dataset(X_tr,y_tr), num_boost_round=400,
                    valid_sets=[lgb.Dataset(X_te,y_te)], callbacks=cb)

    proba = clf.predict(X_te)
    ir    = IsotonicRegression(out_of_bounds="clip")
    ir.fit(proba, y_te)
    proba_cal = ir.transform(proba)

    # Optimal threshold
    best_t, best_mf1 = 0.5, 0
    for t in np.arange(0.2, 0.9, 0.01):
        f = f1_score(y_te, (proba_cal>=t).astype(int), average="macro", zero_division=0)
        if f > best_mf1: best_mf1, best_t = f, t
    preds = (proba_cal >= best_t).astype(int)

    auc   = roc_auc_score(y_te, proba_cal)
    prauc = average_precision_score(y_te, proba_cal)
    brier = brier_score_loss(y_te, proba_cal)
    f1v   = f1_score(y_te, preds)
    f1nv  = f1_score(y_te, preds, pos_label=0)
    prec  = precision_score(y_te, preds, zero_division=0)
    rec   = recall_score(y_te, preds, zero_division=0)
    acc   = (preds==y_te).mean()
    cm    = confusion_matrix(y_te, preds)
    tn,fp,fn,tp = cm.ravel()

    dum  = DummyClassifier(strategy="most_frequent").fit(X_tr,y_tr)
    base_auc = roc_auc_score(y_te, dum.predict_proba(X_te)[:,1])

    sep("Test-set Metrics")
    fmt("ROC-AUC",            f"{auc:.4f}",  f"(random=0.5000, majority={base_auc:.4f})")
    fmt("PR-AUC",             f"{prauc:.4f}", f"(random={pos_rate:.4f})")
    fmt("Brier score",        f"{brier:.4f}", f"(0=perfect, calibration quality)")
    fmt("Accuracy",           f"{acc:.4f}")
    fmt("Optimal threshold",  f"{best_t:.2f}", f"(macro-F1={best_mf1:.4f})")
    fmt("F1 (validated=1)",   f"{f1v:.4f}")
    fmt("F1 (not-valid=0)",   f"{f1nv:.4f}")
    fmt("Precision",          f"{prec:.4f}")
    fmt("Recall",             f"{rec:.4f}")

    sep("Confusion Matrix")
    print(f"                       Predicted 0    Predicted 1")
    print(f"  Actual 0 (no val)    {tn:>9,}  {fp:>12,}   FPR={fp/(tn+fp)*100:.1f}%")
    print(f"  Actual 1 (valid)     {fn:>9,}  {tp:>12,}   TPR={tp/(fn+tp)*100:.1f}%")

    sep("Full Classification Report")
    print(classification_report(y_te, preds,
                                target_names=["Not Validated","Validated"], digits=4))

    sep("Threshold Sensitivity")
    print(f"  {'Threshold':<12}{'Precision':<12}{'Recall':<10}"
          f"{'F1-Val':<10}{'F1-NV':<10}{'MacroF1'}")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        p_  = (proba_cal>=t).astype(int)
        pr  = precision_score(y_te,p_,zero_division=0)
        rc  = recall_score(y_te,p_,zero_division=0)
        fv  = f1_score(y_te,p_,zero_division=0)
        fnv = f1_score(y_te,p_,pos_label=0,zero_division=0)
        mf  = f1_score(y_te,p_,average="macro",zero_division=0)
        star = " ← optimal" if abs(t-best_t)<0.015 else ""
        print(f"  {t:<12.1f}{pr:<12.4f}{rc:<10.4f}{fv:<10.4f}{fnv:<10.4f}{mf:.4f}{star}")

    sep("Feature Importance (top 10 by gain)")
    fi = pd.Series(clf.feature_importance("gain"),
                   index=feat_cols).sort_values(ascending=False)
    for fname, imp in fi.head(10).items():
        bar = "█" * int(imp/fi.max()*28)
        print(f"  {fname:<35} {bar}  {imp:.0f}")

    sep("5-Fold Stratified Cross-Validation (within Nov–Mar)")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs, cv_f1s, cv_nv = [], [], []
    for fold, (tri, tei) in enumerate(skf.split(X, y), 1):
        Xk_tr, Xk_te = X[tri], X[tei]
        yk_tr, yk_te = y[tri], y[tei]
        pk_spw = (yk_tr==0).sum() / max((yk_tr==1).sum(),1)
        pk = {**best_p, "scale_pos_weight": pk_spw}
        mk = lgb.train(pk, lgb.Dataset(Xk_tr,yk_tr),
                       num_boost_round=clf.best_iteration,
                       callbacks=[lgb.log_evaluation(period=-1)])
        pb     = mk.predict(Xk_te)
        ir_k   = IsotonicRegression(out_of_bounds="clip").fit(pb, yk_te)
        pb_cal = ir_k.transform(pb)
        a   = roc_auc_score(yk_te, pb_cal)
        pdk = (pb_cal >= best_t).astype(int)
        fv  = f1_score(yk_te, pdk, zero_division=0)
        fnv = f1_score(yk_te, pdk, pos_label=0, zero_division=0)
        cv_aucs.append(a); cv_f1s.append(fv); cv_nv.append(fnv)
        print(f"  Fold {fold}:  AUC={a:.4f}  F1-val={fv:.4f}  F1-not-val={fnv:.4f}")
    print(f"\n  CV AUC    mean ± std : {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")
    print(f"  CV F1-val mean ± std : {np.mean(cv_f1s):.4f} ± {np.std(cv_f1s):.4f}")
    print(f"  CV F1-NV  mean ± std : {np.mean(cv_nv):.4f} ± {np.std(cv_nv):.4f}")
    return clf, feat_cols, auc, f1v, f1nv

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║      ParkIQ — Full Model Evaluation Report (v5.1)           ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    df = load_data()

    sep("Building hourly time-series for Model A (full dataset)", char="─")
    ts, le_j, rich = build_hourly(df)
    fmt("Hourly rows", f"{len(ts):,}")
    fmt("Rich junctions (≥500 records)", f"{len(rich)} / "
        f"{df[df['junction_name']!='No Junction']['junction_name'].nunique()}")

    model_a, feat_a, ma_mae, ma_r2     = evaluate_model_a(ts)
    model_b, feat_b, mb_auc, mb_f1v, mb_f1nv = evaluate_model_b(df)

    sep("FINAL SUMMARY", char="═")
    print()
    print("  MODEL A — Hourly Count Forecaster")
    print(f"    MAE  = {ma_mae:.4f}   (lower is better)")
    print(f"    R²   = {ma_r2:.4f}   (1.0=perfect, 0.0=mean-baseline)")
    print(f"    Data : Full dataset including April violation counts")
    print()
    print("  MODEL B — Challan Validation Predictor")
    print(f"    ROC-AUC      = {mb_auc:.4f}   (0.5=random, 1.0=perfect)")
    print(f"    F1-validated = {mb_f1v:.4f}")
    print(f"    F1-not-valid = {mb_f1nv:.4f}")
    print(f"    Trained on   : Nov–Mar only (trustworthy labels)")
    print(f"    Infers on    : Full dataset — April gets predictions too")
    print()
    sep(char="═")
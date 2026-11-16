"""
T20 Pressure Reliability Index (PRI) - Reproducible Pipeline
IPL + T20I (22 Jan 2022 to 1 Nov 2026) | 766 Matches | 183,167 Deliveries
Exact Section 4 reproduction hyperparameters & Empirical-Bayes rating logic.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from xgboost import XGBClassifier

XGB_PARAMS = dict(
    n_estimators=350,
    max_depth=4,
    learning_rate=0.045,
    subsample=0.88,
    colsample_bytree=0.90,
    min_child_weight=30,
    reg_lambda=5.0,
    random_state=2026,
    eval_metric="logloss",
    n_jobs=-1,
)

INNINGS1_FEATURES = [
    "format_code", "season", "phase_code", "score", "wickets_lost",
    "balls_remaining", "current_rr", "recent_runs_12", "recent_wkts_12", "elo_diff"
]
INNINGS2_FEATURES = INNINGS1_FEATURES + [
    "target", "runs_required", "required_rr", "rr_gap"
]

def fit_oof_win_prob(df_inn: pd.DataFrame, features: list[str], name: str) -> pd.Series:
    gkf = GroupKFold(n_splits=5)
    oof_wp = pd.Series(index=df_inn.index, dtype=float)
    for tr_idx, va_idx in gkf.split(df_inn[features], df_inn["batting_team_won"], df_inn["match_id"]):
        model = XGBClassifier(**XGB_PARAMS)
        model.fit(df_inn.iloc[tr_idx][features], df_inn.iloc[tr_idx]["batting_team_won"])
        oof_wp.iloc[va_idx] = model.predict_proba(df_inn.iloc[va_idx][features])[:, 1]
    print(f"[{name}] AUC={roc_auc_score(df_inn['batting_team_won'], oof_wp):.3f} | "
          f"Brier={brier_score_loss(df_inn['batting_team_won'], oof_wp):.3f} | "
          f"LogLoss={log_loss(df_inn['batting_team_won'], oof_wp):.3f}")
    return oof_wp

def compute_pri_rankings(
    deliveries: pd.DataFrame,
    role: str,
    pool_min_balls: int = 30,
    pool_min_matches: int = 3,
    top_min_balls: int = 75,
    top_min_matches: int = 8,
    tau2_floor: float = 1.0,
    w_perf: float = 0.75,
    w_lift: float = 0.25,
    export_pool_csv: str | None = None,
) -> pd.DataFrame:
    wpa_col = "wpa_batter" if role == "batter" else "wpa_bowler"
    df = deliveries.dropna(subset=[wpa_col]).copy()
    df["wpa_100"] = df[wpa_col] * 100.0
    df["is_pressure"] = df["li"] >= 2.0

    hi = df[df["is_pressure"]].groupby("player").agg(
        pressure_balls=("wpa_100", "count"),
        matches=("match_id", "nunique"),
        raw_rate=("wpa_100", "mean"),
        s2=("wpa_100", lambda x: np.var(x, ddof=1) if len(x) > 1 else 0.0),
    )
    lo = df[~df["is_pressure"]].groupby("player").agg(
        low_pressure_rate=("wpa_100", "mean")
    )
    pool = hi.join(lo, how="inner")
    pool = pool[(pool["pressure_balls"] >= pool_min_balls) & (pool["matches"] >= pool_min_matches)].copy()

    if export_pool_csv:
        pool.reset_index().assign(role=role)[
            ["player", "role", "matches", "pressure_balls", "raw_rate", "low_pressure_rate", "s2"]
        ].to_csv(export_pool_csv, index=False)

    mu = pool["raw_rate"].mean()
    noise_mean = (pool["s2"] / pool["pressure_balls"]).mean()
    tau2 = max(tau2_floor, pool["raw_rate"].var(ddof=1) - noise_mean)

    pool["lambda"] = tau2 / (tau2 + (pool["s2"] / pool["pressure_balls"]))
    pool["shrunken_rate"] = mu + pool["lambda"] * (pool["raw_rate"] - mu)
    pool["pressure_lift"] = pool["shrunken_rate"] - pool["low_pressure_rate"]

    pool["z_shrunken"] = (pool["shrunken_rate"] - pool["shrunken_rate"].mean()) / pool["shrunken_rate"].std(ddof=1)
    pool["z_lift"] = (pool["pressure_lift"] - pool["pressure_lift"].mean()) / pool["pressure_lift"].std(ddof=1)
    pool["PRI"] = 50.0 + 10.0 * (w_perf * pool["z_shrunken"] + w_lift * pool["z_lift"])

    top10 = pool[(pool["pressure_balls"] >= top_min_balls) & (pool["matches"] >= top_min_matches)]
    return top10.sort_values("PRI", ascending=False).head(10).reset_index()

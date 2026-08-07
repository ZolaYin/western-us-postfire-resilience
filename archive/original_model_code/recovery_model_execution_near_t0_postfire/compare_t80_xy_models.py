from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from esda.moran import Moran
from libpysal.weights import KNN
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from statsmodels.miscmodels.ordinal_model import OrderedModel
from xgboost import XGBClassifier, XGBRegressor


BASE_DIR = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/"
    "recovery_model_execution_near_t0_postfire"
)
INPUT_PATH = BASE_DIR / "MGWR_ready_table_t80_iri_stability_near_t0_postfire.parquet"
OLS_PREDICTORS_PATH = BASE_DIR / "predictors_ols_recovery_baseline.txt"
TREE_PREDICTORS_PATH = BASE_DIR / "predictors_rf_xgb_recovery_baseline.txt"
OUT_DIR = BASE_DIR / "t80_xy_compare_2026-03-31"
RANDOM_STATE = 42
MORAN_K = 8
CLASSES = list(range(2, 11))


def read_predictors(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def add_xy(predictors: list[str], use_xy: bool) -> list[str]:
    out = list(predictors)
    if use_xy:
        for col in ("x", "y"):
            if col not in out:
                out.append(col)
    return out


def compute_moran(coords_df: pd.DataFrame, residuals: np.ndarray) -> dict:
    weights = KNN.from_array(coords_df[["x", "y"]].to_numpy(), k=MORAN_K)
    weights.transform = "R"
    moran = Moran(residuals.astype(float), weights, permutations=0)
    return {
        "k": MORAN_K,
        "n_obs": int(len(coords_df)),
        "moran_i": float(moran.I),
        "z_norm": float(moran.z_norm),
        "p_norm": float(moran.p_norm),
    }


def standardize_linear_importance(work: pd.DataFrame, response: str, predictors: list[str], model) -> pd.DataFrame:
    y_std = work[response].std(ddof=0)
    rows = []
    for predictor in predictors:
        x_std = work[predictor].std(ddof=0)
        std_coef = np.nan if x_std == 0 or y_std == 0 else model.params[predictor] * x_std / y_std
        rows.append(
            {
                "term": predictor,
                "coef": float(model.params[predictor]),
                "pvalue": float(model.pvalues[predictor]),
                "abs_t": float(abs(model.tvalues[predictor])),
                "std_coef": float(std_coef),
                "abs_std_coef": float(abs(std_coef)),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_std_coef", ascending=False)


def ordered_probs_to_expected_year(probs: np.ndarray) -> np.ndarray:
    return probs @ np.array(CLASSES, dtype=float)


def clip_years(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 2.0, 10.0)


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def unique_cols(cols: list[str]) -> list[str]:
    return list(dict.fromkeys(cols))


def run_continuous_ols(df: pd.DataFrame, use_xy: bool) -> dict:
    predictors = add_xy(read_predictors(OLS_PREDICTORS_PATH), use_xy)
    cols = unique_cols(["T80_revised"] + predictors + ["x", "y"])
    work = df[cols].copy()
    work = work.replace([np.inf, -np.inf], np.nan).dropna().copy()
    X = sm.add_constant(work[predictors], has_constant="add")
    model = sm.OLS(work["T80_revised"], X).fit()
    residuals = model.resid.to_numpy()

    run_name = "ols_t80_continuous_plusxy" if use_xy else "ols_t80_continuous_noxy"
    run_dir = OUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{run_name}_summary.txt").write_text(model.summary().as_text())
    standardize_linear_importance(work, "T80_revised", predictors, model).to_csv(
        run_dir / f"{run_name}_importance.csv", index=False
    )
    metrics = {
        "scheme": "continuous",
        "model_family": "ols",
        "use_xy": use_xy,
        "n_rows_used": int(len(work)),
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "rmse_full_fit": float(np.sqrt(mean_squared_error(work["T80_revised"], model.fittedvalues))),
        "full_fit_residual_moran": compute_moran(work[["x", "y"]], residuals),
        "predictors": predictors,
    }
    save_json(run_dir / f"{run_name}_metrics.json", metrics)
    return {
        "scheme": "continuous",
        "model_family": "ols",
        "use_xy": use_xy,
        "score_r2": metrics["r2"],
        "score_rmse": metrics["rmse_full_fit"],
        "moran_i": metrics["full_fit_residual_moran"]["moran_i"],
    }


def run_continuous_rf(df: pd.DataFrame, use_xy: bool) -> dict:
    predictors = add_xy(read_predictors(TREE_PREDICTORS_PATH), use_xy)
    cols = unique_cols(["T80_revised"] + predictors + ["x", "y"])
    work = df[cols].copy()
    work = work.replace([np.inf, -np.inf], np.nan).dropna().copy()
    X_train, X_test, y_train, y_test = train_test_split(
        work[predictors], work["T80_revised"], test_size=0.2, random_state=RANDOM_STATE
    )
    model = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    pred_test = model.predict(X_test)

    full_model = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    full_model.fit(work[predictors], work["T80_revised"])
    full_pred = full_model.predict(work[predictors])
    residuals = work["T80_revised"].to_numpy() - full_pred

    run_name = "rf_t80_continuous_plusxy" if use_xy else "rf_t80_continuous_noxy"
    run_dir = OUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"predictor": predictors, "importance": full_model.feature_importances_}).sort_values(
        "importance", ascending=False
    ).to_csv(run_dir / f"{run_name}_importance.csv", index=False)
    metrics = {
        "scheme": "continuous",
        "model_family": "rf",
        "use_xy": use_xy,
        "n_rows_used": int(len(work)),
        "test_r2": float(r2_score(y_test, pred_test)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "full_fit_residual_moran": compute_moran(work[["x", "y"]], residuals),
        "predictors": predictors,
    }
    save_json(run_dir / f"{run_name}_metrics.json", metrics)
    return {
        "scheme": "continuous",
        "model_family": "rf",
        "use_xy": use_xy,
        "score_r2": metrics["test_r2"],
        "score_rmse": metrics["test_rmse"],
        "moran_i": metrics["full_fit_residual_moran"]["moran_i"],
    }


def run_continuous_xgb(df: pd.DataFrame, use_xy: bool) -> dict:
    predictors = add_xy(read_predictors(TREE_PREDICTORS_PATH), use_xy)
    cols = unique_cols(["T80_revised"] + predictors + ["x", "y"])
    work = df[cols].copy()
    work = work.replace([np.inf, -np.inf], np.nan).dropna().copy()
    X_train, X_test, y_train, y_test = train_test_split(
        work[predictors], work["T80_revised"], test_size=0.2, random_state=RANDOM_STATE
    )
    model = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    pred_test = model.predict(X_test)

    full_model = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    full_model.fit(work[predictors], work["T80_revised"])
    full_pred = full_model.predict(work[predictors])
    residuals = work["T80_revised"].to_numpy() - full_pred

    run_name = "xgb_t80_continuous_plusxy" if use_xy else "xgb_t80_continuous_noxy"
    run_dir = OUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    score = full_model.get_booster().get_score(importance_type="gain")
    pd.DataFrame([{"predictor": k, "importance_gain": v} for k, v in score.items()]).sort_values(
        "importance_gain", ascending=False
    ).to_csv(run_dir / f"{run_name}_importance.csv", index=False)
    metrics = {
        "scheme": "continuous",
        "model_family": "xgb",
        "use_xy": use_xy,
        "n_rows_used": int(len(work)),
        "test_r2": float(r2_score(y_test, pred_test)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "full_fit_residual_moran": compute_moran(work[["x", "y"]], residuals),
        "predictors": predictors,
    }
    save_json(run_dir / f"{run_name}_metrics.json", metrics)
    return {
        "scheme": "continuous",
        "model_family": "xgb",
        "use_xy": use_xy,
        "score_r2": metrics["test_r2"],
        "score_rmse": metrics["test_rmse"],
        "moran_i": metrics["full_fit_residual_moran"]["moran_i"],
    }


def run_ordinal_ols(df: pd.DataFrame, use_xy: bool) -> dict:
    predictors = add_xy(read_predictors(OLS_PREDICTORS_PATH), use_xy)
    cols = unique_cols(["T80_revised"] + predictors + ["x", "y"])
    work = df[cols].copy()
    work = work.replace([np.inf, -np.inf], np.nan).dropna().copy()
    train_idx, test_idx = train_test_split(
        work.index,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=work["T80_revised"],
    )
    train = work.loc[train_idx].copy()
    test = work.loc[test_idx].copy()

    train_cat = pd.Categorical(train["T80_revised"], categories=CLASSES, ordered=True)
    model = OrderedModel(train_cat, train[predictors], distr="logit")
    result = model.fit(method="bfgs", disp=False)
    probs_test = result.model.predict(result.params, exog=test[predictors])
    pred_test = np.array(CLASSES)[np.argmax(probs_test, axis=1)]
    expected_test = ordered_probs_to_expected_year(probs_test)

    full_cat = pd.Categorical(work["T80_revised"], categories=CLASSES, ordered=True)
    full_model = OrderedModel(full_cat, work[predictors], distr="logit")
    full_result = full_model.fit(method="bfgs", disp=False)
    probs_full = full_result.model.predict(full_result.params, exog=work[predictors])
    expected_full = ordered_probs_to_expected_year(probs_full)
    residuals = work["T80_revised"].to_numpy() - expected_full

    run_name = "ols_t80_ordinal_plusxy" if use_xy else "ols_t80_ordinal_noxy"
    run_dir = OUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{run_name}_summary.txt").write_text(full_result.summary().as_text())
    coef_rows = []
    for term in predictors:
        coef_rows.append(
            {
                "term": term,
                "coef": float(full_result.params[term]),
                "pvalue": float(full_result.pvalues[term]),
                "abs_z": float(abs(full_result.tvalues[term])),
            }
        )
    pd.DataFrame(coef_rows).sort_values("abs_z", ascending=False).to_csv(
        run_dir / f"{run_name}_importance.csv", index=False
    )
    metrics = {
        "scheme": "ordinal",
        "model_family": "ols",
        "use_xy": use_xy,
        "n_rows_used": int(len(work)),
        "test_accuracy": float(accuracy_score(test["T80_revised"], pred_test)),
        "test_macro_f1": float(f1_score(test["T80_revised"], pred_test, average="macro")),
        "test_expected_r2": float(r2_score(test["T80_revised"], expected_test)),
        "test_expected_rmse": float(np.sqrt(mean_squared_error(test["T80_revised"], expected_test))),
        "full_fit_residual_moran": compute_moran(work[["x", "y"]], residuals),
        "predictors": predictors,
    }
    save_json(run_dir / f"{run_name}_metrics.json", metrics)
    return {
        "scheme": "ordinal",
        "model_family": "ols",
        "use_xy": use_xy,
        "score_r2": metrics["test_expected_r2"],
        "score_rmse": metrics["test_expected_rmse"],
        "moran_i": metrics["full_fit_residual_moran"]["moran_i"],
        "accuracy": metrics["test_accuracy"],
    }


def run_ordinal_rf(df: pd.DataFrame, use_xy: bool) -> dict:
    predictors = add_xy(read_predictors(TREE_PREDICTORS_PATH), use_xy)
    cols = unique_cols(["T80_revised"] + predictors + ["x", "y"])
    work = df[cols].copy()
    work = work.replace([np.inf, -np.inf], np.nan).dropna().copy()
    train_idx, test_idx = train_test_split(
        work.index,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=work["T80_revised"],
    )
    train = work.loc[train_idx].copy()
    test = work.loc[test_idx].copy()

    model = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(train[predictors], train["T80_revised"])
    probs_test = model.predict_proba(test[predictors])
    pred_test = model.predict(test[predictors])
    expected_test = ordered_probs_to_expected_year(probs_test)

    full_model = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    full_model.fit(work[predictors], work["T80_revised"])
    probs_full = full_model.predict_proba(work[predictors])
    expected_full = ordered_probs_to_expected_year(probs_full)
    residuals = work["T80_revised"].to_numpy() - expected_full

    run_name = "rf_t80_ordinal_plusxy" if use_xy else "rf_t80_ordinal_noxy"
    run_dir = OUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"predictor": predictors, "importance": full_model.feature_importances_}).sort_values(
        "importance", ascending=False
    ).to_csv(run_dir / f"{run_name}_importance.csv", index=False)
    metrics = {
        "scheme": "ordinal",
        "model_family": "rf",
        "use_xy": use_xy,
        "n_rows_used": int(len(work)),
        "test_accuracy": float(accuracy_score(test["T80_revised"], pred_test)),
        "test_macro_f1": float(f1_score(test["T80_revised"], pred_test, average="macro")),
        "test_expected_r2": float(r2_score(test["T80_revised"], expected_test)),
        "test_expected_rmse": float(np.sqrt(mean_squared_error(test["T80_revised"], expected_test))),
        "full_fit_residual_moran": compute_moran(work[["x", "y"]], residuals),
        "predictors": predictors,
    }
    save_json(run_dir / f"{run_name}_metrics.json", metrics)
    return {
        "scheme": "ordinal",
        "model_family": "rf",
        "use_xy": use_xy,
        "score_r2": metrics["test_expected_r2"],
        "score_rmse": metrics["test_expected_rmse"],
        "moran_i": metrics["full_fit_residual_moran"]["moran_i"],
        "accuracy": metrics["test_accuracy"],
    }


def run_ordinal_xgb(df: pd.DataFrame, use_xy: bool) -> dict:
    predictors = add_xy(read_predictors(TREE_PREDICTORS_PATH), use_xy)
    cols = unique_cols(["T80_revised"] + predictors + ["x", "y"])
    work = df[cols].copy()
    work = work.replace([np.inf, -np.inf], np.nan).dropna().copy()
    train_idx, test_idx = train_test_split(
        work.index,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=work["T80_revised"],
    )
    train = work.loc[train_idx].copy()
    test = work.loc[test_idx].copy()

    label_map = {label: idx for idx, label in enumerate(CLASSES)}
    inv_label_map = {idx: label for label, idx in label_map.items()}
    y_train = train["T80_revised"].map(label_map).to_numpy()

    model = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=len(CLASSES),
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(train[predictors], y_train)
    probs_test = model.predict_proba(test[predictors])
    pred_test = np.vectorize(inv_label_map.get)(np.argmax(probs_test, axis=1))
    expected_test = ordered_probs_to_expected_year(probs_test)

    full_y = work["T80_revised"].map(label_map).to_numpy()
    full_model = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=len(CLASSES),
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    full_model.fit(work[predictors], full_y)
    probs_full = full_model.predict_proba(work[predictors])
    expected_full = ordered_probs_to_expected_year(probs_full)
    residuals = work["T80_revised"].to_numpy() - expected_full

    run_name = "xgb_t80_ordinal_plusxy" if use_xy else "xgb_t80_ordinal_noxy"
    run_dir = OUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    score = full_model.get_booster().get_score(importance_type="gain")
    pd.DataFrame([{"predictor": k, "importance_gain": v} for k, v in score.items()]).sort_values(
        "importance_gain", ascending=False
    ).to_csv(run_dir / f"{run_name}_importance.csv", index=False)
    metrics = {
        "scheme": "ordinal",
        "model_family": "xgb",
        "use_xy": use_xy,
        "n_rows_used": int(len(work)),
        "test_accuracy": float(accuracy_score(test["T80_revised"], pred_test)),
        "test_macro_f1": float(f1_score(test["T80_revised"], pred_test, average="macro")),
        "test_expected_r2": float(r2_score(test["T80_revised"], expected_test)),
        "test_expected_rmse": float(np.sqrt(mean_squared_error(test["T80_revised"], expected_test))),
        "full_fit_residual_moran": compute_moran(work[["x", "y"]], residuals),
        "predictors": predictors,
    }
    save_json(run_dir / f"{run_name}_metrics.json", metrics)
    return {
        "scheme": "ordinal",
        "model_family": "xgb",
        "use_xy": use_xy,
        "score_r2": metrics["test_expected_r2"],
        "score_rmse": metrics["test_expected_rmse"],
        "moran_i": metrics["full_fit_residual_moran"]["moran_i"],
        "accuracy": metrics["test_accuracy"],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(INPUT_PATH)
    rows = []
    for use_xy in (False, True):
        rows.append(run_continuous_ols(df, use_xy))
        rows.append(run_continuous_rf(df, use_xy))
        rows.append(run_continuous_xgb(df, use_xy))
        rows.append(run_ordinal_ols(df, use_xy))
        rows.append(run_ordinal_rf(df, use_xy))
        rows.append(run_ordinal_xgb(df, use_xy))

    summary = pd.DataFrame(rows).sort_values(["scheme", "model_family", "use_xy"])
    summary.to_csv(OUT_DIR / "t80_xy_compare_summary.csv", index=False)

    lines = ["T80 xy comparison summary"]
    for _, row in summary.iterrows():
        label = "plusxy" if bool(row["use_xy"]) else "noxy"
        acc = row["accuracy"] if "accuracy" in row and not pd.isna(row["accuracy"]) else None
        if acc is None:
            lines.append(
                f"{row['scheme']} | {row['model_family']} | {label}: "
                f"r2={row['score_r2']:.4f}, rmse={row['score_rmse']:.4f}, moran_i={row['moran_i']:.4f}"
            )
        else:
            lines.append(
                f"{row['scheme']} | {row['model_family']} | {label}: "
                f"expected_r2={row['score_r2']:.4f}, rmse={row['score_rmse']:.4f}, "
                f"accuracy={acc:.4f}, moran_i={row['moran_i']:.4f}"
            )
    (OUT_DIR / "t80_xy_compare_summary.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()

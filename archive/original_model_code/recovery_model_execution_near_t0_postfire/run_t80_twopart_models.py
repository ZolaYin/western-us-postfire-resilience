from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor


BASE_DIR = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/"
    "recovery_model_execution_near_t0_postfire"
)
INPUT_PATH = BASE_DIR / "MGWR_ready_table_t80_iri_stability_near_t0_postfire.parquet"
OLS_PREDICTORS_PATH = BASE_DIR / "predictors_ols_recovery_baseline.txt"
TREE_PREDICTORS_PATH = BASE_DIR / "predictors_rf_xgb_recovery_baseline.txt"
OUT_DIR = BASE_DIR / "t80_twopart_2026-03-31"
RANDOM_STATE = 42


def read_predictors(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def clip_years(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 2.0, 9.0)


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def standardize_ordered_importance(work: pd.DataFrame, response: str, predictors: list[str], model) -> pd.DataFrame:
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


def run_ols(df: pd.DataFrame) -> dict:
    predictors = read_predictors(OLS_PREDICTORS_PATH)
    work = df[["T80_revised", "T80_reached"] + predictors].replace([np.inf, -np.inf], np.nan).dropna().copy()

    train_idx, test_idx = train_test_split(
        work.index,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=work["T80_reached"],
    )
    train = work.loc[train_idx].copy()
    test = work.loc[test_idx].copy()

    X_train_cls = sm.add_constant(train[predictors], has_constant="add")
    X_test_cls = sm.add_constant(test[predictors], has_constant="add")
    cls_model = sm.GLM(train["T80_reached"], X_train_cls, family=sm.families.Binomial()).fit()
    prob_test = cls_model.predict(X_test_cls)
    cls_pred = (prob_test >= 0.5).astype(int)

    reached_train = train.loc[train["T80_reached"] == 1].copy()
    reached_test = test.loc[test["T80_reached"] == 1].copy()
    X_train_reg = sm.add_constant(reached_train[predictors], has_constant="add")
    X_test_reg = sm.add_constant(reached_test[predictors], has_constant="add")
    reg_model = sm.OLS(reached_train["T80_revised"], X_train_reg).fit()
    reg_pred_reached = clip_years(reg_model.predict(X_test_reg).to_numpy())

    expected_year = (1.0 - prob_test.to_numpy()) * 10.0 + prob_test.to_numpy() * clip_years(
        reg_model.predict(sm.add_constant(test[predictors], has_constant="add")).to_numpy()
    )

    model_dir = OUT_DIR / "ols_t80_twopart_2026-03-31"
    model_dir.mkdir(parents=True, exist_ok=True)
    cls_prefix = "ols_twopart_part1_reached"
    reg_prefix = "ols_twopart_part2_years_given_reached"
    (model_dir / f"{cls_prefix}_summary.txt").write_text(cls_model.summary().as_text())
    (model_dir / f"{reg_prefix}_summary.txt").write_text(reg_model.summary().as_text())
    standardize_ordered_importance(train, "T80_reached", predictors, cls_model).to_csv(
        model_dir / f"{cls_prefix}_standardized_importance.csv", index=False
    )
    standardize_ordered_importance(reached_train, "T80_revised", predictors, reg_model).to_csv(
        model_dir / f"{reg_prefix}_standardized_importance.csv", index=False
    )

    part1_metrics = {
        "model_family": "ols",
        "scheme": "two_part",
        "part": "reached_classifier",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "test_accuracy": float(accuracy_score(test["T80_reached"], cls_pred)),
        "test_roc_auc": float(roc_auc_score(test["T80_reached"], prob_test)),
    }
    part2_metrics = {
        "model_family": "ols",
        "scheme": "two_part",
        "part": "years_given_reached_regression",
        "n_train": int(len(reached_train)),
        "n_test": int(len(reached_test)),
        "test_r2": float(r2_score(reached_test["T80_revised"], reg_pred_reached)),
        "test_rmse": float(np.sqrt(mean_squared_error(reached_test["T80_revised"], reg_pred_reached))),
    }
    combined_metrics = {
        "model_family": "ols",
        "scheme": "two_part",
        "part": "combined_expected_t80",
        "n_test": int(len(test)),
        "test_r2": float(r2_score(test["T80_revised"], expected_year)),
        "test_rmse": float(np.sqrt(mean_squared_error(test["T80_revised"], expected_year))),
    }
    save_json(model_dir / f"{cls_prefix}_metrics.json", part1_metrics)
    save_json(model_dir / f"{reg_prefix}_metrics.json", part2_metrics)
    save_json(model_dir / "ols_twopart_combined_metrics.json", combined_metrics)
    return {
        "model_family": "ols",
        "part1_accuracy": part1_metrics["test_accuracy"],
        "part1_roc_auc": part1_metrics["test_roc_auc"],
        "part2_r2": part2_metrics["test_r2"],
        "part2_rmse": part2_metrics["test_rmse"],
        "combined_r2": combined_metrics["test_r2"],
        "combined_rmse": combined_metrics["test_rmse"],
    }


def run_rf(df: pd.DataFrame) -> dict:
    predictors = read_predictors(TREE_PREDICTORS_PATH)
    work = df[["T80_revised", "T80_reached"] + predictors].replace([np.inf, -np.inf], np.nan).dropna().copy()

    train_idx, test_idx = train_test_split(
        work.index,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=work["T80_reached"],
    )
    train = work.loc[train_idx].copy()
    test = work.loc[test_idx].copy()

    cls_model = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    cls_model.fit(train[predictors], train["T80_reached"])
    prob_test = cls_model.predict_proba(test[predictors])[:, 1]
    cls_pred = cls_model.predict(test[predictors])

    reached_train = train.loc[train["T80_reached"] == 1].copy()
    reached_test = test.loc[test["T80_reached"] == 1].copy()
    reg_model = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    reg_model.fit(reached_train[predictors], reached_train["T80_revised"])
    reg_pred_reached = clip_years(reg_model.predict(reached_test[predictors]))
    expected_year = (1.0 - prob_test) * 10.0 + prob_test * clip_years(reg_model.predict(test[predictors]))

    model_dir = OUT_DIR / "rf_t80_twopart_2026-03-31"
    model_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"predictor": predictors, "importance": cls_model.feature_importances_}).sort_values(
        "importance", ascending=False
    ).to_csv(model_dir / "rf_twopart_part1_reached_feature_importance.csv", index=False)
    pd.DataFrame({"predictor": predictors, "importance": reg_model.feature_importances_}).sort_values(
        "importance", ascending=False
    ).to_csv(model_dir / "rf_twopart_part2_years_given_reached_feature_importance.csv", index=False)

    part1_metrics = {
        "model_family": "rf",
        "scheme": "two_part",
        "part": "reached_classifier",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "test_accuracy": float(accuracy_score(test["T80_reached"], cls_pred)),
        "test_roc_auc": float(roc_auc_score(test["T80_reached"], prob_test)),
    }
    part2_metrics = {
        "model_family": "rf",
        "scheme": "two_part",
        "part": "years_given_reached_regression",
        "n_train": int(len(reached_train)),
        "n_test": int(len(reached_test)),
        "test_r2": float(r2_score(reached_test["T80_revised"], reg_pred_reached)),
        "test_rmse": float(np.sqrt(mean_squared_error(reached_test["T80_revised"], reg_pred_reached))),
    }
    combined_metrics = {
        "model_family": "rf",
        "scheme": "two_part",
        "part": "combined_expected_t80",
        "n_test": int(len(test)),
        "test_r2": float(r2_score(test["T80_revised"], expected_year)),
        "test_rmse": float(np.sqrt(mean_squared_error(test["T80_revised"], expected_year))),
    }
    save_json(model_dir / "rf_twopart_part1_reached_metrics.json", part1_metrics)
    save_json(model_dir / "rf_twopart_part2_years_given_reached_metrics.json", part2_metrics)
    save_json(model_dir / "rf_twopart_combined_metrics.json", combined_metrics)
    return {
        "model_family": "rf",
        "part1_accuracy": part1_metrics["test_accuracy"],
        "part1_roc_auc": part1_metrics["test_roc_auc"],
        "part2_r2": part2_metrics["test_r2"],
        "part2_rmse": part2_metrics["test_rmse"],
        "combined_r2": combined_metrics["test_r2"],
        "combined_rmse": combined_metrics["test_rmse"],
    }


def run_xgb(df: pd.DataFrame) -> dict:
    predictors = read_predictors(TREE_PREDICTORS_PATH)
    work = df[["T80_revised", "T80_reached"] + predictors].replace([np.inf, -np.inf], np.nan).dropna().copy()

    train_idx, test_idx = train_test_split(
        work.index,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=work["T80_reached"],
    )
    train = work.loc[train_idx].copy()
    test = work.loc[test_idx].copy()

    cls_model = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    cls_model.fit(train[predictors], train["T80_reached"])
    prob_test = cls_model.predict_proba(test[predictors])[:, 1]
    cls_pred = (prob_test >= 0.5).astype(int)

    reached_train = train.loc[train["T80_reached"] == 1].copy()
    reached_test = test.loc[test["T80_reached"] == 1].copy()
    reg_model = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    reg_model.fit(reached_train[predictors], reached_train["T80_revised"])
    reg_pred_reached = clip_years(reg_model.predict(reached_test[predictors]))
    expected_year = (1.0 - prob_test) * 10.0 + prob_test * clip_years(reg_model.predict(test[predictors]))

    model_dir = OUT_DIR / "xgb_t80_twopart_2026-03-31"
    model_dir.mkdir(parents=True, exist_ok=True)
    cls_score = cls_model.get_booster().get_score(importance_type="gain")
    reg_score = reg_model.get_booster().get_score(importance_type="gain")
    pd.DataFrame([{"predictor": k, "importance_gain": v} for k, v in cls_score.items()]).sort_values(
        "importance_gain", ascending=False
    ).to_csv(model_dir / "xgb_twopart_part1_reached_feature_importance.csv", index=False)
    pd.DataFrame([{"predictor": k, "importance_gain": v} for k, v in reg_score.items()]).sort_values(
        "importance_gain", ascending=False
    ).to_csv(model_dir / "xgb_twopart_part2_years_given_reached_feature_importance.csv", index=False)

    part1_metrics = {
        "model_family": "xgb",
        "scheme": "two_part",
        "part": "reached_classifier",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "test_accuracy": float(accuracy_score(test["T80_reached"], cls_pred)),
        "test_roc_auc": float(roc_auc_score(test["T80_reached"], prob_test)),
    }
    part2_metrics = {
        "model_family": "xgb",
        "scheme": "two_part",
        "part": "years_given_reached_regression",
        "n_train": int(len(reached_train)),
        "n_test": int(len(reached_test)),
        "test_r2": float(r2_score(reached_test["T80_revised"], reg_pred_reached)),
        "test_rmse": float(np.sqrt(mean_squared_error(reached_test["T80_revised"], reg_pred_reached))),
    }
    combined_metrics = {
        "model_family": "xgb",
        "scheme": "two_part",
        "part": "combined_expected_t80",
        "n_test": int(len(test)),
        "test_r2": float(r2_score(test["T80_revised"], expected_year)),
        "test_rmse": float(np.sqrt(mean_squared_error(test["T80_revised"], expected_year))),
    }
    save_json(model_dir / "xgb_twopart_part1_reached_metrics.json", part1_metrics)
    save_json(model_dir / "xgb_twopart_part2_years_given_reached_metrics.json", part2_metrics)
    save_json(model_dir / "xgb_twopart_combined_metrics.json", combined_metrics)
    return {
        "model_family": "xgb",
        "part1_accuracy": part1_metrics["test_accuracy"],
        "part1_roc_auc": part1_metrics["test_roc_auc"],
        "part2_r2": part2_metrics["test_r2"],
        "part2_rmse": part2_metrics["test_rmse"],
        "combined_r2": combined_metrics["test_r2"],
        "combined_rmse": combined_metrics["test_rmse"],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(INPUT_PATH)
    summary = pd.DataFrame([run_ols(df), run_rf(df), run_xgb(df)])
    summary.to_csv(OUT_DIR / "t80_twopart_model_comparison.csv", index=False)
    lines = ["T80 two-part model comparison"]
    for _, row in summary.iterrows():
        lines.append(
            f"{row['model_family']}: part1 acc={row['part1_accuracy']:.4f}, "
            f"part1 auc={row['part1_roc_auc']:.4f}, part2 r2={row['part2_r2']:.4f}, "
            f"part2 rmse={row['part2_rmse']:.4f}, combined r2={row['combined_r2']:.4f}, "
            f"combined rmse={row['combined_rmse']:.4f}"
        )
    (OUT_DIR / "t80_twopart_model_comparison.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()

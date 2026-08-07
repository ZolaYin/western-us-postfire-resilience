from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.miscmodels.ordinal_model import OrderedModel
from xgboost import XGBClassifier


BASE_DIR = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/"
    "recovery_model_execution_near_t0_postfire"
)
INPUT_PATH = BASE_DIR / "MGWR_ready_table_t80_iri_stability_near_t0_postfire.parquet"
OLS_PREDICTORS_PATH = BASE_DIR / "predictors_ols_recovery_baseline.txt"
TREE_PREDICTORS_PATH = BASE_DIR / "predictors_rf_xgb_recovery_baseline.txt"
OUT_DIR = BASE_DIR / "t80_ordinal_2026-03-31"
RANDOM_STATE = 42
CLASSES = list(range(2, 11))


def read_predictors(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def probs_to_expected_year(probs: np.ndarray) -> np.ndarray:
    return probs @ np.array(CLASSES, dtype=float)


def run_ordered_logit(df: pd.DataFrame) -> dict:
    predictors = read_predictors(OLS_PREDICTORS_PATH)
    work = df[["T80_revised"] + predictors].replace([np.inf, -np.inf], np.nan).dropna().copy()
    train_idx, test_idx = train_test_split(
        work.index,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=work["T80_revised"],
    )
    train = work.loc[train_idx].copy()
    test = work.loc[test_idx].copy()
    y_train = pd.Categorical(train["T80_revised"], categories=CLASSES, ordered=True)
    y_test = test["T80_revised"].to_numpy()

    model = OrderedModel(y_train, train[predictors], distr="logit")
    result = model.fit(method="bfgs", disp=False)
    probs = result.model.predict(result.params, exog=test[predictors])
    pred_labels = np.array(CLASSES)[np.argmax(probs, axis=1)]
    expected_year = probs_to_expected_year(probs)

    model_dir = OUT_DIR / "ols_t80_ordinal_2026-03-31"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "ols_t80_ordinal_summary.txt").write_text(result.summary().as_text())
    coef_rows = []
    for term in predictors:
        coef_rows.append(
            {
                "term": term,
                "coef": float(result.params[term]),
                "pvalue": float(result.pvalues[term]),
                "abs_z": float(abs(result.tvalues[term])),
            }
        )
    pd.DataFrame(coef_rows).sort_values("abs_z", ascending=False).to_csv(
        model_dir / "ols_t80_ordinal_importance.csv", index=False
    )
    metrics = {
        "model_family": "ols",
        "scheme": "ordinal",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "test_accuracy": float(accuracy_score(y_test, pred_labels)),
        "test_macro_f1": float(f1_score(y_test, pred_labels, average="macro")),
        "test_expected_r2": float(r2_score(y_test, expected_year)),
        "test_expected_rmse": float(np.sqrt(mean_squared_error(y_test, expected_year))),
    }
    save_json(model_dir / "ols_t80_ordinal_metrics.json", metrics)
    return metrics


def run_rf(df: pd.DataFrame) -> dict:
    predictors = read_predictors(TREE_PREDICTORS_PATH)
    work = df[["T80_revised"] + predictors].replace([np.inf, -np.inf], np.nan).dropna().copy()
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
    probs = model.predict_proba(test[predictors])
    pred_labels = model.predict(test[predictors])
    expected_year = probs_to_expected_year(probs)

    model_dir = OUT_DIR / "rf_t80_ordinal_2026-03-31"
    model_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"predictor": predictors, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False
    ).to_csv(model_dir / "rf_t80_ordinal_feature_importance.csv", index=False)
    metrics = {
        "model_family": "rf",
        "scheme": "ordinal",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "test_accuracy": float(accuracy_score(test["T80_revised"], pred_labels)),
        "test_macro_f1": float(f1_score(test["T80_revised"], pred_labels, average="macro")),
        "test_expected_r2": float(r2_score(test["T80_revised"], expected_year)),
        "test_expected_rmse": float(np.sqrt(mean_squared_error(test["T80_revised"], expected_year))),
    }
    save_json(model_dir / "rf_t80_ordinal_metrics.json", metrics)
    return metrics


def run_xgb(df: pd.DataFrame) -> dict:
    predictors = read_predictors(TREE_PREDICTORS_PATH)
    work = df[["T80_revised"] + predictors].replace([np.inf, -np.inf], np.nan).dropna().copy()
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
    probs = model.predict_proba(test[predictors])
    pred_labels = np.vectorize(inv_label_map.get)(np.argmax(probs, axis=1))
    expected_year = probs_to_expected_year(probs)

    model_dir = OUT_DIR / "xgb_t80_ordinal_2026-03-31"
    model_dir.mkdir(parents=True, exist_ok=True)
    score = model.get_booster().get_score(importance_type="gain")
    pd.DataFrame([{"predictor": k, "importance_gain": v} for k, v in score.items()]).sort_values(
        "importance_gain", ascending=False
    ).to_csv(model_dir / "xgb_t80_ordinal_feature_importance.csv", index=False)
    metrics = {
        "model_family": "xgb",
        "scheme": "ordinal",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "test_accuracy": float(accuracy_score(test["T80_revised"], pred_labels)),
        "test_macro_f1": float(f1_score(test["T80_revised"], pred_labels, average="macro")),
        "test_expected_r2": float(r2_score(test["T80_revised"], expected_year)),
        "test_expected_rmse": float(np.sqrt(mean_squared_error(test["T80_revised"], expected_year))),
    }
    save_json(model_dir / "xgb_t80_ordinal_metrics.json", metrics)
    return metrics


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(INPUT_PATH)
    summary = pd.DataFrame([run_ordered_logit(df), run_rf(df), run_xgb(df)])
    summary.to_csv(OUT_DIR / "t80_ordinal_model_comparison.csv", index=False)
    lines = ["T80 ordinal model comparison"]
    for _, row in summary.iterrows():
        lines.append(
            f"{row['model_family']}: acc={row['test_accuracy']:.4f}, "
            f"macro_f1={row['test_macro_f1']:.4f}, expected_r2={row['test_expected_r2']:.4f}, "
            f"expected_rmse={row['test_expected_rmse']:.4f}"
        )
    (OUT_DIR / "t80_ordinal_model_comparison.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()

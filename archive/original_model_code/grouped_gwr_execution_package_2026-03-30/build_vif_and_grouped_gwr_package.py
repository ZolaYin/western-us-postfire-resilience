import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


ROOT = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km"
)
INPUT = ROOT / "resistance_model_execution_near_t0_aggregated" / "MGWR_ready_table_near_t0_aggregated.parquet"
FULL_SAF_META = ROOT / "resistance_full_saf_models_2026-03-30" / "full_saf_metadata.json"
OUT = ROOT / "grouped_gwr_execution_package_2026-03-30"

RESPONSE = "Resistance"
COORDS = ["x", "y"]
IDS = ["pixel_id", "row", "col", "t0_year"]

GROUPS = {
    "topo_soil": [
        "TS_elev_m_z",
        "TS_slope_deg_z",
        "TS_twi_z",
        "TS_SOC_0_30cm_z",
    ],
    "forest": [
        "FS_TCC_t0_z",
        "FS_CBH_t0agg_z",
        "FS_EVT_t0agg_resistance_proxy",
    ],
    "human": [
        "HUM_popdens_win10km_log_z",
        "HUM_viirs_near_t0_log_z",
        "HUM_imperv_near_t0_z",
    ],
    "climate": [
        "CLIM_eto_sum_pre_z",
        "CLIM_hot_days_35C_pre_z",
        "CLIM_tmmx_std_pre_z",
    ],
}

COMBINATIONS = {
    "A_topo_soil": GROUPS["topo_soil"],
    "B_topo_soil_plus_forest": GROUPS["topo_soil"] + GROUPS["forest"],
    "C_topo_soil_forest_human": GROUPS["topo_soil"] + GROUPS["forest"] + GROUPS["human"],
    "D_topo_soil_forest_human_climate": GROUPS["topo_soil"] + GROUPS["forest"] + GROUPS["human"] + GROUPS["climate"],
    "OLS_full_SAF_reference": [
        "TS_elev_m_z",
        "TS_slope_deg_z",
        "TS_northness_z",
        "TS_eastness_z",
        "TS_twi_z",
        "TS_roughness_z",
        "TS_SOC_0_30cm_z",
        "FS_TCC_t0_z",
        "FS_CBH_t0agg_z",
        "HUM_popdens_win10km_log_z",
        "HUM_viirs_near_t0_log_z",
        "HUM_imperv_near_t0_z",
        "CLIM_pr_sum_pre_z",
        "CLIM_eto_sum_pre_z",
        "CLIM_tmmn_mean_pre_z",
        "CLIM_hot_days_35C_pre_z",
        "CLIM_aridity_pre_z",
        "CLIM_tmmx_std_pre_z",
    ],
}


GWR_SCRIPT = """\
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW


DEFAULT_INPUT = Path("{input_path}")
DEFAULT_PREDICTORS = Path("{predictor_file}")


def read_predictors(path: Path):
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--predictors-file", default=str(DEFAULT_PREDICTORS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bw-min", type=int, default=100)
    args = parser.parse_args()

    input_path = Path(args.input)
    predictors_file = Path(args.predictors_file)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    predictors = read_predictors(predictors_file)
    cols = ["pixel_id", "row", "col", "x", "y", "t0_year", "Resistance"] + predictors
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    coords = work[["x", "y"]].to_numpy(dtype=float)
    y = work[["Resistance"]].to_numpy(dtype=float)
    X = work[predictors].to_numpy(dtype=float)

    selector = Sel_BW(coords, y, X, fixed=False, kernel="bisquare")
    bw = selector.search(bw_min=args.bw_min)
    model = GWR(coords, y, X, bw=bw, fixed=False, kernel="bisquare")
    results = model.fit()

    meta = {{
        "input_file": str(input_path),
        "predictors_file": str(predictors_file),
        "n_rows_used": int(len(work)),
        "predictors": predictors,
        "bandwidth": float(np.atleast_1d(bw)[0]),
        "aic": float(results.aic),
        "bic": float(results.bic),
        "r2": float(results.R2),
        "adj_r2": float(results.adj_R2),
    }}
    (out_dir / "gwr_metrics.json").write_text(json.dumps(meta, indent=2))

    coef_df = work[["pixel_id", "row", "col", "x", "y", "t0_year", "Resistance"]].copy()
    coef_df["fitted"] = results.predy.flatten()
    coef_df["residual"] = results.resid_response.flatten()
    if hasattr(results, "localR2"):
        coef_df["localR2"] = results.localR2.flatten()

    params = results.params
    if params.shape[1] == len(predictors) + 1:
        coef_df["intercept"] = params[:, 0]
        for i, col in enumerate(predictors):
            coef_df[f"coef_{{col}}"] = params[:, i + 1]
    else:
        for i, col in enumerate(predictors):
            coef_df[f"coef_{{col}}"] = params[:, i]

    coef_df.to_parquet(out_dir / "gwr_coefficients.parquet", index=False)
    coef_df.to_csv(out_dir / "gwr_coefficients.csv", index=False)


if __name__ == "__main__":
    main()
"""


def build_full_saf_indicators(df: pd.DataFrame, codes: list[int]) -> pd.DataFrame:
    df = df.copy()
    saf = pd.to_numeric(df["FS_EVT_t0agg_SAF_code"], errors="coerce").astype("Int64")
    for code in codes:
        df[f"FS_EVT_t0agg_SAF_{code}"] = (saf == code).fillna(False).astype(int)
    return df


def compute_vif_table(df: pd.DataFrame, predictors: list[str], label: str) -> pd.DataFrame:
    work = df[predictors].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if work.empty:
        return pd.DataFrame({"set_name": [label], "predictor": ["<empty>"], "vif": [np.nan]})
    # Drop constant/near-constant columns to avoid singular VIF artifacts.
    keep = [c for c in work.columns if work[c].std(ddof=0) > 1e-12]
    dropped = [c for c in work.columns if c not in keep]
    rows = []
    if dropped:
        rows.extend({"set_name": label, "predictor": c, "vif": np.nan, "note": "dropped_constant"} for c in dropped)
    work = work[keep]
    X = work.to_numpy(dtype=float)
    for i, col in enumerate(work.columns):
        try:
            vif = float(variance_inflation_factor(X, i))
        except Exception:
            vif = np.inf
        rows.append({"set_name": label, "predictor": col, "vif": vif, "note": ""})
    return pd.DataFrame(rows)


def write_predictor_file(path: Path, predictors: list[str], header: str):
    text = [f"# {header}"]
    text.extend(predictors)
    path.write_text("\n".join(text) + "\n")


def write_script(path: Path, predictor_file: Path):
    path.write_text(
        GWR_SCRIPT.format(
            input_path=str(INPUT),
            predictor_file=str(predictor_file),
        )
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads(FULL_SAF_META.read_text())
    saf_codes = meta["full_saf_codes"]

    df = pd.read_parquet(INPUT)
    df = build_full_saf_indicators(df, saf_codes)

    full_model_predictors = COMBINATIONS["OLS_full_SAF_reference"] + [f"FS_EVT_t0agg_SAF_{code}" for code in saf_codes]

    vif_tables = []
    vif_tables.append(compute_vif_table(df, full_model_predictors, "official_full_model_with_full_SAF"))
    for name, predictors in GROUPS.items():
        vif_tables.append(compute_vif_table(df, predictors, f"group::{name}"))
    for name, predictors in COMBINATIONS.items():
        vif_tables.append(compute_vif_table(df, predictors, f"combo::{name}"))

    vif_df = pd.concat(vif_tables, ignore_index=True)
    vif_df.to_csv(OUT / "official_vif_summary.csv", index=False)

    overview_rows = []
    for set_name, sub in vif_df.groupby("set_name"):
        vals = sub["vif"].replace([np.inf, -np.inf], np.nan).dropna()
        max_vif = float(vals.max()) if not vals.empty else np.nan
        mean_vif = float(vals.mean()) if not vals.empty else np.nan
        n_gt_5 = int((vals > 5).sum()) if not vals.empty else 0
        n_gt_10 = int((vals > 10).sum()) if not vals.empty else 0
        overview_rows.append(
            {
                "set_name": set_name,
                "n_predictors": int(sub["predictor"].ne("<empty>").sum()),
                "max_vif": max_vif,
                "mean_vif": mean_vif,
                "n_vif_gt_5": n_gt_5,
                "n_vif_gt_10": n_gt_10,
            }
        )
    overview_df = pd.DataFrame(overview_rows).sort_values("set_name")
    overview_df.to_csv(OUT / "official_vif_overview.csv", index=False)

    # predictor files
    for group_name, predictors in GROUPS.items():
        write_predictor_file(
            OUT / f"predictors_{group_name}.txt",
            predictors,
            f"GWR group {group_name}",
        )
    for combo_name, predictors in COMBINATIONS.items():
        write_predictor_file(
            OUT / f"predictors_{combo_name}.txt",
            predictors,
            f"GWR combination {combo_name}",
        )

    # one conservative full-SAF reference set for OLS only, not first GWR run
    write_predictor_file(
        OUT / "predictors_full_SAF_reference.txt",
        full_model_predictors,
        "Full official model with full SAF; reference only, not recommended as first GWR run",
    )

    # scripts
    for group_name in GROUPS:
        pred = OUT / f"predictors_{group_name}.txt"
        write_script(OUT / f"run_gwr_{group_name}.py", pred)
    for combo_name in COMBINATIONS:
        pred = OUT / f"predictors_{combo_name}.txt"
        write_script(OUT / f"run_gwr_{combo_name}.py", pred)

    notes = [
        "Grouped GWR execution package built from the official near-t0 Resistance system.",
        "",
        "What was evaluated:",
        "- full-model VIF for the official near-t0 system with full SAF",
        "- grouped VIF for topo_soil / forest / human / climate",
        "- staged combination VIF for the proposed GWR sequence",
        "",
        "Package contents:",
        "- official_vif_summary.csv",
        "- official_vif_overview.csv",
        "- predictor files for each group and staged combination",
        "- one GWR-ready script per group/combination",
        "",
        "Interpretation rule:",
        "- prioritize combinations with fewer predictors and lower max VIF first",
        "- do not start with the full-SAF 50-predictor model in GWR",
        "- use proxy EVT in the first spatial run; full SAF belongs in OLS or later sensitivity tests",
    ]
    (OUT / "grouped_gwr_package_notes.txt").write_text("\n".join(notes) + "\n")


if __name__ == "__main__":
    main()

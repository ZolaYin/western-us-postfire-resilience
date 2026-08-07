import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from mgwr.gwr import MGWR
from mgwr.sel_bw import Sel_BW


def parse_list_arg(text):
    if text is None or str(text).strip() == "":
        return []
    return [x.strip() for x in str(text).split(",") if x.strip()]


def load_predictor_list(df, predictor_file, response_col, coord_cols, id_cols):
    if predictor_file is not None:
        p = Path(predictor_file)
        if not p.exists():
            raise FileNotFoundError(f"Predictor file not found: {p}")

        if p.suffix.lower() == ".csv":
            pred_df = pd.read_csv(p)
            if "predictor" in pred_df.columns:
                predictor_cols = pred_df["predictor"].dropna().astype(str).tolist()
            else:
                predictor_cols = pred_df.iloc[:, 0].dropna().astype(str).tolist()
        else:
            predictor_cols = [
                line.strip()
                for line in p.read_text().splitlines()
                if line.strip()
            ]
    else:
        non_predictors = set(id_cols + coord_cols + [response_col])
        predictor_cols = [c for c in df.columns if c not in non_predictors]

    predictor_cols = list(dict.fromkeys(predictor_cols))
    return predictor_cols


def safe_summary_text(results):
    try:
        obj = results.summary()
        return obj if isinstance(obj, str) else str(obj)
    except Exception as e:
        return f"Could not render MGWR summary cleanly.\n\nError: {repr(e)}"


def to_serializable_list(x):
    return [float(v) for v in np.atleast_1d(x).tolist()]


def main():
    parser = argparse.ArgumentParser(description="Run MGWR on a prepared model-input table")

    parser.add_argument("--input", required=True, help="Input parquet table")
    parser.add_argument("--output-dir", required=True, help="Directory for MGWR outputs")

    parser.add_argument("--response-col", default="Resistance", help="Response column name")
    parser.add_argument("--coord-cols", default="x,y", help="Comma-separated coordinate columns")
    parser.add_argument("--id-cols", default="pixel_id,row,col,t0_year", help="Comma-separated ID/metadata columns")

    parser.add_argument("--predictors-file", default=None, help="Optional text/csv file listing predictor columns")
    parser.add_argument("--bw-min", type=int, default=20, help="Minimum bandwidth lower bound")
    parser.add_argument("--chunk-label", default=None, help="Optional label for run metadata")
    parser.add_argument("--no-intercept", action="store_true", help="Disable intercept in Sel_BW and MGWR")
    parser.add_argument("--dropna-subset", default=None, help="Optional comma-separated subset of columns for dropna")
    parser.add_argument("--prefix", default="mgwr", help="Prefix for output file names")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    response_col = args.response_col
    coord_cols = parse_list_arg(args.coord_cols)
    id_cols = parse_list_arg(args.id_cols)
    dropna_subset_arg = parse_list_arg(args.dropna_subset)
    use_intercept = not args.no_intercept
    prefix = args.prefix.strip() or "mgwr"

    df = pd.read_parquet(input_path)

    missing_required = [c for c in [response_col] + coord_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    predictor_cols = load_predictor_list(
        df=df,
        predictor_file=args.predictors_file,
        response_col=response_col,
        coord_cols=coord_cols,
        id_cols=id_cols,
    )

    if not predictor_cols:
        raise ValueError("No predictor columns found.")

    missing_predictors = [c for c in predictor_cols if c not in df.columns]
    if missing_predictors:
        raise ValueError(f"Predictor columns missing from input table: {missing_predictors}")

    forbidden_predictors = set(id_cols + coord_cols + [response_col, f"{response_col}_z"])
    bad_predictors = [c for c in predictor_cols if c in forbidden_predictors]
    if bad_predictors:
        raise ValueError(f"Unexpected predictor columns found: {bad_predictors}")

    use_cols = []
    for c in id_cols + coord_cols + [response_col] + predictor_cols:
        if c in df.columns and c not in use_cols:
            use_cols.append(c)

    work = df[use_cols].copy()
    work = work.replace([np.inf, -np.inf], np.nan)

    if dropna_subset_arg:
        dropna_subset = [c for c in dropna_subset_arg if c in work.columns]
        if not dropna_subset:
            raise ValueError("dropna_subset was provided but none of those columns exist in the table.")
    else:
        dropna_subset = [c for c in coord_cols + [response_col] + predictor_cols if c in work.columns]

    before_n = len(work)
    work = work.dropna(subset=dropna_subset).reset_index(drop=True)
    after_n = len(work)

    if after_n == 0:
        raise ValueError("All rows were dropped after removing NaN/inf.")

    coords = work[coord_cols].to_numpy(dtype=float)
    y = work[[response_col]].to_numpy(dtype=float)
    X = work[predictor_cols].to_numpy(dtype=float)

    if X.shape[1] == 0:
        raise ValueError("No predictor columns found after cleaning.")

    selector = Sel_BW(
        coords,
        y,
        X,
        multi=True,
        constant=use_intercept,
    )

    # 更保守的写法，降低不同 mgwr 版本兼容风险
    bw = selector.search(multi_bw_min=[args.bw_min])

    model = MGWR(
        coords,
        y,
        X,
        selector=selector,
        constant=use_intercept,
    )
    results = model.fit()

    summary_txt = safe_summary_text(results)
    (output_dir / f"{prefix}_summary.txt").write_text(summary_txt)

    bw_list = to_serializable_list(bw)

    metadata = {
        "input_file": str(input_path),
        "output_dir": str(output_dir),
        "chunk_label": args.chunk_label,
        "prefix": prefix,
        "n_rows_before_dropna": int(before_n),
        "n_rows_after_dropna": int(after_n),
        "n_predictors": int(len(predictor_cols)),
        "predictor_columns": predictor_cols,
        "response_column": response_col,
        "coordinate_columns": coord_cols,
        "id_columns": id_cols,
        "dropna_subset": dropna_subset,
        "bw_min": int(args.bw_min),
        "bandwidths": bw_list,
        "use_intercept": bool(use_intercept),
        "aic": float(results.aic) if hasattr(results, "aic") and results.aic is not None else None,
        "bic": float(results.bic) if hasattr(results, "bic") and results.bic is not None else None,
    }
    (output_dir / f"{prefix}_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)
    )

    base_cols = [c for c in id_cols + coord_cols + [response_col] if c in work.columns]
    coef_df = work[base_cols].copy()

    if hasattr(results, "predy") and results.predy is not None:
        coef_df["fitted"] = results.predy.flatten()

    if hasattr(results, "resid_response") and results.resid_response is not None:
        coef_df["residual"] = results.resid_response.flatten()

    if hasattr(results, "localR2") and results.localR2 is not None:
        coef_df["localR2"] = results.localR2.flatten()

    params = results.params
    expected_with_intercept = len(predictor_cols) + (1 if use_intercept else 0)

    if params.shape[1] != expected_with_intercept:
        raise ValueError(
            f"Unexpected params shape: {params.shape}, "
            f"predictors={len(predictor_cols)}, use_intercept={use_intercept}"
        )

    offset = 0
    if use_intercept:
        coef_df["intercept"] = params[:, 0]
        offset = 1

    for i, col in enumerate(predictor_cols):
        coef_df[f"coef_{col}"] = params[:, i + offset]

    coef_df.to_parquet(output_dir / f"{prefix}_coefficients.parquet", index=False)
    coef_df.to_csv(output_dir / f"{prefix}_coefficients.csv", index=False)

    if hasattr(results, "tvalues") and results.tvalues is not None:
        tv = results.tvalues
        if tv.shape[1] == expected_with_intercept:
            t_df = work[[c for c in id_cols + coord_cols if c in work.columns]].copy()
            offset = 0
            if use_intercept:
                t_df["intercept_t"] = tv[:, 0]
                offset = 1
            for i, col in enumerate(predictor_cols):
                t_df[f"t_{col}"] = tv[:, i + offset]
            t_df.to_parquet(output_dir / f"{prefix}_tvalues.parquet", index=False)
            t_df.to_csv(output_dir / f"{prefix}_tvalues.csv", index=False)


if __name__ == "__main__":
    main()

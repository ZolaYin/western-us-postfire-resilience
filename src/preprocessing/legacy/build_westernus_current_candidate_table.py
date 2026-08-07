#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol


ROOT = Path(
    "/path/to/google-drive/我的云端硬盘"
)
BASE = ROOT / "US_Fire_and_Ecology_Data" / "WUS_1km"
RESP_PATH = (
    BASE
    / "westernus_response_legacy_nlcd_20260410"
    / "westernus_response_metrics_legacy_nlcd.parquet"
)
EVT_MODE_PATH = BASE / "WesternUS_EVT2022_mode_code_1km.tif"
EVT_LOOKUP_XLSX = ROOT / "中转" / "EVT_WUS_Table.xlsx"
EVT_LOOKUP_SHEET = "EVT2022_WUS_Table"

TOPO_PATH = ROOT / "WesternUS_drivers_raw" / "TOPO_STATIC_WesternUS11_1km_5070_v2.tif"
SOIL_PATH = ROOT / "WesternUS_drivers_raw" / "SOIL_ALL_WesternUS11_1km_5070_v1.tif"
POP_PATH = ROOT / "WesternUS_drivers_raw" / "HUMAN_GPWv411_PopDensity10km_WesternUS11_1km_5070_v1.tif"
VIIRS_PATH = ROOT / "WesternUS_drivers_raw" / "HUMAN_VIIRS_AnnualMean_WesternUS11_1km_5070_v1.tif"
IMPERV_PATH = ROOT / "WesternUS_drivers_raw" / "HUMAN_NLCD_Impervious_WesternUS11_1km_5070_v1.tif"
CLIM_TILE_TOP = ROOT / "WesternUS_drivers_raw" / "GRIDMET_STACK_2000_2023_WesternUS_11states-0000000000-0000000000.tif"
CLIM_TILE_BOTTOM = ROOT / "WesternUS_drivers_raw" / "GRIDMET_STACK_2000_2023_WesternUS_11states-0000002048-0000000000.tif"

OUT_TABLE = BASE / "westernus_current_candidate_table_legacy_nlcd.parquet"
OUT_SAMPLE = BASE / "westernus_current_candidate_table_legacy_nlcd_sample.csv"
OUT_VIF = BASE / "westernus_current_candidate_table_legacy_nlcd_vif.csv"
OUT_REPORT = BASE / "westernus_current_candidate_table_legacy_nlcd_report.md"
SAMPLE_N = 10000
SEED = 42

CLIMATE_VARIABLES = [
    "pr_sum",
    "eto_sum",
    "vpd_mean",
    "tmmx_mean",
    "tmmn_mean",
    "tmmx_max",
    "hot_days_30C",
    "hot_days_35C",
    "water_deficit",
    "aridity",
    "tmmx_std",
    "vpd_std",
]

EXCLUDED_DRIVERS = [
    {
        "name": "HUM_roaddens_r10km",
        "reason": "verified file is still on the old coast-era 1936x817 grid",
        "path": str(ROOT / "WesternUS_drivers_raw" / "westernus_road_trail_build_20260406" / "road_density_WesternUS_filtered_r10km_1km_5070.tif"),
    },
    {
        "name": "HUM_traildens_r10km",
        "reason": "verified file is still on the old coast-era 1936x817 grid",
        "path": str(ROOT / "WesternUS_drivers_raw" / "westernus_road_trail_build_20260406" / "trail_density_WesternUS_r10km_1km_5070.tif"),
    },
    {
        "name": "FS_CBH_1km",
        "reason": "current WesternUS response-grid CBH product not verified",
        "path": str(ROOT / "WUS_drivers_raw_v2" / "forest" / "CBH_1km_5070_aligned.tif"),
    },
    {
        "name": "FS_TCC_t0",
        "reason": "current WesternUS response-grid TCC product not verified",
        "path": str(BASE),
    },
    {
        "name": "FS_GEDI_p95_near_t0",
        "reason": "current WesternUS response-grid GEDI product not verified",
        "path": str(BASE),
    },
]


def parse_xlsx_sheet(xlsx_path: Path, sheet_name: str) -> list[dict]:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    rows_out: list[dict] = []
    with zipfile.ZipFile(xlsx_path) as zf:
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        sheets = {
            s.attrib["name"]: s.attrib[rel_ns]
            for s in workbook.find("a:sheets", ns)
        }
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        relmap = {rel.attrib["Id"]: rel.attrib["Target"].lstrip("/") for rel in rels}
        worksheet = ET.fromstring(zf.read(relmap[sheets[sheet_name]]))
        rows = []
        for row in worksheet.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row"):
            values = []
            for cell in row.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c"):
                inline = cell.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}is")
                value = cell.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v")
                if inline is not None:
                    texts = [
                        t.text or ""
                        for t in inline.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")
                    ]
                    values.append("".join(texts))
                elif value is not None:
                    values.append(value.text or "")
                else:
                    values.append("")
            rows.append(values)
    header = rows[0]
    for raw in rows[1:]:
        padded = raw + [""] * (len(header) - len(raw))
        rows_out.append(dict(zip(header, padded[: len(header)])))
    return rows_out


def classify_evt_group(row: dict) -> str:
    text_fields = [
        row.get("EVT_NAME", ""),
        row.get("SAF_SRM", ""),
        row.get("EVT_PHYS", ""),
        row.get("EVT_ORDER", row.get("NVCSORDER", "")),
        row.get("EVT_CLASS", row.get("NVCSCLASS", "")),
        row.get("EVT_SBCLS", row.get("NVCSSUBCLASS", row.get("NVCSSUBCLA", ""))),
    ]
    text = " | ".join(str(x) for x in text_fields).lower()
    order = str(row.get("EVT_ORDER", row.get("NVCSORDER", ""))).strip().lower()
    if "shrub" in text or order == "shrub-dominated":
        return "shrub"
    if "mixed evergreen-deciduous" in text or "mixed forest" in text or "mixed woodland" in text:
        return "mixed"
    if "deciduous" in text:
        return "deciduous"
    if "evergreen" in text or "conifer" in text or "woodland" in text:
        return "conifer"
    if order == "tree-dominated":
        return "mixed"
    return "nonwoody"


def build_evt_lookup(rows: list[dict]) -> dict[int, dict[str, object]]:
    resistance_scores = {
        "nonwoody": 0.0,
        "shrub": 1.0,
        "deciduous": 2.0,
        "mixed": 3.0,
        "conifer": 4.0,
    }
    regeneration_scores = {
        "nonwoody": 0.0,
        "shrub": 4.0,
        "deciduous": 3.0,
        "mixed": 2.0,
        "conifer": 1.0,
    }
    out: dict[int, dict[str, object]] = {}
    for row in rows:
        value_text = row.get("Value", row.get("VALUE", row.get("value", "")))
        try:
            code = int(float(value_text))
        except ValueError:
            continue
        order = str(row.get("EVT_ORDER", row.get("NVCSORDER", ""))).strip().lower()
        group = classify_evt_group(row)
        old_tree_or_shrub = int(order in {"tree-dominated", "shrub-dominated"})
        out[code] = {
            "group": group,
            "old_tree_or_shrub": old_tree_or_shrub,
            "resistance_proxy": resistance_scores[group] if old_tree_or_shrub else 0.0,
            "regeneration_proxy": regeneration_scores[group] if old_tree_or_shrub else 0.0,
        }
    return out


def compute_nearest_year_map(available_years: list[int], targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    years = np.array(sorted(available_years), dtype=np.int16)
    pos = np.searchsorted(years, targets, side="left")
    left_idx = np.clip(pos - 1, 0, len(years) - 1)
    right_idx = np.clip(pos, 0, len(years) - 1)
    left_year = years[left_idx]
    right_year = years[right_idx]
    choose_right = np.abs(right_year - targets) < np.abs(left_year - targets)
    chosen_idx = np.where(choose_right, right_idx, left_idx)
    return years[chosen_idx], chosen_idx


def sample_multiband_points(path: Path, band_indices: list[int], xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    out = np.full((xs.shape[0], len(band_indices)), np.nan, dtype=np.float32)
    with rasterio.open(path) as src:
        inside = (
            (xs >= src.bounds.left)
            & (xs <= src.bounds.right)
            & (ys >= src.bounds.bottom)
            & (ys <= src.bounds.top)
        )
        coords = list(zip(xs[inside], ys[inside]))
        if coords:
            vals = np.array(list(src.sample(coords, indexes=band_indices)), dtype=np.float32)
            nodata = src.nodata
            if nodata is not None:
                vals[np.isclose(vals, nodata)] = np.nan
            vals[~np.isfinite(vals)] = np.nan
            out[inside, :] = vals
    return out


def sample_two_tile_multiband(paths: list[Path], band_indices: list[int], xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    out = np.full((xs.shape[0], len(band_indices)), np.nan, dtype=np.float32)
    filled = np.zeros(xs.shape[0], dtype=bool)
    for path in paths:
        vals = sample_multiband_points(path, band_indices, xs, ys)
        valid = np.isfinite(vals).any(axis=1)
        take = valid & (~filled)
        out[take, :] = vals[take, :]
        filled[take] = True
    return out


def sample_two_tile_multiband_by_index(paths: list[Path], band_indices: list[int], xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    out = np.full((xs.shape[0], len(band_indices)), np.nan, dtype=np.float32)
    filled = np.zeros(xs.shape[0], dtype=bool)
    for path in paths:
        with rasterio.open(path) as src:
            inside = (
                (xs >= src.bounds.left)
                & (xs <= src.bounds.right)
                & (ys >= src.bounds.bottom)
                & (ys <= src.bounds.top)
                & (~filled)
            )
            if not inside.any():
                continue
            rr, cc = rowcol(src.transform, xs[inside], ys[inside])
            rr = np.asarray(rr, dtype=int)
            cc = np.asarray(cc, dtype=int)
            stack = src.read(band_indices).astype(np.float32, copy=False)
            vals = stack[:, rr, cc].T
            nodata = src.nodata
            if nodata is not None:
                vals[np.isclose(vals, nodata)] = np.nan
            vals[~np.isfinite(vals)] = np.nan
            out[inside, :] = vals
            filled[inside] = True
    return out


def parse_year_from_label(desc: str) -> int | None:
    m = re.search(r"(19|20)\d{2}", str(desc))
    return int(m.group(0)) if m else None


def build_year_band_map_from_descs(descs: tuple[str | None, ...]) -> dict[int, int]:
    out = {}
    for i, desc in enumerate(descs, start=1):
        year = parse_year_from_label(desc or "")
        if year is not None:
            out[year] = i
    return out


def build_climate_year_band_map(descs: tuple[str | None, ...], variable_name: str) -> dict[int, int]:
    out = {}
    for i, desc in enumerate(descs, start=1):
        label = desc or ""
        m = re.match(r"(\d+)_(.+)", label)
        if not m:
            continue
        year = 2000 + int(m.group(1))
        var = m.group(2)
        if var == variable_name:
            out[year] = i
    return out


def compute_vif(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    X = df[columns].astype(float).to_numpy()
    rows = []
    for i, col in enumerate(columns):
        y = X[:, i]
        X_other = np.delete(X, i, axis=1)
        X_design = np.column_stack([np.ones(X_other.shape[0]), X_other])
        beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
        yhat = X_design @ beta
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot == 0:
            r2 = np.nan
            vif = np.inf
        else:
            r2 = 1.0 - ss_res / ss_tot
            vif = np.inf if r2 >= 0.999999999 else 1.0 / (1.0 - r2)
        rows.append({"variable": col, "r2_against_others": r2, "vif": vif})
    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


def main() -> None:
    print("Loading WesternUS response table...", flush=True)
    df = pd.read_parquet(RESP_PATH).copy()
    xs = df["x"].to_numpy(dtype=float)
    ys = df["y"].to_numpy(dtype=float)
    rows = df["row"].to_numpy(dtype=int)
    cols = df["col"].to_numpy(dtype=int)
    t0_year = df["t0_year"].to_numpy(dtype=np.int16)

    print("Mapping EVT 2022 mode codes and proxies...", flush=True)
    with rasterio.open(EVT_MODE_PATH) as src:
        evt_codes = src.read(1).astype(np.int32, copy=False)
        evt_nodata = src.nodata
    evt_sample = evt_codes[rows, cols]
    if evt_nodata is not None:
        evt_sample = np.where(evt_sample == evt_nodata, -9999, evt_sample)

    evt_lookup = build_evt_lookup(parse_xlsx_sheet(EVT_LOOKUP_XLSX, EVT_LOOKUP_SHEET))
    df["FS_EVT2022_code"] = evt_sample
    df["FS_EVT_resistance_proxy"] = np.array(
        [evt_lookup.get(int(code), {}).get("resistance_proxy", np.nan) if code != -9999 else np.nan for code in evt_sample],
        dtype=np.float32,
    )
    df["FS_EVT_regeneration_proxy"] = np.array(
        [evt_lookup.get(int(code), {}).get("regeneration_proxy", np.nan) if code != -9999 else np.nan for code in evt_sample],
        dtype=np.float32,
    )
    df["FS_EVT_group_class"] = [
        evt_lookup.get(int(code), {}).get("group") if code != -9999 else None for code in evt_sample
    ]

    print("Sampling topography...", flush=True)
    topo_bands = sample_multiband_points(TOPO_PATH, [1, 2, 4, 5, 6, 7], xs, ys)
    df["TS_elev_m"] = topo_bands[:, 0]
    df["TS_slope_deg"] = topo_bands[:, 1]
    df["TS_northness"] = topo_bands[:, 2]
    df["TS_eastness"] = topo_bands[:, 3]
    df["TS_twi"] = topo_bands[:, 4]
    df["TS_roughness"] = topo_bands[:, 5]

    print("Sampling soil...", flush=True)
    soil_bands = sample_multiband_points(SOIL_PATH, [4], xs, ys)
    df["TS_SOC_0_30cm"] = soil_bands[:, 0]

    print("Sampling population...", flush=True)
    with rasterio.open(POP_PATH) as src:
        pop_year_map = build_year_band_map_from_descs(src.descriptions)
    pop_years = sorted(pop_year_map)
    pop_source_year, pop_idx = compute_nearest_year_map(pop_years, t0_year)
    pop_band_indices = [pop_year_map[y] for y in pop_years]
    pop_stack = sample_multiband_points(POP_PATH, pop_band_indices, xs, ys)
    df["HUM_popdens_source_year"] = pop_source_year.astype(np.int16)
    df["HUM_popdens_win10km"] = pop_stack[np.arange(len(df)), pop_idx]

    print("Sampling VIIRS...", flush=True)
    with rasterio.open(VIIRS_PATH) as src:
        viirs_year_map = build_year_band_map_from_descs(src.descriptions)
    viirs_years = sorted(viirs_year_map)
    viirs_source_year, viirs_idx = compute_nearest_year_map(viirs_years, t0_year)
    viirs_band_indices = [viirs_year_map[y] for y in viirs_years]
    viirs_stack = sample_multiband_points(VIIRS_PATH, viirs_band_indices, xs, ys)
    df["HUM_viirs_source_year"] = viirs_source_year.astype(np.int16)
    df["HUM_viirs_near_t0"] = viirs_stack[np.arange(len(df)), viirs_idx]

    print("Sampling impervious...", flush=True)
    with rasterio.open(IMPERV_PATH) as src:
        imperv_year_map = build_year_band_map_from_descs(src.descriptions)
    imperv_years = sorted(imperv_year_map)
    imperv_source_year, imperv_idx = compute_nearest_year_map(imperv_years, t0_year)
    imperv_band_indices = [imperv_year_map[y] for y in imperv_years]
    imperv_stack = sample_multiband_points(IMPERV_PATH, imperv_band_indices, xs, ys)
    df["HUM_imperv_source_year"] = imperv_source_year.astype(np.int16)
    df["HUM_imperv_near_t0"] = imperv_stack[np.arange(len(df)), imperv_idx]

    print("Sampling climate windows variable by variable...", flush=True)
    with rasterio.open(CLIM_TILE_TOP) as src:
        clim_descs = src.descriptions
    climate_tile_paths = [CLIM_TILE_TOP, CLIM_TILE_BOTTOM]
    for var in CLIMATE_VARIABLES:
        print(f"  climate variable: {var}", flush=True)
        year_band_map = build_climate_year_band_map(clim_descs, var)
        years = sorted(year_band_map)
        sampled = sample_two_tile_multiband_by_index(
            climate_tile_paths, [year_band_map[y] for y in years], xs, ys
        )
        year_to_pos = {y: i for i, y in enumerate(years)}
        pre = np.full(len(df), np.nan, dtype=np.float32)
        post = np.full(len(df), np.nan, dtype=np.float32)
        for year in np.unique(t0_year):
            mask = t0_year == year
            pre_years = [int(year) - 3, int(year) - 2, int(year) - 1]
            post_years = [int(year), int(year) + 1]
            if all(y in year_to_pos for y in pre_years):
                pre[:, ...]
                pre[mask] = np.nanmean(sampled[mask][:, [year_to_pos[y] for y in pre_years]], axis=1).astype(np.float32)
            if all(y in year_to_pos for y in post_years):
                post[mask] = np.nanmean(sampled[mask][:, [year_to_pos[y] for y in post_years]], axis=1).astype(np.float32)
        df[f"CLIM_{var}_pre"] = pre
        df[f"CLIM_{var}_post"] = post

    print("Computing complete-case VIF...", flush=True)
    predictor_cols = [
        "TS_elev_m",
        "TS_slope_deg",
        "TS_northness",
        "TS_eastness",
        "TS_twi",
        "TS_roughness",
        "TS_SOC_0_30cm",
        "FS_EVT_resistance_proxy",
        "FS_EVT_regeneration_proxy",
        "HUM_popdens_win10km",
        "HUM_viirs_near_t0",
        "HUM_imperv_near_t0",
    ] + [f"CLIM_{var}_pre" for var in CLIMATE_VARIABLES] + [f"CLIM_{var}_post" for var in CLIMATE_VARIABLES]

    complete_mask = df[predictor_cols].notna().all(axis=1)
    complete_df = df.loc[complete_mask, predictor_cols].copy()
    vif_df = compute_vif(complete_df, predictor_cols)
    vif_df["n_complete_cases"] = int(len(complete_df))
    vif_df.to_csv(OUT_VIF, index=False)

    print("Writing candidate table outputs...", flush=True)
    df.to_parquet(OUT_TABLE, index=False)
    df.sample(min(SAMPLE_N, len(df)), random_state=SEED).to_csv(OUT_SAMPLE, index=False)

    missing_summary = (
        df[predictor_cols]
        .isna()
        .sum()
        .sort_values(ascending=False)
        .rename("n_missing")
    )

    report_lines = [
        "# WesternUS Current Candidate Table",
        "",
        "## Build Rule",
        "- Base table: `westernus_response_metrics_legacy_nlcd.parquet`",
        "- Current usable drivers were sampled or mapped onto the current WesternUS response pixels using each pixel center coordinate.",
        "- No old-grid road/trail or unverified current-grid CBH/TCC/GEDI products were forced into this table.",
        "",
        "## Output Files",
        f"- Candidate table: `{OUT_TABLE}`",
        f"- Sample CSV: `{OUT_SAMPLE}`",
        f"- Final VIF CSV: `{OUT_VIF}`",
        "",
        "## Included Predictors",
    ]
    report_lines.extend(f"- `{col}`" for col in predictor_cols)
    report_lines.extend(
        [
            "",
            "## Excluded Drivers",
        ]
    )
    report_lines.extend(
        f"- `{item['name']}`: {item['reason']}" for item in EXCLUDED_DRIVERS
    )
    report_lines.extend(
        [
            "",
            "## Coverage And Completeness",
            f"- Response rows in base table: `{len(df)}`",
            f"- Complete cases for final VIF across included predictors: `{len(complete_df)}`",
            "- Missing-count summary by predictor:",
        ]
    )
    report_lines.extend(
        f"- `{idx}`: {int(val)} missing" for idx, val in missing_summary.items()
    )
    report_lines.extend(
        [
            "",
            "## Final VIF",
        ]
    )
    for _, row in vif_df.iterrows():
        vif_str = "inf" if not np.isfinite(row["vif"]) else f"{row['vif']:.3f}"
        report_lines.append(
            f"- `{row['variable']}`: VIF={vif_str}, R2={row['r2_against_others']:.6f}"
        )

    OUT_REPORT.write_text("\n".join(report_lines), encoding="utf-8")

    summary = {
        "candidate_table": str(OUT_TABLE),
        "candidate_sample_csv": str(OUT_SAMPLE),
        "vif_csv": str(OUT_VIF),
        "report_md": str(OUT_REPORT),
        "response_rows": int(len(df)),
        "complete_cases_for_vif": int(len(complete_df)),
        "included_predictor_count": int(len(predictor_cols)),
        "excluded_drivers": EXCLUDED_DRIVERS,
        "top_vif": vif_df.head(10).to_dict(orient="records"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

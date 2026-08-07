import os
import numpy as np
import pandas as pd

from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW

# =========================
# 0) USER SETTINGS
# =========================
IN_PARQUET = "df_mgwr_resistance_z.parquet"   # ← 改成你的实际路径
Y_name = "Resistance"

# 输出目录
OUT_DIR = "gwr_grouped_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# 1) VARIABLE GROUPS (RAW NAMES)
# =========================
GROUPS = {
    "climate": [
        "CLIM_pr_gs_sum_mm_pre",
        "CLIM_vpd_gs_mean_kpa_pre",
    ],
    "forest": [
        "FS_TCC_t0",
        "FS_AIproxy_t0",
    ],
    "topo_soil": [
        "TS_elev_m",
        "TS_slope_deg",
        "TS_SOC_0_30cm",
    ],
    "human": [
        "HUM_imperv_near_t0",
        "HUM_popdens_near_t0",
        "HUM_viirs_near_t0",
    ],
}

USE_Z = True  # 用 *_z 列（推荐）

# =========================
# 2) LOAD + CHECK
# =========================
df = pd.read_parquet(IN_PARQUET)

base_need = ["x", "y", Y_name]
missing_base = [c for c in base_need if c not in df.columns]
if missing_base:
    raise KeyError(f"缺少基础列: {missing_base}")

# 组内变量检查
missing_detail = {}
for g, cols in GROUPS.items():
    cols_use = [c + "_z" for c in cols] if USE_Z else cols
    miss = [c for c in cols_use if c not in df.columns]
    if miss:
        missing_detail[g] = miss
if missing_detail:
    raise KeyError(f"以下分组缺列（请确认 parquet 里是否有 *_z）：\n{missing_detail}")

# 组织 coords / y
coords = df[["x", "y"]].to_numpy(np.float64)
y = df[[Y_name]].to_numpy(np.float64)

# 去掉 NaN/inf（所有模型统一用同一批行）
bad = (
    np.isnan(coords).any(axis=1) | np.isnan(y).any(axis=1) |
    np.isinf(coords).any(axis=1) | np.isinf(y).any(axis=1)
)
# 还要把所有分组X里的坏行也统一剔除
all_x_cols = []
for cols in GROUPS.values():
    all_x_cols.extend([c + "_z" for c in cols] if USE_Z else cols)
X_all = df[all_x_cols].to_numpy(np.float64)
bad = bad | np.isnan(X_all).any(axis=1) | np.isinf(X_all).any(axis=1)

if bad.any():
    df = df.loc[~bad].copy()
    coords = coords[~bad]
    y = y[~bad]

print(f"✅ Data ready: n={len(df)}")

# =========================
# 3) RUN ONE GROUP
# =========================
def run_gwr_for_group(group_name: str, raw_cols: list[str]):
    cols_use = [c + "_z" for c in raw_cols] if USE_Z else raw_cols
    X = df[cols_use].to_numpy(np.float64)

    # 带宽选择：adaptive + gaussian（更稳）
    selector = Sel_BW(
        coords, y, X,
        fixed=False,
        kernel="gaussian",
        constant=True
    )
    bw = selector.search()
    print(f"✅ [{group_name}] bw = {bw}")

    # 拟合
    gwr = GWR(
        coords, y, X,
        bw=bw,
        fixed=False,
        kernel="gaussian",
        constant=True
    )
    res = gwr.fit()

    # 导出：params/tvalues/localR2
    param_cols = ["Intercept"] + cols_use
    params = pd.DataFrame(res.params, columns=param_cols, index=df.index)
    tvals  = pd.DataFrame(res.tvalues, columns=param_cols, index=df.index)
    localR2 = pd.Series(res.localR2.flatten(), index=df.index, name="localR2")

    out = df[["x", "y", Y_name]].copy()
    out = pd.concat(
        [out, params.add_prefix("b_"), tvals.add_prefix("t_"), localR2],
        axis=1
    )

    # 保存
    out_path = os.path.join(OUT_DIR, f"GWR_{group_name}_results.parquet")
    out.to_parquet(out_path, index=False)

    txt_path = os.path.join(OUT_DIR, f"GWR_{group_name}_summary.txt")
    with open(txt_path, "w") as f:
        f.write(str(res.summary()))

    # 记录对比指标
    metrics = {
        "model": group_name,
        "k": X.shape[1],
        "bw": float(bw),
        "AICc": float(res.aicc),
        "AIC": float(res.aic),
        "R2": float(res.R2),
        "Adj_R2": float(res.adj_R2),
        "sigma": float(res.sigma2),
        "out_path": out_path,
        "summary_path": txt_path,
    }
    return metrics

# =========================
# 4) LOOP ALL GROUPS
# =========================
metrics_list = []
for g, cols in GROUPS.items():
    print(f"\n=== Running group: {g} ===")
    m = run_gwr_for_group(g, cols)
    metrics_list.append(m)

compare = pd.DataFrame(metrics_list).sort_values("AICc")
compare_path = os.path.join(OUT_DIR, "GWR_group_compare.csv")
compare.to_csv(compare_path, index=False)

print("\n✅ DONE.")
print("Saved compare table:", compare_path)
print(compare)

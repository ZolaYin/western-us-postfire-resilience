import pandas as pd
from libpysal.weights import KNN

INP = "outputs_resistance/MGWR_Resistance_with_residual_bwmin20.parquet"

df = pd.read_parquet(INP)
# 你这个文件里坐标列是 x_beta/y_beta（你之前 merge 后就是这样）
coords = df[["x_beta", "y_beta"]].astype(float).to_numpy()

def comp_stats(w):
    labels = w.component_labels
    n_comp = len(set(labels))
    # 最大组件大小
    import pandas as pd
    sizes = pd.Series(labels).value_counts().sort_values(ascending=False)
    return n_comp, int(sizes.iloc[0]), sizes.head(10).tolist()

print("n =", coords.shape[0])

for k in [4,6,8,10,12,15,20,30,40]:
    w = KNN.from_array(coords, k=k)
    n_comp, max_size, top10 = comp_stats(w)
    print(f"k={k:>2}  n_components={n_comp:<4}  largest_comp={max_size:<6}  top10_sizes={top10}")

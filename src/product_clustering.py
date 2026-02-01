from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def build_product_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    sales_daily.csv（日次の取引明細/集計）から、
    商品（product_code）単位の特徴量テーブルを作る。
    """
    # --- 基本の安全処理 ---
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 0除算を避けつつ、行単位の指標を作る
    # AOV = sales_amount / transactions（客単価）
    # UPT = sales_qty / transactions（1客あたり点数）
    df["aov"] = np.where(df["transactions"] > 0, df["sales_amount"] / df["transactions"], np.nan)
    df["upt"] = np.where(df["transactions"] > 0, df["sales_qty"] / df["transactions"], np.nan)

    # --- 商品ごとに集計（特徴量化） ---
    g = df.groupby("product_code", as_index=True)

    feat = pd.DataFrame(
        {
            # 売上の「水準」
            "mean_daily_amount": g["sales_amount"].mean(),
            "mean_daily_qty": g["sales_qty"].mean(),
            "mean_daily_txn": g["transactions"].mean(),

            # 売上の「安定性」：標準偏差 / 平均（変動係数CV）
            "cv_amount": g["sales_amount"].std(ddof=0) / (g["sales_amount"].mean() + 1e-9),
            "cv_qty": g["sales_qty"].std(ddof=0) / (g["sales_qty"].mean() + 1e-9),

            # 客単価・点数（商品が“高単価寄りか/まとめ買いされるか”の傾向）
            "mean_aov": g["aov"].mean(),
            "mean_upt": g["upt"].mean(),
        }
    )

    # gender比率（Men / Women）も特徴量にする：Men比率（0〜1）
    # 例：Men売上が多い商品なのか、Women中心なのか
    if "gender" in df.columns:
        pivot = (
            df.pivot_table(
                index="product_code",
                columns="gender",
                values="sales_amount",
                aggfunc="sum",
                fill_value=0,
            )
        )
        # Men列が無い可能性もあるので安全に
        men_amt = pivot["Men"] if "Men" in pivot.columns else 0
        total_amt = pivot.sum(axis=1)
        feat["men_share_amount"] = np.where(total_amt > 0, men_amt / total_amt, 0.5)

    # 欠損が出たら、中央値で埋める（KMeansは欠損NG）
    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.fillna(feat.median(numeric_only=True))

    return feat


def main() -> None:
    # ====== ここだけ自分の環境に合わせてOK ======
    # 例：sales_dashboard/data/sales_daily.csv
    CSV_PATH = Path("data/sales_daily.csv")  # ←ここを必要なら修正
    OUT_DIR = Path("outputs")
    K = 3  # まずは3クラスタで。あとで2〜6で試す
    SEED = 42
    # =========================================

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH.resolve()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    # 1) 特徴量作成（商品単位）
    feat = build_product_features(df)

    # 2) 標準化（超重要：尺度が違うとクラスタが壊れる）
    scaler = StandardScaler()
    X = scaler.fit_transform(feat.values)

    # 3) KMeansでクラスタリング
    km = KMeans(n_clusters=K, random_state=SEED, n_init=10)
    labels = km.fit_predict(X)

    result = feat.copy()
    result["cluster"] = labels

    # 4) cluster別の平均値（解釈用テーブル）
    cluster_profile = result.groupby("cluster").mean(numeric_only=True).round(3)

    # 5) 2次元に圧縮して散布図（PCA）
    pca = PCA(n_components=2, random_state=SEED)
    X2 = pca.fit_transform(X)
    plot_df = pd.DataFrame(X2, columns=["pc1", "pc2"], index=result.index)
    plot_df["cluster"] = labels

    # ====== 出力 ======
    result_path = OUT_DIR / "product_clusters.csv"
    profile_path = OUT_DIR / "cluster_profile.csv"
    fig_path = OUT_DIR / "cluster_scatter_pca.png"

    result.sort_values(["cluster", "mean_daily_amount"], ascending=[True, False]).to_csv(result_path)
    cluster_profile.to_csv(profile_path)

    # プロット
    plt.figure()
    for c in sorted(plot_df["cluster"].unique()):
        d = plot_df[plot_df["cluster"] == c]
        plt.scatter(d["pc1"], d["pc2"], label=f"cluster {c}")
    plt.title("Product Clusters (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    # コンソール出力（ここが面接で説明しやすい）
    print("=== Saved ===")
    print(result_path)
    print(profile_path)
    print(fig_path)

    print("\n=== Cluster Profile (mean) ===")
    print(cluster_profile)

    print("\n=== Sample: top products per cluster (by mean_daily_amount) ===")
    for c in sorted(result["cluster"].unique()):
        top = result[result["cluster"] == c].sort_values("mean_daily_amount", ascending=False).head(5)
        print(f"\n[cluster {c}]")
        print(top[["mean_daily_amount", "mean_daily_qty", "mean_aov", "mean_upt"]])


if __name__ == "__main__":
    main()

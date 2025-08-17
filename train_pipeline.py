# train_pipeline.py
# EDA (plots), cleaning, training (trending + content-sim + also-bought), and artifact export

import os
import argparse
import json
import logging
from typing import List, Dict, Tuple
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ CONFIG ------------------
OUTPUT_MODELS_DIR = "models"
OUTPUT_REPORTS_DIR = "reports"
OUTPUT_DATA_DIR   = "data_out"

REQUIRED_COLS = [
    "order_id",
    "gender_category",
    "product_line",
    "product_name",
    "size",
    "units_sold",
    "mrp",
    "discount_applied",
    "revenue",
    "order_date",
    "sales_channel",
    "region",
    "profit",
]

# ------------------ LOGGING ------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("train")

# ------------------ UTILS ------------------
def ensure_dirs():
    for d in [OUTPUT_MODELS_DIR, OUTPUT_REPORTS_DIR, OUTPUT_DATA_DIR]:
        os.makedirs(d, exist_ok=True)

def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    # string-ish
    for c in ["gender_category","product_line","product_name","size","sales_channel","region"]:
        df[c] = df[c].astype(str).str.strip()
    # numbers
    for c in ["units_sold","mrp","discount_applied","revenue","profit"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    # order_id as int if possible
    df["order_id"] = pd.to_numeric(df["order_id"], errors="coerce").fillna(-1).astype(int)
    # dates
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    return df

def clip_outliers(df: pd.DataFrame, cols: List[str], lo_q: float = 0.01, hi_q: float = 0.99) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        lo = out[c].quantile(lo_q)
        hi = out[c].quantile(hi_q)
        out[c] = out[c].clip(lower=lo, upper=hi)
    return out

def safe_minmax(s: pd.Series) -> pd.Series:
    s = s.fillna(0)
    if s.nunique(dropna=True) <= 1:
        return pd.Series(0.0, index=s.index)
    return (s - s.min()) / (s.max() - s.min())

# ------------------ EDA PLOTS ------------------
def plot_bar(series: pd.Series, title: str, ylabel: str, fname: str):
    plt.figure()
    series.plot(kind="bar")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_REPORTS_DIR, fname))
    plt.close()

def plot_line(x, y, title: str, ylabel: str, fname: str):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_REPORTS_DIR, fname))
    plt.close()

def run_eda(df: pd.DataFrame):
    # 1) Revenue by gender
    rev_by_gender = df.groupby("gender_category")["revenue"].sum().sort_values(ascending=False)
    plot_bar(rev_by_gender, "Revenue by Gender", "Revenue", "rev_by_gender.png")

    # 2) Revenue by product line
    rev_by_line = df.groupby("product_line")["revenue"].sum().sort_values(ascending=False)
    plot_bar(rev_by_line, "Revenue by Product Line", "Revenue", "rev_by_product_line.png")

    # 3) Top products by revenue (top 10)
    top_prod = df.groupby("product_name")["revenue"].sum().sort_values(ascending=True).tail(10)
    plt.figure()
    top_prod.plot(kind="barh")
    plt.title("Top 10 Products by Revenue")
    plt.xlabel("Revenue")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_REPORTS_DIR, "top10_products_revenue.png"))
    plt.close()

    # 4) Monthly revenue trend
    df_m = df.dropna(subset=["order_date"]).copy()
    if not df_m.empty:
        df_m["ym"] = df_m["order_date"].dt.to_period("M").dt.to_timestamp()
        monthly = df_m.groupby("ym")["revenue"].sum().sort_index()
        plot_line(monthly.index, monthly.values, "Monthly Revenue Trend", "Revenue", "monthly_revenue_trend.png")

# ------------------ TRAINING ------------------
def train_trending(df: pd.DataFrame, w_revenue: float = 0.7, w_profit: float = 0.3) -> pd.DataFrame:
    agg_cols = ["gender_category", "product_line", "product_name"]
    trending = (
        df.groupby(agg_cols, as_index=False)
          .agg(
              total_units=("units_sold","sum"),
              total_revenue=("revenue","sum"),
              total_profit=("profit","sum"),
              avg_discount=("discount_applied","mean"),
              mrp_median=("mrp","median"),
          )
    )
    trending["revenue_norm"] = safe_minmax(trending["total_revenue"])
    trending["profit_norm"]  = safe_minmax(trending["total_profit"])
    trending["popularity_score"] = w_revenue * trending["revenue_norm"] + w_profit * trending["profit_norm"]
    trending_sorted = trending.sort_values(
        ["gender_category","product_line","popularity_score"],
        ascending=[True, True, False]
    ).reset_index(drop=True)
    return trending_sorted

def build_text(row: pd.Series) -> str:
    return " ".join([
        str(row["product_name"]),
        str(row["product_line"]),
        str(row["gender_category"]),
        f"size_{row['size']}",
    ])

def train_content_similarity(df: pd.DataFrame) -> Tuple[pd.DataFrame, TfidfVectorizer, np.ndarray, Dict[str, List[int]]]:
    items = (
        df[["product_name","product_line","gender_category","size"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    items["text"] = items.apply(build_text, axis=1)

    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X = vectorizer.fit_transform(items["text"])
    cos = cosine_similarity(X)

    name_to_indices: Dict[str, List[int]] = {}
    for i, name in enumerate(items["product_name"]):
        name_to_indices.setdefault(str(name).lower(), []).append(i)

    return items, vectorizer, cos, name_to_indices

def train_also_bought(df: pd.DataFrame, top_n: int = 20) -> dict:
    """Simple co-occurrence (lift-like) neighbors per product_name using order_id baskets."""
    baskets = (
        df.groupby("order_id")["product_name"]
          .apply(lambda s: sorted(set(map(str, s))))
          .tolist()
    )
    freq = Counter()
    co = Counter()
    for items in baskets:
        for i in range(len(items)):
            a = items[i]
            freq[a] += 1
            for j in range(i+1, len(items)):
                b = items[j]
                pair = tuple(sorted((a, b)))
                co[pair] += 1

    neighbors = defaultdict(list)
    for (a, b), c in co.items():
        # "lift-like" score (bounded, simple)
        score = c / (freq[a] * freq[b])
        neighbors[a].append((b, score, c))
        neighbors[b].append((a, score, c))

    out = {}
    for k, lst in neighbors.items():
        top = sorted(lst, key=lambda x: (-x[1], -x[2]))[:top_n]
        out[k.lower()] = [
            {"product_name": n, "score": float(s), "co_count": int(c)} for (n, s, c) in top
        ]
    return out

# ------------------ PERSIST ------------------
def save_artifacts(trending_sorted: pd.DataFrame,
                   items: pd.DataFrame,
                   vectorizer: TfidfVectorizer,
                   cosine_sim: np.ndarray,
                   also_bought: dict,
                   meta: Dict):
    trending_path = os.path.join(OUTPUT_MODELS_DIR, "trending_sorted.csv")
    items_path     = os.path.join(OUTPUT_MODELS_DIR, "items_df.joblib")
    vec_path       = os.path.join(OUTPUT_MODELS_DIR, "tfidf_vectorizer.joblib")
    cos_path       = os.path.join(OUTPUT_MODELS_DIR, "cosine_sim.npy")
    meta_path      = os.path.join(OUTPUT_MODELS_DIR, "meta.json")
    also_path      = os.path.join(OUTPUT_MODELS_DIR, "also_bought.json")
    catalog_path   = os.path.join(OUTPUT_MODELS_DIR, "catalog_options.json")

    trending_sorted.to_csv(trending_path, index=False)
    joblib.dump(items, items_path)
    joblib.dump(vectorizer, vec_path)
    np.save(cos_path, cosine_sim)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(also_path, "w", encoding="utf-8") as f:
        json.dump(also_bought, f, indent=2)

    # Save catalog options for UI/bootstrap
    genders = sorted(trending_sorted["gender_category"].dropna().astype(str).unique().tolist())
    lines   = sorted(trending_sorted["product_line"].dropna().astype(str).unique().tolist())
    products = sorted(items["product_name"].dropna().astype(str).unique().tolist())
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump({"genders": genders, "product_lines": lines, "product_names": products}, f, indent=2)

# ------------------ MAIN ------------------
def main():
    parser = argparse.ArgumentParser(description="EDA → Clean → Train recommender artifacts")
    parser.add_argument("--csv", type=str, default="Nike_Sales_Cleaned.csv", help="Path to the input CSV")
    parser.add_argument("--w_revenue", type=float, default=0.7, help="Weight for revenue in popularity score")
    parser.add_argument("--w_profit", type=float, default=0.3, help="Weight for profit in popularity score")
    parser.add_argument("--also_topn", type=int, default=20, help="Top-N neighbors to keep in also-bought")
    args = parser.parse_args()

    ensure_dirs()

    # Load + clean
    log.info("Loading CSV...")
    df = read_csv(args.csv)
    df = coerce_types(df)

    log.info("Clipping numeric outliers...")
    df = clip_outliers(df, cols=["units_sold","mrp","discount_applied","revenue","profit"])

    # Save cleaned snapshot
    cleaned_path = os.path.join(OUTPUT_DATA_DIR, "cleaned.csv")
    df.to_csv(cleaned_path, index=False)

    # EDA plots
    log.info("Creating EDA reports...")
    run_eda(df)

    # Training
    log.info("Training trending model...")
    trending_sorted = train_trending(df, w_revenue=args.w_revenue, w_profit=args.w_profit)

    log.info("Training content-similarity model...")
    items, vectorizer, cosine_sim, name_to_indices = train_content_similarity(df)

    log.info("Training also-bought neighbors...")
    also_bought = train_also_bought(df, top_n=args.also_topn)

    # Persist artifacts
    meta = {
        "source_csv": os.path.basename(args.csv),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(df.shape[0]),
        "n_items_catalog": int(items.shape[0]),
        "weights": {"revenue": args.w_revenue, "profit": args.w_profit},
        "also_topn": args.also_topn
    }
    log.info("Saving artifacts...")
    save_artifacts(trending_sorted, items, vectorizer, cosine_sim, also_bought, meta)

    # Console summary
    print("=== DONE ===")
    print(f"Cleaned data  → {cleaned_path}")
    print(f"Reports       → {OUTPUT_REPORTS_DIR}/*.png")
    print(f"Models        → {OUTPUT_MODELS_DIR}/ (trending_sorted.csv, items_df.joblib, tfidf_vectorizer.joblib, cosine_sim.npy, also_bought.json, meta.json, catalog_options.json)")
    # Quick sample output
    print("\nSample top-5 trending per first gender x line:")
    try:
        seg = trending_sorted.groupby(["gender_category","product_line"], as_index=False).head(5)
        print(seg.head(15).to_string(index=False))
    except Exception:
        pass

if __name__ == "__main__":
    main()

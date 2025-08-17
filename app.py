# app.py
# FastAPI app serving: catalog, trending/similar/also-bought (with dynamic weights),
# and a static UI + reports. No logging/telemetry.

import os
import json
from functools import lru_cache
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Query, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ------------------ SETTINGS ------------------
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
WEB_DIR = os.environ.get("WEB_DIR", "web")
API_KEY = os.environ.get("API_KEY", "").strip()  # optional; leave empty to disable auth

def require_key(request: Request):
    if not API_KEY:
        return True
    key = request.headers.get("X-API-Key", "")
    if key != API_KEY:
        raise HTTPException(401, detail="Unauthorized")
    return True

# ------------------ LOAD ARTIFACTS ------------------
trending_path = os.path.join(MODELS_DIR, "trending_sorted.csv")
items_path    = os.path.join(MODELS_DIR, "items_df.joblib")
vec_path      = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
cos_path      = os.path.join(MODELS_DIR, "cosine_sim.npy")
also_path     = os.path.join(MODELS_DIR, "also_bought.json")
catalog_path  = os.path.join(MODELS_DIR, "catalog_options.json")

if not (os.path.exists(trending_path) and os.path.exists(items_path) and
        os.path.exists(vec_path) and os.path.exists(cos_path)):
    raise RuntimeError("Model artifacts not found. Run: python train_pipeline.py")

trending = pd.read_csv(trending_path)  # has revenue_norm, profit_norm, popularity_score
items     = joblib.load(items_path)
vectorizer = joblib.load(vec_path)
cosine_sim = np.load(cos_path)

ALSO_BOUGHT: Dict[str, Any] = {}
if os.path.exists(also_path):
    with open(also_path, "r", encoding="utf-8") as f:
        ALSO_BOUGHT = json.load(f)

if os.path.exists(catalog_path):
    with open(catalog_path, "r", encoding="utf-8") as f:
        CATALOG_OPTS = json.load(f)
else:
    CATALOG_OPTS = {
        "genders": sorted(trending["gender_category"].dropna().astype(str).unique().tolist()),
        "product_lines": sorted(trending["product_line"].dropna().astype(str).unique().tolist()),
        "product_names": sorted(items["product_name"].dropna().astype(str).unique().tolist()),
    }

# reverse index: product_name -> indices (case-insensitive)
name_to_indices: Dict[str, List[int]] = {}
for i, name in enumerate(items["product_name"]):
    name_to_indices.setdefault(str(name).lower(), []).append(i)

# ------------------ APP & STATIC ------------------
app = FastAPI(title="Nike Recommender API", version="2.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

os.makedirs(WEB_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
if os.path.isdir("reports"):
    app.mount("/reports", StaticFiles(directory="reports"), name="reports")

@app.get("/", response_class=FileResponse)
def root() -> FileResponse:
    index_path = os.path.join(WEB_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(404, detail=f"UI not found: {index_path}")
    return FileResponse(index_path)

# ------------------ HELPERS ------------------
def _reasons_for_trending(row: Dict[str, Any]) -> List[str]:
    out = [f"Popular in {row.get('product_line')} ({row.get('gender_category')})"]
    try:
        out.append(f"High revenue (${int(row.get('total_revenue', 0)):,})")
    except Exception:
        pass
    return out

def _reasons_for_sim(seed: str) -> List[str]:
    return [f"Similar to “{seed}”"]

@lru_cache(maxsize=2048)
def _also_bought_lookup(name_lower: str, top_k: int):
    lst = ALSO_BOUGHT.get(name_lower, [])
    return lst[:top_k]

def diversify_by_product_line(records: List[Dict[str, Any]], desired_k: int) -> List[Dict[str, Any]]:
    used = set()
    out: List[Dict[str, Any]] = []
    pool = records.copy()
    while pool and len(out) < desired_k:
        pick_idx = None
        for i, r in enumerate(pool):
            if r.get("product_line") not in used:
                pick_idx = i
                break
        if pick_idx is None:
            pick_idx = 0
        chosen = pool.pop(pick_idx)
        out.append(chosen)
        used.add(chosen.get("product_line"))
    return out

# ------------------ ROUTES ------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "items": int(items.shape[0]),
        "segments": int(trending[["gender_category","product_line"]].drop_duplicates().shape[0]),
        "also_bought_items": len(ALSO_BOUGHT),
    }

@app.get("/catalog/options")
def catalog_options() -> Dict[str, Any]:
    return CATALOG_OPTS

@app.get("/recommend/trending", dependencies=[Depends(require_key)])
def recommend_trending(
    gender: str = Query(..., description="e.g., Men, Women, Kids"),
    product_line: Optional[str] = Query(None, description="e.g., Running, Basketball; omit for all"),
    top_k: int = Query(10, ge=1, le=100),
    diversify: bool = Query(True, description="Diversify across product_line"),
    w_revenue: float = Query(0.7, ge=0.0, le=1.0, description="Weight for revenue in score"),
    w_profit: float = Query(0.3, ge=0.0, le=1.0, description="Weight for profit in score"),
) -> Dict[str, Any]:
    df = trending.copy()
    df = df[df["gender_category"].str.lower() == gender.lower()]
    if product_line:
        df = df[df["product_line"].str.lower() == product_line.lower()]
    if df.empty:
        raise HTTPException(404, detail="No results for given filters.")

    # dynamic score from precomputed norms (no retrain required)
    df["score"] = w_revenue * df["revenue_norm"] + w_profit * df["profit_norm"]
    df = df.sort_values("score", ascending=False).head(max(top_k, 30))
    records = df.to_dict(orient="records")

    # tidy numbers a bit (UI still formats nicely)
    for r in records:
        r["total_revenue"] = float(r.get("total_revenue", 0))
        r["total_profit"]  = float(r.get("total_profit", 0))
        r["avg_discount"]  = float(r.get("avg_discount", 0))
        r["score"]         = float(r.get("score", 0))

    if diversify:
        records = diversify_by_product_line(records, top_k)
    else:
        records = records[:top_k]

    for rank, r in enumerate(records, start=1):
        r["rank"] = rank
        r["reasons"] = _reasons_for_trending(r)
    return {"results": records, "weights": {"revenue": w_revenue, "profit": w_profit}}

@app.get("/recommend/similar", dependencies=[Depends(require_key)])
def recommend_similar(
    product_name: str = Query(..., description="Seed product name (exact as in catalog)"),
    gender: Optional[str] = Query(None, description="Optional filter: Men/Women/Kids/etc"),
    top_k: int = Query(10, ge=1, le=100),
) -> Dict[str, Any]:
    key = product_name.lower().strip()
    if key not in name_to_indices:
        raise HTTPException(404, detail="Seed product not found in items catalog.")
    idxs = name_to_indices[key]
    sims = cosine_sim[idxs].mean(axis=0)
    order = np.argsort(-sims)

    results = []
    for j in order:
        if j in idxs:
            continue
        if gender and str(items.loc[j, "gender_category"]).lower() != gender.lower():
            continue
        results.append({
            "product_name": str(items.loc[j,"product_name"]),
            "product_line": str(items.loc[j,"product_line"]),
            "gender_category": str(items.loc[j,"gender_category"]),
            "size": str(items.loc[j,"size"]),
            "similarity": float(sims[j]),
            "reasons": _reasons_for_sim(product_name),
        })
        if len(results) >= top_k:
            break
    return {"seed": product_name, "results": results}

@app.get("/recommend/also_bought", dependencies=[Depends(require_key)])
def recommend_also_bought(
    product_name: str = Query(..., description="Seed product (exact as in catalog)"),
    top_k: int = Query(10, ge=1, le=50),
):
    key = product_name.lower().strip()
    neighbors = _also_bought_lookup(key, top_k)
    if not neighbors:
        raise HTTPException(404, detail="No also-bought neighbors for this product.")
    results = []
    for n in neighbors:
        results.append({
            "product_name": n["product_name"],
            "score": float(n["score"]),
            "co_count": int(n["co_count"]),
            "reasons": [f"Frequently bought with “{product_name}”"],
        })
    return {"seed": product_name, "results": results}

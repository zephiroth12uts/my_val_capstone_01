"""myVal recommender web app.

Run from the project root:  uv run uvicorn server:app --reload
Then open http://127.0.0.1:8000
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from fastapi import Body, FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402

from my_val_capstone_01.modeling.predict import predict_next_categories  # noqa: E402

app = FastAPI(title="myVal Next-Category Recommender")
CATS = [f"CAT-00{i}" for i in range(1, 8)]


@app.get("/")
def index():
    return FileResponse(ROOT / "index.html")


@app.post("/api/predict")
def predict(payload: dict = Body(...)):
    row = dict(payload)
    try:
        # Totals are derived from the per-category counts, as in the training data.
        doc = [float(row.get(f"documented_count_{c}", 0)) for c in CATS]
        ai = [float(row.get(f"ai_count_{c}", 0)) for c in CATS]
        row["documented_asset_count"] = sum(doc)
        row["documented_category_count"] = sum(d > 0 for d in doc)
        row["ai_asset_count"] = sum(ai)
        return predict_next_categories(pd.DataFrame([row]))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"{type(exc).__name__}: {exc}")

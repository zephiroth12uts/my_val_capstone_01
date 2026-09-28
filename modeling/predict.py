# -------------------------------------------------------------------
# Next-Category Recommendation Inference
# -------------------------------------------------------------------

from pathlib import Path

import joblib
import pandas as pd


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

TOP1_MODEL_PATH = (
    MODELS_DIR / "next_category_top1_random_forest.joblib"
)

RANKING_MODEL_PATH = (
    MODELS_DIR / "next_category_ranking_random_forest.joblib"
)


# -------------------------------------------------------------------
# Load Persisted Models
# -------------------------------------------------------------------

top1_model = joblib.load(TOP1_MODEL_PATH)
ranking_model = joblib.load(RANKING_MODEL_PATH)


# -------------------------------------------------------------------
# Features Excluded From Prediction
# -------------------------------------------------------------------

NON_MODEL_COLUMNS = [
    "Property_ID",
    "Customer_ID",
    "Prediction_Time",
]


# -------------------------------------------------------------------
# Prediction Function
# -------------------------------------------------------------------

def predict_next_categories(
    feature_row: pd.DataFrame,
) -> dict:
    """
    Predict the next recommended household categories.

    Parameters
    ----------
    feature_row : pd.DataFrame
        A single-row DataFrame containing the same features used
        during model training.

    Returns
    -------
    dict
        Top-1 recommendation together with Top-2 and Top-3 ranked
        recommendations and their predicted probabilities.
    """

    if len(feature_row) != 1:
        raise ValueError(
            "feature_row must contain exactly one row."
        )

    # Remove identifiers and metadata excluded during training.
    model_input = feature_row.drop(
        columns=NON_MODEL_COLUMNS,
        errors="ignore",
    ).copy()

    # ---------------------------------------------------------------
    # Top-1 Recommendation
    # ---------------------------------------------------------------

    top1_category = top1_model.predict(
        model_input
    )[0]

    top1_probabilities = top1_model.predict_proba(
        model_input
    )[0]

    top1_classes = top1_model.named_steps[
        "classifier"
    ].classes_

    top1_probability_map = dict(
        zip(
            top1_classes,
            top1_probabilities,
        )
    )

    top1_probability = top1_probability_map[
        top1_category
    ]

    # ---------------------------------------------------------------
    # Ranked Recommendations
    # ---------------------------------------------------------------

    ranking_probabilities = ranking_model.predict_proba(
        model_input
    )[0]

    ranking_classes = ranking_model.named_steps[
        "classifier"
    ].classes_

    ranked_results = sorted(
        zip(
            ranking_classes,
            ranking_probabilities,
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    # ---------------------------------------------------------------
    # API-Friendly Response
    # ---------------------------------------------------------------

    return {
        "top_1": {
            "category": str(top1_category),
            "probability": round(
                float(top1_probability),
                4,
            ),
        },
        "top_2": [
            {
                "category": str(category),
                "probability": round(
                    float(probability),
                    4,
                ),
            }
            for category, probability
            in ranked_results[:2]
        ],
        "top_3": [
            {
                "category": str(category),
                "probability": round(
                    float(probability),
                    4,
                ),
            }
            for category, probability
            in ranked_results[:3]
        ],
    }
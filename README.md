# myVal Next-Category Recommendation

Machine learning prototype developed for the myVal Capstone Project.

The objective of this project is to recommend the **next household-content category** that may be relevant for a customer to document, using the information already available within myVal.

The recommendation is sequential: as the customer adds more assets and the household state evolves, the model can generate a new recommendation.

---

## Project Objective

The modelling task is defined as:

> Given the information already documented and analysed for a household, predict the next previously undocumented category that may be relevant for the customer to capture.

The model combines:

- documented household assets;
- existing myVal AI-analysis outputs;
- customer and household information;
- property context;
- optional property-video information.

The system is designed as a recommendation tool, not as a claim that a category is definitely missing.

---

## Main Workflow

```text
Customer documents an asset
        ↓
myVal AI analyses the asset
        ↓
Household state is updated
        ↓
Next-category recommendation model
        ↓
Top-1 / Top-2 / Top-3 recommendations
```

A single property can generate multiple prediction opportunities as more assets are documented.

---

## Repository Structure

The project follows a Cookiecutter Data Science structure.

```text
.
├── data/
│   ├── processed/
│   │   └── model_state.csv
│   └── raw/
│       └── myVal_Synthetic_Datasets_release_v2.xlsx
│
├── models/
│   ├── next_category_top1_random_forest.joblib
│   └── next_category_ranking_random_forest.joblib
│
├── my_val_capstone_01/
│   └── modeling/
│       ├── predict.py
│       └── train.py
│
├── notebooks/
│   ├── 01_fm_data_inventory.ipynb
│   ├── 02_fm_target_definition.ipynb
│   ├── 03_fm_modelling.ipynb
│   └── 04_fm_inference.ipynb
│
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Notebook Overview

### `01_fm_data_inventory.ipynb`

Initial exploration of the source workbook.

Main tasks:

- workbook and table inspection;
- key and relationship validation;
- category distribution analysis;
- temporal sequence exploration;
- AI-analysis and video coverage checks.

### `02_fm_target_definition.ipynb`

Construction of the supervised learning target.

The target was defined as the **next previously undocumented category** for a property.

This notebook also:

- reconstructs historical household states;
- handles simultaneous category events;
- prevents future information from entering the predictors;
- combines asset, AI, customer, property and video information;
- creates the final processed modelling dataset.

Final processed dataset:

```text
431 prediction states
234 properties
60 columns
```

### `03_fm_modelling.ipynb`

Model development and evaluation.

The dataset was split by `Property_ID` to ensure that historical states from the same household did not appear across train, validation and test partitions.

Models evaluated included:

- Majority-Class Baseline
- Logistic Regression
- Random Forest
- Tuned CatBoost
- Regularised CatBoost
- Regularised Random Forest

The notebook also includes:

- overfitting analysis;
- grouped cross-validation;
- hyperparameter tuning;
- AI-feature ablation;
- feature importance;
- permutation importance;
- final holdout evaluation.

### `04_fm_inference.ipynb`

Validation of the persisted models and reusable inference function.

The inference pipeline returns:

- Top-1 recommended category;
- Top-2 ranked recommendations;
- Top-3 ranked recommendations;
- predicted probabilities.

Reusable inference logic is located in:

```text
my_val_capstone_01/modeling/predict.py
```

---

## Data Leakage Prevention

Data leakage prevention was a key part of the modelling design.

The main controls were:

- features only use information available at the historical prediction point;
- future asset and category information is excluded;
- simultaneous category events are not artificially ordered;
- preprocessing is fitted only on training data;
- train, validation and test sets are grouped by `Property_ID`;
- the final test set remained untouched during model selection and tuning.

The grouped split produced:

```text
Train:       309 rows | 163 properties
Validation:   67 rows |  35 properties
Test:         55 rows |  36 properties
```

Property overlap between all partitions was zero.

---

## Modelling Results

The strongest Top-1 validation result came from the standard Random Forest.

However, the model showed substantial overfitting, so a regularised Random Forest was also evaluated for ranking recommendations.

### Final Holdout Results

**Top-1 Random Forest**

```text
Accuracy:    0.5091
Macro F1:    0.3223
Weighted F1: 0.4431
```

**Regularised Random Forest Ranking**

```text
Top-2 Accuracy: 0.7091
Top-3 Accuracy: 0.8364
```

These results suggest that the problem works particularly well as a **ranked recommendation task**, where the model provides several relevant category suggestions instead of relying only on one prediction.

---

## AI Contribution

An ablation experiment was performed to test whether the existing myVal AI outputs added predictive value.

With AI features:

```text
Validation Macro F1: 0.3676
```

Without AI features:

```text
Validation Macro F1: 0.3324
```

This suggests that the existing AI-analysis outputs provide useful additional information for the recommendation model.

---

## Feature Interpretation

Random Forest feature importance and validation permutation importance were used to understand which variables contributed most strongly to the predictions.

Important signals included:

- documented category composition;
- number of assets already documented;
- documented household value;
- AI-detected category counts;
- AI estimated asset values;
- AI suggestion acceptance;
- AI category agreement;
- property valuation;
- customer and household context;
- selected video-derived information.

The strongest validation permutation signal was the existing documented category profile.

The analysis was used for interpretation only and should not be treated as evidence of causal relationships.

---

## Final Model Selection

Two final models were retained because the project supports both a primary recommendation and ranked alternatives.

### Top-1 Recommendation

**Standard Random Forest**

Selected because it achieved the strongest Top-1 validation performance.

### Top-2 / Top-3 Ranking

**Regularised Random Forest**

Selected because it provided stronger ranking behaviour and reduced the generalisation gap.

---

## Per-Class Performance

Performance was not uniform across categories.

The strongest Top-1 performance was observed for:

- Electronics;
- Furniture;
- Other Items;
- Appliances.

The most difficult categories included:

- Clothings;
- Jewellery and Personal Items;
- Vehicles.

These categories had substantially fewer observations, which limited the model's ability to generalise reliably.

---

## Inference

Two trained models are persisted in:

```text
models/
├── next_category_top1_random_forest.joblib
└── next_category_ranking_random_forest.joblib
```

Reusable inference logic is implemented in:

```text
my_val_capstone_01/modeling/predict.py
```

The prediction function returns:

```python
{
    "top_1": {
        "category": "...",
        "probability": ...
    },
    "top_2": [...],
    "top_3": [...]
}
```

An end-to-end inference test was successfully performed using an unseen holdout observation.

Example:

```text
Actual category: Other Items

Top-1:
Electronics

Top-2:
1. Electronics
2. Other Items

Top-3:
1. Electronics
2. Other Items
3. Clothings
```

This example illustrates the value of the ranked recommendation formulation: although the first recommendation was incorrect, the true next category appeared in the second position.

---

## Limitations

The application and dataset are still at an early stage.

The current modelling dataset contains a limited number of historical prediction opportunities, and some categories are underrepresented.

The models also showed noticeable overfitting, especially when model complexity increased.

For this reason, the current implementation should be considered a **technical proof of concept**, rather than a production-ready recommendation system.

Future improvements would benefit from:

- more household histories;
- more real customer interaction data;
- better representation of minority categories;
- additional behavioural signals;
- repeated retraining as the application grows.

---

## Environment

The project uses:

- Python;
- pandas;
- scikit-learn;
- CatBoost;
- Jupyter;
- joblib;
- uv.

Dependencies are managed through:

```text
pyproject.toml
uv.lock
```

---

## Summary

This project demonstrates that myVal's historical household information and existing AI-analysis outputs can be combined to generate useful next-category recommendations.

The strongest results were obtained when the problem was treated as a ranked recommendation task.

The final ranking model placed the observed next category within the Top-2 recommendations in **70.91%** of unseen test cases and within the Top-3 recommendations in **83.64%**.

At the same time, the experiments exposed important limitations related to sample size, class imbalance and overfitting.

The current implementation should therefore be considered a machine learning proof of concept that establishes technical feasibility and provides a foundation for future improvement as more real customer behaviour becomes available.
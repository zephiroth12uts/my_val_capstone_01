Getting started
===============

Install the dependencies with `uv sync` (Python 3.12). The data folder is not tracked by git,
so the raw workbook `myVal_Synthetic_Datasets_release_v2.xlsx` has to be placed in `data/raw/`
before anything is run.

The two models depend on each other, so the notebooks are run in this order:

1. `notebooks/classification/02_fm_target_definition.ipynb` creates `data/processed/model_state.csv`.
2. `notebooks/regression/01_ky` to `07_ky` build and evaluate the household-value regression.
3. `notebooks/classification/03_ky_regression_feature_merging.ipynb` creates
   `data/processed/classification/model_state_with_reg.csv`.
4. `notebooks/classification/04_fm_modelling.ipynb` (about 50 minutes) and then `05_fm_inference.ipynb`.

The regression code can also be run from the project root with
`python -m my_val_regression.dataset`, `python -m my_val_regression.modeling.train`,
`python -m my_val_regression.features` and `python -m my_val_regression.plots`.
See `regression.md` for the details of every notebook and folder.

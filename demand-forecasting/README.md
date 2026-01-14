# 7-Eleven Demand Forecasting (Multi-store / Multi-item) — ML + **Temporal Fusion Transformer (TFT)**



This project builds and compares multiple forecasting approaches for **daily product demand** and culminates in a **deep learning TFT model** as the main solution.  

The focus is **modeling + optimization + comparison** (not descriptive analysis).



---



## 1) Problem \& Goal



Given historical sales for store–item combinations, the goal is to predict future demand over a **90-day horizon**.



Key challenges (what drives the modeling choices):

\- **Multiple related time series** (store × item) with different scales and patterns

\- Strong **seasonality / calendar effects**

\- Need for **robust generalization** (avoid leakage, validate with time-aware splits)

\- Preference for models that can learn **nonlinear effects + interactions** across series



---



## 2) What I built (high level)



I implemented and evaluated forecasting models in three tiers:



### A) Classical time-series baselines (univariate, aggregated series)

\- **Prophet**

\- **SARIMAX / ARIMA-style**



Purpose: provide a **lower-bound** baseline and sanity check.



### B) Feature-based Machine Learning models (tabular formulation)

\- **LightGBM** (baseline + Optuna tuning)

\- **XGBoost** (baseline + Optuna tuning)



Purpose: strong tabular learners using **lag/rolling/calendar features**, trained with **time-series cross-validation** and **early stopping**.



### C) Deep learning main model (panel forecasting)

\- **Temporal Fusion Transformer (TFT)** using `pytorch-forecasting` + PyTorch Lightning



Purpose: a production-grade approach that handles **many series jointly**, learns **long/short-term patterns**, and outputs **probabilistic forecasts** (quantiles).



---



## 3) Evaluation design (how I measured models)



### Holdout strategy (time-aware)

\- Final **90 days** are held out for evaluation (no leakage)

\- Models are trained on earlier data and evaluated on the final window



### Metrics

\- **RMSE**

\- **MAE**

\- **sMAPE** (robust percentage error for demand)



> I do not use accuracy-style metrics because forecasting is regression and errors are asymmetric in business impact.



---



## 4) Feature Engineering for ML models (LightGBM / XGBoost)



To convert forecasting into supervised learning, I created lag/rolling features and calendar signals.



### Feature set v1 (compact, strong baseline)

\- Calendar: `dayofweek`, `month`, `is_weekend`

\- Lags: `lag_1`, `lag_7`, `lag_30`

\- Rolling means (shifted to avoid leakage): `roll_7`, `roll_30`



### Feature set v2 (richer / more aggressive)

\- Trend: `t` (days since start)

\- Calendar: `weekofyear`, `quarter`, `month_end`, etc.

\- More rolling signals: `roll_mean_7/14/28/30`, `roll_std_7/28`, `ewm_14`

\- Multiple lags



**Important note:** richer features don’t always improve performance—v2 can introduce:

\- more noise / higher variance (overfitting risk)

\- feature redundancy

\- sensitivity to the split and scaling



---



## 5) Hyperparameter optimization (Optuna)



### LightGBM tuning

\- Optimized with **Optuna** using **time-series cross-validation** (TimeSeriesSplit)

\- Early stopping with high `n_estimators` and pruning

\- Tuned parameters include:

&nbsp; - `learning_rate`, `num_leaves`, `max_depth`

&nbsp; - `min_data_in_leaf`, `feature_fraction`, `bagging_fraction`, `bagging_freq`

&nbsp; - `lambda_l1`, `lambda_l2`



### XGBoost tuning

\- Optimized with **Optuna** using time-series CV

\- Early stopping based on validation RMSE

\- Tuned parameters include:

&nbsp; - tree complexity: `max_depth`, `min_child_weight`

&nbsp; - learning rate: `learning_rate`

&nbsp; - regularization: `reg_alpha`, `reg_lambda`

&nbsp; - subsampling: `subsample`, `colsample_bytree`

&nbsp; - split threshold: `gamma`



---



## 6) Model comparison (results summary)



### Aggregated series (Prophet / SARIMAX / LightGBM / XGBoost)

Below is the final holdout performance (90-day horizon):

| Model | RMSE | MAE | sMAPE | Notes |
|------|------|-----|-------|-------|
| Prophet (baseline) | 9141.43 | 8047.37 | 26.74 | Not suited here, large errors |
| SARIMAX (baseline) | 4465.52 | 3515.09 | 14.06 | Better than Prophet but still weak |
| LightGBM (baseline, FE v1) | 991.28 | 689.76 | 2.68 | Strong jump vs classical models |
| LightGBM (Optuna tuned, FE v1) | **748.37** | **538.31** | **2.08** | Best among tabular models |
| LightGBM (FE v2 baseline) | 1050.41 | 722.92 | 2.88 | Rich features did not help baseline |
| LightGBM (FE v2 tuned) | 2032.34 | 1244.25 | 4.66 | Overfit / unstable tuning |
| LightGBM (FE v2 tuned v2) | 788.63 | 553.21 | 2.16 | Improved but still behind FE v1 tuned |
| XGBoost (baseline) | 1274.80 | 762.25 | 2.92 | Underperforms tuned LGBM |
| XGBoost (Optuna tuned) | 1220.09 | 779.31 | 3.09 | Small gains |
| XGBoost (Optuna tuned v2) | 1030.70 | 650.69 | 2.62 | Better but still behind tuned LGBM |



**Takeaway:**  

\- Classical models (Prophet/SARIMAX) struggled on this dataset.  

\- Feature-based ML is much stronger; **LightGBM tuned (FE v1)** is the best tabular model.



---



## 7) Why TFT is the main model in this project



Tabular GBMs are excellent, but they rely on **handcrafted features** and typically operate on one aggregated series or a limited formulation.  

For the real business setting (store × item), we need a model that can learn patterns across **many related series** and produce stable multi-horizon forecasts.



### TFT advantages (why it stands out)

\- **Global model** across many store–item series (learns shared structure)

\- Combines:

&nbsp; - RNN-style temporal processing

&nbsp; - **attention** for long-range dependencies

&nbsp; - **gating + variable selection** for robustness

\- Supports **probabilistic forecasting** via **QuantileLoss**

\- Can incorporate **static categoricals** (store, item) + known covariates (calendar)



### TFT setup used in this notebook

\- Forecast horizon: **90 days**

\- Encoder length: **365 days**

\- Inputs:

&nbsp; - static categoricals: `store`, `item`

&nbsp; - known reals: `time_idx`, `dayofweek`, `month`, `is_weekend`

&nbsp; - unknown reals: `sales`

\- Normalization: `GroupNormalizer` per series (`softplus`)

\- Loss: `QuantileLoss` (multi-quantile output)

\- Training:

&nbsp; - Early stopping on `val_loss`

&nbsp; - Checkpoint best epoch

&nbsp; - Gradient clipping

&nbsp; - Mixed precision (`bf16-mixed`) when available



### TFT sanity check vs naive baseline (panel forecasting)

To ensure TFT improvement is real and not leakage/mismatched windows, the notebook includes a naive baseline:



| Model | RMSE | MAE |

|---|---:|---:|

| Naive (repeat first value across horizon) | 15.40 | 11.61 |

| TFT | **7.55** | **5.81** |



**Interpretation:** TFT delivers a clear improvement over a naive baseline in the **panel setting**.



> Note: The TFT panel metrics are not directly comparable to the aggregated-series RMSE values above because the scale/target definition differs (panel series vs aggregated total). The right comparison for TFT is against baselines evaluated on the same panel formulation (e.g., naive, simple global models, or per-series classical baselines).



---





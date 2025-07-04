# import time
# import pandas as pd
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OrdinalEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.metrics import make_scorer, r2_score
# import joblib
# import numpy as np
#
# # --- Start total timer ----------------------------------------------
# script_start = time.time()
#
# # --- Load data ------------------------------------------------------
# df = pd.read_csv("Indicators.csv")
# X = df.drop("Value", axis=1)
# y = df["Value"]
#
# cat_cols = ['CountryName', 'CountryCode', 'IndicatorName', 'IndicatorCode']
# num_cols = ['Year']
#
# # --- Define pipeline ------------------------------------------------
# pipe = Pipeline([
#     ("prep", ColumnTransformer([
#         ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
#                                unknown_value=-1), cat_cols),
#         ("num", "passthrough", num_cols)
#     ])),
#     ("rf", RandomForestRegressor(
#         random_state=52,
#         n_jobs=-1,
#     ))
# ])
#
# param_dist = {
#     "rf__n_estimators":     [200, 400, 800, 1200],
#     "rf__max_depth":        [None, 10, 15, 20, 30],
#     "rf__min_samples_leaf": [1, 2, 5, 10],
#     "rf__max_features": ["sqrt", "log2", 0.6, 0.8],
# }
#
# # --- Train/test split -----------------------------------------------
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=1
# )
#
# # --- Set up and run RandomizedSearchCV ------------------------------
# search = RandomizedSearchCV(
#     pipe,
#     param_dist,
#     n_iter=3,
#     cv=3,
#     scoring=make_scorer(r2_score),
#     n_jobs=-1,
#     verbose=2,
#     error_score="raise",
#     random_state=52,
# )
#
# print("\n▶ Starting RandomizedSearchCV...")
# cv_start = time.time()
# search.fit(X_train, y_train)
# cv_elapsed = time.time() - cv_start
# print(f"\n✅ RandomizedSearchCV completed in {cv_elapsed:.2f} seconds\n")
#
# # --- Print per‐iteration timings from cv_results_ --------------------
# for idx, (params, mean_t, std_t) in enumerate(zip(
#         search.cv_results_["params"],
#         search.cv_results_["mean_fit_time"],
#         search.cv_results_["std_fit_time"],
#     ), start=1):
#     print(f"Iteration {idx:2d}/{len(search.cv_results_['params']):2d} — "
#           f"{mean_t:.2f}s ± {std_t:.2f}s   →  {params}")
#
# # --- Report best scores ---------------------------------------------
# print(f"\nBest CV R²   : {search.best_score_:.4f}")
# print("Best settings :", search.best_params_)
#
# best_model = search.best_estimator_
# print(f"Test-set R²  : {best_model.score(X_test, y_test):.4f}")
#
# # --- Persist the tuned pipeline --------------------------------------
# joblib.dump(best_model,
#             "co2_model_light_new.pkl",
#             compress=3)
# print("✓ Saved tuned pipeline to device")
#
# # save_path = "/co2_model_light.pkl"
# # joblib.dump(pipe, save_path, compress=3)
# # print(f"✓ Saved base pipeline to: {save_path}")
#
# # --- Print total script time -----------------------------------------
# total_elapsed = time.time() - script_start
# print(f"\n Total script run time: {total_elapsed:.2f} seconds")



# # --------OLD
# import time, joblib, pandas as pd, numpy as np
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OrdinalEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import make_scorer, r2_score
#
# # ── enable experimental search ─────────────────────────────────────
# from sklearn.experimental import enable_halving_search_cv   # noqa
# from sklearn.model_selection import HalvingRandomSearchCV
# from sklearn.metrics import (
#     r2_score,
#     mean_absolute_error,
#     mean_squared_error,
# )
#
# # ── timing helper ──────────────────────────────────────────────────
# t0 = time.time()
#
# # ── data ───────────────────────────────────────────────────────────
# df = pd.read_csv("Indicators.csv")
# print(f"Length: {len(df.values)}")
# X, y = df.drop("Value", axis=1), df["Value"]
#
# cat_cols = ['CountryName', 'CountryCode', 'IndicatorName', 'IndicatorCode']
# num_cols = ['Year']
#
# # ── constants that tame thermals ───────────────────────────────────
# BASE_TREES  = 100           # first stage
# MAX_TREES   = 600           # final stage
# CPU_CORES   = 4             # ≤ high-perf cores on your M3 Pro
#
# # ── pipeline (encoder + RF) ────────────────────────────────────────
# base_rf = RandomForestRegressor(
#     n_estimators=BASE_TREES,      # will be multiplied by Halving search
#     random_state=52,
#     n_jobs=CPU_CORES,
# )
#
# pipe = Pipeline([
#     ("prep", ColumnTransformer([
#         ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
#                                unknown_value=-1), cat_cols),
#         ("num", "passthrough", num_cols)
#     ])),
#     ("rf", base_rf)
# ])
#
# # ── search space (NO n_estimators here!) ───────────────────────────
# param_dist = {
#     "rf__max_depth":        [None, 12, 18],
#     "rf__min_samples_leaf": [1, 3, 8],
#     "rf__max_features":     ["sqrt", 0.7],
# }
#
# # ── split ──────────────────────────────────────────────────────────
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=1
# )
#
# # ── HalvingRandomSearchCV set-up ───────────────────────────────────
# search = HalvingRandomSearchCV(
#     pipe,
#     param_dist,
#     factor=3,                       # drop ⅔ configs each round
#     resource="rf__n_estimators",    # grows: 100 ➜ 300 ➜ 600
#     max_resources=MAX_TREES,
#     min_resources=BASE_TREES,
#     scoring=make_scorer(r2_score),
#     cv=3,
#     n_jobs=CPU_CORES,
#     random_state=52,
#     verbose=2,
# )
#
# print("▶ Hyper-parameter search …")
# t_search = time.time()
# search.fit(X_train, y_train)
# print(f"⏱︎ search finished in {time.time()-t_search:.1f}s")
#
# # ── evaluate on hold-out ───────────────────────────────────────────
# def show_metrics(label: str, y_true, y_pred):
#     r2   = r2_score(y_true, y_pred)
#     mae  = mean_absolute_error(y_true, y_pred)
#     rmse = mean_squared_error(y_true, y_pred, squared=False)
#     print(f"{label:<10}  R²: {r2:.4f}   MAE: {mae:.4f}   RMSE: {rmse:.4f}")
#     return {"R2": r2, "MAE": mae, "RMSE": rmse}
#
# print("\n▸ Metrics on training folds (same rows CV used)")
# train_pred = search.best_estimator_.predict(X_train)
# val_metrics = show_metrics("Train /CV", y_train, train_pred)
#
# print("\n▸ Metrics on held-out test set")
# test_pred = search.best_estimator_.predict(X_test)
# test_metrics = show_metrics("Test set", y_test, test_pred)
# import json, pathlib
# pathlib.Path("artifacts").mkdir(exist_ok=True)
# with open("artifacts/metrics.json", "w") as f:
#     json.dump({"validation": val_metrics, "test": test_metrics}, f, indent=2)
#
# # ── save full pipeline
# joblib.dump(search.best_estimator_, "co2_model_light_new.pkl", compress=3)
# print("✔ model saved to co2_model_light_new.pkl")
#
# print(f"Total script time: {time.time()-t0:.1f}s")


"""
train_co2_model.py  —  fast & resumable training on a 5-million-row dataset
──────────────────────────────────────────────────────────────────────────
• 10 % sub-sample for hyper-param search (HalvingRandomSearchCV)       (~×25 faster)
• Caches search results  → rerun picks up where it left off
• Final single fit on full data with best params
──────────────────────────────────────────────────────────────────────────
"""

# import time, json, pathlib, joblib, pandas as pd, numpy as np
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OrdinalEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
#
# # enable experimental search
# from sklearn.experimental import enable_halving_search_cv  # noqa
# from sklearn.model_selection import HalvingRandomSearchCV
#
# # ------------------------------------------------------------------ paths
# ROOT          = pathlib.Path(".")
# ARTIFACT_DIR  = ROOT / "artifacts";  ARTIFACT_DIR.mkdir(exist_ok=True)
# CACHE_DIR     = ROOT / ".cache";     CACHE_DIR.mkdir(exist_ok=True)
# SEARCH_PKL    = ARTIFACT_DIR / "search_result.pkl"
# MODEL_PKL     = ARTIFACT_DIR / "co2_model_light.pkl"
# METRICS_JSON  = ARTIFACT_DIR / "metrics.json"
#
# # ------------------------------------------------------------------ config
# CPU_CORES      = 4
# BASE_TREES     = 100
# MAX_TREES      = 300
# SUB_SAMPLE_FR  = 0.10        # 10 % of rows for search
# RANDOM_STATE   = 52
#
#
# cat_cols = ['CountryName', 'CountryCode', 'IndicatorName', 'IndicatorCode']
# num_cols = ['Year']
#
# # ------------------------------------------------------------------ data
# t0 = time.time()
# print("▶ Loading dataset …")
# df = pd.read_csv("Indicators.csv")
# print(f"   rows = {len(df):,}")
#
# X, y = df.drop("Value", axis=1), df["Value"]
#
# # ------------------------------------------------------------------ subsample for search
# X_sub, _, y_sub, _ = train_test_split(
#     X, y, train_size=SUB_SAMPLE_FR, random_state=RANDOM_STATE
# )
# print(f"   using {len(X_sub):,} rows for hyper-param search")
#
# # ------------------------------------------------------------------ pipeline definition
# encoder = ColumnTransformer([
#     ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
#     ("num", "passthrough", num_cols)
# ])
#
# base_rf = RandomForestRegressor(
#     n_estimators=BASE_TREES,
#     random_state=RANDOM_STATE,
#     n_jobs=CPU_CORES,
# )
#
# pipe = Pipeline([("prep", encoder), ("rf", base_rf)])
#
# param_dist = {
#     "rf__max_depth":        [None, 18],
#     "rf__min_samples_leaf": [1, 3],
#     "rf__max_features":     ["sqrt"],
# }
#
# # ------------------------------------------------------------------ search (resumable)
# if SEARCH_PKL.exists():
#     print("▶ Found cached hyper-param search → loading …")
#     search = joblib.load(SEARCH_PKL)
# else:
#     print("▶ Running HalvingRandomSearchCV …")
#     search = HalvingRandomSearchCV(
#         pipe,
#         param_dist,
#         factor=3,
#         resource="rf__n_estimators",
#         max_resources=MAX_TREES,
#         min_resources=BASE_TREES,
#         cv=2,                              # 2-fold for speed
#         n_jobs=CPU_CORES,
#         random_state=RANDOM_STATE,
#         verbose=2,
#         scoring="r2",
#     )
#     search.fit(X_sub, y_sub)
#     joblib.dump(search, SEARCH_PKL)
#     print("   search cached to", SEARCH_PKL)
#
# print("Best params :", search.best_params_)
# print(f"CV R²       : {search.best_score_:.4f}")
#
# # ------------------------------------------------------------------ split full data once
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=RANDOM_STATE
# )
#
# # ------------------------------------------------------------------ final single fit
# best_params = search.best_params_.copy()
# best_params["n_estimators"] = MAX_TREES        # beef up trees for final model
# best_rf = RandomForestRegressor(
#     warm_start=True,
#     n_estimators=0,                 # start at 0
#     random_state=RANDOM_STATE,
#     n_jobs=-1
# )
# final_pipe = Pipeline([
#     ("prep", search.best_estimator_.named_steps["prep"]),
#     ("rf",   best_rf)
# ])
#
# print("▶ Fitting final model in chunks …")
# for chunk in range(0, MAX_TREES, 100):         # 100-tree steps
#     best_rf.set_params(n_estimators=chunk+100)
#     final_pipe.fit(X_train, y_train)
#     print(f"   …{chunk+100} trees done")
#
#
# # ------------------------------------------------------------------ metrics
# def metrics(y_true, y_pred):
#     mse  = mean_squared_error(y_true, y_pred)   # plain MSE
#     rmse = np.sqrt(mse)                         # convert to RMSE
#     return {
#         "R2"  : r2_score(y_true, y_pred),
#         "MAE" : mean_absolute_error(y_true, y_pred),
#         "RMSE": rmse,
#     }
#
# val_metrics  = metrics(y_train, final_pipe.predict(X_train))
# test_metrics = metrics(y_test,  final_pipe.predict(X_test))
#
# print("\n▸ Train metrics : ", val_metrics)
# print("▸ Test  metrics : ", test_metrics)
#
# with open(METRICS_JSON, "w") as f:
#     json.dump({"validation": val_metrics, "test": test_metrics}, f, indent=2)
#
# # ------------------------------------------------------------------ persist model
# joblib.dump(final_pipe, MODEL_PKL, compress=3)
# print("\n✔ Saved final pipeline →", MODEL_PKL)
# print("⏱︎ Total wall-clock time:", f"{time.time()-t0:.1f} s")


# fast_rf_train.py  –  RAM-lean, early-stopping Random-Forest pipeline
import time, json, pathlib, joblib, pandas as pd, numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV

# ─── paths ─────────────────────────────────────────────────────────
ART = pathlib.Path("artifacts"); ART.mkdir(exist_ok=True)
MODEL_PKL   = ART / "co2_model_light.pkl"
METRICS_JS  = ART / "metrics.json"

# ─── config ────────────────────────────────────────────────────────
CPU_CORES   = 4
BASE_TREES  = 50        # very cheap for first halving round
MAX_TREES   = 600       # upper cap if OOB keeps improving
CHUNK       = 50        # warm-start step
SUB_FR      = 0.10      # 10 % rows for search
SEED        = 52

cat_cols = ['CountryName', 'CountryCode', 'IndicatorName', 'IndicatorCode']
num_cols = ['Year']

t0 = time.time()
print("▶ load data …")
df = pd.read_csv("Indicators.csv")
X, y = df.drop("Value", axis=1), df["Value"]
print("   rows:", len(df))

# ─── sample for hyper-param search ─────────────────────────────────
from sklearn.model_selection import train_test_split
X_sub, _, y_sub, _ = train_test_split(X, y, train_size=SUB_FR,
                                      random_state=SEED)

# ─── pipeline for search ──────────────────────────────────────────
encoder = ColumnTransformer([
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value',
                           unknown_value=-1), cat_cols),
    ('num', 'passthrough', num_cols)
])

search_pipe = Pipeline([
    ('prep', encoder),
    ('rf',  RandomForestRegressor(
              n_estimators=BASE_TREES,
              max_samples=0.6,          # ↓ RAM
              oob_score=True,
              n_jobs=CPU_CORES,
              random_state=SEED))
])

param_dist = {
    "rf__max_depth"       : [None, 12, 18],
    "rf__min_samples_leaf": [1, 3],
    "rf__max_features"    : ["sqrt"],
}

print("▶ Hyper-param search …")
search = HalvingRandomSearchCV(
    search_pipe, param_dist,
    factor=3, resource='rf__n_estimators',
    max_resources=BASE_TREES,  # stays 50 in all rounds (cheap)
    min_resources=BASE_TREES,
    cv=2, n_jobs=CPU_CORES, random_state=SEED,
    scoring='r2', verbose=2)
search.fit(X_sub, y_sub)
print("best params:", search.best_params_,
      "CV R²:", f"{search.best_score_:.3f}")

# ─── final warm-start forest with OOB early-stop ───────────────────
best = {k.replace("rf__", ""): v for k, v in search.best_params_.items()}
best.update(dict(max_samples=0.6, oob_score=True,
                 warm_start=True, n_jobs=-1, random_state=SEED,
                 n_estimators=0))

rf = RandomForestRegressor(**best)
pipe = Pipeline([('prep', encoder), ('rf', rf)])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED)

print("▶ growing forest in 50-tree blocks (early-stop on OOB)…")
oob_prev = -1
for n in range(CHUNK, MAX_TREES + CHUNK, CHUNK):
    rf.set_params(n_estimators=n)
    pipe.fit(X_train, y_train)
    oob = rf.oob_score_
    print(f"   {n:3d} trees  |  OOB R² = {oob:.4f}")
    if n >= 150 and (oob - oob_prev) < 0.002:   # plateau
        print("   ↳ OOB gain < 0.002 → stop early")
        break
    oob_prev = oob

# ─── metrics ───────────────────────────────────────────────────────
def metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"R2": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mse)}

val  = metrics(y_train, pipe.predict(X_train))
test = metrics(y_test,  pipe.predict(X_test))
print("train:", val, "\ntest :", test)

json.dump({"val": val, "test": test}, open(METRICS_JS, "w"), indent=2)
joblib.dump(pipe, MODEL_PKL, compress=3)
print("✓ saved", MODEL_PKL, "   ⏱", f"{time.time()-t0:.1f}s total")







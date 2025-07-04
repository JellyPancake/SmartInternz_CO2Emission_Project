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







"""Train and evaluate PD, LGD and EAD models; compute Expected Loss and recovery tiers.

    python src/train_evaluate.py                      # writes benchmarks/metrics.json, artifacts/shap_summary.json

Design (fixed before the first run):

* Development sample = snapshot 2022-06 (performance to 2023-06). Out-of-time (OOT) test = snapshot 2023-06
  (performance to 2024-06). Nothing from the OOT sample is used to fit, calibrate or select anything.
* PD candidates: logistic regression, LightGBM and XGBoost, fixed hyperparameters (no tuning). Each is
  evaluated with stratified 5-fold CV on the development sample, raw and isotonic-calibrated (calibration
  fitted with an inner stratified 3-fold CV). Selection rule: lowest mean CV Brier score of the calibrated
  model. The selected model is refitted on the full development sample (isotonic, 5-fold) and scored OOT.
* LGD and EAD are fitted on development-sample defaults only (segment-mean baseline vs LightGBM, selected
  by 5-fold CV RMSE) and applied to every OOT loan.
* EL_i = PD_i x LGD_i x EAD_i. Backtested against realised OOT losses; loans are ranked into recovery tiers.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score, roc_curve)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_pipeline import (CATEGORICAL_FEATURES, COLUMN_BY_NAME, FEATURE_COLUMNS, NUMERIC_FEATURES,  # noqa: E402
                           GeneratorParams, assert_no_leakage, build_dataset, dataset_sha256, describe_params)

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
MONEY = ("monthly_income", "credit_limit")
OTHER_NUM = tuple(f for f in NUMERIC_FEATURES if f not in MONEY)
TIERS = (  # (name, share of loans, recovery action), ranked by expected loss
    ("Tier 1", 0.05, "Dedicated collector now: restructuring or settlement offer, collateral review"),
    ("Tier 2", 0.10, "Proactive outreach before the next due date; payment-plan options"),
    ("Tier 3", 0.25, "Automated reminders plus agent follow-up on any missed instalment"),
    ("Tier 4", 0.60, "Standard servicing and monitoring"),
)


# --------------------------------------------------------------------------- models


def _linear_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("money", Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                            ("scale", StandardScaler())]), list(MONEY)),
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler())]), list(OTHER_NUM)),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(CATEGORICAL_FEATURES)),
    ])


def _tree_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", "passthrough", list(NUMERIC_FEATURES)),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), list(CATEGORICAL_FEATURES)),
    ], verbose_feature_names_out=False).set_output(transform="pandas")


def make_pd_model(name: str) -> Pipeline:
    if name == "logistic_regression":
        return Pipeline([("pre", _linear_preprocessor()), ("model", LogisticRegression(C=1.0, max_iter=5000))])
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return Pipeline([("pre", _tree_preprocessor()), ("model", LGBMClassifier(
            n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=50, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, n_jobs=4,
            deterministic=True, force_row_wise=True, verbose=-1))])
    if name == "xgboost":
        from xgboost import XGBClassifier

        return Pipeline([("pre", _tree_preprocessor()), ("model", XGBClassifier(
            n_estimators=400, learning_rate=0.03, max_depth=4, min_child_weight=5, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", random_state=SEED, n_jobs=4,
            eval_metric="logloss"))])
    raise ValueError(name)


PD_MODELS = ("logistic_regression", "lightgbm", "xgboost")


def calibrated(name: str, folds: int) -> CalibratedClassifierCV:
    return CalibratedClassifierCV(make_pd_model(name), method="isotonic",
                                  cv=StratifiedKFold(folds, shuffle=True, random_state=SEED))


class SegmentMean(BaseEstimator, RegressorMixin):
    """Baseline: mean target per product."""

    def fit(self, X, y):
        y = pd.Series(np.asarray(y, dtype=float), index=X.index)
        self.means_ = y.groupby(X["product"]).mean().to_dict()
        self.overall_ = float(y.mean())
        return self

    def predict(self, X):
        return X["product"].map(self.means_).fillna(self.overall_).to_numpy(dtype=float)


def make_regressor(name: str, objective: str) -> BaseEstimator:
    if name == "segment_mean":
        return SegmentMean()
    from lightgbm import LGBMRegressor

    return Pipeline([("pre", _tree_preprocessor()), ("model", LGBMRegressor(
        objective=objective, n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, n_jobs=4,
        deterministic=True, force_row_wise=True, verbose=-1))])


# --------------------------------------------------------------------------- metrics


def r(x: float, nd: int = 4) -> float:
    return round(float(x), nd)


def expected_calibration_error(y, p, bins: int = 10) -> float:
    order = np.argsort(p, kind="stable")
    return float(sum(abs(p[c].mean() - y[c].mean()) * len(c) for c in np.array_split(order, bins) if len(c)) / len(y))


def pd_metrics(y, p) -> dict:
    y, p = np.asarray(y), np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    base = y.mean()
    brier = brier_score_loss(y, p)
    return {
        "roc_auc": r(auc), "gini": r(2 * auc - 1), "pr_auc": r(average_precision_score(y, p)),
        "brier": r(brier, 5), "brier_skill_vs_base_rate": r(1 - brier / (base * (1 - base))),
        "log_loss": r(log_loss(y, p)), "ks": r(np.max(tpr - fpr)), "ece_10_bins": r(expected_calibration_error(y, p), 5),
        "mean_pd": r(p.mean()), "observed_default_rate": r(base), "n": int(len(y)), "defaults": int(y.sum()),
    }


def summarize(folds: list[dict]) -> dict:
    keys = ("roc_auc", "pr_auc", "brier", "log_loss", "ece_10_bins")
    return {k: {"mean": r(np.mean([f[k] for f in folds]), 5), "std": r(np.std([f[k] for f in folds]), 5)} for k in keys}


def regression_metrics(y, pred) -> dict:
    return {"mae": r(mean_absolute_error(y, pred)), "rmse": r(np.sqrt(mean_squared_error(y, pred))),
            "r2": r(r2_score(y, pred)), "mean_actual": r(np.mean(y)), "mean_predicted": r(np.mean(pred)), "n": int(len(y))}


def calibration_table(y, p, bins: int = 10) -> list[dict]:
    order = np.argsort(p, kind="stable")
    return [{"decile": i + 1, "n": int(len(c)), "mean_pd": r(p[c].mean()), "observed_default_rate": r(y[c].mean()),
             "defaults": int(y[c].sum())} for i, c in enumerate(np.array_split(order, bins))]


def bootstrap_ci(y, p, n_boot: int = 500) -> dict:
    rng = np.random.default_rng(SEED)
    stats = {"roc_auc": [], "pr_auc": [], "brier": []}
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        stats["roc_auc"].append(roc_auc_score(y[i], p[i]))
        stats["pr_auc"].append(average_precision_score(y[i], p[i]))
        stats["brier"].append(brier_score_loss(y[i], p[i]))
    return {k: [r(np.percentile(v, 2.5), 5), r(np.percentile(v, 97.5), 5)] for k, v in stats.items()} | {"resamples": n_boot}


# --------------------------------------------------------------------------- PD


def evaluate_pd(dev: pd.DataFrame, oot: pd.DataFrame, log) -> tuple[dict, CalibratedClassifierCV, np.ndarray]:
    X, y = dev[list(FEATURE_COLUMNS)], dev["default_flag"].to_numpy()
    outer = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
    cv = {}
    for name in PD_MODELS:
        raw_folds, cal_folds = [], []
        for train, test in outer:
            raw = make_pd_model(name).fit(X.iloc[train], y[train])
            raw_folds.append(pd_metrics(y[test], raw.predict_proba(X.iloc[test])[:, 1]))
            cal = calibrated(name, 3).fit(X.iloc[train], y[train])
            cal_folds.append(pd_metrics(y[test], cal.predict_proba(X.iloc[test])[:, 1]))
        cv[name] = {"raw": summarize(raw_folds), "calibrated": summarize(cal_folds)}
        log(f"  PD CV {name}: AUC {cv[name]['calibrated']['roc_auc']['mean']}  Brier {cv[name]['calibrated']['brier']['mean']}")
    selected = min(PD_MODELS, key=lambda m: cv[m]["calibrated"]["brier"]["mean"])

    Xo, yo = oot[list(FEATURE_COLUMNS)], oot["default_flag"].to_numpy()
    finals, oot_metrics = {}, {}
    for name in PD_MODELS:
        finals[name] = calibrated(name, 5).fit(X, y)
        oot_metrics[name] = pd_metrics(yo, finals[name].predict_proba(Xo)[:, 1])
    p_sel = finals[selected].predict_proba(Xo)[:, 1]

    # Leakage control: same pipeline, shuffled development labels, repeated. A model fitted on shuffled
    # labels is still a random function of risk-related features, so a single draw can land away from 0.5;
    # the mean over draws should not.
    controls = []
    for k in range(5):
        shuffled = np.random.default_rng([SEED, k]).permutation(y)
        controls.append(roc_auc_score(yo, calibrated(selected, 5).fit(X, shuffled).predict_proba(Xo)[:, 1]))

    result = {
        "candidates": list(PD_MODELS),
        "hyperparameters": "fixed in code, not tuned",
        "calibration": "isotonic; inner stratified 3-fold CV during model selection, 5-fold for the final model",
        "selection_rule": "lowest mean 5-fold CV Brier score of the calibrated model (development sample only)",
        "cv_development": cv,
        "selected": selected,
        "oot": oot_metrics,
        "oot_selected_bootstrap_95ci": bootstrap_ci(yo, p_sel),
        "oracle_oot": pd_metrics(yo, oot["oracle_pd"].to_numpy()),
        "oracle_note": "generator's own default probability from hidden state; the ceiling for any model on this data",
        "calibration_table_oot": calibration_table(yo, p_sel),
        "label_shuffle_control_oot_roc_auc": {"draws": [r(c) for c in controls], "mean": r(np.mean(controls))},
    }
    return result, finals, p_sel


# --------------------------------------------------------------------------- LGD / EAD


def evaluate_severity(dev, oot, target: str, objective: str, log) -> tuple[dict, BaseEstimator, str]:
    d_dev, d_oot = dev[dev["default_flag"] == 1], oot[oot["default_flag"] == 1]
    X, y = d_dev[list(FEATURE_COLUMNS)], d_dev[target].to_numpy()
    folds = list(KFold(5, shuffle=True, random_state=SEED).split(X))
    cv, fitted = {}, {}
    for name in ("segment_mean", "lightgbm"):
        scores = [regression_metrics(y[te], np.clip(make_regressor(name, objective).fit(X.iloc[tr], y[tr]).predict(X.iloc[te]), 0, None))
                  for tr, te in folds]
        cv[name] = {k: {"mean": r(np.mean([s[k] for s in scores])), "std": r(np.std([s[k] for s in scores]))}
                    for k in ("mae", "rmse", "r2")}
        fitted[name] = make_regressor(name, objective).fit(X, y)
        log(f"  {target} CV {name}: RMSE {cv[name]['rmse']['mean']}  R2 {cv[name]['r2']['mean']}")
    selected = min(cv, key=lambda m: cv[m]["rmse"]["mean"])
    Xo, yo = d_oot[list(FEATURE_COLUMNS)], d_oot[target].to_numpy()
    oot_metrics = {name: regression_metrics(yo, np.clip(m.predict(Xo), 0, None)) for name, m in fitted.items()}
    return {"trained_on": f"{len(d_dev)} development-sample defaults", "cv_development": cv,
            "selection_rule": "lowest mean 5-fold CV RMSE", "selected": selected,
            "oot_defaults": oot_metrics}, fitted, selected


# --------------------------------------------------------------------------- expected loss and tiers


def expected_loss(oot: pd.DataFrame, pd_hat, lgd_hat, ead_hat) -> tuple[dict, pd.DataFrame]:
    scored = oot[["loan_id", "product", "default_flag", "exposure_base", "realized_loss"]].copy()
    scored["pd"], scored["lgd_hat"], scored["ead_hat"] = pd_hat, lgd_hat, ead_hat
    scored["expected_loss"] = scored["pd"] * scored["lgd_hat"] * scored["ead_hat"]
    scored = scored.sort_values("expected_loss", ascending=False, kind="stable").reset_index(drop=True)
    n = len(scored)
    bounds = np.cumsum([0] + [round(share * n) for _, share, _ in TIERS])
    bounds[-1] = n
    scored["tier"] = ""
    for (name, _, _), lo, hi in zip(TIERS, bounds[:-1], bounds[1:]):
        scored.loc[lo:hi - 1, "tier"] = name
    total_loss = scored["realized_loss"].sum()
    tiers = []
    for name, share, action in TIERS:
        t = scored[scored["tier"] == name]
        tiers.append({"tier": name, "share_of_loans": share, "action": action, "loans": int(len(t)),
                      "predicted_el_inr": r(t["expected_loss"].sum(), 0), "realized_loss_inr": r(t["realized_loss"].sum(), 0),
                      "share_of_realized_loss": r(t["realized_loss"].sum() / total_loss),
                      "mean_pd": r(t["pd"].mean()), "observed_default_rate": r(t["default_flag"].mean())})

    def capture(order_by: str, k: float) -> float:
        top = scored.sort_values(order_by, ascending=False, kind="stable").head(round(k * n))
        return r(top["realized_loss"].sum() / total_loss)

    by_product = [{"product": p, "predicted_el_inr": r(g["expected_loss"].sum(), 0), "realized_loss_inr": r(g["realized_loss"].sum(), 0),
                   "el_to_realized": r(g["expected_loss"].sum() / g["realized_loss"].sum()) if g["realized_loss"].sum() else None}
                  for p, g in scored.groupby("product")]
    return {
        "formula": "EL_i = PD_i x LGD_i x EAD_i (12-month horizon, INR)",
        "oot_portfolio": {"loans": n, "exposure_base_inr": r(scored["exposure_base"].sum(), 0),
                          "predicted_el_inr": r(scored["expected_loss"].sum(), 0), "realized_loss_inr": r(total_loss, 0),
                          "el_to_realized": r(scored["expected_loss"].sum() / total_loss)},
        "by_product": by_product,
        "tiers": tiers,
        "loss_capture_top_share": {f"top_{int(k * 100)}pct": {"by_expected_loss": capture("expected_loss", k), "by_pd_only": capture("pd", k),
                                                              "by_exposure_only": capture("exposure_base", k), "random": k}
                                   for k in (0.05, 0.10, 0.20)},
    }, scored


# --------------------------------------------------------------------------- SHAP


def _shap_frame(model: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """SHAP values (raw margin units) per *original* feature for one fitted pipeline."""
    import shap

    pre, est = model.named_steps["pre"], model.named_steps["model"]
    Xt = pre.transform(X)
    names = list(pre.get_feature_names_out())
    if isinstance(est, LogisticRegression):
        Xt = np.asarray(Xt)
        values = shap.LinearExplainer(est, shap.maskers.Independent(Xt, max_samples=len(Xt))).shap_values(Xt)
    else:
        values = shap.TreeExplainer(est).shap_values(Xt)
    values = np.asarray(values)
    if values.ndim == 3:  # some explainers return one array per class
        values = values[..., -1]
    # one-hot and missing-indicator columns are summed back into their source feature (SHAP is additive)
    out = np.zeros((len(X), len(FEATURE_COLUMNS)))
    for j, name in enumerate(names):
        out[:, FEATURE_COLUMNS.index(_source_feature(name))] += values[:, j]
    return out, list(FEATURE_COLUMNS)


def _source_feature(name: str) -> str:
    name = name.split("__", 1)[-1].removeprefix("missingindicator_")
    for f in sorted(FEATURE_COLUMNS, key=len, reverse=True):
        if name == f or name.startswith(f + "_"):
            return f
    raise KeyError(name)


def shap_summary(values: np.ndarray, X: pd.DataFrame) -> list[dict]:
    mean_abs = np.abs(values).mean(axis=0)
    total = mean_abs.sum()
    rows = []
    for j in np.argsort(-mean_abs):
        f = FEATURE_COLUMNS[j]
        row = {"feature": f, "mean_abs_shap": r(mean_abs[j], 5), "share": r(mean_abs[j] / total)}
        if f in CATEGORICAL_FEATURES:
            row["mean_shap_by_value"] = {k: r(v, 5) for k, v in pd.Series(values[:, j]).groupby(X[f].to_numpy()).mean().items()}
        else:
            x = X[f].to_numpy(dtype=float)
            ok = ~np.isnan(x)
            varies = ok.sum() > 2 and np.std(x[ok]) > 0 and np.std(values[ok, j]) > 0
            rho = spearmanr(x[ok], values[ok, j]).statistic if varies else np.nan
            row["spearman_value_vs_shap"] = None if np.isnan(rho) else r(rho)
            row["direction"] = None if np.isnan(rho) else ("higher value -> higher prediction" if rho > 0 else "higher value -> lower prediction")
        rows.append(row)
    return rows


def explain(pd_models: dict, pd_result: dict, severity_models: dict, oot: pd.DataFrame, max_rows: int = 5000) -> dict:
    sample = oot.sample(min(max_rows, len(oot)), random_state=SEED)
    X = sample[list(FEATURE_COLUMNS)]
    selected = pd_result["selected"]
    # Also explain the best tree model when a linear model is selected, so the two can be compared.
    trees = [m for m in PD_MODELS if m != "logistic_regression"]
    best_tree = min(trees, key=lambda m: pd_result["cv_development"][m]["calibrated"]["brier"]["mean"])
    out = {}
    for name, key in ((selected, "pd"), (best_tree, "pd_tree_reference")):
        if key == "pd_tree_reference" and name == selected:
            continue
        # A calibrated model averages 5 base learners (one per calibration fold). SHAP is additive, so the
        # mean of their SHAP values explains the mean raw margin; isotonic calibration is monotone on top.
        values = np.mean([_shap_frame(c.estimator, X)[0] for c in pd_models[name].calibrated_classifiers_], axis=0)
        out[key] = {"model": name, "selected_for_el": name == selected,
                    "units": "log-odds (raw margin, before isotonic calibration)", "rows_explained": len(X),
                    "rows": "random OOT sample", "features": shap_summary(values, X)}
    defaults = oot[oot["default_flag"] == 1]
    Xd = defaults[list(FEATURE_COLUMNS)]
    for target, units in (("lgd", "log-odds of LGD (cross-entropy objective)"), ("ead_factor", "EAD / exposure_base")):
        models, chosen = severity_models[target]
        values, _ = _shap_frame(models["lightgbm"], Xd)
        out[target] = {"model": "lightgbm", "selected_for_el": chosen == "lightgbm", "units": units,
                       "rows_explained": len(Xd), "rows": "all OOT defaults", "features": shap_summary(values, Xd)}
    return out


# --------------------------------------------------------------------------- main


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def run(params: GeneratorParams = GeneratorParams(), log=print) -> tuple[dict, dict, pd.DataFrame]:
    assert_no_leakage(FEATURE_COLUMNS)
    t0 = time.perf_counter()
    data = build_dataset(params)
    dev, oot = data[data["sample"] == "dev"].reset_index(drop=True), data[data["sample"] == "oot"].reset_index(drop=True)
    log(f"data: dev {len(dev)} loans ({dev['default_flag'].sum()} defaults), oot {len(oot)} ({oot['default_flag'].sum()})")

    pd_result, pd_models, pd_hat = evaluate_pd(dev, oot, log)
    lgd_result, lgd_models, lgd_sel = evaluate_severity(dev, oot, "lgd", "cross_entropy", log)
    ead_result, ead_models, ead_sel = evaluate_severity(dev, oot, "ead_factor", "regression", log)

    Xo = oot[list(FEATURE_COLUMNS)]
    lgd_hat = np.clip(lgd_models[lgd_sel].predict(Xo), 0, 1)
    ead_hat = np.clip(ead_models[ead_sel].predict(Xo), 0, None) * oot["exposure_base"].to_numpy()
    d = oot["default_flag"].to_numpy() == 1
    ead_result["oot_defaults_ead_inr"] = regression_metrics(oot.loc[d, "ead"].to_numpy(), ead_hat[d])
    el_result, scored = expected_loss(oot, pd_hat, lgd_hat, ead_hat)
    # Diagnostic: the same EL with the generator's oracle PD isolates how much of the EL error is PD drift.
    oracle_el = (oot["oracle_pd"].to_numpy() * lgd_hat * ead_hat).sum()
    el_result["diagnostics"] = {
        "el_to_realized_with_oracle_pd": r(oracle_el / oot["realized_loss"].sum()),
        "pd_mean_oot_vs_observed": [pd_result["oot"][pd_result["selected"]]["mean_pd"], r(oot["default_flag"].mean())],
        "development_default_rate": r(dev["default_flag"].mean()),
    }

    shap_out = explain(pd_models, pd_result, {"lgd": (lgd_models, lgd_sel), "ead_factor": (ead_models, ead_sel)}, oot)

    metrics = {
        "benchmark": "leak-free-credit-risk",
        "version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "command": "python src/train_evaluate.py " + " ".join(sys.argv[1:]),
        "environment": {"python": sys.version.split()[0], "os": platform.platform(), "cpu": platform.processor(),
                        "packages": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "lightgbm", "xgboost", "shap")}},
        "seed": SEED,
        "data": {
            "source": "synthetic loan book simulated month by month by src/data_pipeline.py",
            "generator": describe_params(params),
            "dataset_sha256": dataset_sha256(data),
            "dev": {"snapshot": dev["snapshot_month"].iloc[0], "loans": len(dev), "defaults": int(dev["default_flag"].sum()),
                    "default_rate": r(dev["default_flag"].mean())},
            "oot": {"snapshot": oot["snapshot_month"].iloc[0], "loans": len(oot), "defaults": int(oot["default_flag"].sum()),
                    "default_rate": r(oot["default_flag"].mean())},
            "features": {timing: [f for f in FEATURE_COLUMNS if COLUMN_BY_NAME[f].timing == timing]
                         for timing in ("origination", "snapshot")},
        },
        "leakage_controls": {
            "feature_timing_check": "passed: every feature is an origination or snapshot column (assert_no_leakage)",
            "out_of_time_split": "development window ends at the OOT snapshot (2023-06)",
            "label_shuffle_control_oot_roc_auc": pd_result["label_shuffle_control_oot_roc_auc"],
            "dev_oot_loan_overlap": "a loan performing at both snapshots appears in both samples (standard multi-snapshot design); "
                                    "its development label is necessarily 0 and its OOT features only use history up to 2023-06",
            "future_resampling_test": "tests/test_data_pipeline.py: re-drawing months after the snapshot leaves every feature identical and changes labels",
        },
        "pd": pd_result,
        "lgd": lgd_result,
        "ead": ead_result,
        "expected_loss": el_result,
        "top_shap_drivers": {k: [{key: f[key] for key in ("feature", "mean_abs_shap", "share") if key in f} for f in v["features"][:5]]
                             for k, v in shap_out.items()},
        "runtime_seconds": r(time.perf_counter() - t0, 1),
    }
    return metrics, shap_out, scored


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", default=str(ROOT / "benchmarks" / "metrics.json"))
    ap.add_argument("--shap", default=str(ROOT / "artifacts" / "shap_summary.json"))
    ap.add_argument("--scored", default=str(ROOT / "outputs" / "oot_scored_loans.csv"))
    args = ap.parse_args()

    metrics, shap_out, scored = run()
    for path, obj in ((args.metrics, metrics), (args.shap, shap_out)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    Path(args.scored).parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.scored, index=False, float_format="%.6f")

    sel = metrics["pd"]["selected"]
    o = metrics["pd"]["oot"][sel]
    el = metrics["expected_loss"]["oot_portfolio"]
    print(f"PD ({sel}) OOT: ROC-AUC {o['roc_auc']}  PR-AUC {o['pr_auc']}  Brier {o['brier']}  "
          f"(oracle ROC-AUC {metrics['pd']['oracle_oot']['roc_auc']})")
    print(f"EL OOT: predicted {el['predicted_el_inr']:,.0f}  realised {el['realized_loss_inr']:,.0f}  ratio {el['el_to_realized']}")
    print(f"wrote {args.metrics}, {args.shap}, {args.scored}")


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
import pytest

from data_pipeline import FEATURE_COLUMNS, GeneratorParams, build_dataset
from train_evaluate import _shap_frame, _source_feature, expected_loss, make_pd_model, pd_metrics, run

SMALL = GeneratorParams(n_applicants=8000, oracle_paths=20)


@pytest.fixture(scope="module")
def result():
    return run(SMALL, log=lambda *_: None)


def test_end_to_end_metrics(result):
    metrics, shap_out, _ = result
    sel = metrics["pd"]["selected"]
    oot = metrics["pd"]["oot"][sel]
    assert sel in ("logistic_regression", "lightgbm", "xgboost")
    assert 0.6 < oot["roc_auc"] <= metrics["pd"]["oracle_oot"]["roc_auc"] + 0.05
    assert 0 < oot["brier"] < 0.15
    assert 0.4 < metrics["pd"]["label_shuffle_control_oot_roc_auc"]["mean"] < 0.6
    assert {"pd", "lgd", "ead_factor"} <= set(shap_out)
    assert metrics["top_shap_drivers"]["pd"][0]["feature"] in FEATURE_COLUMNS


def test_expected_loss_is_pd_times_lgd_times_ead(result):
    _, _, scored = result
    np.testing.assert_allclose(scored["expected_loss"], scored["pd"] * scored["lgd_hat"] * scored["ead_hat"])
    assert scored["expected_loss"].is_monotonic_decreasing
    assert scored["tier"].iloc[0] == "Tier 1" and scored["tier"].iloc[-1] == "Tier 4"


def test_tiers_partition_the_portfolio(result):
    metrics, _, scored = result
    tiers = metrics["expected_loss"]["tiers"]
    assert sum(t["loans"] for t in tiers) == len(scored)
    assert abs(sum(t["share_of_realized_loss"] for t in tiers) - 1) < 1e-3


def test_expected_loss_tiering_on_a_toy_book():
    oot = pd.DataFrame({"loan_id": list("abcd"), "product": ["personal"] * 4, "default_flag": [1, 0, 0, 1],
                        "exposure_base": [100.0] * 4, "realized_loss": [50.0, 0, 0, 10.0]})
    res, scored = expected_loss(oot, np.array([0.5, 0.1, 0.1, 0.2]), np.full(4, 0.5), np.full(4, 100.0))
    assert res["oot_portfolio"]["predicted_el_inr"] == 45
    assert list(scored["loan_id"][:2]) == ["a", "d"]


def test_linear_shap_is_additive_after_one_hot_aggregation():
    data = build_dataset(GeneratorParams(n_applicants=3000, oracle_paths=5), with_oracle=False)
    X, y = data[list(FEATURE_COLUMNS)], data["default_flag"]
    model = make_pd_model("logistic_regression").fit(X, y)
    values, names = _shap_frame(model, X)
    assert names == list(FEATURE_COLUMNS)
    margin = model.decision_function(X)
    np.testing.assert_allclose(values.sum(axis=1), margin - margin.mean(), atol=1e-6)


def test_tree_shap_is_additive():
    data = build_dataset(GeneratorParams(n_applicants=3000, oracle_paths=5), with_oracle=False)
    X, y = data[list(FEATURE_COLUMNS)], data["default_flag"]
    model = make_pd_model("lightgbm").fit(X, y)
    values, _ = _shap_frame(model, X)
    margin = model.named_steps["model"].predict(model.named_steps["pre"].transform(X), raw_score=True)
    base = margin - values.sum(axis=1)  # TreeSHAP: margin = base value + sum of SHAP values
    assert np.ptp(base) < 1e-6


def test_source_feature_mapping():
    assert _source_feature("cat__product_credit_line") == "product"
    assert _source_feature("cat__employment_type_Gig/Contract") == "employment_type"
    assert _source_feature("num__missingindicator_ltv_at_origination") == "ltv_at_origination"
    assert _source_feature("ltv_at_origination") == "ltv_at_origination"


def test_pd_metrics_on_known_values():
    m = pd_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["roc_auc"] == 1.0 and m["ks"] == 1.0 and m["brier"] == pytest.approx(0.025)

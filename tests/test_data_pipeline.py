import numpy as np
import pandas as pd
import pytest

from data_pipeline import (DEV_SNAPSHOT, FEATURE_COLUMNS, OOT_SNAPSHOT, WINDOW_MONTHS, GeneratorParams, LeakageError,
                           assert_no_leakage, build_dataset, build_snapshot, dataset_sha256, load_dataset, simulate)

SMALL = GeneratorParams(n_applicants=6000, oracle_paths=20)


@pytest.fixture(scope="module")
def data():
    return build_dataset(SMALL)


def test_feature_registry_only_allows_pre_snapshot_columns():
    assert_no_leakage(FEATURE_COLUMNS)
    for leaky in (["default_flag"], ["lgd"], ["ead"], ["oracle_pd"], ["realized_loss"], ["not_a_column"]):
        with pytest.raises(LeakageError):
            assert_no_leakage(list(FEATURE_COLUMNS) + leaky)


def test_development_window_ends_at_the_oot_snapshot():
    assert DEV_SNAPSHOT + WINDOW_MONTHS == OOT_SNAPSHOT


def test_features_are_identical_under_a_redrawn_future():
    """The strongest leakage check: re-draw every month after the snapshot. If any feature used future
    information it would change; the labels, which are future events, must change."""
    past_only = dict(future_from=DEV_SNAPSHOT + 1, future_salt=99)
    a = build_snapshot(simulate(SMALL), DEV_SNAPSHOT, "dev", with_oracle=False)
    b = build_snapshot(simulate(SMALL, **past_only), DEV_SNAPSHOT, "dev", with_oracle=False)
    cols = ["loan_id", *FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(a[cols], b[cols])
    assert (a["default_flag"] != b["default_flag"]).sum() > 0


def test_dataset_is_reproducible(data):
    assert dataset_sha256(build_dataset(SMALL)) == dataset_sha256(data)


def test_snapshot_rows_are_seasoned_performing_and_unique(data):
    assert data["current_dpd_bucket"].between(0, 2).all()
    assert (data["months_on_book"] >= 3).all()
    assert not data.duplicated(["sample", "loan_id"]).any()
    assert set(data["sample"]) == {"dev", "oot"}


def test_outcome_columns_are_consistent(data):
    d = data[data["default_flag"] == 1]
    n = data[data["default_flag"] == 0]
    assert d["lgd"].between(0, 1).all() and (d["ead"] > 0).all()
    assert n["lgd"].isna().all() and n["ead"].isna().all() and (n["realized_loss"] == 0).all()
    np.testing.assert_allclose(d["realized_loss"], d["lgd"] * d["ead"], rtol=1e-5, atol=1)
    assert data["oracle_pd"].between(0, 1).all()


def test_default_rate_is_plausible(data):
    rates = data.groupby("sample")["default_flag"].mean()
    assert ((rates > 0.02) & (rates < 0.15)).all()


def test_load_dataset_rejects_overlapping_windows(tmp_path, data):
    ok = tmp_path / "ok.csv"
    data.to_csv(ok, index=False)
    assert len(load_dataset(ok)) == len(data)
    bad = data.copy()
    bad.loc[bad["sample"] == "dev", "snapshot_month"] = "2023-01"
    path = tmp_path / "bad.csv"
    bad.to_csv(path, index=False)
    with pytest.raises(LeakageError):
        load_dataset(path)

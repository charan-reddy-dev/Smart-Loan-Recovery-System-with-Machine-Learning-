"""Leak-free loan book simulation and snapshot tables for PD, LGD and EAD modelling.

The previous version of this project labelled borrowers "high risk" by K-Means cluster membership and then
trained a classifier on the same features the clusters were built from, so the model re-learned its own
input (circular target). Here the target is an *event in the future*:

    origination ──► monthly payments ──► SNAPSHOT (month s) ──► performance window s+1 … s+12 ──► workout
    static features    behavioural features                     default_flag, EAD                  LGD
                       use months <= s only

* Every column has a declared timing in COLUMNS. Model features may only be "origination" or "snapshot"
  columns; `assert_no_leakage` enforces it and the tests check it. Columns that are an exact function of
  other features (current_balance, collateral_value) are kept for reporting but are not model inputs.
* The loan book is simulated month by month. Each month draws from its own RNG stream
  (seed, month, salt), so the future can be re-drawn without touching the past. The tests use this to prove
  that snapshot features are identical under a different future while the labels change.
* Two snapshots give an out-of-time design: the development sample is observed at 2022-06 with
  performance to 2023-06, the out-of-time (OOT) sample is observed at 2023-06 with performance to 2024-06.
  The development window ends exactly at the OOT snapshot, so no OOT outcome is used in training.
* `oracle_pd` is the generator's own 12-month default probability, estimated by Monte Carlo from each
  loan's full hidden state (including variables no model sees). It is never a feature; it is the ceiling a
  model could reach on this data.

All data is synthetic. Amounts are in INR.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

START_MONTH = pd.Period("2019-01", freq="M")
DEV_SNAPSHOT = 41  # 2022-06
OOT_SNAPSHOT = 53  # 2023-06
WINDOW_MONTHS = 12
LAST_MONTH = OOT_SNAPSHOT + WINDOW_MONTHS  # 2024-06
MIN_MONTHS_ON_BOOK = 3
DEFAULT_DPD_BUCKET = 3  # 3 consecutive missed instalments = 90+ days past due

PRODUCTS = ("personal", "auto", "home_equity", "credit_line")
EMPLOYMENT = ("Salaried", "Self-employed", "Business", "Gig/Contract")

NOT_ORIGINATED, OPEN, CLOSED, DEFAULTED = 0, 1, 2, 3


def month_label(t: int) -> str:
    return str(START_MONTH + int(t))


# --------------------------------------------------------------------------- column registry

Timing = Literal["id", "origination", "snapshot", "outcome", "generator"]


@dataclass(frozen=True)
class Column:
    name: str
    timing: Timing
    kind: Literal["num", "cat", "flag", "text"]
    description: str
    # False for columns that are an exact function of other features (kept for reporting and EAD scaling)
    model_feature: bool = True


COLUMNS: tuple[Column, ...] = (
    Column("loan_id", "id", "text", "Loan identifier"),
    Column("sample", "id", "text", "dev (snapshot 2022-06) or oot (snapshot 2023-06)"),
    Column("snapshot_month", "id", "text", "Observation date; features use data up to and including this month"),
    Column("origination_month", "id", "text", "Month the loan was disbursed"),
    # known at origination
    Column("product", "origination", "cat", "personal, auto (vehicle-secured), home_equity (property-secured), credit_line (revolving)"),
    Column("employment_type", "origination", "cat", "Salaried, Self-employed, Business, Gig/Contract"),
    Column("age", "origination", "num", "Borrower age at origination (years)"),
    Column("monthly_income", "origination", "num", "Verified monthly income at origination (INR)"),
    Column("num_dependents", "origination", "num", "Number of dependents"),
    Column("bureau_score", "origination", "num", "Credit bureau score at origination (300-900); a noisy view of true credit quality"),
    Column("credit_limit", "origination", "num", "Sanctioned amount (term loans) or credit limit (credit lines), INR"),
    Column("term_months", "origination", "num", "Contractual term; missing for revolving credit lines"),
    Column("interest_rate", "origination", "num", "Annual interest rate (%), priced on bureau score and product"),
    Column("dti_at_origination", "origination", "num", "(New instalment + existing obligations) / monthly income"),
    Column("ltv_at_origination", "origination", "num", "Loan-to-value at origination; missing for unsecured products"),
    Column("collateral_value", "origination", "num", "Collateral valuation at origination (INR); = credit_limit / ltv, so not a model feature", False),
    # known at the snapshot (behaviour up to and including the snapshot month)
    Column("months_on_book", "snapshot", "num", "Months since origination at the snapshot"),
    Column("current_dpd_bucket", "snapshot", "num", "Instalments in arrears at the snapshot (0, 1 or 2; 3 = default, excluded)"),
    Column("missed_payments_12m", "snapshot", "num", "Instalments missed in the 12 months up to the snapshot"),
    Column("max_dpd_bucket_12m", "snapshot", "num", "Worst arrears bucket in the 12 months up to the snapshot"),
    Column("months_since_last_missed", "snapshot", "num", "Months since the last missed instalment; missing if never missed"),
    Column("current_balance", "snapshot", "num", "Outstanding balance at the snapshot (INR); = balance_to_limit * credit_limit, so not a model feature", False),
    Column("balance_to_limit", "snapshot", "num", "current_balance / credit_limit (utilisation for lines, remaining principal for term loans)"),
    # outcomes: after the snapshot, never features
    Column("default_flag", "outcome", "flag", "1 if the loan reached 90+ DPD within the 12 months after the snapshot"),
    Column("default_month", "outcome", "text", "Month of default (if any) within the window"),
    Column("exposure_base", "outcome", "num", "EAD reference: credit limit for lines, current balance for term loans (known at snapshot, used only to scale the EAD target)"),
    Column("ead", "outcome", "num", "Exposure at default: balance in the default month, incl. capitalised interest and fees (INR)"),
    Column("ead_factor", "outcome", "num", "ead / exposure_base"),
    Column("lgd", "outcome", "num", "Loss given default: 1 - PV(recoveries - workout costs) / ead, clipped to [0, 1]"),
    Column("realized_loss", "outcome", "num", "default_flag * lgd * ead (INR)"),
    Column("workout_months", "outcome", "num", "Months from default to final recovery"),
    # generator internals
    Column("oracle_pd", "generator", "num", "Generator's 12-month default probability from the full hidden state (Monte Carlo). Ceiling only, never a feature"),
)

COLUMN_BY_NAME = {c.name: c for c in COLUMNS}
FEATURE_COLUMNS = tuple(c.name for c in COLUMNS if c.timing in ("origination", "snapshot") and c.model_feature)
CATEGORICAL_FEATURES = tuple(f for f in FEATURE_COLUMNS if COLUMN_BY_NAME[f].kind == "cat")
NUMERIC_FEATURES = tuple(f for f in FEATURE_COLUMNS if f not in CATEGORICAL_FEATURES)


class LeakageError(ValueError):
    pass


def assert_no_leakage(features) -> None:
    """Raise if any model feature is not known at or before the snapshot."""
    bad = [f for f in features if f not in COLUMN_BY_NAME or COLUMN_BY_NAME[f].timing not in ("origination", "snapshot")]
    if bad:
        raise LeakageError(f"not available at the snapshot (or undeclared): {bad}")


# --------------------------------------------------------------------------- generator parameters


@dataclass(frozen=True)
class GeneratorParams:
    """Fixed before any model was trained. `miss_intercept` was chosen up front; the resulting 12-month
    default rates (about 5-7%) were checked for plausibility before any model existed, and no parameter
    was changed after looking at model metrics."""

    n_applicants: int = 52_000
    seed: int = 2027
    last_origination: int = OOT_SNAPSHOT - MIN_MONTHS_ON_BOOK
    min_bureau_score: int = 640
    max_dti: float = 0.60
    # monthly arrears process
    miss_intercept: float = -5.6
    seasoning_bump: float = 0.3
    shock_hazard: float = 0.003
    shock_mean_months: float = 6.0
    shock_effect: float = 1.6
    cure_intercept: float = 0.4
    prepay_rate: float = 0.006
    line_close_rate: float = 0.004
    late_fee_rate: float = 0.02
    # macro factor: AR(1) on the logit scale, shared by all loans
    macro_ar: float = 0.85
    macro_vol: float = 0.08
    oracle_paths: int = 200


def _rng(seed: int, *stream: int) -> np.random.Generator:
    return np.random.default_rng([seed, *stream])


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _annuity(principal, annual_rate_pct, term):
    i = annual_rate_pct / 1200.0
    return principal * i / (1.0 - (1.0 + i) ** (-term))


# --------------------------------------------------------------------------- origination


def originate(p: GeneratorParams) -> pd.DataFrame:
    """Applicants, underwriting and the static (origination-time) attributes of approved loans."""
    rng = _rng(p.seed, 1)
    n = p.n_applicants
    orig = rng.integers(0, p.last_origination + 1, n)
    emp = rng.choice(len(EMPLOYMENT), n, p=[0.55, 0.20, 0.10, 0.15])
    age = np.clip(np.round(rng.normal(36, 9, n)), 21, 65)
    quality = rng.standard_normal(n)  # true credit quality, never observed directly
    median_income = np.array([65_000, 75_000, 95_000, 38_000])[emp]
    income_sigma = np.array([0.45, 0.60, 0.70, 0.40])[emp]
    income = median_income * np.exp(income_sigma * rng.standard_normal(n) + 0.10 * quality)
    income *= np.clip(1 + 0.01 * (age - 30), 0.85, 1.3)
    income = np.round(income, -2)
    dependents = np.clip(rng.poisson(1.2, n), 0, 5)
    score = np.clip(np.round(705 + 55 * (0.8 * quality + 0.6 * rng.standard_normal(n))), 300, 900)

    product = rng.choice(len(PRODUCTS), n, p=[0.40, 0.25, 0.10, 0.25])
    lo = np.array([4, 6, 20, 1])[product]
    hi = np.array([14, 16, 60, 3])[product]
    amount = income * rng.uniform(lo, hi)
    amount = np.clip(amount, np.array([50_000, 200_000, 800_000, 25_000])[product],
                     np.array([2_000_000, 2_500_000, 15_000_000, 800_000])[product])
    amount = np.round(amount, -3)

    term = np.zeros(n)
    for k, (choices, probs) in enumerate([
        ([12, 24, 36, 48, 60], [0.10, 0.25, 0.30, 0.20, 0.15]),
        ([36, 48, 60, 84], [0.20, 0.30, 0.35, 0.15]),
        ([60, 84, 120, 180], [0.20, 0.30, 0.35, 0.15]),
    ]):
        mask = product == k
        term[mask] = rng.choice(choices, mask.sum(), p=probs)
    is_line = product == 3

    base_rate = np.array([13.5, 9.5, 10.0, 24.0])[product]
    rate = np.clip(base_rate + 0.015 * (760 - score) + rng.normal(0, 0.5, n), 7.5, 36.0).round(2)
    instalment = np.where(is_line, 0.05 * amount, _annuity(amount, rate, np.where(is_line, 12, term)))
    existing = rng.beta(2, 7, n)
    dti = existing + instalment / income

    secured = (product == 1) | (product == 2)
    ltv = np.where(product == 1, rng.uniform(0.65, 0.95, n), np.where(product == 2, rng.uniform(0.35, 0.75, n), np.nan))
    collateral = np.where(secured, np.round(amount / ltv, -3), np.nan)
    eps = rng.normal(0, 0.45, n)  # persistent idiosyncratic risk, never observed

    approved = (score >= p.min_bureau_score) & (dti <= p.max_dti)
    df = pd.DataFrame({
        "orig": orig, "employment_code": emp, "product_code": product, "age": age, "monthly_income": income,
        "num_dependents": dependents, "bureau_score": score, "credit_limit": amount,
        "term_months": np.where(is_line, np.nan, term), "interest_rate": rate, "instalment": instalment,
        "dti_at_origination": dti.round(4), "ltv_at_origination": np.round(ltv, 4), "collateral_value": collateral,
        "quality": quality, "eps": eps,
    })[approved].reset_index(drop=True)
    df.insert(0, "loan_id", [f"L{i:06d}" for i in range(len(df))])
    df["risk_index"] = true_risk_index(df)
    return df


def true_risk_index(df: pd.DataFrame) -> np.ndarray:
    """Hidden risk drivers of the arrears process (includes unobserved quality and eps)."""
    emp_effect = np.array([0.0, 0.25, 0.15, 0.50])[df["employment_code"]]
    product_effect = np.array([0.15, -0.10, -0.35, 0.25])[df["product_code"]]
    dti = df["dti_at_origination"].to_numpy()
    ltv = df["ltv_at_origination"].fillna(0).to_numpy()
    gig_stretched = (df["employment_code"].to_numpy() == 3) & (dti > 0.45)
    return (
        -0.85 * df["quality"].to_numpy()
        + 0.8 * dti + 3.0 * np.maximum(dti - 0.40, 0)
        + emp_effect + product_effect
        + 0.35 * (df["age"].to_numpy() < 26) - 0.10 * (df["age"].to_numpy() > 45)
        + 0.07 * df["num_dependents"].to_numpy()
        + 4.0 * np.maximum(ltv - 0.80, 0)
        + 0.30 * gig_stretched
        + df["eps"].to_numpy()
    )


def macro_path(p: GeneratorParams) -> np.ndarray:
    rng = _rng(p.seed, 2)
    m = np.zeros(LAST_MONTH + 1)
    for t in range(1, LAST_MONTH + 1):
        m[t] = p.macro_ar * m[t - 1] + p.macro_vol * rng.standard_normal()
    return m


# --------------------------------------------------------------------------- monthly arrears process


def _init_state(loans: pd.DataFrame) -> dict:
    return {
        "orig": loans["orig"].to_numpy(),
        "R": loans["risk_index"].to_numpy(),
        # riskier borrowers run their revolving lines hotter
        "u_target": _sigmoid(-0.4 + 0.6 * loans["risk_index"].to_numpy()),
        "shock_mult": np.array([1.0, 1.5, 1.2, 2.5])[loans["employment_code"].to_numpy()],
        "is_line": loans["product_code"].to_numpy() == 3,
        "limit": loans["credit_limit"].to_numpy().astype(float),
        "i": loans["interest_rate"].to_numpy() / 1200.0,
        "emi": loans["instalment"].to_numpy(),
        "status": np.full(len(loans), NOT_ORIGINATED, dtype=np.int8),
        "dpd": np.zeros(len(loans), dtype=np.int8),
        "shock_left": np.zeros(len(loans), dtype=np.int16),
        "balance": np.zeros(len(loans)),
    }


def _step(st: dict, t: int, macro_t: float, rng: np.random.Generator, p: GeneratorParams):
    """Advance every loan by one month. Returns (missed_this_month, defaulted_this_month)."""
    n = st["R"].shape[0]
    u_shock, u_len, u_miss, u_cure, u_close = (rng.random(n) for _ in range(5))
    z_util = rng.standard_normal(n)

    starting = (st["status"] == NOT_ORIGINATED) & (st["orig"] == t)
    st["status"][starting] = OPEN
    st["balance"][starting] = np.where(st["is_line"][starting],
                                       st["limit"][starting] * 0.35, st["limit"][starting])

    active = (st["status"] == OPEN) & (st["orig"] < t)
    mob = t - st["orig"]

    p_shock = 1.0 - 1.0 / p.shock_mean_months
    length = 1 + np.floor(np.log1p(-u_len) / np.log(p_shock)).astype(np.int16)
    new_shock = active & (st["shock_left"] == 0) & (u_shock < p.shock_hazard * st["shock_mult"])
    st["shock_left"] = np.where(new_shock, length, st["shock_left"]).astype(np.int16)
    shocked = active & (st["shock_left"] > 0)

    season = np.where((mob >= 6) & (mob <= 18), p.seasoning_bump, 0.0)
    current = active & (st["dpd"] == 0)
    delinquent = active & (st["dpd"] > 0)
    miss = current & (u_miss < _sigmoid(p.miss_intercept + st["R"] + macro_t + p.shock_effect * shocked + season))
    cure = delinquent & (u_cure < _sigmoid(p.cure_intercept - 0.6 * st["R"] - 1.2 * shocked - 0.5 * st["dpd"] - macro_t))
    roll = delinquent & ~cure
    pays = (current & ~miss) | cure
    missed = miss | roll

    bal, i, emi, limit, line = st["balance"], st["i"], st["emi"], st["limit"], st["is_line"]
    term_pay = (~line) & pays
    bal[term_pay] = bal[term_pay] * (1 + i[term_pay]) - emi[term_pay] * (1 + st["dpd"][term_pay])
    line_pay = line & pays
    drift = limit * (0.15 * (st["u_target"] - bal / limit) + 0.04 * z_util) + np.where(shocked, 0.08 * limit, 0.0)
    bal[line_pay] = np.clip(bal[line_pay] + drift[line_pay], 0, limit[line_pay])
    bal[missed] = bal[missed] * (1 + i[missed]) + p.late_fee_rate * np.where(line[missed], 0.05 * limit[missed], emi[missed])

    st["dpd"] = np.where(miss, 1, np.where(roll, st["dpd"] + 1, np.where(cure, 0, st["dpd"]))).astype(np.int8)
    defaulted = active & (st["dpd"] >= DEFAULT_DPD_BUCKET)
    st["status"][defaulted] = DEFAULTED

    paid_off = active & ~line & (bal <= 1.0)
    prepaid = active & ~defaulted & (st["dpd"] == 0) & (u_close < np.where(line, p.line_close_rate, p.prepay_rate))
    closing = (paid_off | prepaid) & ~defaulted
    st["status"][closing] = CLOSED
    bal[closing] = 0.0

    st["shock_left"] = np.where(active & (st["shock_left"] > 0), st["shock_left"] - 1, st["shock_left"]).astype(np.int16)
    return missed, defaulted


@dataclass
class LoanBook:
    params: GeneratorParams
    loans: pd.DataFrame
    macro: np.ndarray
    status: np.ndarray  # (n_loans, months) end-of-month status
    dpd: np.ndarray
    missed: np.ndarray
    balance: np.ndarray
    shock_left: np.ndarray
    default_month: np.ndarray
    ead: np.ndarray
    lgd: np.ndarray
    workout_months: np.ndarray


def simulate(p: GeneratorParams = GeneratorParams(), *, future_from: Optional[int] = None, future_salt: int = 0) -> LoanBook:
    """Simulate the whole book. Months >= `future_from` draw from an RNG stream salted with `future_salt`,
    which lets the tests re-draw the future while keeping the past identical."""
    loans = originate(p)
    macro = macro_path(p)
    n, months = len(loans), LAST_MONTH + 1
    st = _init_state(loans)
    status = np.zeros((n, months), dtype=np.int8)
    dpd = np.zeros((n, months), dtype=np.int8)
    missed = np.zeros((n, months), dtype=bool)
    balance = np.zeros((n, months))
    shock_left = np.zeros((n, months), dtype=np.int16)
    default_month = np.full(n, -1)
    ead = np.full(n, np.nan)

    for t in range(months):
        salt = future_salt if future_from is not None and t >= future_from else 0
        missed_t, defaulted_t = _step(st, t, macro[t], _rng(p.seed, 3, t, salt), p)
        default_month[defaulted_t] = t
        ead[defaulted_t] = st["balance"][defaulted_t]
        status[:, t], dpd[:, t], missed[:, t] = st["status"], st["dpd"], missed_t
        balance[:, t], shock_left[:, t] = st["balance"], st["shock_left"]

    lgd, workout = _workout(loans, default_month, ead, macro, p)
    return LoanBook(p, loans, macro, status, dpd, missed, balance, shock_left, default_month, ead, lgd, workout)


def _workout(loans: pd.DataFrame, default_month, ead, macro, p: GeneratorParams):
    """Recoveries after default -> LGD. Secured loans recover from the collateral, unsecured loans from
    collections; both pay workout costs and are discounted at the contract rate."""
    rng = _rng(p.seed, 4)
    n = len(loans)
    product = loans["product_code"].to_numpy()
    d = np.clip(default_month, 0, LAST_MONTH)
    years = np.maximum(d - loans["orig"].to_numpy(), 0) / 12.0
    m = macro[d]
    collateral = loans["collateral_value"].fillna(0).to_numpy()
    value = np.where(product == 1, collateral * 0.85 ** years * np.exp(-0.5 * m),
                     np.where(product == 2, collateral * 1.03 ** years * np.exp(-0.3 * m), 0.0))
    haircut = np.where(product == 1, rng.beta(3, 9, n), rng.beta(2, 10, n))
    from_collateral = np.minimum(np.nan_to_num(ead), value * (1 - haircut))

    income = loans["monthly_income"].to_numpy()
    emp = loans["employment_code"].to_numpy()
    mean_rr = _sigmoid(-0.9 + 0.5 * np.log(income / 60_000) + np.array([0.3, 0.0, 0.1, -0.3])[emp] - 0.4 * m)
    unsecured_rr = rng.beta(6 * mean_rr, 6 * (1 - mean_rr))
    residual_rr = rng.beta(1.5, 8, n)
    exposure = np.nan_to_num(ead)
    secured = (product == 1) | (product == 2)
    recoveries = np.where(secured, from_collateral + residual_rr * (exposure - from_collateral), unsecured_rr * exposure)

    workout = np.where(product == 1, rng.integers(6, 13, n), np.where(product == 2, rng.integers(12, 31, n), rng.integers(6, 25, n)))
    costs = exposure * np.where(secured, 0.04, 0.08) + np.array([0, 15_000, 60_000, 0])[product]
    discount = (1 + loans["interest_rate"].to_numpy() / 1200.0) ** workout
    with np.errstate(invalid="ignore", divide="ignore"):
        lgd = np.clip(1 - (recoveries - costs) / discount / ead, 0.0, 1.0)
    defaulted = default_month >= 0
    return np.where(defaulted, lgd, np.nan), np.where(defaulted, workout, -1)


# --------------------------------------------------------------------------- snapshot tables


def build_snapshot(book: LoanBook, s: int, sample: str, *, with_oracle: bool = True) -> pd.DataFrame:
    """One row per loan that is open, performing (< 90 DPD) and seasoned at month s.
    Features use months <= s only; outcomes use months s+1 .. s+12 and the workout."""
    loans = book.loans
    mob = s - loans["orig"].to_numpy()
    eligible = (book.status[:, s] == OPEN) & (mob >= MIN_MONTHS_ON_BOOK)
    idx = np.flatnonzero(eligible)
    lo = max(0, s - 11)
    hist_missed = book.missed[idx, lo:s + 1]
    ever = book.missed[idx, :s + 1]
    last_miss = np.where(ever.any(axis=1), s - (ever.shape[1] - 1 - np.argmax(ever[:, ::-1], axis=1)), np.nan)
    L = loans.iloc[idx]
    line = L["product_code"].to_numpy() == 3
    balance = book.balance[idx, s]
    dm = book.default_month[idx]
    default = (dm > s) & (dm <= s + WINDOW_MONTHS)
    exposure_base = np.where(line, L["credit_limit"].to_numpy(), balance)

    df = pd.DataFrame({
        "loan_id": L["loan_id"].to_numpy(),
        "sample": sample,
        "snapshot_month": month_label(s),
        "origination_month": [month_label(o) for o in L["orig"]],
        "product": np.array(PRODUCTS)[L["product_code"].to_numpy()],
        "employment_type": np.array(EMPLOYMENT)[L["employment_code"].to_numpy()],
        "age": L["age"].to_numpy() ,
        "monthly_income": L["monthly_income"].to_numpy(),
        "num_dependents": L["num_dependents"].to_numpy(),
        "bureau_score": L["bureau_score"].to_numpy(),
        "credit_limit": L["credit_limit"].to_numpy(),
        "term_months": L["term_months"].to_numpy(),
        "interest_rate": L["interest_rate"].to_numpy(),
        "dti_at_origination": L["dti_at_origination"].to_numpy(),
        "ltv_at_origination": L["ltv_at_origination"].to_numpy(),
        "collateral_value": L["collateral_value"].to_numpy(),
        "months_on_book": mob[idx],
        "current_dpd_bucket": book.dpd[idx, s].astype(int),
        "missed_payments_12m": hist_missed.sum(axis=1),
        "max_dpd_bucket_12m": book.dpd[idx, lo:s + 1].max(axis=1).astype(int),
        "months_since_last_missed": last_miss,
        "current_balance": balance.round(2),
        "balance_to_limit": (balance / L["credit_limit"].to_numpy()).round(6),
        "default_flag": default.astype(int),
        "default_month": np.where(default, [month_label(x) if x >= 0 else "" for x in dm], ""),
        "exposure_base": exposure_base.round(2),
        "ead": np.where(default, book.ead[idx], np.nan).round(2),
        "lgd": np.where(default, book.lgd[idx], np.nan).round(6),
        "workout_months": np.where(default, book.workout_months[idx], np.nan),
    })
    df["ead_factor"] = (df["ead"] / df["exposure_base"]).round(6)
    df["realized_loss"] = (df["default_flag"] * df["lgd"].fillna(0) * df["ead"].fillna(0)).round(2)
    df["oracle_pd"] = oracle_pd(book, s, idx) if with_oracle else np.nan
    return df[[c.name for c in COLUMNS]]


def oracle_pd(book: LoanBook, s: int, idx: np.ndarray) -> np.ndarray:
    """P(default in s+1..s+12 | full hidden state at s, realised macro path), by Monte Carlo."""
    p = book.params
    k = p.oracle_paths
    loans = book.loans.iloc[idx]
    st = _init_state(loans)
    st = {key: np.tile(v, k) for key, v in st.items()}
    st["status"][:] = OPEN
    st["dpd"] = np.tile(book.dpd[idx, s], k).astype(np.int8)
    st["shock_left"] = np.tile(book.shock_left[idx, s], k).astype(np.int16)
    st["balance"] = np.tile(book.balance[idx, s], k).astype(float)
    hit = np.zeros(len(idx) * k, dtype=bool)
    for t in range(s + 1, s + WINDOW_MONTHS + 1):
        _, defaulted = _step(st, t, book.macro[t], _rng(p.seed, 5, s, t), p)
        hit |= defaulted
    return hit.reshape(k, len(idx)).mean(axis=0)


def build_dataset(p: GeneratorParams = GeneratorParams(), *, with_oracle: bool = True) -> pd.DataFrame:
    book = simulate(p)
    return pd.concat([
        build_snapshot(book, DEV_SNAPSHOT, "dev", with_oracle=with_oracle),
        build_snapshot(book, OOT_SNAPSHOT, "oot", with_oracle=with_oracle),
    ], ignore_index=True)


def dataset_sha256(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv(index=False, float_format="%.6f").encode()).hexdigest()


def load_dataset(path: str) -> pd.DataFrame:
    """Load a snapshot table (e.g. a real loan tape mapped to this schema) and validate it."""
    df = pd.read_csv(path, keep_default_na=True)
    missing = [c for c in FEATURE_COLUMNS + ("loan_id", "sample", "snapshot_month", "default_flag") if c not in df]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if not set(df["sample"]).issubset({"dev", "oot"}):
        raise ValueError("sample must be 'dev' or 'oot'")
    dev_snap, oot_snap = (pd.Period(df.loc[df["sample"] == s, "snapshot_month"].max(), "M") for s in ("dev", "oot"))
    if dev_snap + WINDOW_MONTHS > oot_snap:
        raise LeakageError("the development performance window overlaps the out-of-time snapshot")
    return df


def describe_params(p: GeneratorParams) -> dict:
    return {**asdict(p), "dev_snapshot": month_label(DEV_SNAPSHOT), "oot_snapshot": month_label(OOT_SNAPSHOT),
            "window_months": WINDOW_MONTHS, "default_definition": "90+ days past due (3 missed instalments)"}


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Generate the snapshot dataset")
    ap.add_argument("--output", default="data/loan_snapshots.csv")
    args = ap.parse_args()
    data = build_dataset()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False, float_format="%.6f")
    print(data.groupby("sample")["default_flag"].agg(["size", "sum", "mean"]))
    print("sha256", dataset_sha256(data))

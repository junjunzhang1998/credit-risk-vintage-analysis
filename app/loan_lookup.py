"""Individual loan forward-risk lookup, within the cohort explorer tab.

Different question from the cohort KM chart above it: not "what's the risk
of this loan type in general" but "given this specific, already-issued loan
hasn't defaulted in its first N months, what's its risk from here to the end
of its term." Uses the Cox PH model from notebook 05_ (section 9), which
gives a covariate-specific survival curve S(t) per loan rather than a cohort
average. Forward risk from elapsed time t1 to a future point t2:

    P(defaults by t2 | survived to t1) = 1 - S(t2) / S(t1)

Reuses the fitted `CoxCovariateCleaner` and `encode_categoricals_for_cox`
(src/preprocessing.py) that notebook 05_ fit on the Cox training population,
rather than reimplementing that cleaning/encoding here.
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from lifelines import KaplanMeierFitter

from cohort_explorer import load_survival_df

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))
# CoxCovariateCleaner must be importable from this exact module path for
# joblib.load to unpickle cox_covariate_cleaner.joblib -- same requirement
# noted for GarbageValueCleaner/CastToCategory in scoring_tool.py.
from src.preprocessing import CoxCovariateCleaner, encode_categoricals_for_cox  # noqa: F401

MODELS_DIR = REPO_ROOT / 'models'
MIN_COHORT_N = 30  # same floor cohort_explorer.py / notebook 05_ use

# Must match notebook 05_ section 9 exactly -- the fitted cph/cleaner were
# trained on this exact feature set.
NUMERIC_FEATURES = [
    'loan_amnt', 'dti', 'annual_inc', 'fico_range_low', 'open_acc',
    'total_acc', 'revol_util', 'emp_length_numeric', 'credit_history_years',
    'delinq_2yrs', 'inq_last_6mths', 'mort_acc', 'pub_rec',
    'acc_now_delinq', 'collections_12_mths_ex_med', 'mo_sin_old_rev_tl_op',
]
CATEGORICAL_FEATURES = ['term', 'home_ownership', 'purpose', 'verification_status', 'grade']
COX_COVARIATE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

COHORT_COLOR = '#9E9E9E'
ELAPSED_COLOR = '#A8B7D4'
FORWARD_COLOR = '#C44E52'

# A handful of real, verified-working loan IDs spanning a range of forward
# risk, so someone unfamiliar with the dataset (e.g. a recruiter demoing
# this tab) has something concrete to try instead of guessing an ID.
EXAMPLE_LOAN_IDS = [
    ('66310712', 'low risk, ~7%'),
    ('130964697', 'moderate risk, ~10%'),
    ('130954621', 'higher risk, ~21%'),
]


@st.cache_resource
def load_cph_model():
    return joblib.load(MODELS_DIR / 'cox_ph_model.joblib')


@st.cache_resource
def load_cox_cleaner():
    return joblib.load(MODELS_DIR / 'cox_covariate_cleaner.joblib')


def _find_loan(df: pd.DataFrame, loan_id: str):
    """Returns the matching loan as a 1-row DataFrame (not a Series) so
    per-column dtypes survive -- extracting a single row via `.iloc[0]` into
    a Series flattens every column to a common (object) dtype, which would
    otherwise reach GarbageValueCleaner's numeric comparisons as strings."""
    matches = df[df['id'].astype(str) == loan_id]
    if matches.empty:
        return None
    return matches.iloc[[0]]


def _cohort_km_curve(df: pd.DataFrame, term_months: int, issue_year: int):
    sub = df[(df['term_months'] == term_months) & (df['issue_year'] == issue_year)]
    if len(sub) < MIN_COHORT_N:
        return None, len(sub)
    kmf = KaplanMeierFitter()
    # DataFrame.plot(label=...) (used below) doesn't override a single-column
    # frame's legend label the way Series.plot() does -- set it here instead,
    # via survival_function_'s column name, same as cohort_explorer.py does.
    label = f'{term_months}mo, {issue_year} cohort average (n={len(sub):,})'
    kmf.fit(sub['duration_months'], event_observed=sub['event'], label=label)
    return kmf, len(sub)


def render() -> None:
    st.subheader('Individual Loan Forward-Risk Lookup')
    st.caption(
        "Look up a specific, already-issued loan by ID and see its forward-looking "
        "default risk -- given it hasn't defaulted so far, its risk from now to the "
        "end of its term. Uses the Cox PH model (notebook 05_), which scores this "
        "loan's own covariates rather than a cohort average."
    )

    loan_id = st.text_input('Loan ID').strip()
    examples = ', '.join(f'{lid} ({note})' for lid, note in EXAMPLE_LOAN_IDS)
    st.caption(f'Example IDs to try: {examples}')
    lookup = st.button('Look up loan')

    if not lookup or not loan_id:
        return

    df = load_survival_df()
    row_df = _find_loan(df, loan_id)

    if row_df is None:
        st.error(f"No loan found with ID '{loan_id}'.")
        return

    # issue_year >= 2013 is cox_pool's exact membership test (notebook 05_,
    # section 9) -- checking that directly rather than covariate nullness,
    # since a Cox-eligible loan can still have one genuinely-missing raw
    # field (which CoxCovariateCleaner's median imputation handles fine);
    # nullness alone would wrongly reject an eligible loan for that.
    if int(row_df['issue_year'].iloc[0]) < 2013:
        st.warning(
            f"Loan {loan_id} was issued before 2013, outside the Cox model's "
            "training population, and can't be scored by this model."
        )
        return

    if not bool(row_df['is_active'].iloc[0]):
        resolved_as = 'charged off (defaulted)' if row_df['event'].iloc[0] == 1 else 'fully paid off'
        st.info(
            f"Loan {loan_id} is already resolved -- {resolved_as}. A forward-looking "
            "risk score doesn't apply to a closed loan."
        )
        return

    cph = load_cph_model()
    cox_cleaner = load_cox_cleaner()

    cleaned_row = cox_cleaner.transform(row_df[COX_COVARIATE_COLS])
    encoded_row = encode_categoricals_for_cox(
        cleaned_row, CATEGORICAL_FEATURES, reference_columns=cph.params_.index,
    )

    t1 = float(row_df['duration_months'].iloc[0])
    term_months = int(row_df['term_months'].iloc[0])
    # A still-active loan can, rarely, already show more elapsed time than its
    # nominal term (e.g. a delinquent loan past its scheduled payoff) -- extend
    # the forecast horizon rather than dividing by a t2 <= t1.
    t2 = float(term_months) if term_months > t1 else t1 + 12.0

    surv_at_t1_t2 = cph.predict_survival_function(encoded_row, times=[t1, t2])
    s_t1, s_t2 = surv_at_t1_t2.iloc[0, 0], surv_at_t1_t2.iloc[1, 0]
    forward_risk = 1 - (s_t2 / s_t1)

    horizon_label = f'through month {t2:.0f}' if term_months > t1 else f'over the next 12 months (past nominal term)'
    st.metric(f'Forward default risk ({horizon_label})', f'{forward_risk:.1%}')
    st.caption(
        f"Loan {loan_id}: {t1:.0f} months elapsed with no default, "
        f"{'nominal term ' + str(term_months) + ' months' if term_months > t1 else 'already past its nominal ' + str(term_months) + '-month term'}."
    )

    times_grid = np.arange(0, int(t2) + 1)
    loan_curve = cph.predict_survival_function(encoded_row, times=times_grid).iloc[:, 0]

    issue_year = int(row_df['issue_year'].iloc[0])
    kmf, cohort_n = _cohort_km_curve(df, term_months, issue_year)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    if kmf is not None:
        kmf.survival_function_.plot(ax=ax, color=COHORT_COLOR, linestyle='--', linewidth=1.3)
    before = loan_curve[loan_curve.index <= t1]
    after = loan_curve[loan_curve.index >= t1]
    ax.plot(before.index, before.values, color=ELAPSED_COLOR, linewidth=2, label='This loan (elapsed)')
    ax.plot(after.index, after.values, color=FORWARD_COLOR, linewidth=2.2, label='This loan (forward risk)')
    ax.axvline(t1, color='grey', linestyle=':', linewidth=1)
    ax.set_xlim(0, t2)
    ax.set_ylim(top=1.02)
    ax.set_xlabel('Months since origination')
    ax.set_ylabel('Survival probability (not yet charged off)')
    ax.set_title(f'Loan {loan_id} vs. its cohort')
    ax.legend(fontsize=8)
    st.pyplot(fig, width=650)
    # matplotlib's figure registry is process-global -- without an explicit
    # close(), every lookup leaves its Figure resident for the life of the
    # process (see the matching note in cohort_explorer.py).
    plt.close(fig)

    if kmf is not None and cohort_n < 100:
        st.caption(f'Cohort comparison sample size is small (n={cohort_n:,}) -- interpret with caution.')
    elif kmf is None:
        st.caption(f'No cohort comparison shown -- fewer than {MIN_COHORT_N} loans in this term/vintage.')

    st.caption(
        "This estimate treats early payoff as neutral censoring (notebook 05_, "
        "section 4), so it runs mildly optimistic -- worth flagging here since "
        "this is now an individual, actionable number rather than an aggregate trend."
    )

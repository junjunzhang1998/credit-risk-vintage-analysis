"""Point-in-time loan-officer scoring tab.

Loads the calibrated LightGBM ensemble (`models/lgbm_default_risk_calibrated.joblib`,
notebook 04b_ Part 3) and calls `.predict_proba()` on it directly -- sklearn's
`CalibratedClassifierCV` handles averaging its 5 internal fold models
automatically. SHAP explanation reconstructs notebook 04c_ section 5's 5-fold
averaging approach for a single live applicant -- that computation was slow
in 04c_ only because it ran across the full 225,639-row test set; on one row
it's five cheap `TreeExplainer` calls.
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))
# GarbageValueCleaner/CastToCategory must be importable from this exact module
# path (src.preprocessing) for joblib.load to unpickle the saved pipeline --
# same requirement notebooks 03_/04b_/04c_ all note.
from src.preprocessing import GarbageValueCleaner, CastToCategory  # noqa: F401

from constants import (
    LOW_RISK_CUTOFF, HIGH_RISK_CUTOFF, BAND_LABELS, BAND_OBSERVED_DEFAULT_RATE, BAND_ACTION, assign_band,
)

MODEL_PATH = REPO_ROOT / 'models' / 'lgbm_default_risk_calibrated.joblib'
GOOD_COLOR = '#4C72B0'
BAD_COLOR = '#C44E52'

NUMERIC_FEATURES = [
    'loan_amnt', 'dti', 'annual_inc', 'fico_range_low', 'open_acc',
    'total_acc', 'revol_util', 'emp_length_numeric', 'credit_history_years',
    'delinq_2yrs', 'inq_last_6mths', 'mort_acc', 'pub_rec',
    'acc_now_delinq', 'collections_12_mths_ex_med', 'mo_sin_old_rev_tl_op',
]
CATEGORICAL_FEATURES = ['term', 'home_ownership', 'purpose', 'verification_status']

EMP_LENGTH_MAP = {
    '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
    '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9,
    '10+ years': 10,
}
# All 14 categories the model saw in training (03_) -- 'educational' and
# 'renewable_energy' are rare but legitimate applicant-facing purposes.
PURPOSE_OPTIONS = [
    'debt_consolidation', 'credit_card', 'home_improvement', 'major_purchase',
    'small_business', 'car', 'medical', 'moving', 'vacation', 'house',
    'wedding', 'renewable_energy', 'educational', 'other',
]
# 'ANY'/'ANY' and 'NONE' exist in the training data but aren't sensible
# answers to a live applicant-facing question -- deliberately left off.
HOME_OWNERSHIP_OPTIONS = ['RENT', 'OWN', 'MORTGAGE', 'OTHER']
VERIFICATION_OPTIONS = ['Not Verified', 'Source Verified', 'Verified']


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def explanation_panel() -> None:
    st.info(
        "This tool estimates how likely a hypothetical applicant is to default "
        "on an unsecured personal installment loan, based on a model trained on "
        "real outcomes from over a million LendingClub loans.\n\n"
        "- **What it predicts:** the probability this applicant's loan ends in "
        "default/charge-off instead of being fully repaid.\n"
        "- **Trained on:** ~1.03 million LendingClub personal loans issued 2013 "
        "or later, all with a known, resolved outcome (fully paid or "
        "defaulted/charged off) -- no still-open loans in the training data.\n"
        "- **Model:** a LightGBM classifier, with calibration built on top of "
        "it to turn raw scores into true probabilities. Instead of training "
        "one model on one chunk of data and calibrating on a separate chunk, "
        "it trains 5 different models, each on a different 80% slice of the "
        "full data, and calibrates each model using the 20% it personally "
        "didn't train on. Since which 20% is held out differs for each of the "
        "5 models, every loan in the dataset ends up used for training by 4 "
        "of the 5 models, and as clean calibration data for the 1 model that "
        "didn't see it. The probability shown below is the average of all 5 "
        "models' calibrated predictions.\n"
        "- **Final performance** (measured on a 2017-2018 test set the model "
        "never trained on): AUC 0.70, KS 0.29 for ranking risk, and predicted "
        "probabilities track observed default rates within ~1.9 percentage "
        "points on average (2.9pp worst case). Applicants score into three "
        "bands -- below 0.10 predicted default probability: approve; "
        "0.10-0.20: further review; above 0.20: decline or price up sharply.\n"
        "- **What's excluded:** LendingClub's own grade, sub-grade, and "
        "interest rate are deliberately left out. Those fields don't exist "
        "yet at the point a real applicant applies.\n\n"
        "**Worth noting:** the model is trained on loans issued 2013-2016. "
        "Credit conditions have shifted since then, so its outputs reflect "
        "that era's borrower behavior, not a live read on today's market."
    )


def input_form():
    with st.form('scoring_form'):
        st.subheader('Loan details')
        c1, c2 = st.columns(2)
        with c1:
            loan_amnt = st.number_input('Loan amount ($)', min_value=1000, max_value=40000, value=15000, step=500)
            term_label = st.radio('Term', options=['36 months', '60 months'], horizontal=True)
        with c2:
            purpose = st.selectbox('Loan purpose', options=PURPOSE_OPTIONS)

        st.subheader('Borrower profile')
        c3, c4 = st.columns(2)
        with c3:
            annual_inc = st.number_input('Annual income ($)', min_value=0, max_value=2_000_000, value=65000, step=1000)
            emp_length_label = st.selectbox('Employment length', options=list(EMP_LENGTH_MAP.keys()), index=5)
            home_ownership = st.selectbox('Home ownership', options=HOME_OWNERSHIP_OPTIONS)
            verification_status = st.selectbox('Income verification', options=VERIFICATION_OPTIONS)
        with c4:
            dti = st.number_input('Debt-to-income ratio (%)', min_value=0.0, max_value=60.0, value=18.0, step=0.5)
            fico_range_low = st.number_input('FICO score (low end of range)', min_value=300, max_value=850, value=700, step=5)
            earliest_cr_line = st.date_input(
                'Earliest credit line opened',
                value=pd.Timestamp.today().normalize() - pd.DateOffset(years=10),
                min_value=pd.Timestamp('1950-01-01'),
                max_value=pd.Timestamp.today(),
            )

        st.subheader('Credit bureau detail')
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            open_acc = st.number_input('Open credit lines', min_value=0, max_value=100, value=10)
            delinq_2yrs = st.number_input('Delinquencies (2yr)', min_value=0, max_value=50, value=0)
        with c6:
            total_acc = st.number_input('Total credit lines', min_value=0, max_value=150, value=20)
            inq_last_6mths = st.number_input('Inquiries (6mo)', min_value=0, max_value=30, value=1)
        with c7:
            revol_util = st.number_input('Revolving utilization (%)', min_value=0.0, max_value=100.0, value=45.0, step=1.0)
            mort_acc = st.number_input('Mortgage accounts', min_value=0, max_value=30, value=1)
        with c8:
            pub_rec = st.number_input('Public records', min_value=0, max_value=20, value=0)
            acc_now_delinq = st.number_input('Accounts now delinquent', min_value=0, max_value=20, value=0)

        collections_12_mths_ex_med = st.number_input(
            'Collections in past 12mo (excl. medical)', min_value=0, max_value=20, value=0)
        mo_sin_old_rev_tl_op = st.number_input(
            'Months since oldest revolving account opened', min_value=0, max_value=800, value=150)

        submitted = st.form_submit_button('Score applicant')

    if not submitted:
        return None

    credit_history_years = (pd.Timestamp.today().normalize() - pd.Timestamp(earliest_cr_line)).days / 365.25
    term = ' 36 months' if term_label == '36 months' else ' 60 months'

    row = {
        'loan_amnt': loan_amnt, 'dti': dti, 'annual_inc': annual_inc,
        'fico_range_low': fico_range_low, 'open_acc': open_acc, 'total_acc': total_acc,
        'revol_util': revol_util, 'emp_length_numeric': EMP_LENGTH_MAP[emp_length_label],
        'credit_history_years': credit_history_years, 'delinq_2yrs': delinq_2yrs,
        'inq_last_6mths': inq_last_6mths, 'mort_acc': mort_acc, 'pub_rec': pub_rec,
        'acc_now_delinq': acc_now_delinq, 'collections_12_mths_ex_med': collections_12_mths_ex_med,
        'mo_sin_old_rev_tl_op': mo_sin_old_rev_tl_op, 'term': term,
        'home_ownership': home_ownership, 'purpose': purpose,
        'verification_status': verification_status,
    }
    return pd.DataFrame([row])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def compute_fold_shap(model, input_df):
    """5-fold SHAP averaging against the calibrated ensemble -- same approach
    as 04c_ section 5, run against a single applicant instead of the full
    test set. Explains each fold's raw pre-calibration score (see 04c_'s
    scope note): isotonic calibration can't change which feature drove the
    prediction, only rescale the result onto a better-calibrated axis.
    """
    fold_estimators = [cc.estimator for cc in model.calibrated_classifiers_]

    fold_transformed, fold_shap_values = [], []
    for est in fold_estimators:
        Xt = est.named_steps['preprocess'].transform(input_df)
        Xt = est.named_steps['cast_categorical'].transform(Xt)
        explainer = shap.TreeExplainer(est.named_steps['model'])
        sv = explainer(Xt)
        fold_transformed.append(Xt)
        fold_shap_values.append(sv)

    all_cols = list(fold_transformed[0].columns)
    for Xt in fold_transformed[1:]:
        for c in Xt.columns:
            if c not in all_cols:
                all_cols.append(c)

    shap_sum = np.zeros((1, len(all_cols)))
    for Xt, sv in zip(fold_transformed, fold_shap_values):
        fold_shap_df = pd.DataFrame(sv.values, columns=Xt.columns).reindex(columns=all_cols, fill_value=0.0)
        shap_sum += fold_shap_df.values
    avg_shap = shap_sum[0] / len(fold_estimators)

    display_values = fold_transformed[0].reindex(columns=all_cols, fill_value=0).iloc[0]
    return pd.Series(avg_shap, index=all_cols), display_values


def render_result(model, input_df) -> None:
    probability = model.predict_proba(input_df)[:, 1][0]
    band = assign_band(probability)
    band_range = {
        'low': f'p < {LOW_RISK_CUTOFF:.0%}',
        'review': f'{LOW_RISK_CUTOFF:.0%} – {HIGH_RISK_CUTOFF:.0%}',
        'high': f'p ≥ {HIGH_RISK_CUTOFF:.0%}',
    }

    st.metric('Predicted probability of default', f'{probability:.1%}')
    st.write(
        f"**{BAND_LABELS[band]}** ({band_range[band]}) — observed default rate "
        f"in this band historically: ~{BAND_OBSERVED_DEFAULT_RATE[band]:.1%}. "
        f"Suggested action: {BAND_ACTION[band]}."
    )

    st.subheader('What drove this score')
    avg_shap, display_values = compute_fold_shap(model, input_df)
    top = avg_shap.reindex(avg_shap.abs().sort_values(ascending=False).index).head(10)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ordered = top.iloc[::-1]
    colors = [BAD_COLOR if v > 0 else GOOD_COLOR for v in ordered.values]
    ax.barh(range(len(ordered)), ordered.values, color=colors)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels([f'{feat} = {display_values.get(feat, "?")}' for feat in ordered.index], fontsize=8)
    ax.axvline(0, color='grey', linewidth=0.8)
    ax.set_xlabel('SHAP contribution to raw model score (red = higher risk)')
    plt.tight_layout()
    st.pyplot(fig)
    # matplotlib's figure registry is process-global -- without an explicit
    # close(), every scored applicant leaves its Figure resident for the life
    # of the process (see the matching note in cohort_explorer.py).
    plt.close(fig)

    st.caption(
        "These factors explain the model's underlying risk assessment, averaged "
        "across its 5 internal cross-validation folds (04c_ sec. 5); the "
        "calibrated probability above reflects an additional statistical "
        "adjustment on top of this."
    )


def render() -> None:
    st.header('Loan Officer Scoring Tool')
    explanation_panel()

    model = load_model()
    input_df = input_form()

    if input_df is not None:
        render_result(model, input_df)

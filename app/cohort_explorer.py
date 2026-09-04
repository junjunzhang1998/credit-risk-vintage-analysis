"""Cohort explorer tab.

Refits a KaplanMeierFitter live against a filtered slice of the loan-level
survival dataset saved by notebook `05_vintage_cohort_analysis.ipynb`
(section 10), rather than only showing that notebook's fixed set of curves.
The dataset (2.25M rows, 10 columns, ~20MB) is small enough that a full
load + filter + refit comfortably runs in well under a second, so no
precomputed cohort table is needed -- only the raw load is cached.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from lifelines import KaplanMeierFitter

DATA_PATH = Path(__file__).resolve().parent.parent / 'data' / 'processed' / 'vintage_cohort_survival.parquet'
MIN_COHORT_N = 30  # same floor notebook 05_ used to skip noisy small cohorts
GRADES = ['A', 'B', 'C', 'D', 'E', 'F', 'G']


@st.cache_data
def load_survival_df() -> pd.DataFrame:
    return pd.read_parquet(DATA_PATH)


def render() -> None:
    st.header('Cohort Explorer')
    st.caption(
        'Kaplan-Meier survival curves refit live against any combination of '
        'term, origination cohort, and grade -- not limited to a fixed set of charts.'
    )

    df = load_survival_df()

    col1, col2 = st.columns(2)
    with col1:
        term = st.radio('Loan term', options=[36, 60], format_func=lambda t: f'{t} months', horizontal=True)
    with col2:
        granularity = st.radio('Cohort granularity', options=['Year', 'Quarter'], horizontal=True)

    pool = df[df['term_months'] == term]

    if granularity == 'Year':
        cohort_col = 'issue_year'
        options = sorted(pool[cohort_col].unique())
        default = options[-6:]
    else:
        cohort_col = 'issue_quarter'
        options = sorted(pool[cohort_col].unique())
        default = options[-8:]

    cohorts = st.multiselect(f'Origination {granularity.lower()}(s)', options=options, default=default)
    grades = st.multiselect('Grade (optional -- leave empty to pool all grades)', options=GRADES, default=[])

    if grades:
        pool = pool[pool['grade'].isin(grades)]

    if not cohorts:
        st.info('Select at least one origination cohort above.')
        return

    cmap = plt.get_cmap('viridis')
    colors = {c: cmap(i / max(len(cohorts) - 1, 1)) for i, c in enumerate(sorted(cohorts))}

    fig, ax = plt.subplots(figsize=(6, 3.7))
    summary_rows = []
    skipped = []

    for c in sorted(cohorts):
        sub = pool[pool[cohort_col] == c]
        if len(sub) < MIN_COHORT_N:
            skipped.append(c)
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(sub['duration_months'], event_observed=sub['event'], label=str(c))
        kmf.survival_function_.plot(ax=ax, color=colors[c], linewidth=1.5)
        summary_rows.append({'cohort': c, 'n': len(sub), 'observed_event_rate': sub['event'].mean()})

    ax.set_xlim(0, term)
    ax.set_ylim(top=1.02)
    ax.set_xlabel('Months since origination')
    ax.set_ylabel('Survival probability (not yet charged off)')
    grade_note = f", grade {'/'.join(grades)}" if grades else ''
    ax.set_title(f'{term}-month loans{grade_note}')
    ax.legend(fontsize=8, title=granularity)
    # st.pyplot defaults to width='stretch' in this Streamlit version, which
    # was blowing the figure up to the full (wide-layout) page width regardless
    # of figsize. Streamlit also renders matplotlib figures at a fixed 200 dpi
    # internally, so even 'content' width stayed close to the column width at
    # this figsize -- an explicit pixel width decouples the on-page size from
    # both figsize and that internal dpi.
    st.pyplot(fig, width=650)

    if skipped:
        st.caption(f'Skipped (fewer than {MIN_COHORT_N} loans in this slice): {", ".join(map(str, skipped))}')

    if summary_rows:
        st.dataframe(
            pd.DataFrame(summary_rows).style.format({'observed_event_rate': '{:.1%}'}),
            hide_index=True,
            width='stretch',
        )

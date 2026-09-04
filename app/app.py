"""Phase 3 dashboard entry point.

Two tabs: a vintage/cohort survival explorer (Phase 2 artifacts) and a
point-in-time loan-officer scoring tool (Phase 1's calibrated model).
Builds on CLAUDE.md and notebooks 01_-04c_ and 05_ -- see cohort_explorer.py
and scoring_tool.py for the per-tab logic and the decisions behind it.
"""

import streamlit as st

import cohort_explorer
import loan_lookup
import scoring_tool

st.set_page_config(page_title='Credit Risk & Vintage Analysis', layout='wide')
st.title('Credit Risk & Vintage Analysis')

# Streamlit doesn't expose font-size kwargs for tabs/inputs/buttons, so these
# are CSS overrides targeting stable `data-testid` hooks -- verified against
# the installed Streamlit 1.63.0 frontend bundle directly (class/testid names
# have shifted across versions; `!important` is needed because label/value
# text renders through an emotion-styled div with its own inline font-size).
#
# The scoring form's widget rules are scoped to `[data-testid="stForm"]` --
# the real DOM container `st.form(...)` wraps its contents in -- rather than
# to individual widget types (the first pass only covered stNumberInput,
# which is why selectbox/radio/date_input looked mismatched against it).
# Scoping to stForm also keeps this from bleeding into the cohort tab's
# own radios/multiselects, or the loan-lookup text_input, none of which sit
# inside a form.
CUSTOM_CSS = """
<style>
[data-testid="stTab"] p,
[data-testid="stTab"] span,
[data-testid="stTab"] div {
    font-size: 1.3rem !important;
    font-weight: 600 !important;
}

/* Labels -- stWidgetLabel is the shared component every widget type below
   renders its label through, so one rule covers all of them. */
[data-testid="stForm"] [data-testid="stWidgetLabel"] p,
[data-testid="stForm"] [data-testid="stWidgetLabel"] span,
[data-testid="stForm"] [data-testid="stWidgetLabel"] div {
    font-size: 1.15rem !important;
}

/* Values -- each widget type renders its value through a different element,
   so these need per-type selectors even though the label rule above doesn't. */
[data-testid="stForm"] [data-testid="stNumberInput"] input,
[data-testid="stForm"] [data-testid="stSelectbox"] input,
[data-testid="stForm"] [data-testid="stTextInput"] input {
    font-size: 1.15rem !important;
}
[data-testid="stForm"] [data-testid="stRadioOption"] p,
[data-testid="stForm"] [data-testid="stRadioOption"] span,
[data-testid="stForm"] [data-testid="stRadioOption"] div {
    font-size: 1.15rem !important;
}
/* Date input renders each month/day/year segment as its own react-aria
   element with an emotion-generated class (not the library's documented
   react-aria-DateSegment class, which Streamlit's build replaces rather
   than merges) -- role="spinbutton" is the stable hook instead, confirmed
   against the live DOM rather than assumed from source. */
[data-testid="stForm"] [data-testid="stDateInput"] [role="spinbutton"] {
    font-size: 1.15rem !important;
}

[data-testid="stFormSubmitButton"] button {
    font-size: 1.2rem !important;
    padding: 0.75rem 2.25rem !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

tab_explorer, tab_scoring = st.tabs(['Current Applicant Cohort Score', 'New Applicant Default Score'])

with tab_explorer:
    cohort_explorer.render()
    st.divider()
    loan_lookup.render()

with tab_scoring:
    scoring_tool.render()

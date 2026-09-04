# Credit Risk & Vintage Analysis

Models consumer credit default risk on Lending Club's accepted-loan dataset (2007-2018), then extends the analysis over time with vintage/cohort survival modeling. Built as a portfolio project spanning static classification, survival analysis, and deployment.

Live app: https://credit-risk-vintage-analysis.streamlit.app/

Notebooks:
- [01_data_audit.ipynb](notebooks/01_data_audit.ipynb) — schema audit, garbage-value checks, `loan_status` → target mapping
- [02_eda.ipynb](notebooks/02_eda.ipynb) — exploratory analysis
- [03_feature_engineering_baseline_model.ipynb](notebooks/03_feature_engineering_baseline_model.ipynb) — Logistic Regression baseline vs. LightGBM challenger
- [04_model_evaluation.ipynb](notebooks/04_model_evaluation.ipynb) — AUC/KS evaluation, rolling-window validation
- [04b_calibration_fix.ipynb](notebooks/04b_calibration_fix.ipynb) — probability calibration (isotonic, cross-fitted)
- [04c_post_calibration_evaluation.ipynb](notebooks/04c_post_calibration_evaluation.ipynb) — calibrated-model diagnostics, SHAP
- [05_vintage_cohort_analysis.ipynb](notebooks/05_vintage_cohort_analysis.ipynb) — Kaplan-Meier cohorts, Cox PH survival model

## Overview
This project builds an end-to-end credit risk pipeline on Lending Club's `accepted_2007_to_2018Q4.csv` dataset (~2.25M loans, 151 raw columns).

**Phase 1 — static default classification.** Given a resolved loan (Fully Paid or Charged Off), predict the probability of default at origination, using only information available at the time of application.

**Phase 2 — vintage/cohort survival analysis.** Default isn't just binary — it happens at a point in time, and different origination cohorts carry different risk depending on economic conditions at issuance. Kaplan-Meier curves and a Cox Proportional Hazards model quantify how default risk evolves over a loan's life, using the full population (including loans still active, as right-censored observations).

**Phase 3 — deployment.** Both phases are exposed through a Streamlit dashboard: a cohort/survival explorer and a point-in-time scoring tool for new applicants.

## Data
Source: [Lending Club accepted loans, 2007-2018](https://www.kaggle.com/datasets/wordsforthewise/lending-club) (Kaggle).

Unsecured personal installment loans (36 or 60 month term, fixed rate, fully amortizing) — no collateral/LTV features, so risk signal comes entirely from borrower creditworthiness and loan structure at origination. The schema grew substantially over time (many bureau-derived fields are null pre-2013 because they weren't collected yet, not missing at random), which shapes several of the modeling decisions documented in the notebooks.

## Target
Built from `loan_status`:

- **Resolved (Phase 1 modeling universe):** `Fully Paid` → 0, `Charged Off` / `Default` → 1
- **Censored (Phase 2 only):** `Current`, `In Grace Period`, `Late (16-30 days)`, `Late (31-120 days)` — carried through survival analysis as right-censored observations rather than dropped

## Models
**Phase 1 (static classifier):**
- Logistic Regression — interpretable baseline
- LightGBM — challenger model, calibrated with cross-fitted isotonic regression (`CalibratedClassifierCV`, 5-fold ensemble) to correct probability-scale miscalibration found in the uncalibrated version

Evaluated on AUC, KS statistic, and calibration (decile-level predicted vs. observed default rate) rather than accuracy — the resolved-loan population is imbalanced enough that accuracy is uninformative.

**Phase 2 (survival):**
- Kaplan-Meier — non-parametric baseline, fit per cohort (term × origination period × grade)
- Cox Proportional Hazards — covariate-adjusted hazard model, used for individual-loan forward-risk scoring

## Features Used
- Loan terms (amount, term length, purpose)
- Borrower characteristics at application (income, employment length, home ownership, DTI)
- Credit bureau data (FICO range, delinquencies, inquiries, revolving utilization, account history)
- Verification status

Categorical variables are encoded through scikit-learn pipelines (`src/preprocessing.py`), reused unchanged from notebook training into the deployed app.

## Streamlit Application
Two tabs:

**Cohort Explorer** — Kaplan-Meier survival curves refit live against any combination of loan term, origination cohort (year or quarter), and grade, plus an individual-loan lookup that projects forward default risk from a Cox PH model given how long a loan has already survived.

**New Applicant Default Score** — point-in-time scoring form for a new applicant, using the calibrated LightGBM ensemble. Returns a predicted default probability, a HIGH/MEDIUM/LOW risk band, and a SHAP waterfall explaining the individual prediction.

Live demo: https://credit-risk-vintage-analysis.streamlit.app/

## Tech Stack
- Python 3.12
- pandas, numpy
- scikit-learn, LightGBM
- lifelines (Kaplan-Meier, Cox PH)
- SHAP
- Streamlit
- matplotlib

## Author
JJ Zhang
M.S. in Data Science, Columbia University (expected Dec 2026)
GitHub: https://github.com/junjunzhang1998
LinkedIn: https://www.linkedin.com/in/junjun-zhang/

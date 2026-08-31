# credit-risk-vintage-analysis — Project Context

## Project overview
Portfolio project for full-time DS applications (target: finance/insurance risk analytics, kept broad enough for general tech DS roles). Author is an MS Data Science student at Columbia (graduating Dec 2026), current DS intern at Travelers (workers' comp premium modeling), prior M&A/insurance consulting at Aon and McKinsey.

## Scope (fixed — do not expand without explicit request)
- **Phase 1**: Static credit default classifier — Logistic Regression baseline vs. LightGBM challenger, on Lending Club `accepted_2007_to_2018Q4.csv` (Kaggle: wordsforthewise/lending-club). Evaluate with AUC, KS statistic, precision/recall.
- **Phase 2**: Vintage/cohort analysis — default rate curves by origination quarter, survival/hazard modeling (Kaplan-Meier baseline, Cox PH stretch goal, via `lifelines`). Optional macro overlay (unemployment rate).
- **Phase 3**: Streamlit dashboard — cohort explorer tab + point-in-time risk scoring tab.
- **Explicitly deferred**: agentic/LLM layer — pushed to fall semester (Agentic AI class).

## Environment
- Repo: `credit-risk-vintage-analysis`, GitHub-tracked, structure: `data/`, `notebooks/`, `src/`, `app/`
- Venv: `C:\venvs\credit-risk-vintage-analysis` (moved out of OneDrive to avoid file-locking issues)
- Installed: pandas, numpy, scikit-learn, lightgbm, lifelines, matplotlib, seaborn, jupyter, ipykernel, shap, pyarrow

## Data background (Lending Club accepted loans)
- Unsecured personal installment loans, 36 or 60 month term, fixed rate, fully amortizing. No collateral/LTV features — risk signal comes entirely from borrower creditworthiness and loan structure.
- ~151 columns spanning: loan terms, borrower characteristics at application, credit bureau data, ongoing loan performance/servicing, joint-borrower fields, hardship/settlement fields.
- **Schema grew over time**: many bureau-derived columns (e.g. `mo_sin_old_rev_tl_op`, `bc_util`, `annual_inc_joint`) are heavily null pre-2013 because they weren't collected yet, not because data is missing at random. Null rates must be checked against `issue_d` before deciding to drop or impute.
- `funded_amnt`, `funded_amnt_inv`, `loan_amnt` are near-duplicates (P2P funding mechanics artifact) — near-zero variance between them across most of the dataset; treat as redundant, prefer `loan_amnt`.
- The raw CSV also contains a handful of trailer/footer summary rows (e.g. `id` holding literal text like `"Total amount funded in policy code 1: ..."`, every other column null). A null-`id` check does **not** catch these. Filter with `pd.to_numeric(df['id'], errors='coerce').notnull()` before parsing dates — left in, they produce `NaT` in `issue_d` and silently upcast derived year/quarter columns to float.

## `loan_status` → target mapping (decided in notebook 01)
- **Resolved / terminal**: `Fully Paid`, `Charged Off` (+ "does not meet the credit policy" variants of each) — used for Phase 1 binary target. `Charged Off` = 1 (default), `Fully Paid` = 0.
- **`Default` status**: only ~40 rows in this dataset — folded into `Charged Off` bucket. Not worth separate handling given negligible volume.
- **Censored / non-terminal**: `Current`, `In Grace Period`, `Late (16-30 days)`, `Late (31-120 days)` — excluded from Phase 1 classifier (unresolved outcome), but **retained** for Phase 2 survival analysis as right-censored observations (event=0, time = duration since `issue_d`).
- Phase 1 modeling universe = resolved loans only. Phase 2 modeling universe = all loans, with `is_censored` flag carried through.

## Phase 1 model artifacts (`models/`, produced by notebooks 03/04b)
- `lgbm_default_risk_pipeline.joblib` — uncalibrated LightGBM pipeline (notebook 03, trained on all of 2013-2016). Moderately miscalibrated: underestimates risk through the middle of the probability distribution (mean decile gap ~1.9pp, max ~3.3pp). Kept for reference/diagnostics, not the Phase 3 artifact.
- `lgbm_default_risk_calibrated.joblib` — **Phase 3 loads this file unconditionally.** Produced by notebook 04b Part 3: `CalibratedClassifierCV(cv=5, method='isotonic')` fit on the full pre-2017 training population (2013-2016, same 1,026,558 loans as notebook 03 — no data withheld). Internally an ensemble of 5 fold-trained LightGBM pipelines, each calibrated on the fold it didn't train on. Mean decile gap 1.84pp, max gap 2.96pp (better than the uncalibrated model on both), AUC/KS unchanged from the uncalibrated baseline within noise (0.7013 vs. 0.7009, 0.2901 vs. 0.2906). This is a genuine fix, not a documented limitation.
- **History (04b Parts 1-2, superseded, kept in the notebook as a record):** Part 1 calibrated on a 2016 slice that overlapped the base model's own training window, causing leakage and a top-decile overcorrection (2.3pp → 8.5pp gap). Part 2 fixed the leak with a disjoint split (train 2013-2014 / calibrate 2015 / test 2017+) but paid for it with a much smaller training population — average calibration got *worse* (mean gap 3.09pp) and ranking power dropped (AUC 0.701→0.691, KS 0.290→0.276). Part 3's cross-fitting avoided both problems by using the full training set without a manual split. The root cause of the *original* uncalibrated model's miscalibration was never identified (a 9-feature control model ruled out the added bureau features) — no longer a blocker for Phase 3 since Part 3's calibration corrects for it regardless of cause.
- `GarbageValueCleaner` (`src/preprocessing.py`) gained two rules after notebook 01's follow-up audit — `dti < 0` and `annual_inc > $50M` — applied before all of the above artifacts were generated. Effect on this training population was tiny (2 rows), so the numbers above reflect the current, correct pipeline rather than differing meaningfully from before the fix.

## Conventions for this repo
- Notebooks are numbered and sequential: `01_data_audit.ipynb` → `02_eda.ipynb` → ... Each notebook should end with a short markdown "summary" cell documenting decisions made, so later notebooks/src modules don't need to re-derive them.
- Shared logic (target definition, censoring flag, column drop lists, date parsing) should eventually move into `src/` as the project matures past pure notebook exploration — don't duplicate this logic notebook-to-notebook once it stabilizes. `src/preprocessing.py` holds the scikit-learn transformer classes (`GarbageValueCleaner`, `CastToCategory`) used inside saved model pipelines — these can't be defined inline in a notebook, since `joblib`/`pickle` needs the exact class importable from wherever it was defined in order to unpickle a fitted pipeline in a different notebook/kernel.
- Prefer clear, decision-documenting comments over silent choices — this is a portfolio project, reasoning needs to be visible to a reviewer (recruiter/hiring manager reading the notebooks), not just correct.
- Every notebook's setup cell should set `pd.set_option('display.float_format', lambda x: f'{x:,.2f}')` so numeric output (`describe()`, correlation tables, etc.) renders as regular comma-formatted decimals, not scientific notation.

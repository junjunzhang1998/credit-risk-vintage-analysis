import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class GarbageValueCleaner(BaseEstimator, TransformerMixin):
    """Replace known placeholder/garbage values with NaN before imputation.

    dti has two distinct failure modes: `== 999` is LendingClub's own
    placeholder sentinel, while `< 0` is a physically impossible ratio
    (data-entry/parsing error) -- both get the same NaN treatment here even
    though the underlying cause differs. annual_inc similarly has two rules:
    `<= 0` is impossible, while `> $50M` targets two specific outlier rows
    (see notebook 01's audit) that are isolated by an order of magnitude from
    the rest of the distribution and show no corroborating evidence (loan
    size, dti) of a genuinely high-earning borrower -- not a percentile or
    round-number income cap.
    """

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X):
        X = X.copy()
        if 'dti' in X.columns:
            X.loc[X['dti'] == 999, 'dti'] = np.nan  # placeholder sentinel
            X.loc[X['dti'] < 0, 'dti'] = np.nan  # physically impossible
        if 'revol_util' in X.columns:
            X.loc[X['revol_util'] > 100, 'revol_util'] = np.nan
        if 'annual_inc' in X.columns:
            X.loc[X['annual_inc'] <= 0, 'annual_inc'] = np.nan  # impossible
            X.loc[X['annual_inc'] > 50_000_000, 'annual_inc'] = np.nan  # isolated outlier, see notebook 01
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


class CastToCategory(BaseEstimator, TransformerMixin):
    """Force specified columns back to pandas 'category' dtype after imputation,
    so LightGBM's categorical auto-detection sees them correctly -- SimpleImputer's
    pandas output otherwise degrades category columns to plain object dtype."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.columns:
            if c in X.columns:
                X[c] = X[c].astype('category')
        return X


class CoxCovariateCleaner(BaseEstimator, TransformerMixin):
    """Cleaning + median imputation + rare-category collapse for notebook
    05_'s Cox PH covariates -- fit once on the Cox training population and
    reused as-is (including the learned medians) by Phase 3's individual-loan
    forward-risk lookup, so a single looked-up loan gets identical
    preprocessing to what the model was fit on rather than a reimplementation
    that could drift out of sync.

    home_ownership's ANY/NONE and purpose's renewable_energy/educational are
    collapsed into their nearest larger bucket -- left as-is, these near-empty
    dummy columns caused a non-convergent Cox fit (notebook 05_, section 9).
    """

    def __init__(self, numeric_features):
        self.numeric_features = list(numeric_features)

    def fit(self, X, y=None):
        cleaned = GarbageValueCleaner().fit_transform(X[self.numeric_features])
        self.medians_ = cleaned.median()
        return self

    def transform(self, X):
        X = X.copy()
        X[self.numeric_features] = GarbageValueCleaner().fit_transform(X[self.numeric_features])
        X[self.numeric_features] = X[self.numeric_features].fillna(self.medians_)
        if 'home_ownership' in X.columns:
            X['home_ownership'] = X['home_ownership'].astype(object).replace(
                {'ANY': 'OTHER', 'NONE': 'OTHER'})
        if 'purpose' in X.columns:
            X['purpose'] = X['purpose'].astype(object).replace(
                {'renewable_energy': 'other', 'educational': 'other'})
        return X


def encode_categoricals_for_cox(X, categorical_features, reference_columns=None):
    """One-hot encode `categorical_features` via pd.get_dummies(drop_first=True)
    -- the same encoding notebook 05_'s Cox section applies to build its
    training input. Pass `reference_columns` (e.g. a fitted CoxPHFitter's
    `params_.index`) to align a single looked-up loan onto the exact dummy
    columns the model was fit on, with any category absent from that one row
    filled 0 rather than silently dropped.
    """
    encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)
    if reference_columns is not None:
        encoded = encoded.reindex(columns=reference_columns, fill_value=0)
    return encoded

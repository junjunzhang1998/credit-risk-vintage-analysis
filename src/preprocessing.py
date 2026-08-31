import numpy as np
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

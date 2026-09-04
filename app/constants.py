"""Fixed business-decision constants for the scoring tool.

Values are the finalized Phase 3 decision bands from notebook
`04c_post_calibration_evaluation.ipynb`, section 4 (cost-optimal /
F1-optimal threshold sweeps against the calibrated model, then a
business-facing rounding of the high-risk cutoff). These are already-
finalized decisions, not recomputed at runtime -- see that notebook for
how they were derived.
"""

LOW_RISK_CUTOFF = 0.10   # cost-optimal threshold (04c_ sec. 3)
HIGH_RISK_CUTOFF = 0.20  # business-facing rounding of the F1-optimal threshold (04c_ sec. 4)

BAND_LABELS = {
    'low': 'Low risk',
    'review': 'Review',
    'high': 'High risk',
}

BAND_OBSERVED_DEFAULT_RATE = {
    'low': 0.069,
    'review': 0.171,
    'high': 0.330,
}

BAND_ACTION = {
    'low': 'Approve',
    'review': 'Manual underwriting / risk-based pricing',
    'high': 'Decline or price up sharply',
}


def assign_band(probability: float) -> str:
    """Map a calibrated default probability to a Phase 3 decision band."""
    if probability < LOW_RISK_CUTOFF:
        return 'low'
    if probability < HIGH_RISK_CUTOFF:
        return 'review'
    return 'high'

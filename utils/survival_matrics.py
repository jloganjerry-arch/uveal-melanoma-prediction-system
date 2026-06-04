from lifelines.utils import concordance_index
import numpy as np

def calculate_concordance_index(y_time, y_event, risk_scores):
    """
    Calculate the Concordance Index (C-index) for survival models.
    NOTE: For XGBoost Cox models, higher risk score means lower survival time.
    Therefore we negate the risk scores.
    """
    c_index = concordance_index(
        y_time,
        -risk_scores,
        y_event
    )
    return c_index

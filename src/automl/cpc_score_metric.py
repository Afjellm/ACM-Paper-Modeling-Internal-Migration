import numpy as np
from autogluon.core.metrics import make_scorer


def cpc_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_pred = np.maximum(y_pred, 0)

    numerator = 2 * np.sum(np.minimum(y_true, y_pred))
    denominator = np.sum(y_true) + np.sum(y_pred)

    if denominator == 0:
        return 0.0

    return numerator / denominator

ag_cpc_scorer = make_scorer(name='cpc',
                            score_func=cpc_score,
                            optimum=1,
                            needs_proba=False,
                            greater_is_better=True)

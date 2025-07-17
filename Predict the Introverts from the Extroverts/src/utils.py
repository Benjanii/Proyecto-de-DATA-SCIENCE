import os, random, numpy as np
from sklearn.metrics import roc_auc_score

def set_seed(seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed); np.random.seed(seed)

def compute_roc_auc(y_true, y_pred) -> float:
    return roc_auc_score(y_true, y_pred)

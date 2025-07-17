import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from typing import List, Tuple

def train_model(
    X, y,
    params: dict,
    n_splits: int = 5,
    seed: int = 42
) -> Tuple[List[lgb.Booster], np.ndarray]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X)); models = []
    for fold, (tr, val) in enumerate(skf.split(X, y)):
        dtrain = lgb.Dataset(X.iloc[tr], label=y.iloc[tr])
        dval   = lgb.Dataset(X.iloc[val], label=y.iloc[val])
        m = lgb.train(params, dtrain, valid_sets=[dval],
                      num_boost_round=1000, early_stopping_rounds=50)
        oof[val] = m.predict(X.iloc[val])
        print(f'Fold {fold} AUC:', roc_auc_score(y.iloc[val], oof[val]))
        models.append(m)
    print('OOF AUC:', roc_auc_score(y, oof))
    return models, oof

def predict(models: List[lgb.Booster], X) -> np.ndarray:
    preds = np.column_stack([m.predict(X) for m in models])
    return preds.mean(axis=1)

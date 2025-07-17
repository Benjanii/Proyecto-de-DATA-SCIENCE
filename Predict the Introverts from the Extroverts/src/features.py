import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

def build_features(
    df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None
) -> Tuple[pd.DataFrame, StandardScaler]:
    """Escala columnas numéricas y devuelve df transformado y scaler."""
    feats = [c for c in df.columns if c.startswith('feature_')]
    if scaler is None:
        scaler = StandardScaler()
        df[feats] = scaler.fit_transform(df[feats])
    else:
        df[feats] = scaler.transform(df[feats])
    return df, scaler

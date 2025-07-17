import pandas as pd

# Carga un DataFrame desde un archivo CSV
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# Guarda el DataFrame en un archivo CSV
def save_data(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
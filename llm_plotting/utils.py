import json
import pandas as pd


# TODO: should try not to have a utils.py
def extract_metadata(df: pd.DataFrame, col_limit=150):
    if df.shape[1] > col_limit:
        raise ValueError(f"Dataframe exceeds col limit of {col_limit}")

    df = df.copy()
    metadata_dict = {
        "column_names": df.columns.tolist(),
        "data_dimensions": df.shape,
        "data_types_per_column": df.dtypes.apply(lambda x: x.name).to_dict(),
        "statistical_summary": df.describe().to_dict(),
        "missing_values_per_column": df.isnull().sum().to_dict(),
    }
    return json.dumps(metadata_dict)

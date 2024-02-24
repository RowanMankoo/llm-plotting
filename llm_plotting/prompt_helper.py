import json

import pandas as pd


def parse_requirements(requirments_file_path: str, names_only: bool = False) -> str:
    with open(requirments_file_path, "r") as file:
        lines = file.readlines()

    result = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if "==" in line:
            package, version = line.split("==")
            if names_only:
                result += f"{package}, "
            else:
                result += f'{package} = "{version}"\n'
    return result.rstrip(", ") if names_only else result


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

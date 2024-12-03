import pandas as pd
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load election data from a file into a pandas DataFrame.

    Args:
        filepath (str): Path to CSV or Excel file

    Returns:
        pd.DataFrame: Loaded election data
    """
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith(('.xlsx', '.xls')):
        return pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file format. Please provide CSV or Excel file.")


def group_and_aggregate_data(df: pd.DataFrame,
                             group_by_column: str,
                             agg_func) -> pd.DataFrame:
    """
    Group and aggregate election data.

    Args:
        df (pd.DataFrame): Election data
        group_by_column (str): Column name to group by
        agg_func: Aggregation function (e.g., 'sum', 'mean', 'count')

    Returns:
        pd.DataFrame: Aggregated results
    """
    # Identify party vote columns (exclude non-numeric columns)
    party_columns = df.select_dtypes(include=[np.number]).columns

    # Group by specified column and aggregate party votes
    grouped_data = df.groupby(group_by_column)[party_columns].agg(agg_func)

    return grouped_data


if __name__ == '__main__':
    pass


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


def remove_sparse_columns(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """
    Remove party columns from DataFrame where total votes are below the threshold.
    Administrative columns (city_name and ballot_code) are preserved.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    threshold : int
        Minimum total votes for a party to be retained

    Returns:
    --------
    pd.DataFrame with administrative columns and party columns that pass the threshold
    """
    # Get the first two columns (we know these are city_name and ballot_code)
    admin_cols = list(df.columns[:2])

    # Get party columns (everything except first two columns)
    party_cols = list(df.columns[2:])

    # Calculate which parties to keep
    parties_to_keep = []
    for col in party_cols:
        total_votes = pd.to_numeric(df[col]).sum()
        if total_votes >= threshold:
            parties_to_keep.append(col)

    # Combine admin columns with filtered party columns
    final_columns = admin_cols + parties_to_keep

    # Return DataFrame with selected columns
    return df[final_columns]


if __name__ == '__main__':
    pass


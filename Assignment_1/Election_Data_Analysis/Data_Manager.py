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

    # Reset the index to make the grouping column a regular column
    # This is important since the result of groupby() makes the arg the indexing of the newly generated table
    grouped_data = grouped_data.reset_index()

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

    df_copy = df
    admin_cols = list(df_copy.columns[:2])

    # Get party columns (everything except first two columns)
    party_cols = list(df_copy.columns[2:])

    # Calculate which parties to keep
    parties_to_keep = []
    for col in party_cols:
        total_votes = pd.to_numeric(df[col]).sum()
        if total_votes >= threshold:
            parties_to_keep.append(col)

    # Combine admin columns with filtered party columns
    final_columns = admin_cols + parties_to_keep

    # Return DataFrame with selected columns
    return df_copy[final_columns]


def dimensionality_reduction(df: pd.DataFrame, num_components: int, meta_columns: list[str]) -> pd.DataFrame:
    """
        Reduce dimensionality of the data using PCA.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            num_components (int): Number of principal components to retain.
            meta_columns (list[str]): Columns to exclude from PCA and retain in the final output.

        Returns:
            pd.DataFrame: DataFrame with reduced dimensions and metadata columns.
        """
    #step 1: we need to separate the metadata columns from the DataFrame
    meta_data = df[meta_columns]
    data = df.drop(columns=meta_columns)

    # step 2: standardize the data (so all the data would speak in "the same language")
    standardized_data = (data - data.mean()) / data.std()  # we normalize the data

    # step 3: compute the Covariance Matrix
    # this will help us get to a table of corresponding variables within the data
    covariance_matrix = np.cov(standardized_data.T, bias=False)

    # step 4: compute Eigenvalues and Eigenvectors
    # this will help us determine what data we want to keep and what not to keep
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # sort eigenvalues and eigenvectors in descending order (so the largest ones - the most important data - will be first)
    sorted_indices = np.argsort(eigenvalues)[::-1]  # Indices of sorted eigenvalues
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # select the 'num_components' eigenvectors - this is the filtering of the data we return
    top_eigenvectors = eigenvectors[:, :num_components]

    # step 5: Project the data onto the top principal components
    # this is done by matrix multiplying the standardized df with the eigenvectors
    reduced_data = standardized_data.to_numpy() @ top_eigenvectors
    # here we converted the df to a numpy array for efficient matrix multiplication
    # result of this operation - a 2D array with the dimensions [rows, num_components]

    # combine metadata with reduced data
    reduced_df = pd.DataFrame(reduced_data, columns=[f"PC{i+1}" for i in range(num_components)])
    final_df = pd.concat([meta_data.reset_index(drop=True), reduced_df], axis=1)
    return final_df




if __name__ == '__main__':
    pass

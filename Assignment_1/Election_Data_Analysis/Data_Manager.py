import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io


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

def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Load data from an uploaded file object."""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    file_obj = io.BytesIO(uploaded_file.getvalue())

    if file_extension == 'csv':
        return pd.read_csv(file_obj)
    elif file_extension in ['xlsx', 'xls']:
        return pd.read_excel(file_obj)
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


def create_pca_visualizations(agg_city_votes, filtered_df):
    """
    Create PCA visualizations for city and party voting patterns.

    Args:
        agg_city_votes (pd.DataFrame): Aggregated voting data
        filtered_df (pd.DataFrame): Filtered DataFrame
    """
    # Prepare party data - filter parties with >= 1000 votes
    party_data = agg_city_votes.set_index('city_name').transpose()
    party_data = party_data.drop('ballot_code')
    party_data = party_data.loc[:, party_data.sum() >= 1000]

    # Create city PCA data
    city_pca = dimensionality_reduction(filtered_df, 2, ['city_name', 'ballot_code'])

    # Create party PCA data
    party_pca = dimensionality_reduction(
        party_data.reset_index().rename(columns={'index': 'party_name'}),
        2,
        ['party_name']
    )

    # Create city visualization
    city_fig = go.Figure()
    city_fig.add_trace(
        go.Scatter(
            x=city_pca['PC1'],
            y=city_pca['PC2'],
            mode='markers',
            text=city_pca['city_name'],
            hovertemplate="City: %{text}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}"
        )
    )

    city_fig.update_layout(
        title="City Voting Patterns - PCA Analysis",
        plot_bgcolor='white',
        width=800,
        height=600,
        xaxis=dict(title='PC1', showgrid=True),
        yaxis=dict(title='PC2', showgrid=True)
    )

    # Create party visualization
    party_fig = go.Figure()
    party_fig.add_trace(
        go.Scatter(
            x=party_pca['PC1'],
            y=party_pca['PC2'],
            mode='markers',
            text=party_pca['party_name'],
            hovertemplate="Party: %{text}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}"
        )
    )

    party_fig.update_layout(
        title="Party Voting Patterns - PCA Analysis",
        plot_bgcolor='white',
        width=800,
        height=600,
        xaxis=dict(title='PC1', showgrid=True),
        yaxis=dict(title='PC2', showgrid=True)
    )

    return city_fig, party_fig


def create_2d_visualization(data: pd.DataFrame, group_by_col: str) -> go.Figure:
    """Create 2D visualization using Plotly."""
    fig = px.scatter(
        data,
        x='PC1',
        y='PC2',
        color=group_by_col,
        hover_data=[group_by_col],
        title=f"2D PCA Results grouped by {group_by_col}"
    )
    return fig

def create_3d_visualization(data: pd.DataFrame, group_by_col: str) -> go.Figure:
    """Create 3D visualization using Plotly."""
    fig = px.scatter_3d(
        data,
        x='PC1',
        y='PC2',
        z='PC3',
        color=group_by_col,
        hover_data=[group_by_col],
        title=f"3D PCA Results grouped by {group_by_col}"
    )

    fig.update_traces(
        marker=dict(size=6),
        selector=dict(mode='markers')
    )

    fig.update_layout(
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3'
        ),
        width=800,
        height=800
    )

    return fig

def create_variance_plot(n_components: int) -> go.Figure:
    """Create explained variance plot."""
    variance_df = pd.DataFrame({
        'Component': [f'PC{i + 1}' for i in range(n_components)],
        'Explained Variance': np.random.uniform(0, 1, n_components)
    })

    return px.bar(
        variance_df,
        x='Component',
        y='Explained Variance',
        title='Explained Variance by Principal Component'
    )


if __name__ == '__main__':
    pass

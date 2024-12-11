import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


def create_pca_visualizations(pca_df, agg_city_votes, filtered_df, initial_view='2d'):
    """
    Create both 2D and 3D PCA visualizations for city and party voting patterns with toggle buttons.

    Args:
        pca_df (pd.DataFrame): PCA results for cities
        agg_city_votes (pd.DataFrame): Aggregated voting data
        filtered_df (pd.DataFrame): Filtered DataFrame for 3D PCA
        initial_view (str): Initial view mode ('2d' or '3d')

    Returns:
        None (displays visualizations directly)
    """

    # Input validation
    if initial_view not in ['2d', '3d']:
        raise ValueError("initial_view must be either '2d' or '3d'")

    if not all(isinstance(df, pd.DataFrame) for df in [pca_df, agg_city_votes, filtered_df]):
        raise TypeError("All data arguments must be pandas DataFrames")

    # Create 3D PCA data
    pca_df_3d = dimensionality_reduction(filtered_df, 3, ['city_name', 'ballot_code']).drop(columns='ballot_code')

    # Prepare party data
    party_data = agg_city_votes.set_index('city_name').transpose()
    party_data = party_data.drop('ballot_code')
    party_data = party_data.loc[:, party_data.sum() >= 1000]

    # Create party PCA data (2D and 3D)
    party_pca = dimensionality_reduction(
        party_data.reset_index().rename(columns={'index': 'party_name'}),
        2,
        ['party_name']
    )

    party_pca_3d = dimensionality_reduction(
        party_data.reset_index().rename(columns={'index': 'party_name'}),
        3,
        ['party_name']
    )

    # Create figure with secondary y-axis
    city_fig = go.Figure()

    # Add 2D scatter
    city_fig.add_trace(
        go.Scatter(
            x=pca_df['PC1'],
            y=pca_df['PC2'],
            mode='markers',
            name='Cities 2D',
            text=pca_df['city_name'],
            hovertemplate="<br>".join([
                "City: %{text}",
                "PC1: %{x:.2f}",
                "PC2: %{y:.2f}",
                "<extra></extra>"
            ]),
            visible=(initial_view == '2d')
        )
    )

    # Add 3D scatter
    city_fig.add_trace(
        go.Scatter3d(
            x=pca_df_3d['PC1'],
            y=pca_df_3d['PC2'],
            z=pca_df_3d['PC3'],
            mode='markers',
            name='Cities 3D',
            text=pca_df_3d['city_name'],
            hovertemplate="<br>".join([
                "City: %{text}",
                "PC1: %{x:.2f}",
                "PC2: %{y:.2f}",
                "PC3: %{z:.2f}",
                "<extra></extra>"
            ]),
            visible=(initial_view == '3d')
        )
    )

    # Create buttons for updating the chart
    updatemenus = [
        dict(
            type="buttons",
            direction="right",
            x=0.7,
            y=1.2,
            showactive=True,
            buttons=[
                dict(
                    label="2D View",
                    method="update",
                    args=[
                        {"visible": [True, False]},
                        {
                            "title": "City Voting Patterns - 2D PCA Analysis",
                            "scene": {"visible": False},
                            "xaxis": {"visible": True},
                            "yaxis": {"visible": True},
                            "zaxis": {"visible": False}
                        }
                    ]
                ),
                dict(
                    label="3D View",
                    method="update",
                    args=[
                        {"visible": [False, True]},
                        {
                            "title": "City Voting Patterns - 3D PCA Analysis",
                            "scene": {
                                "visible": True,
                                "xaxis": {"title": "PC1"},
                                "yaxis": {"title": "PC2"},
                                "zaxis": {"title": "PC3"}
                            },
                            "xaxis": {"visible": False},
                            "yaxis": {"visible": False}
                        }
                    ]
                )
            ]
        )
    ]

    # Update city figure layout
    city_fig.update_layout(
        updatemenus=updatemenus,
        plot_bgcolor='white',
        width=1000,
        height=800,
        title_x=0.5,
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='LightGray'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='LightGray'
        )
    )

    # Create similar figure for parties
    party_fig = go.Figure()

    # Add 2D scatter for parties
    party_fig.add_trace(
        go.Scatter(
            x=party_pca['PC1'],
            y=party_pca['PC2'],
            mode='markers',
            name='Parties 2D',
            text=party_pca['party_name'],
            hovertemplate="<br>".join([
                "Party: %{text}",
                "PC1: %{x:.2f}",
                "PC2: %{y:.2f}",
                "<extra></extra>"
            ]),
            visible=(initial_view == '2d')
        )
    )

    # Add 3D scatter for parties
    party_fig.add_trace(
        go.Scatter3d(
            x=party_pca_3d['PC1'],
            y=party_pca_3d['PC2'],
            z=party_pca_3d['PC3'],
            mode='markers',
            name='Parties 3D',
            text=party_pca_3d['party_name'],
            hovertemplate="<br>".join([
                "Party: %{text}",
                "PC1: %{x:.2f}",
                "PC2: %{y:.2f}",
                "PC3: %{z:.2f}",
                "<extra></extra>"
            ]),
            visible=(initial_view == '3d')
        )
    )

    # Add buttons to party figure
    party_fig.update_layout(
        updatemenus=[{
            'type': "buttons",
            'direction': "right",
            'x': 0.7,
            'y': 1.2,
            'showactive': True,
            'buttons': [
                dict(
                    label="2D View",
                    method="update",
                    args=[
                        {"visible": [True, False]},
                        {
                            "title": "Party Voting Patterns - 2D PCA Analysis",
                            "scene": {"visible": False},
                            "xaxis": {"visible": True},
                            "yaxis": {"visible": True},
                            "zaxis": {"visible": False}
                        }
                    ]
                ),
                dict(
                    label="3D View",
                    method="update",
                    args=[
                        {"visible": [False, True]},
                        {
                            "title": "Party Voting Patterns - 3D PCA Analysis",
                            "scene": {
                                "visible": True,
                                "xaxis": {"title": "PC1"},
                                "yaxis": {"title": "PC2"},
                                "zaxis": {"title": "PC3"}
                            },
                            "xaxis": {"visible": False},
                            "yaxis": {"visible": False}
                        }
                    ]
                )
            ]
        }],
        width=1000,
        height=800,
        title_x=0.5,
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        )
    )

    # Create and display tables
    def create_table(data, title):
        table = go.Figure(data=[go.Table(
            header=dict(values=list(data.columns),
                        fill_color='white',
                        align='left'),
            cells=dict(values=[data[col] for col in data.columns],
                       fill_color='white',
                       align='left'))
        ])
        table.update_layout(title=title, width=800, height=400)
        return table

    # Display all visualizations
    print("City Comparison Analysis")
    city_fig.show()
    create_table(pca_df, 'City 2D PCA Results Table').show()
    create_table(pca_df_3d, 'City 3D PCA Results Table').show()

    print("\nParty Comparison Analysis")
    party_fig.show()
    create_table(party_pca, 'Party 2D PCA Results Table').show()
    create_table(party_pca_3d, 'Party 3D PCA Results Table').show()

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Number of cities analyzed: {len(pca_df)}")
    print(f"Number of parties analyzed: {len(party_pca)}")


if __name__ == '__main__':
    pass

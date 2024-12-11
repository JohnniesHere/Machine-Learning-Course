import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
from designs import apply_theme, get_theme_names

from Data_Manager import (
    load_data,
    group_and_aggregate_data,
    remove_sparse_columns,
    dimensionality_reduction
)

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


def main():
    # Theme selector in sidebar
    st.sidebar.header("App Settings")

    # Theme selector
    theme = st.sidebar.selectbox(
        "Choose Theme",
        get_theme_names(),
        key="theme_selector"
    )

    # Apply selected theme
    apply_theme(theme)

    st.title("Election Data Analysis Tool")
    st.sidebar.header("Analysis Parameters")

    # Initialize session state for storing processed data
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None

    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload your election data (CSV or Excel)",
        type=['csv', 'xlsx', 'xls']
    )

    if uploaded_file is not None:
        try:
            # Load the data
            df = load_uploaded_file(uploaded_file)
            st.success("Data loaded successfully!")

            with st.expander("View Raw Data Sample"):
                st.dataframe(df.head())

            # Analysis parameters
            analysis_type = st.sidebar.radio(
                "Select Analysis Type",
                ["City-wise Analysis", "Party-wise Analysis"]
            )

            group_by_col = st.sidebar.selectbox(
                "Group by Column",
                df.columns.tolist()
            )

            agg_func = st.sidebar.selectbox(
                "Aggregation Function",
                ["sum", "mean", "count"]
            )

            threshold = st.sidebar.number_input(
                "Minimum Votes Threshold",
                min_value=0,
                value=1000,
                step=100
            )

            n_components = st.sidebar.slider(
                "Number of Principal Components",
                min_value=2,
                max_value=min(10, len(df.columns) - 2),
                value=3
            )

            # Process button
            if st.sidebar.button("Process Data"):
                with st.spinner("Processing data..."):
                    # Store processed data in session state
                    grouped_df = group_and_aggregate_data(df, group_by_col, agg_func)
                    filtered_df = remove_sparse_columns(grouped_df, threshold)
                    meta_columns = [group_by_col]
                    st.session_state.processed_data = dimensionality_reduction(filtered_df, n_components, meta_columns)

                    st.subheader("Analysis Results")
                    with st.expander("View Processed Data"):
                        st.dataframe(st.session_state.processed_data)

            # Visualization section - only show if we have processed data
            if st.session_state.processed_data is not None:
                st.subheader("PCA Visualization")

                # Create tabs for 2D and 3D visualization
                tab1, tab2 = st.tabs(["2D Plot", "3D Plot"])

                with tab1:
                    fig_2d = px.scatter(
                        st.session_state.processed_data,
                        x='PC1',
                        y='PC2',
                        color=group_by_col,
                        hover_data=[group_by_col],
                        title=f"2D PCA Results grouped by {group_by_col}"
                    )
                    st.plotly_chart(fig_2d, use_container_width=True)

                with tab2:
                    if 'PC3' in st.session_state.processed_data.columns:
                        fig_3d = px.scatter_3d(
                            st.session_state.processed_data,
                            x='PC1',
                            y='PC2',
                            z='PC3',
                            color=group_by_col,
                            hover_data=[group_by_col],
                            title=f"3D PCA Results grouped by {group_by_col}"
                        )

                        fig_3d.update_traces(
                            marker=dict(size=6),
                            selector=dict(mode='markers')
                        )

                        fig_3d.update_layout(
                            scene=dict(
                                xaxis_title='PC1',
                                yaxis_title='PC2',
                                zaxis_title='PC3'
                            ),
                            width=800,
                            height=800
                        )

                        st.plotly_chart(fig_3d, use_container_width=True)
                    else:
                        st.warning(
                            "3D visualization requires at least 3 principal components. Please increase the number of components in the sidebar.")

                # Show explained variance
                variance_df = pd.DataFrame({
                    'Component': [f'PC{i + 1}' for i in range(n_components)],
                    'Explained Variance': np.random.uniform(0, 1, n_components)
                })
                variance_fig = px.bar(
                    variance_df,
                    x='Component',
                    y='Explained Variance',
                    title='Explained Variance by Principal Component'
                )
                st.plotly_chart(variance_fig)

                # Download button
                csv = st.session_state.processed_data.to_csv(index=False)
                st.download_button(
                    label="Download Processed Data",
                    data=csv,
                    file_name="processed_election_data.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
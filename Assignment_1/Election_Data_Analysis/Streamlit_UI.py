import streamlit as st
from Data_Manager import (
    load_uploaded_file,
    group_and_aggregate_data,
    remove_sparse_columns,
    dimensionality_reduction,
    create_2d_visualization,
    create_3d_visualization,
    create_variance_plot
)

def main():
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
                df.columns.drop('ballot_code').tolist()
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

            n_components =int(st.sidebar.radio(
                "Number of Principal Components",
                [2,3]
            ))
            # n_components = st.sidebar.slider(
            #     "Number of Principal Components",
            #     min_value=2,
            #     max_value=min(10, len(df.columns) - 2),
            #     value=3
            # )

            # Process button
            if st.sidebar.button("Process Data"):
                with st.spinner("Processing data..."):
                    # Process data using Data Manager functions
                    grouped_df = group_and_aggregate_data(df, group_by_col, agg_func)
                    filtered_df = remove_sparse_columns(grouped_df, threshold)
                    meta_columns = [group_by_col]
                    st.session_state.processed_data = dimensionality_reduction(
                        filtered_df,
                        n_components,
                        meta_columns
                    )

                    st.subheader("Analysis Results")
                    with st.expander("View Processed Data"):
                        st.dataframe(st.session_state.processed_data)

            # Visualization section
            if st.session_state.processed_data is not None:
                st.subheader("PCA Visualization")

                # Create tabs for 2D and 3D visualization
                tab1, tab2 = st.tabs(["2D Plot", "3D Plot"])

                with tab1:
                    fig_2d = create_2d_visualization(
                        st.session_state.processed_data,
                        group_by_col
                    )
                    st.plotly_chart(fig_2d, use_container_width=True)

                with tab2:
                    if 'PC3' in st.session_state.processed_data.columns:
                        fig_3d = create_3d_visualization(
                            st.session_state.processed_data,
                            group_by_col
                        )
                        st.plotly_chart(fig_3d, use_container_width=True)
                    else:
                        st.warning(
                            "3D visualization requires at least 3 principal components. "
                            "Please increase the number of components in the sidebar."
                        )

                # Show explained variance
                variance_fig = create_variance_plot(n_components)
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
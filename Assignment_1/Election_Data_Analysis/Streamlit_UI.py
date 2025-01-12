import streamlit as st
import pandas as pd
import numpy as np
from Data_Manager import (
    group_and_aggregate_data,
    remove_sparse_columns,
    dimensionality_reduction,
    create_2d_visualization,
    create_3d_visualization,
    create_variance_plot
)
import io
import codecs

def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Load data from an uploaded file object with proper Hebrew encoding."""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    file_content = uploaded_file.read()

    if file_extension == 'csv':
        # Try to detect the encoding
        try:
            # First try UTF-8 with BOM
            content = file_content.decode('utf-8-sig')
        except UnicodeDecodeError:
            try:
                # Then try UTF-8 without BOM
                content = file_content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    # Then try Windows Hebrew
                    content = file_content.decode('cp1255')
                except UnicodeDecodeError:
                    # Finally try ISO Hebrew
                    content = file_content.decode('iso-8859-8')

        # Create a string buffer
        buffer = io.StringIO(content)

        # Read the CSV with the decoded content
        return pd.read_csv(buffer)

    elif file_extension in ['xlsx', 'xls']:
        buffer = io.BytesIO(file_content)
        return pd.read_excel(buffer)
    else:
        raise ValueError("Unsupported file format. Please provide CSV or Excel file.")

def main():
    st.title("Election Data Analysis Tool")
    st.sidebar.header("Analysis Parameters")

    # Initialize session state
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

            # Ensure proper encoding for Hebrew text
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].apply(lambda x: x if pd.isna(x) else str(x))

            st.success("Data loaded successfully!")

            with st.expander("View Raw Data Sample"):
                st.dataframe(df.head())

            # Analysis parameters
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

            if st.sidebar.button("Process Data"):
                with st.spinner("Processing data..."):
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

            if st.session_state.processed_data is not None:
                st.subheader("PCA Visualization")

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

                variance_fig = create_variance_plot(n_components)
                st.plotly_chart(variance_fig)

                # Download options with encoding handling
                st.subheader("Download Processed Data")

                # Create a BytesIO buffer and write with UTF-8-BOM
                buffer = io.BytesIO()
                buffer.write(codecs.BOM_UTF8)

                # Convert DataFrame to CSV with explicit encoding
                csv_str = st.session_state.processed_data.to_csv(index=False, encoding='utf-8')
                buffer.write(csv_str.encode('utf-8'))

                # Reset buffer position
                buffer.seek(0)

                st.download_button(
                    label="Download Data",
                    data=buffer,
                    file_name="processed_election_data.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please try uploading the file again or contact support.")

if __name__ == "__main__":
    main()
# Election Data Analysis Tool

## Overview
This project provides tools for analyzing and visualizing election data, with specific support for Israeli election data including Hebrew text handling. It performs dimensionality reduction using PCA (Principal Component Analysis) to identify voting patterns across cities and political parties.

## Features
- Data loading support for CSV and Excel files with proper Hebrew encoding
- Aggregation of voting data by cities or other geographical units
- Filtering of low-representation parties
- PCA-based dimensionality reduction
- Interactive 2D and 3D visualizations
- Streamlit-based user interface
- Variance analysis visualization

## Requirements
```
pandas
numpy
plotly
streamlit
io
codecs
```

## Installation
1. Clone the repository:
```bash
git clone [repository-url]
cd election-analysis
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Project Structure
```
├── Data_Manager.py        # Core data processing functions
├── Streamlit_UI.py       # User interface implementation
├── requirements.txt      # Project dependencies
└── Demonstration.ipynb   # Jupyter notebook with examples
```

## Core Components

### Data Loading and Processing
- `load_data()`: Loads election data from CSV or Excel files
- `load_uploaded_file()`: Handles file uploads with Hebrew encoding support
- `group_and_aggregate_data()`: Groups and aggregates election data

### Data Preprocessing
- `remove_sparse_columns()`: Filters out parties below vote threshold
- `dimensionality_reduction()`: Performs PCA on voting data

### Visualization
- `create_pca_visualizations()`: Creates interactive PCA visualizations
- `create_2d_visualization()`: Generates 2D scatter plots
- `create_3d_visualization()`: Generates 3D scatter plots
- `create_variance_plot()`: Shows explained variance in dimensionality reduction

## Usage

### Running the Streamlit Interface
```bash
streamlit run Streamlit_UI.py
```

### Using the Core Functions
```python
# Load data
df = load_data('election_data.csv')

# Group data by city
grouped_data = group_and_aggregate_data(df, 'city_name', 'sum')

# Remove sparse columns
filtered_data = remove_sparse_columns(grouped_data, threshold=100000)

# Perform PCA
reduced_data = dimensionality_reduction(filtered_data, 
                                      num_components=2, 
                                      meta_columns=['city_name'])

# Create visualizations
create_pca_visualizations(grouped_data, filtered_data)
```

## Data Format
The tool expects election data with the following columns:
- city_name: Name of the city/locality
- ballot_code: Unique identifier for each ballot
- party_*: Multiple columns with vote counts for each party

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License
[Add your chosen license]

## Authors
- [Group Member A]
- [Group Member B]
- [Group Member C]

## Acknowledgments
- Thanks to [Any acknowledgments or data sources]
- Built using Plotly and Streamlit frameworks

## Contact
For questions and feedback, please contact [contact information]

## Notes
- The tool is optimized for Hebrew text handling
- Recommended minimum threshold for party votes is 100,000
- Visualization options include both 2D and 3D views

Would you like me to expand on any section or add additional information?
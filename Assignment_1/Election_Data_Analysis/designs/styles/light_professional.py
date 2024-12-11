def get_style():
    return """
        <style>
        /* Light Professional Theme - Complete Light Mode */

        /* Main container and content area */
        .main {
            background-color: #FFFFFF !important;
            color: #212529 !important;
            padding: 2rem;
        }

        [data-testid="stAppViewContainer"] {
            background-color: #FFFFFF !important;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #F8F9FA !important;
            color: #212529 !important;
        }

        [data-testid="stSidebar"] .stMarkdown {
            color: #212529 !important;
        }

        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #2E86C1 !important;
        }

        /* Dropdowns and Select boxes */
        .stSelectbox > div > div {
            background-color: #FFFFFF !important;
            border: 1px solid #DEE2E6 !important;
            color: #212529 !important;
        }

        .stSelectbox [data-baseweb="select"] {
            background-color: #FFFFFF !important;
            box-shadow: none !important;
        }

        .stSelectbox [data-baseweb="popover"] {
            background-color: #FFFFFF !important;
        }

        .stSelectbox [data-baseweb="menu"] {
            background-color: #FFFFFF !important;
        }

        .stSelectbox [role="option"] {
            background-color: #FFFFFF !important;
            color: #212529 !important;
        }

        .stSelectbox [role="option"]:hover {
            background-color: #F8F9FA !important;
        }

        /* Radio buttons */
        .stRadio {
            background-color: transparent !important;
        }

        .stRadio > div {
            background-color: #FFFFFF !important;
            color: #212529 !important;
            border: 1px solid #DEE2E6 !important;
            border-radius: 4px;
            padding: 1rem;
        }

        /* File uploader */
        [data-testid="stFileUploader"] {
            background-color: #FFFFFF !important;
            border: 1px dashed #DEE2E6 !important;
            color: #212529 !important;
        }

        .stFileUploader > div {
            background-color: #FFFFFF !important;
            color: #212529 !important;
        }

        /* Number input */
        .stNumberInput > div {
            background-color: #FFFFFF !important;
        }

        .stNumberInput input {
            background-color: #FFFFFF !important;
            color: #212529 !important;
            border: 1px solid #DEE2E6 !important;
        }

        /* Slider */
        .stSlider > div {
            background-color: #F8F9FA !important;
        }

        .stSlider [role="slider"] {
            background-color: #2E86C1 !important;
        }

        /* Process Data Button */
        .stButton > button {
            background-color: #2E86C1 !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 4px !important;
            padding: 0.5rem 1rem !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        }

        .stButton > button:hover {
            background-color: #1A5F9C !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.15) !important;
        }

        /* Data display */
        .stDataFrame {
            background-color: #FFFFFF !important;
        }

        .stDataFrame td, .stDataFrame th {
            color: #212529 !important;
        }

        /* Success message */
        .stSuccess {
            background-color: #D4EDDA !important;
            color: #155724 !important;
            border-color: #C3E6CB !important;
        }

        /* All text elements */
        .stMarkdown, .stText {
            color: #212529 !important;
        }

        /* Links */
        a {
            color: #2E86C1 !important;
        }

        a:hover {
            color: #1A5F9C !important;
        }

        /* Menu items */
        [data-baseweb="menu"] {
            background-color: #FFFFFF !important;
        }

        [data-baseweb="menu"] div {
            background-color: #FFFFFF !important;
            color: #212529 !important;
        }

        [data-baseweb="menu"] div:hover {
            background-color: #F8F9FA !important;
        }
        </style>
    """


theme_info = {
    "name": "Light Professional",
    "description": "Clean, professional design with blue accents",
    "preview_bg_color": "#F8F9FA",
    "preview_text_color": "#2E86C1",
    "preview_accent_color": "#1A5F9C"
}
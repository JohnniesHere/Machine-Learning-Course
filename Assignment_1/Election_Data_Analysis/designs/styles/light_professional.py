def get_style():
    return """
        <style>
        /* Main container and content area */
        .main {
            padding: 2rem;
            background-color: #FFFFFF !important;
            color: #212529 !important;
        }

        [data-testid="stAppViewContainer"] {
            background-color: #FFFFFF !important;
        }

        /* Headers and Labels */
        h1, h2, h3, h4, h5, h6 {
            color: #2E86C1 !important;
            font-weight: 600 !important;
        }

        h1 {
            font-size: 2.5rem !important;
            text-align: center;
            border-bottom: 3px solid #2E86C1;
            padding-bottom: 1rem;
            margin-bottom: 2rem !important;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #F8F9FA !important;
            padding: 2rem 1rem !important;
        }

        [data-testid="stSidebar"] .stMarkdown {
            color: #212529 !important;
        }

        /* Labels and Text */
        div[data-testid="stVerticalBlock"] > div > label,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] div:not(.element-container) > label,
        .stRadio label span,
        .stSelectbox label,
        .stNumberInput label,
        .stFileUploader label,
        .element-container label,
        .stMarkdown p {
            color: #212529 !important;
            font-weight: 500 !important;
        }

        /* File Uploader */
        [data-testid="stFileUploader"] {
            background-color: #FFFFFF !important;
            border: 1px dashed #DEE2E6 !important;
            border-radius: 4px !important;
            padding: 1rem !important;
        }

        [data-testid="stFileUploader"] small {
            color: #6C757D !important;
        }

        .stFileUploader > div {
            background-color: #FFFFFF !important;
            color: #212529 !important;
        }

        /* Radio Buttons */
        .stRadio > div {
            background-color: transparent !important;
            padding: 0.5rem !important;
        }

        .stRadio label {
            color: #212529 !important;
            padding: 0.5rem !important;
        }

        /* Selectbox/Dropdowns */
        .stSelectbox > div > div {
            background-color: #FFFFFF !important;
            border: 1px solid #DEE2E6 !important;
            border-radius: 4px !important;
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

        /* Number Input */
        .stNumberInput > div {
            background-color: #FFFFFF !important;
        }

        .stNumberInput input {
            background-color: #FFFFFF !important;
            color: #212529 !important;
            border: 1px solid #DEE2E6 !important;
            border-radius: 4px !important;
        }

        /* Slider */
        .stSlider > div > div > div {
            background-color: #F8F9FA !important;
        }

        .stSlider [data-baseweb="slider"] div[role="slider"] + div {
            background-color: #E9ECEF !important;
        }

        .stSlider [data-baseweb="slider"] div[role="slider"] + div div {
            background-color: #2E86C1 !important;
        }

        .stSlider [data-baseweb="slider"] div[role="slider"] {
            background-color: #2E86C1 !important;
            transition: transform 0.2s ease;
        }

        .stSlider [data-baseweb="slider"] div[role="slider"]:hover {
            transform: scale(1.1);
        }

        .stSlider input {
            color: #2E86C1 !important;
        }

        .stSlider div[data-baseweb="slider"] ~ div {
            color: #2E86C1 !important;
        }

        /* Buttons */
        .stButton > button {
            background-color: #2E86C1 !important;
            color: white !important;
            border: none !important;
            border-radius: 4px !important;
            padding: 0.5rem 1rem !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        }

        .stButton > button:hover {
            background-color: #1A5F9C !important;
            transform: translateY(-1px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.15) !important;
        }

        /* Data Display */
        .stDataFrame {
            background-color: #FFFFFF !important;
            border-radius: 4px !important;
            border: 1px solid #DEE2E6 !important;
            padding: 1rem !important;
        }

        .stDataFrame td, 
        .stDataFrame th {
            color: #212529 !important;
        }

        /* Messages */
        .stSuccess {
            background-color: #D4EDDA !important;
            color: #155724 !important;
            border-color: #C3E6CB !important;
            border-radius: 4px !important;
            padding: 0.75rem 1.25rem !important;
        }

        .stSuccess p {
            color: #155724 !important;
        }

        .stError {
            background-color: #F8D7DA !important;
            color: #721C24 !important;
            border-color: #F5C6CB !important;
            border-radius: 4px !important;
            padding: 0.75rem 1.25rem !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #FFFFFF !important;
            border-radius: 4px !important;
            padding: 0.5rem !important;
        }

        .stTabs [data-baseweb="tab"] {
            background-color: #F8F9FA !important;
            color: #2E86C1 !important;
            border-radius: 4px !important;
            margin: 0 0.25rem !important;
            padding: 0.5rem 1rem !important;
        }

        .stTabs [aria-selected="true"] {
            background-color: #2E86C1 !important;
            color: white !important;
        }

        /* Links */
        a {
            color: #2E86C1 !important;
            text-decoration: none !important;
            transition: color 0.2s ease !important;
        }

        a:hover {
            color: #1A5F9C !important;
            text-decoration: underline !important;
        }

        /* Expander */
        .streamlit-expanderHeader {
            background-color: #FFFFFF !important;
            color: #212529 !important;
            border: 1px solid #DEE2E6 !important;
            border-radius: 4px !important;
        }

        .streamlit-expanderContent {
            background-color: #FFFFFF !important;
            color: #212529 !important;
            border: 1px solid #DEE2E6 !important;
            border-radius: 0 0 4px 4px !important;
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
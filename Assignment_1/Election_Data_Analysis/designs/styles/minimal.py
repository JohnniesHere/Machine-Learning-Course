def get_style():
    return """
        <style>
        .main { 
            padding: 2rem; 
            background-color: #FAFAFA; 
        }
        h1 {
            color: #333333;
            text-align: center;
            font-weight: 400 !important;
            font-size: 2.2rem !important;
            letter-spacing: 0.5px;
            margin-bottom: 2rem !important;
        }
        .stButton>button {
            background-color: #333333;
            color: white;
            border: none;
            border-radius: 2px;
            padding: 0.5rem 1rem;
            font-weight: 400;
            transition: all 0.2s ease;
        }
        .stButton>button:hover {
            background-color: #555555;
        }
        .stExpander {
            background-color: #FFFFFF;
            border: 1px solid #EEEEEE;
            border-radius: 4px;
            padding: 1rem;
            margin: 1rem 0;
        }
        .stSelectbox, .stNumberInput {
            background-color: #FFFFFF;
            border: 1px solid #EEEEEE;
            border-radius: 2px;
        }
        </style>
    """

theme_info = {
    "name": "Minimal",
    "description": "Clean, minimalist design focusing on content",
    "preview_bg_color": "#FAFAFA",
    "preview_text_color": "#333333",
    "preview_accent_color": "#555555"
}
def get_style():
    return """
        <style>
        .main { padding: 2rem; background-color: #0E1117; }
        h1 {
            color: #FF4B4B;
            text-align: center;
            font-weight: 800 !important;
            font-size: 3rem !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .stButton>button {
            background: linear-gradient(45deg, #FF4B4B, #FF6B6B);
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        .stExpander {
            background: rgba(38, 39, 48, 0.8);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        </style>
    """

theme_info = {
    "name": "Modern Dark",
    "description": "A sleek dark theme with red accents and modern effects",
    "preview_bg_color": "#0E1117",
    "preview_text_color": "#FF4B4B",
    "preview_accent_color": "#FF6B6B"
}
def get_style():
    return """
        <style>
        .main {
            padding: 2rem;
            background: linear-gradient(135deg, #1A1A2E, #16213E);
        }
        h1 {
            background: linear-gradient(45deg, #6C63FF, #FF6584);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            font-weight: 700 !important;
            font-size: 2.8rem !important;
            margin-bottom: 2rem !important;
        }
        .stButton>button {
            background: linear-gradient(45deg, #6C63FF, #FF6584);
            border: none;
            border-radius: 25px;
            padding: 0.7rem 1.5rem;
            font-weight: 600;
            box-shadow: 0 4px 15px rgba(108,99,255,0.2);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(108,99,255,0.4);
        }
        .stExpander {
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.1);
            padding: 1.5rem;
            margin: 1rem 0;
        }
        .stSelectbox, .stNumberInput {
            background: rgba(255,255,255,0.1);
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.2);
        }
        </style>
    """

theme_info = {
    "name": "Vibrant Gradient",
    "description": "Modern design with vibrant gradients and glass effects",
    "preview_bg_color": "#1A1A2E",
    "preview_text_color": "#6C63FF",
    "preview_accent_color": "#FF6584"
}
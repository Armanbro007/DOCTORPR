from __future__ import annotations

from io import BytesIO

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from src.gemini_vision import GeminiDecodeResult, decode_prescription_with_gemini


def inject_responsive_styles(dark_mode: bool) -> None:
    # Color scheme based on theme
    if dark_mode:
        # Dark theme: Deep charcoal with electric teal accents
        bg_color = "#0f172a"
        surface_bg = "#1e293b"
        accent_color = "#14b8a6"
        accent_light = "#2dd4bf"
        text_primary = "#f1f5f9"
        text_secondary = "#cbd5e1"
        border_color = "rgba(20, 184, 166, 0.1)"
        button_hover = "#0d9488"
    else:
        # Light theme: Clean white-smoke with soft emerald accents
        bg_color = "#f8fafc"
        surface_bg = "#ffffff"
        accent_color = "#059669"
        accent_light = "#10b981"
        text_primary = "#1e293b"
        text_secondary = "#64748b"
        border_color = "rgba(16, 185, 129, 0.1)"
        button_hover = "#047857"

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        * {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}

        html, body, [data-testid="stAppViewContainer"] {{
            background-color: {bg_color} !important;
            color: {text_primary} !important;
        }}

        [data-testid="stDecoration"],
        .stDeployButton,
        footer {{
            display: none !important;
            visibility: hidden !important;
        }}

        .block-container {{
            max-width: 900px;
            padding-left: 1.5rem;
            padding-right: 1.5rem;
            padding-top: 2rem;
        }}

        /* Header Styling */
        h1 {{
            color: {text_primary} !important;
            font-weight: 700 !important;
            font-size: 2.2rem !important;
            margin-bottom: 0.5rem !important;
            letter-spacing: -0.02em;
        }}

        .stCaption {{
            color: {text_secondary} !important;
            font-size: 1rem !important;
            font-weight: 500 !important;
            margin-bottom: 2.5rem !important;
        }}

        h2, h3 {{
            color: {text_primary} !important;
            font-weight: 600 !important;
            margin-top: 1.5rem !important;
            margin-bottom: 1rem !important;
        }}

        /* File Uploader Styling */
        [data-testid="stFileUploader"] {{
            border: 2px dashed {accent_light} !important;
            border-radius: 12px !important;
            padding: 2.5rem !important;
            background: {surface_bg} !important;
            transition: all 0.3s ease !important;
        }}

        [data-testid="stFileUploader"]:hover {{
            background: rgba({accent_color.lstrip('#'): <4}, 0.05) !important;
            border-color: {accent_light} !important;
        }}

        /* Button Styling */
        div.stButton > button {{
            width: 100%;
            min-height: 48px;
            white-space: normal;
            background-color: {accent_color} !important;
            color: white !important;
            font-weight: 600 !important;
            border: none !important;
            border-radius: 8px !important;
            transition: all 0.2s ease !important;
            font-size: 0.95rem !important;
        }}

        div.stButton > button:hover {{
            background-color: {button_hover} !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 8px 16px rgba({accent_color.lstrip('#'): <4}, 0.3) !important;
        }}

        div.stButton > button:active {{
            transform: translateY(0) !important;
        }}

        /* Download Button Styling */
        div.stDownloadButton > button {{
            width: 100%;
            min-height: 48px;
            white-space: normal;
            background-color: {accent_color} !important;
            color: white !important;
            font-weight: 600 !important;
            border: none !important;
            border-radius: 8px !important;
            transition: all 0.2s ease !important;
            font-size: 0.95rem !important;
        }}

        div.stDownloadButton > button:hover {{
            background-color: {button_hover} !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 8px 16px rgba({accent_color.lstrip('#'): <4}, 0.3) !important;
        }}

        /* Info/Success/Warning/Error Messages */
        [data-testid="stAlert"] {{
            background-color: rgba({accent_color.lstrip('#'): <4}, 0.08) !important;
            border: 1px solid {border_color} !important;
            border-radius: 8px !important;
            color: {text_primary} !important;
            padding: 1.25rem !important;
            font-size: 0.95rem !important;
        }}

        /* Image Container */
        [data-testid="stImage"] {{
            border-radius: 10px !important;
            overflow: hidden !important;
            border: 1px solid {border_color} !important;
            background: {surface_bg} !important;
            padding: 0.5rem !important;
        }}

        [data-testid="stImage"] img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px !important;
        }}

        /* Text Area Styling */
        textarea {{
            background-color: {surface_bg} !important;
            color: {text_primary} !important;
            border: 1px solid {border_color} !important;
            border-radius: 8px !important;
            font-size: 14px !important;
            font-family: 'Monaco', 'Courier New', monospace !important;
            padding: 1rem !important;
        }}

        textarea:focus {{
            border-color: {accent_light} !important;
            box-shadow: 0 0 0 3px rgba({accent_color.lstrip('#'): <4}, 0.1) !important;
        }}

        /* General text and markdown */
        .stMarkdown {{
            color: {text_primary} !important;
            line-height: 1.6;
        }}

        .stMarkdown p {{
            margin-bottom: 1rem;
        }}

        /* Spinners and progress indicators */
        [data-testid="stSpinner"] {{
            color: {accent_light} !important;
        }}

        /* Dividers */
        hr {{
            border-color: {border_color} !important;
        }}

        @media (max-width: 768px) {{
            .block-container {{
                padding-left: 1rem;
                padding-right: 1rem;
                padding-top: 1rem;
            }}

            h1 {{
                font-size: 1.75rem !important;
            }}

            h2 {{
                font-size: 1.25rem !important;
            }}

            [data-testid="stFileUploader"] {{
                padding: 1.5rem !important;
            }}

            div.stButton > button,
            div.stDownloadButton > button {{
                min-height: 44px;
                font-size: 0.9rem !important;
            }}

            textarea {{
                font-size: 13px !important;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_pdf(image: Image.Image, result: GeminiDecodeResult) -> bytes:
    page_width, page_height = 1240, 1754
    margin = 90
    page = Image.new("RGB", (page_width, page_height), "white")
    draw = ImageDraw.Draw(page)

    # Font setup
    try:
        title_font = ImageFont.load_default(size=48)
        heading_font = ImageFont.load_default(size=32)
        subheading_font = ImageFont.load_default(size=28)
        body_font = ImageFont.load_default(size=24)
        small_font = ImageFont.load_default(size=20)
    except:
        title_font = ImageFont.load_default()
        heading_font = ImageFont.load_default()
        subheading_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Color palette
    primary_color = (20, 184, 166)  # Teal
    secondary_color = (100, 116, 139)  # Gray
    dark_color = (15, 23, 42)  # Dark slate

    y = margin

    # Header with accent line
    draw.rectangle([(margin, y), (page_width - margin, y + 3)], fill=primary_color)
    y += 20

    # Title
    draw.text((margin, y), "DOCTOR'S PRESCRIPTION DIGITIZER", fill=dark_color, font=title_font)
    y += 65

    # Subtitle
    draw.text((margin, y), "Digital Prescription Report", fill=secondary_color, font=heading_font)
    y += 50

    # Confidence score with visual indicator
    confidence_pct = result.confidence * 100
    draw.text((margin, y), "CONFIDENCE SCORE", fill=dark_color, font=subheading_font)
    y += 35
    draw.text((margin + 20, y), f"{confidence_pct:.1f}%", fill=primary_color, font=heading_font)
    y += 50

    # Separator line
    draw.line([(margin, y), (page_width - margin, y)], fill=secondary_color, width=1)
    y += 30

    # Medicines Section
    draw.text((margin, y), "DETECTED MEDICINES", fill=dark_color, font=subheading_font)
    y += 45

    if result.medicines:
        for idx, medicine in enumerate(result.medicines, 1):
            draw.text((margin + 20, y), f"{idx}. {medicine}", fill=dark_color, font=body_font)
            y += 40
    else:
        draw.text((margin + 20, y), "No clear medicine names detected.", fill=secondary_color, font=body_font)
        y += 40

    y += 20

    # Separator line
    draw.line([(margin, y), (page_width - margin, y)], fill=secondary_color, width=1)
    y += 30

    # Full Transcription Section
    draw.text((margin, y), "FULL TRANSCRIPTION", fill=dark_color, font=subheading_font)
    y += 45

    wrapped_lines: list[str] = []
    for original_line in result.transcription.splitlines() or [""]:
        line = original_line.strip()
        while len(line) > 85:
            wrapped_lines.append(line[:85])
            line = line[85:]
        if line:
            wrapped_lines.append(line)

    for line in wrapped_lines[:18]:
        draw.text((margin + 20, y), line, fill=dark_color, font=body_font)
        y += 34

    y += 30

    # Safety Warning
    draw.rectangle([(margin, y), (page_width - margin, y + 120)], outline=primary_color, width=2)
    y += 15
    draw.text((margin + 20, y), "⚠ SAFETY NOTICE", fill=primary_color, font=subheading_font)
    y += 40
    warning_text = "This digitization is AI-generated. Verify all information with a doctor or pharmacist before using any medicine."
    draw.multiline_text((margin + 20, y), warning_text, fill=dark_color, font=small_font, spacing=8)

    # Footer
    footer_y = page_height - 60
    draw.text((margin, footer_y), "Generated by Doctor's Prescription Digitizer", fill=secondary_color, font=small_font)

    output = BytesIO()
    page.save(output, format="PDF", resolution=100.0)
    return output.getvalue()


st.set_page_config(
    page_title="Doctor's Prescription Digitizer",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Initialize dark mode state
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# Inject responsive styles based on current theme
inject_responsive_styles(st.session_state.dark_mode)

# Create a floating theme toggle in sidebar
with st.sidebar:
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🌙" if not st.session_state.dark_mode else "☀️", key="theme_toggle", help="Toggle dark/light mode"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

# Hero Section
st.markdown("<div style='text-align: center; margin-bottom: 1rem;'>", unsafe_allow_html=True)
st.title("📋 Doctor's Prescription Digitizer")
st.markdown("<div class='stCaption' style='text-align: center;'>Transform handwritten prescriptions into digital text with AI-powered accuracy</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Upload Section with visual hierarchy
st.markdown("### 📸 Upload Prescription")
uploaded_file = st.file_uploader(
    "Drag and drop your prescription image here",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <p style='font-size: 1.1rem; margin-bottom: 0.5rem;'>Ready to digitize?</p>
        <p style='opacity: 0.7;'>Upload a clear photo of your handwritten prescription</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Preview Section
st.markdown("### 👁️ Preview")
col1, col2 = st.columns([2, 1])
with col1:
    st.image(uploaded_file, caption="Your uploaded prescription", use_container_width=True)
with col2:
    st.metric("File Status", "✓ Ready", delta="Upload Complete")

# Decode Button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    decode_clicked = st.button("🔍 Decode Prescription", type="primary", use_container_width=True)

if not decode_clicked:
    st.stop()

# Processing
image = Image.open(uploaded_file)

with st.spinner("🤖 Reading the handwritten prescription..."):
    try:
        result = decode_prescription_with_gemini(image)
    except Exception as exc:
        message = str(exc)
        st.error(f"❌ Error: {message}")
        if "GEMINI_API_KEY is not set" in message:
            st.info(
                "To enable this free API mode, get a Gemini API key from Google AI Studio, "
                "then run: `setx GEMINI_API_KEY \"your_key_here\"`. "
                "Restart PowerShell and Streamlit after setting it."
            )
        else:
            st.info(
                "This usually means the service is busy or your free-tier quota is temporarily limited. "
                "Try again after a minute, or set `GEMINI_VISION_MODEL` to `gemini-2.0-flash`."
            )
        st.stop()

# Results Section
st.success("✅ Prescription decoding completed")
st.metric("Confidence Score", f"{result.confidence * 100:.1f}%", delta="High Accuracy")

# Detected Medicines
st.markdown("### 💊 Detected Medicines")
if result.medicines:
    for idx, medicine in enumerate(result.medicines, 1):
        st.markdown(f"**{idx}.** {medicine}")
else:
    st.warning("⚠️ No clear medicine names detected. Please verify with a pharmacist.")

# Full Transcription
st.markdown("### 📝 Full Transcription")
st.text_area(
    "Decoded text",
    result.transcription,
    height=220,
    disabled=True,
    label_visibility="collapsed"
)

if result.notes:
    st.info(f"📌 Notes: {result.notes}")

# PDF Export
pdf_bytes = build_pdf(image, result)
col1, col2 = st.columns(2)
with col1:
    st.download_button(
        label="📥 Download PDF Report",
        data=pdf_bytes,
        file_name="prescription_decode_result.pdf",
        mime="application/pdf",
        use_container_width=True
    )

with col2:
    if st.button("🔄 Decode Another", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# Safety Notice
st.warning(
    "⚠️ **Safety Reminder**\n\n"
    "This system digitizes handwritten medical text using AI. "
    "Always have a pharmacist or doctor verify the transcription before using any medicine."
)

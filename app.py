from __future__ import annotations

from io import BytesIO

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from src.gemini_vision import GeminiDecodeResult, decode_prescription_with_gemini


def inject_responsive_styles() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stDecoration"],
        .stDeployButton,
        footer {
            display: none !important;
            visibility: hidden !important;
        }

        .block-container {
            max-width: 920px;
            padding-left: 1.2rem;
            padding-right: 1.2rem;
        }

        [data-testid="stImage"] img {
            max-width: 100%;
            height: auto;
        }

        div.stButton > button,
        div.stDownloadButton > button {
            width: 100%;
            min-height: 44px;
            white-space: normal;
        }

        textarea {
            font-size: 15px !important;
        }

        @media (max-width: 640px) {
            .block-container {
                padding-top: 1rem;
                padding-left: 0.8rem;
                padding-right: 0.8rem;
            }

            h1 {
                font-size: 1.65rem !important;
                line-height: 1.25 !important;
            }

            h2, h3 {
                font-size: 1.15rem !important;
            }

            [data-testid="stCaptionContainer"],
            .stMarkdown,
            .stAlert {
                font-size: 0.95rem;
            }

            [data-testid="stFileUploader"] {
                width: 100%;
            }

            textarea {
                font-size: 14px !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_pdf(image: Image.Image, result: GeminiDecodeResult) -> bytes:
    page_width, page_height = 1240, 1754
    margin = 90
    page = Image.new("RGB", (page_width, page_height), "white")
    draw = ImageDraw.Draw(page)

    title_font = ImageFont.load_default(size=42)
    heading_font = ImageFont.load_default(size=30)
    body_font = ImageFont.load_default(size=24)

    y = margin
    draw.text((margin, y), "Doctor's Prescription Digitizer", fill="black", font=title_font)
    y += 75
    draw.text((margin, y), "Handwriting Decode Result", fill="black", font=heading_font)
    y += 55
    draw.text((margin, y), f"Accuracy / Confidence: {result.confidence * 100:.2f}%", fill="black", font=body_font)
    y += 50

    draw.text((margin, y), "Detected medicines:", fill="black", font=heading_font)
    y += 45
    if result.medicines:
        for medicine in result.medicines:
            draw.text((margin, y), f"- {medicine}", fill="black", font=body_font)
            y += 34
    else:
        draw.text((margin, y), "No clear medicine names detected.", fill="black", font=body_font)
        y += 34

    y += 30
    draw.text((margin, y), "Full transcription:", fill="black", font=heading_font)
    y += 45
    wrapped_lines: list[str] = []
    for original_line in result.transcription.splitlines() or [""]:
        line = original_line.strip()
        while len(line) > 80:
            wrapped_lines.append(line[:80])
            line = line[80:]
        wrapped_lines.append(line)

    for line in wrapped_lines[:22]:
        draw.text((margin, y), line, fill="black", font=body_font)
        y += 32

    y += 25
    warning = "Safety note: Verify the transcription with a doctor or pharmacist before using any medicine."
    draw.multiline_text((margin, y), warning, fill="black", font=body_font, spacing=8)

    output = BytesIO()
    page.save(output, format="PDF", resolution=100.0)
    return output.getvalue()


st.set_page_config(
    page_title="Bangladeshi Prescription Digitizer",
    layout="centered",
)

inject_responsive_styles()

st.title("Doctor's Prescription Digitizer")
st.caption("Upload a prescription image to digitize readable handwritten text.")

uploaded_file = st.file_uploader(
    "Upload handwritten prescription image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is None:
    st.info("Upload a prescription image to start.")
    st.stop()

image = Image.open(uploaded_file)
st.image(image, caption="Uploaded image", use_container_width=True)

if st.button("Decode Prescription", type="primary"):
    with st.spinner("Reading the handwritten prescription..."):
        try:
            result = decode_prescription_with_gemini(image)
        except Exception as exc:
            message = str(exc)
            st.error(message)
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

    st.success("Prescription decoding completed")
    st.info(f"Accuracy / confidence: {result.confidence * 100:.2f}%")

    st.subheader("Detected medicines")
    if result.medicines:
        for medicine in result.medicines:
            st.write(f"- {medicine}")
    else:
        st.warning("No clear medicine names detected.")

    st.subheader("Full transcription")
    st.text_area("Decoded handwritten text", result.transcription, height=220)

    if result.notes:
        st.caption(f"Notes: {result.notes}")

    pdf_bytes = build_pdf(image, result)
    st.download_button(
        label="Download PDF Result",
        data=pdf_bytes,
        file_name="prescription_decode_result.pdf",
        mime="application/pdf",
    )

    st.warning(
        "This system only digitizes handwritten medical text. "
        "Final medicine verification must be done by a doctor or pharmacist."
    )

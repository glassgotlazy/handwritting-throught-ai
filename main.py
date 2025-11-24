import streamlit as st
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
import textwrap
import tempfile
import os
from pathlib import Path

# Initialize OpenAI client with Streamlit secrets
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))

def generate_assignment_answer(question: str, pages: int = 2) -> str:
    approx_words_per_page = 180
    target_words = pages * approx_words_per_page

    prompt = f"""
You are an Indian first-year LLB student.
Write an exam-style answer to the following question.

Question: {question}

Constraints:
- Around {target_words} words.
- Simple vocabulary.
- Clear headings and short paragraphs.
- No bullet points, just paragraphs.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # change if you need a different model
            messages=[{"role": "user", "content": prompt}],
            max_tokens=min(target_words * 2, 2048),
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return ""

def load_font_from_path(font_path: str | Path | None, font_size: int):
    """
    Try to load a TTF/OTF font from disk. If it fails, return PIL's default font.
    """
    if font_path is None:
        st.warning("No font path provided — using default font.")
        return ImageFont.load_default()

    try:
        return ImageFont.truetype(str(font_path), font_size)
    except OSError as e:
        st.warning(f"Could not load font at '{font_path}': {e}")
        st.info("Upload a TTF/OTF font in the sidebar or place a font at 'fonts/handwriting.ttf' in the repo.")
        return ImageFont.load_default()
    except Exception as e:
        st.error(f"Unexpected font loading error: {e}")
        return ImageFont.load_default()

def wrap_text_to_pixel_width(draw: ImageDraw.Draw, text: str, font: ImageFont.FreeTypeFont, max_width: int):
    """Wrap text to fit into max_width (pixels) using the provided draw/font."""
    lines = []
    paragraphs = text.split("\n")
    for para in paragraphs:
        words = para.strip().split()
        if not words:
            lines.append("")  # blank line for paragraph break
            continue
        current_line = words[0]
        for word in words[1:]:
            test_line = current_line + " " + word
            bbox = draw.textbbox((0, 0), test_line, font=font)
            line_width = bbox[2] - bbox[0]
            if line_width <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
    return lines

def render_handwritten_image(
    text: str,
    font_obj: ImageFont.ImageFont,
    img_width: int = 1240,
    img_height: int = 1754,
    margin_left: int = 120,
    margin_top: int = 120,
    line_spacing: int = 10,
):
    image = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(image)

    max_text_width = img_width - 2 * margin_left
    lines = wrap_text_to_pixel_width(draw, text, font_obj, max_text_width)

    x, y = margin_left, margin_top
    for line in lines:
        if y > img_height - margin_top:
            break
        draw.text((x, y), line, font=font_obj, fill="black")
        bbox = draw.textbbox((0, 0), line or "A", font=font_obj)
        line_height = bbox[3] - bbox[1]
        y += line_height + line_spacing

    return image

# Streamlit UI
st.title("AI Handwritten Assignment Generator ✍️")

# Sidebar font options
st.sidebar.header("Font options")
uploaded_font = st.sidebar.file_uploader("Upload a TTF/OTF font to use (optional)", type=["ttf", "otf"])
font_size = st.sidebar.slider("Font size", min_value=12, max_value=80, value=32, step=1)

# Resolve bundled font path relative to this file so deployment finds it
try:
    repo_font_path = Path(__file__).parent / "fonts" / "handwriting.ttf"
except NameError:
    # __file__ may not be defined in some streamlit environments; fall back to cwd
    repo_font_path = Path(os.getcwd()) / "fonts" / "handwriting.ttf"

font_path_to_use = None
if uploaded_font is not None:
    try:
        tmp_dir = tempfile.gettempdir()
        tmp_font_path = Path(tmp_dir) / uploaded_font.name
        with open(tmp_font_path, "wb") as f:
            f.write(uploaded_font.getbuffer())
        font_path_to_use = tmp_font_path
        st.sidebar.success(f"Using uploaded font: {uploaded_font.name}")
    except Exception as e:
        st.sidebar.error(f"Failed to save uploaded font: {e}")
        font_path_to_use = None
else:
    if repo_font_path.exists():
        font_path_to_use = repo_font_path
        st.sidebar.info(f"Using bundled font: {repo_font_path.name}")
    else:
        st.sidebar.warning("No bundled font found at 'fonts/handwriting.ttf'. Upload a font or add one to the repo.")

# Debugging helpers
st.sidebar.write("Font debug")
st.sidebar.write("Resolved font path:", str(font_path_to_use) if font_path_to_use else "None")
if font_path_to_use:
    st.sidebar.write("Exists on disk:", Path(font_path_to_use).exists())

question = st.text_area("Enter your assignment question", height=150)
pages = st.number_input("Approximate pages (handwritten)", min_value=1, max_value=10, value=2)

if st.button("Generate Handwritten Assignment"):
    if not question.strip():
        st.warning("Please enter an assignment question.")
    else:
        with st.spinner("Generating answer with OpenAI GPT..."):
            answer_text = generate_assignment_answer(question, int(pages))

        if not answer_text:
            st.error("No answer generated. See errors above.")
        else:
            font_obj = load_font_from_path(font_path_to_use, int(font_size))

            with st.spinner("Rendering handwritten image..."):
                img = render_handwritten_image(answer_text, font_obj=font_obj, img_width=1240, img_height=1754)

            st.image(img, caption="Generated handwritten assignment page", use_column_width=True)

            output_path = "assignment_page.png"
            try:
                img.save(output_path)
                st.success(f"Rendered page saved as {output_path}")
                with open(output_path, "rb") as file:
                    st.download_button(
                        label="Download Handwritten Assignment Image",
                        data=file,
                        file_name=output_path,
                        mime="image/png",
                    )
            except Exception as e:
                st.error(f"Failed to save image: {e}")

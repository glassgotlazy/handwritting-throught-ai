import streamlit as st
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import textwrap
import tempfile
import os
from pathlib import Path
import random
import io
import zipfile
from typing import List

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
            model="gpt-4o",
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

def create_paper_background(img_width: int, img_height: int, paper_color: tuple, texture_strength: float = 0.12):
    """Create a slightly textured 'paper' background using noise and blur."""
    base = Image.new("RGB", (img_width, img_height), paper_color)
    # Add noise
    noise = Image.effect_noise((img_width, img_height), 64).convert("L")
    noise = noise.point(lambda p: p * texture_strength)
    noise_rgb = Image.merge("RGB", (noise, noise, noise))
    textured = ImageChops_add = Image.blend(base, noise_rgb, alpha=0.14)
    # Soften and return
    return textured.filter(ImageFilter.GaussianBlur(radius=0.5))

def draw_ruled_lines(draw: ImageDraw.Draw, img_width: int, img_height: int, margin_left: int, margin_top: int, margin_right: int, line_spacing: int, line_color: tuple):
    y = margin_top
    while y < img_height - margin_top:
        draw.line([(margin_left - 20, y), (img_width - margin_right + 20, y)], fill=line_color, width=1)
        y += line_spacing

def render_handwritten_image(
    text: str,
    font_obj: ImageFont.ImageFont,
    img_width: int = 1240,
    img_height: int = 1754,
    margin_left: int = 120,
    margin_top: int = 120,
    line_spacing: int = 12,
    ink_color: tuple = (10, 10, 10),
    paper_color: tuple = (245, 242, 230),
    ruled: bool = True,
    rotation_jitter: float = 1.5,
    header: str | None = None,
    footer: str | None = None,
):
    # Create textured paper base
    try:
        base = Image.new("RGB", (img_width, img_height), paper_color)
        noise = Image.effect_noise((img_width, img_height), 64).convert("L")
        noise = noise.point(lambda p: p * 0.07)
        noise_rgb = Image.merge("RGB", (noise, noise, noise))
        base = Image.blend(base, noise_rgb, alpha=0.12)
    except Exception:
        # Fallback: plain paper
        base = Image.new("RGB", (img_width, img_height), paper_color)

    # Transparent layer for handwriting
    text_layer = Image.new("RGBA", (img_width, img_height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(text_layer)

    max_text_width = img_width - margin_left - margin_left
    lines = wrap_text_to_pixel_width(draw, text, font_obj, max_text_width)

    # Optional ruled lines
    if ruled:
        ruled_color = (180, 200, 215)  # subtle blue
        draw_ruled_lines(draw, img_width, img_height, margin_left, margin_top, margin_left, int(font_obj.size * 1.9), ruled_color)

    # Write header
    header_y = margin_top - int(font_obj.size * 1.6)
    if header:
        draw.text((margin_left, header_y), header, font=font_obj, fill=tuple(min(255, c + 20) for c in ink_color))

    # Draw text lines
    x, y = margin_left, margin_top
    for line in lines:
        if y > img_height - margin_top - font_obj.size:
            break
        # slight random horizontal jitter for authenticity
        jitter_x = random.randint(-2, 2)
        draw.text((x + jitter_x, y), line, font=font_obj, fill=ink_color)
        bbox = draw.textbbox((0, 0), line or "A", font=font_obj)
        line_height = bbox[3] - bbox[1]
        y += line_height + line_spacing

    # Footer
    if footer:
        footer_text_bbox = draw.textbbox((0, 0), footer, font=font_obj)
        footer_w = footer_text_bbox[2] - footer_text_bbox[0]
        draw.text((img_width - margin_left - footer_w, img_height - margin_top + int(font_obj.size * 0.2)), footer, font=font_obj, fill=ink_color)

    # Add subtle ink smudge/noise
    text_layer = text_layer.filter(ImageFilter.GaussianBlur(radius=0.2))

    # Rotate slightly for realism
    angle = random.uniform(-rotation_jitter, rotation_jitter)
    rotated = text_layer.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(255,255,255,0))

    # Composite onto paper
    final = base.convert("RGBA")
    final = Image.alpha_composite(final, rotated).convert("RGB")

    # Add vignette / burn edges
    vignette = Image.new("L", (img_width, img_height), 0)
    vdraw = ImageDraw.Draw(vignette)
    for i in range(0, max(img_width, img_height) // 2, 10):
        alpha = int(3 * i / (max(img_width, img_height) // 2))
        vdraw.ellipse([-i, -i, img_width + i, img_height + i], fill=alpha)
    final = Image.composite(final, Image.new("RGB", final.size, (230, 225, 210)), vignette.filter(ImageFilter.GaussianBlur(radius=120)))

    # Slight color/contrast tweak to make it "cool"
    final = ImageOps.autocontrast(final, cutoff=1)

    return final

def split_text_into_pages(text: str, pages: int):
    words = text.split()
    if pages <= 1:
        return [text]
    per_page = max(100, len(words) // pages)
    chunks = []
    i = 0
    for p in range(pages):
        chunk = words[i:i + per_page]
        i += per_page
        if p == pages - 1:
            # append remaining
            chunk += words[i:]
        chunks.append(" ".join(chunk).strip())
    # make sure no empty chunk
    return [c if c else " " for c in chunks]

# Streamlit UI
st.set_page_config(page_title="AI Handwritten Assignment Generator", layout="wide")
st.title("AI Handwritten Assignment Generator ✍️ — Multi-page & Styled")

# Sidebar controls
st.sidebar.header("Font & Style options")
uploaded_font = st.sidebar.file_uploader("Upload a TTF/OTF font to use (optional)", type=["ttf", "otf"])
font_size = st.sidebar.slider("Font size (px)", min_value=18, max_value=72, value=32, step=1)
pages = st.sidebar.number_input("Approximate pages (handwritten)", min_value=1, max_value=10, value=2)
ink_color_choice = st.sidebar.selectbox("Ink color", options=["Black", "Dark Blue", "Brown", "Gray"])
style_ruled = st.sidebar.checkbox("Add ruled lines", value=True)
rotation_jitter = st.sidebar.slider("Rotation jitter (degrees)", min_value=0.0, max_value=5.0, value=1.5, step=0.1)
paper_color_choice = st.sidebar.selectbox("Paper color", options=["White", "Ivory", "Aged (beige)"])

# Resolve bundled font path relative to this file
try:
    repo_font_path = Path(__file__).parent / "fonts" / "handwriting.ttf"
except NameError:
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

# Map color choices to RGB
ink_colors_map = {
    "Black": (20, 20, 20),
    "Dark Blue": (8, 35, 86),
    "Brown": (60, 30, 10),
    "Gray": (60, 60, 60),
}
paper_colors_map = {
    "White": (255, 255, 255),
    "Ivory": (245, 242, 230),
    "Aged (beige)": (238, 230, 210),
}

st.sidebar.markdown(" ")
st.sidebar.markdown("Generated pages will be saved as images and offered as a ZIP and combined PDF.")

# Input area
question = st.text_area("Enter your assignment question", height=150)

if st.button("Generate Styled Handwritten Assignment"):
    if not question.strip():
        st.warning("Please enter an assignment question.")
    else:
        with st.spinner("Generating answer with OpenAI GPT..."):
            answer_text = generate_assignment_answer(question, int(pages))

        if not answer_text:
            st.error("No answer generated. See errors above.")
        else:
            font_obj = load_font_from_path(font_path_to_use, int(font_size))

            # Split text into page-wise chunks
            chunks = split_text_into_pages(answer_text, int(pages))

            images: List[Image.Image] = []
            tmp_dir = Path(tempfile.mkdtemp(prefix="handwritten_pages_"))

            st.info(f"Rendering {len(chunks)} page(s)...")
            progress = st.progress(0)
            for idx, chunk in enumerate(chunks, start=1):
                with st.spinner(f"Rendering page {idx}/{len(chunks)}..."):
                    # Slight variation per page
                    ink_variation = tuple(max(0, min(255, c + random.randint(-10, 10))) for c in ink_colors_map.get(ink_color_choice, (0,0,0)))
                    paper_color = paper_colors_map.get(paper_color_choice, (245,242,230))
                    header = f"Answer — Page {idx}" if len(chunks) > 1 else None
                    footer = "Generated with AI Handwriter"  # simple footer label

                    img = render_handwritten_image(
                        chunk,
                        font_obj=font_obj,
                        img_width=1240,
                        img_height=1754,
                        margin_left=100,
                        margin_top=120,
                        line_spacing=int(font_obj.size * 0.45),
                        ink_color=ink_variation,
                        paper_color=paper_color,
                        ruled=style_ruled,
                        rotation_jitter=float(rotation_jitter),
                        header=header,
                        footer=footer,
                    )
                    out_path = tmp_dir / f"assignment_page_{idx:02d}.png"
                    img.save(out_path)
                    images.append(img)
                progress.progress(idx / len(chunks))

            st.success("Rendering complete!")

            # Show thumbnails
            st.write("Preview (click to enlarge):")
            cols = st.columns(min(4, len(images)))
            for i, img in enumerate(images):
                with cols[i % len(cols)]:
                    st.image(img.resize((300, int(300 * img.height / img.width))), use_column_width=False, caption=f"Page {i+1}")

            # Prepare ZIP
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for i, img in enumerate(images, start=1):
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format="PNG")
                    img_bytes.seek(0)
                    zf.writestr(f"assignment_page_{i:02d}.png", img_bytes.read())
            zip_buffer.seek(0)

            st.download_button(
                label="Download all pages as ZIP",
                data=zip_buffer,
                file_name="handwritten_assignment_pages.zip",
                mime="application/zip",
            )

            # Prepare combined PDF
            if images:
                pdf_bytes = io.BytesIO()
                rgb_images = [im.convert("RGB") for im in images]
                rgb_images[0].save(pdf_bytes, format="PDF", save_all=True, append_images=rgb_images[1:])
                pdf_bytes.seek(0)
                st.download_button(
                    label="Download combined PDF",
                    data=pdf_bytes,
                    file_name="handwritten_assignment.pdf",
                    mime="application/pdf",
                )

            st.success(f"Saved {len(images)} images to {tmp_dir}")
            st.write("If you want different styling (more ink variation, different font), tweak the sidebar and regenerate.")

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
from typing import List, Union

# Optional matplotlib usage for rendering LaTeX math to images
try:
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    MATPLOTLIB_AVAILABLE = True
    # Use LaTeX-like mathtext via mathtext renderer
    rcParams["mathtext.fontset"] = "dejavusans"
except Exception:
    MATPLOTLIB_AVAILABLE = False

# Initialize OpenAI client with Streamlit secrets
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))

def generate_assignment_answer(question: str, pages: int = 2) -> str:
    """
    Generate an exam-style answer while preserving math in LaTeX.
    Ask the model to keep math in $...$ or $$...$$ and not modify it.
    """
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
- IMPORTANT: If you need to include any mathematical formula or expression, please use LaTeX math delimiters: inline math with $...$ or display math with $$...$$. Do NOT wrap math in markdown code fences. Do not include any other markup. Keep the answer plain text with math kept exactly between $ or $$ so it can be correctly rendered later.
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

def load_font_from_path(font_path: Union[str, Path, None], font_size: int):
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

# Math rendering/cache helpers
_math_image_cache = {}

def render_math_to_image(math_tex: str, font_size: int = 28, color=(0, 0, 0)):
    """
    Render math (LaTeX-like) to a transparent PIL image using matplotlib if available.
    math_tex should include delimiters like $...$ or $$...$$; we'll strip them.
    """
    key = (math_tex, font_size, color)
    if key in _math_image_cache:
        return _math_image_cache[key]

    if not MATPLOTLIB_AVAILABLE:
        # Fallback: return an image with the raw math string rendered as plain text
        font = ImageFont.load_default()
        dummy = Image.new("RGBA", (10, 10), (255, 255, 255, 0))
        d = ImageDraw.Draw(dummy)
        bbox = d.textbbox((0, 0), math_tex, font=font)
        img = Image.new("RGBA", (bbox[2] - bbox[0] + 4, bbox[3] - bbox[1] + 4), (255, 255, 255, 0))
        d = ImageDraw.Draw(img)
        d.text((2, 2), math_tex, font=font, fill=color)
        _math_image_cache[key] = img
        return img

    # Strip delimiters
    content = math_tex
    if content.startswith("$$") and content.endswith("$$"):
        content = content[2:-2]
    elif content.startswith("$") and content.endswith("$"):
        content = content[1:-1]

    # Create a matplotlib figure and render the math text
    fig = plt.figure(figsize=(0.01, 0.01), dpi=200)
    text_kwargs = dict(fontsize=font_size, color='#%02x%02x%02x' % color, ha='left', va='bottom')
    # Invisible axes
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    # Place text at (0,0)
    t = ax.text(0, 0, f"${content}$", **text_kwargs)
    # Draw and get bounding box
    fig.canvas.draw()
    bbox = t.get_window_extent(renderer=fig.canvas.get_renderer())
    width, height = int(bbox.width) + 6, int(bbox.height) + 6
    # Adjust figure to bbox and re-render
    plt.close(fig)
    fig = plt.figure(figsize=(width / 200.0, height / 200.0), dpi=200)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.text(0, 0, f"${content}$", **text_kwargs)
    fig.canvas.draw()
    # Extract RGBA buffer from canvas
    buf = fig.canvas.tostring_argb()
    w, h = fig.canvas.get_width_height()
    plt.close(fig)
    import numpy as np
    arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
    # Convert ARGB -> RGBA
    arr = arr[:, :, [1, 2, 3, 0]]
    pil_img = Image.fromarray(arr, mode="RGBA")
    # Trim transparent borders
    bbox = pil_img.getbbox()
    if bbox:
        pil_img = pil_img.crop(bbox)
    _math_image_cache[key] = pil_img
    return pil_img

def split_text_preserving_math(text: str):
    """
    Split text into tokens where math (between $...$ or $$...$$) is kept as a single token.
    Returns a list of tokens (strings). Math tokens include their delimiters.
    """
    tokens = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "$":
            # detect single or double delimiter
            if i + 1 < n and text[i+1] == "$":
                # $$...$$
                j = text.find("$$", i+2)
                if j == -1:
                    # no closing delimiter, take rest
                    tokens.append(text[i:])
                    break
                tokens.append(text[i:j+2])
                i = j + 2
            else:
                # $...$
                j = text.find("$", i+1)
                if j == -1:
                    tokens.append(text[i:])
                    break
                tokens.append(text[i:j+1])
                i = j + 1
        else:
            # accumulate until next $
            j = text.find("$", i)
            if j == -1:
                tokens.append(text[i:])
                break
            tokens.append(text[i:j])
            i = j
    return tokens

def wrap_text_to_pixel_width(draw: ImageDraw.Draw, text: str, font: ImageFont.FreeTypeFont, max_width: int):
    """
    Wrap text into lines based on pixel width, preserving math tokens as indivisible.
    Returns list of lines, where each line is a list of tokens. Token is a tuple (type, content).
    type = 'text' or 'math'
    """
    tokens = split_text_preserving_math(text)
    lines = []
    current_line = []
    current_width = 0

    for tok in tokens:
        if tok.startswith("$"):
            # math token: treat as single unit
            math_img = render_math_to_image(tok, font_size=int(font.size * 0.9), color=(0,0,0))
            tok_width = math_img.width
            tok_type = "math"
            tok_content = tok
        else:
            # plain text: split into words to wrap
            words = tok.replace("\n", " \n ").split()  # preserve line breaks as tokens '\n'
            for w in words:
                if w == "\n":
                    # force line break
                    if current_line:
                        lines.append(current_line)
                    current_line = []
                    current_width = 0
                    continue
                w_bbox = draw.textbbox((0, 0), w, font=font)
                w_width = w_bbox[2] - w_bbox[0]
                spacer = draw.textbbox((0, 0), " ", font=font)[2] - draw.textbbox((0,0)," ",font=font)[0]
                if current_width == 0:
                    # first word in line
                    current_line.append(("text", w))
                    current_width = w_width
                else:
                    if current_width + spacer + w_width <= max_width:
                        current_line.append(("text", " " + w))
                        current_width += spacer + w_width
                    else:
                        lines.append(current_line)
                        current_line = [("text", w)]
                        current_width = w_width
            continue

        # handle math token insertion (after words handling)
        spacer = draw.textbbox((0, 0), " ", font=font)[2] - draw.textbbox((0,0)," ",font=font)[0]
        if current_width == 0:
            current_line.append(("math", tok_content))
            current_width = tok_width
        else:
            if current_width + spacer + tok_width <= max_width:
                current_line.append(("text", " "))
                current_line.append(("math", tok_content))
                current_width += spacer + tok_width
            else:
                lines.append(current_line)
                current_line = [("math", tok_content)]
                current_width = tok_width

    if current_line:
        lines.append(current_line)

    return lines

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
    ruled: bool = False,
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
        base = Image.new("RGB", (img_width, img_height), paper_color)

    # Transparent layer for handwriting
    text_layer = Image.new("RGBA", (img_width, img_height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(text_layer)

    max_text_width = img_width - margin_left - margin_left
    lines = wrap_text_to_pixel_width(draw, text, font_obj, max_text_width)

    # Optional ruled lines (default disabled)
    if ruled:
        ruled_color = (180, 200, 215)
        draw_ruled_lines(draw, img_width, img_height, margin_left, margin_top, margin_left, int(font_obj.size * 1.9), ruled_color)

    # Write header (if any) slightly lighter ink
    header_y = margin_top - int(font_obj.size * 1.6)
    if header:
        draw.text((margin_left, header_y), header, font=font_obj, fill=tuple(min(255, c + 20) for c in ink_color))

    # Draw lines (each line is list of tokens)
    x_start, y = margin_left, margin_top
    spacer_width = draw.textbbox((0,0)," ", font=font_obj)[2] - draw.textbbox((0,0)," ", font=font_obj)[0]
    for line in lines:
        if y > img_height - margin_top - font_obj.size:
            break
        x = x_start
        # slight per-line horizontal jitter
        jitter_line = random.randint(-2, 2)
        for token_type, token_content in line:
            if token_type == "text":
                # token_content may include leading spaces (we preserved them)
                jitter_x = random.randint(-2, 2)
                draw.text((x + jitter_x + jitter_line, y), token_content, font=font_obj, fill=ink_color)
                bb = draw.textbbox((0,0), token_content or "A", font=font_obj)
                token_w = bb[2] - bb[0]
                x += token_w
            else:  # math token
                math_img = render_math_to_image(token_content, font_size=int(font_obj.size * 0.9), color=ink_color)
                # Optionally jitter math placement a little vertically to blend in
                v_jitter = random.randint(-2, 2)
                text_layer.paste(math_img, (int(x + jitter_line), int(y + v_jitter)), math_img)
                x += math_img.width + 2  # small spacing after math
        # determine line height approximately by font size and token heights
        y += int(font_obj.size * 1.0) + line_spacing

    # Footer omitted or optional — watermark removed by default
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

    # Slight color/contrast tweak
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
            chunk += words[i:]
        chunks.append(" ".join(chunk).strip())
    return [c if c else " " for c in chunks]

# Streamlit UI
st.set_page_config(page_title="AI Handwritten Assignment Generator", layout="wide")
st.title("AI Handwritten Assignment Generator ✍️ — Multi-page & Math-aware")

# Sidebar controls
st.sidebar.header("Font & Style options")
uploaded_font = st.sidebar.file_uploader("Upload a TTF/OTF font to use (optional)", type=["ttf", "otf"])
font_size = st.sidebar.slider("Font size (px)", min_value=18, max_value=72, value=32, step=1)
pages = st.sidebar.number_input("Approximate pages (handwritten) to GENERATE", min_value=1, max_value=10, value=2)
ink_color_choice = st.sidebar.selectbox("Ink color", options=["Black", "Dark Blue", "Brown", "Gray"])
style_ruled = st.sidebar.checkbox("Add ruled lines", value=False)  # default OFF (blueprint removed)
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
if not MATPLOTLIB_AVAILABLE:
    st.sidebar.warning("Matplotlib not available: math will be rendered as plain text. Install matplotlib for proper LaTeX rendering.")

# Input area
question = st.text_area("Enter your assignment question (you may include math in LaTeX, e.g. $E=mc^2$ )", height=150)

if st.button("Generate Styled Handwritten Assignment"):
    if not question.strip():
        st.warning("Please enter an assignment question.")
    else:
        with st.spinner("Generating answer with OpenAI GPT..."):
            answer_text = generate_assignment_answer(question, int(pages))

        if not answer_text:
            st.error("No answer generated. See errors above.")
        else:
            # Show approximate pages required for the generated text
            approx_words_per_page = 180
            word_count = len(answer_text.split())
            approx_pages_needed = max(1, (word_count + approx_words_per_page - 1) // approx_words_per_page)
            st.info(f"Generated answer contains ~{word_count} words. Approx pages needed (handwritten): {approx_pages_needed} pages (with ~{approx_words_per_page} words per page).")

            font_obj = load_font_from_path(font_path_to_use, int(font_size))

            # Split text into page-wise chunks (based on user 'pages' input)
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
                    footer = None  # watermark removed

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

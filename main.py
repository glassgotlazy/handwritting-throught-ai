import streamlit as st
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import tempfile
import os
from pathlib import Path
import random
import io
import zipfile
from typing import List, Union, Tuple, Dict
import math
import numpy as np
from dataclasses import dataclass
import base64
import traceback

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

@dataclass
class HandwritingConfig:
    char_rotation_range: float = 1.3
    char_scale_variance: float = 0.03
    char_shear_range: float = 2.0
    char_spacing_variance: float = 0.43
    baseline_wave_amplitude: float = 0.7
    baseline_wave_frequency: float = 180.0
    baseline_drift: float = 0.09
    ink_pressure_layers: int = 3       # Only 3 for minimal dotting
    ink_bleeding: float = 0.0          # Default: no blur
    paper_noise_intensity: float = 0.025 # Subtle
    ligature_detection: bool = True
    margin_irregularity: float = 2.2

def load_font_from_path(font_path: Union[str, Path, None], font_size: int):
    if font_path is None:
        return ImageFont.load_default()
    try:
        return ImageFont.truetype(str(font_path), font_size)
    except Exception:
        return ImageFont.load_default()

def create_textured_paper(width: int, height: int, base_color: Tuple[int,int,int], texture_intensity: float = 0.025) -> Image.Image:
    base = Image.new("RGB", (width, height), base_color)
    try:
        noise = Image.effect_noise((width // 2, height // 2), 64).convert("L")
        noise = noise.resize((width, height), Image.LANCZOS)
        noise = noise.point(lambda p: int(p * texture_intensity))
        noise_rgb = Image.merge("RGB", (noise, noise, noise))
        base = Image.blend(base, noise_rgb, alpha=0.06)
    except Exception:
        pass
    return base

def should_ligate(prev_char: str, curr_char: str) -> bool:
    ligature_pairs = [
        ('f', 'i'), ('f', 'l'), ('t', 'h'), ('c', 'h'),
        ('o', 'n'), ('i', 'n'), ('r', 'e'), ('t', 'o'), ('a', 'n'), ('e', 'r')
    ]
    return (prev_char.lower(), curr_char.lower()) in ligature_pairs

def split_text_into_pages(text: str, pages: int) -> List[str]:
    words = text.split()
    if pages <= 1:
        return [text]
    paragraphs = text.split("\n\n")
    if len(paragraphs) >= pages:
        chunks = [[] for _ in range(pages)]
        for i, para in enumerate(paragraphs):
            chunks[i % pages].append(para)
        return ["\n\n".join(chunk) for chunk in chunks]
    per_page = max(80, len(words) // pages)
    return [" ".join(words[p * per_page:(p + 1) * per_page if p < pages - 1 else len(words)]) for p in range(pages)]

def estimate_pages_needed(text: str, font_size: int = 32) -> Dict:
    page_width, page_height, margin = 1240, 1754, 120
    avg_char_width = font_size * 0.62
    avg_chars_per_line = (page_width - 2 * margin) / avg_char_width
    line_height = font_size * 1.5
    lines_per_page = (page_height - 2 * margin - 100) / line_height
    char_count = len(text)
    word_count = len(text.split())
    estimated_lines = char_count / avg_chars_per_line
    estimated_pages = math.ceil(estimated_lines / lines_per_page)
    return {
        "estimated_pages": max(1, estimated_pages),
        "char_count": char_count,
        "word_count": word_count,
        "words_per_page": int(word_count / max(1, estimated_pages)),
        "estimated_lines": int(estimated_lines)
    }

def render_character_final(char: str, font: ImageFont.FreeTypeFont, ink_color: Tuple[int,int,int], config: HandwritingConfig) -> Image.Image:
    if char == " ":
        try:
            w = font.getbbox(" ")[2]
        except Exception:
            w = font.size // 2
        return Image.new("RGBA", (max(w, 6), max(font.size, 12)), (255, 255, 255, 0))
    temp_img = Image.new("RGBA", (font.size * 4, font.size * 4), (255, 255, 255, 0))
    d = ImageDraw.Draw(temp_img)
    try:
        bbox = d.textbbox((0, 0), char, font=font)
        w = max(bbox[2] - bbox[0] + 12, 6)
        h = max(bbox[3] - bbox[1] + 12, 12)
    except Exception:
        w, h = font.size * 2, font.size * 2
    char_img = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    cd = ImageDraw.Draw(char_img)
    for i in range(config.ink_pressure_layers):
        offset_x = random.gauss(0, 0.07)
        offset_y = random.gauss(0, 0.07)
        alpha = 255
        fill_color = (ink_color[0], ink_color[1], ink_color[2], alpha)
        cd.text((6 + offset_x, 6 + offset_y), char, font=font, fill=fill_color)
    if config.ink_bleeding > 0.009:
        char_img = char_img.filter(ImageFilter.GaussianBlur(radius=config.ink_bleeding))
    bbox = char_img.getbbox()
    if bbox:
        char_img = char_img.crop(bbox)
    return char_img

def apply_char_transform(char_img: Image.Image, rotation: float, shear: float, scale: float) -> Image.Image:
    w, h = char_img.size
    if abs(scale - 1.0) > 0.01:
        char_img = char_img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        w, h = char_img.size
    if abs(shear) > 0.09:
        sh = math.tan(math.radians(shear))
        char_img = char_img.transform((int(w + abs(sh) * h) + 2, h), Image.AFFINE, (1, sh, 0, 0, 1, 0), Image.LANCZOS, (255, 255, 255, 0))
    if abs(rotation) > 0.09:
        char_img = char_img.rotate(rotation, Image.LANCZOS, True, (255, 255, 255, 0))
    return char_img

def wrap_text_to_pixel_width(draw: ImageDraw.Draw, text: str, font: ImageFont.FreeTypeFont, max_width: int, ink_color: Tuple[int,int,int]) -> List[List[Tuple]]:
    # plain naive word wrap
    words = text.split()
    lines = []
    current_line = []
    current_width = 0
    try:
        space_w = draw.textbbox((0, 0), " ", font=font)[2]
    except Exception:
        space_w = font.size // 2
    for word in words:
        try:
            w_width = draw.textbbox((0, 0), word, font=font)[2]
        except Exception:
            w_width = len(word) * font.size // 2
        if current_width == 0:
            current_line.append(("text", word, None))
            current_width = w_width
        elif current_width + space_w + w_width <= max_width:
            current_line.append(("text", " " + word, None))
            current_width += space_w + w_width
        else:
            lines.append(current_line)
            current_line = [("text", word, None)]
            current_width = w_width
    if current_line:
        lines.append(current_line)
    return lines

def render_handwritten_page(text: str, font_obj: ImageFont.ImageFont, config: HandwritingConfig,
                          img_width: int = 1240, img_height: int = 1754,
                          margin_left: int = 120, margin_top: int = 120, line_spacing: int = 14,
                          ink_color: Tuple[int,int,int] = (8, 8, 8),
                          paper_color: Tuple[int,int,int] = (245, 242, 230),
                          ruled: bool = False, page_number: int = 1, total_pages: int = 1,
                          header_text: str = None, upscale: int = 3) -> Image.Image:
    # upscale all measurements
    W, H = img_width * upscale, img_height * upscale
    ML, MT, LS = margin_left * upscale, margin_top * upscale, line_spacing * upscale
    F_obj = load_font_from_path(font_obj.path, int(font_obj.size * upscale)) if hasattr(font_obj, 'path') else font_obj
    base = create_textured_paper(W, H, paper_color, config.paper_noise_intensity)
    text_layer = Image.new("RGBA", (W, H), (255, 255, 255, 0))
    draw = ImageDraw.Draw(text_layer)
    max_text_width = W - 2 * ML
    lines = wrap_text_to_pixel_width(draw, text, F_obj, max_text_width, ink_color)
    x_start = ML + random.uniform(-config.margin_irregularity * upscale, config.margin_irregularity * upscale)
    y = MT
    prev_char = None
    for line_idx, line in enumerate(lines):
        if y > H - MT - F_obj.size * 2:
            break
        x = x_start + random.uniform(-1.2 * upscale, 1.2 * upscale)
        baseline_offset = 0
        for token_type, token_content, extra_data in line:
            if token_type == "text":
                for char in token_content:
                    if char == " ":
                        try:
                            sp = draw.textbbox((0, 0), " ", font=F_obj)[2]
                        except Exception:
                            sp = F_obj.size // 2
                        x += sp * random.uniform(0.98, 1.03)
                        prev_char = None
                        continue
                    wave = config.baseline_wave_amplitude * upscale * math.sin((x + line_idx * 17) / config.baseline_wave_frequency * 2 * math.pi)
                    baseline_offset += config.baseline_drift * upscale * random.uniform(-0.5, 0.5)
                    jitter_x = random.gauss(0, F_obj.size * 0.012 * upscale)
                    jitter_y = random.gauss(0, F_obj.size * 0.018 * upscale)
                    rotation = random.gauss(0, config.char_rotation_range) * 0.5
                    shear = random.gauss(0, config.char_shear_range) * 0.1
                    scale = random.gauss(1.0, config.char_scale_variance)
                    char_img = render_character_final(char, F_obj, ink_color, config)
                    char_img = apply_char_transform(char_img, rotation, shear, scale)
                    px = int(x + jitter_x)
                    py = int(y + jitter_y + wave + baseline_offset)
                    if 0 <= px < W and 0 <= py < H:
                        text_layer.paste(char_img, (px, py), char_img)
                    spacing = char_img.width + random.gauss(0.9 * upscale, config.char_spacing_variance * upscale)
                    if config.ligature_detection and prev_char and should_ligate(prev_char, char):
                        spacing *= 0.74
                    x += spacing
                    prev_char = char
        y += int(F_obj.size * 1.12) + LS + random.uniform(-0.25 * upscale, 0.25 * upscale)
    if total_pages > 1:
        footer_text = f"— {page_number} —"
        try:
            footer_w = draw.textbbox((0, 0), footer_text, font=F_obj)[2]
        except Exception:
            footer_w = len(footer_text) * F_obj.size // 2
        draw.text(((W - footer_w) // 2, H - MT // 2),
                 footer_text, font=F_obj, fill=ink_color)
    text_layer = text_layer.rotate(random.uniform(-0.22, 0.22), Image.LANCZOS, False, (255, 255, 255, 0))
    final = Image.alpha_composite(base.convert("RGBA"), text_layer).convert("RGB")
    enhancer = ImageEnhance.Contrast(final)
    final = enhancer.enhance(1.12)
    # Resize to actual output
    final = final.resize((img_width, img_height), Image.LANCZOS)
    return final

# ==================== STREAMLIT UI ====================
st.set_page_config(page_title="Perfect Handwriting Pro (Sharp)", layout="wide")
st.markdown("""
<style>
    .main-header {
        text-align: center; padding: 1.5rem;
        background: linear-gradient(135deg,#667eea 0%,#764ba2 100%);
        color: white; border-radius: 10px; margin-bottom: 2rem;
    }
    .feature-box {background-color: #f0f2f6; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;}
    .estimate-box {background: linear-gradient(135deg,#ffecd2 0%,#fcb69f 100%);
                   padding: 1rem; border-radius: 8px; margin: 1rem 0; font-weight: bold;}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<div class="main-header">
    <h1>✍️ Perfect Handwriting Pro (Sharp)</h1>
    <p>✨ Crystal Clear • Natural Flow • Export Ready</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Configuration")
    uploaded_font = st.file_uploader("Upload Font", type=["ttf", "otf"])
    font_size = st.slider("Font Size", 18, 72, 32)
    ink_layers = st.slider("Ink Pressure Layers (less = smoother)", min_value=2,max_value=6,value=3)
    ink_bleed = st.slider("Ink Bleed (blur, 0 = sharp)", min_value=0.0, max_value=0.04, value=0.0, step=0.004)
    paper_choice = st.selectbox("Paper", ["Ivory", "White", "Aged"])
    ink_choice = st.selectbox("Ink Color", ["Black", "Blue Black", "Brown"])
    ruled = st.checkbox("Add Ruled Lines", value=False)

    config = HandwritingConfig(
        ink_pressure_layers=ink_layers,
        ink_bleeding=ink_bleed,
        paper_noise_intensity=0.025
    )

    ink_colors = {
        "Black": (8, 8, 8),
        "Blue Black": (10, 25, 65),
        "Brown": (45, 25, 10)
    }
    paper_colors = {
        "White": (255, 255, 255),
        "Ivory": (245, 242, 230),
        "Aged": (238, 230, 210)
    }

try:
    repo_font_path = Path(__file__).parent / "fonts" / "handwriting.ttf"
except Exception:
    repo_font_path = Path(os.getcwd()) / "fonts" / "handwriting.ttf"
font_path_to_use = None
if uploaded_font:
    tmp_font_path = Path(tempfile.gettempdir()) / uploaded_font.name
    with open(tmp_font_path, "wb") as f:
        f.write(uploaded_font.getbuffer())
    font_path_to_use = tmp_font_path
elif repo_font_path.exists():
    font_path_to_use = repo_font_path

col1, col2 = st.columns([2, 1])
with col1:
    question = st.text_area("📝 Enter Text", height=250, placeholder="Paste your essay or answer here...")
    if question.strip():
        estimate = estimate_pages_needed(question, int(font_size))
        st.markdown(f"""
        <div class="estimate-box">
        Pages: {estimate['estimated_pages']}, Words: {estimate['word_count']} (~{estimate['words_per_page']}/page)
        </div>
        """, unsafe_allow_html=True)
        suggested_pages = estimate['estimated_pages']
    else:
        suggested_pages = 1
    pages = st.number_input("Pages", 1, 20, suggested_pages)
with col2:
    st.markdown("""
    <div class="feature-box">
    ✅ Sharp, high-resolution output<br>
    ✅ Smooth, natural lines<br>
    ✅ Export ZIP and PDF
    </div>
    """, unsafe_allow_html=True)

if st.button("Generate Handwritten Output", use_container_width=True):
    if not question.strip():
        st.warning("⚠️ Please enter text!")
    else:
        st.success(f"Generating {pages} page(s)...")
        font_obj = load_font_from_path(font_path_to_use, int(font_size))
        chunks = split_text_into_pages(question, int(pages))
        images = []
        progress = st.progress(0)
        status = st.empty()
        for idx, chunk in enumerate(chunks, start=1):
            status.text(f"✍️ Rendering page {idx}/{len(chunks)} ...")
            img = render_handwritten_page(
                chunk, font_obj, config,
                ink_color=ink_colors[ink_choice], paper_color=paper_colors[paper_choice],
                ruled=ruled, page_number=idx, total_pages=len(chunks), header_text=None, upscale=3
            )
            images.append(img)
            progress.progress(idx / len(chunks))
        status.text("✅ Complete!")
        st.subheader("Preview")
        cols = st.columns(min(3, len(images)))
        for i, img in enumerate(images):
            with cols[i % len(cols)]:
                st.image(img.resize((340, int(340 * img.height / img.width))), caption=f"Page {i+1}", use_container_width=True)
        st.subheader("⬇️ Download")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for i, img in enumerate(images, 1):
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, "PNG", quality=98)
                    img_bytes.seek(0)
                    zf.writestr(f"page_{i:02d}.png", img_bytes.read())
            zip_buf.seek(0)
            st.download_button("Download ZIP", zip_buf, "handwritten.zip", "application/zip", use_container_width=True)
        with col_d2:
            pdf_buf = io.BytesIO()
            rgb = [im.convert("RGB") for im in images]
            rgb[0].save(pdf_buf, "PDF", save_all=True, append_images=rgb[1:], quality=98)
            pdf_buf.seek(0)
            st.download_button("Download PDF", pdf_buf, "handwritten.pdf", "application/pdf", use_container_width=True)
        st.balloons()

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#666; padding:2rem;">
    <p><b>✍️ Perfect Handwriting Pro - Sharp Production Output</b></p>
    <p>✨ Resolution Adapted • High DPI • Export-Ready</p>
</div>
""", unsafe_allow_html=True)

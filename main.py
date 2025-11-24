# main.py (patched)
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
import math

# Optional matplotlib usage for rendering LaTeX math to images
try:
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
    rcParams["mathtext.fontset"] = "dejavusans"
except Exception:
    MATPLOTLIB_AVAILABLE = False

client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))

# --- Caches to improve performance and consistency ---
_char_image_cache = {}
_math_image_cache = {}

def generate_assignment_answer(question: str, pages: int = 2) -> str:
    approx_words_per_page = 180
    target_words = pages * approx_words_per_page
    prompt = f"""You are an Indian first-year LLB student.
Write an exam-style answer to the following question.

Question: {question}

Constraints:
- Around {target_words} words.
- Simple vocabulary.
- Clear headings and short paragraphs.
- No bullet points, just paragraphs.
- IMPORTANT: If you need to include any mathematical formula or expression, please use LaTeX math delimiters: inline math with $...$ or display math with $$...$$. Do NOT wrap math in markdown code fences. Keep the answer plain text with math kept exactly between $ or $$ so it can be correctly rendered later.
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

# ---- Math rendering: ensure math images are scaled to match font height ----
def render_math_to_image(math_tex: str, font_size: int = 28, color=(0,0,0), target_height: int | None = None):
    key = (math_tex, font_size, color, target_height)
    if key in _math_image_cache:
        return _math_image_cache[key]

    if not MATPLOTLIB_AVAILABLE:
        font = ImageFont.load_default()
        dummy = Image.new("RGBA", (10, 10), (255, 255, 255, 0))
        d = ImageDraw.Draw(dummy)
        bbox = d.textbbox((0, 0), math_tex, font=font)
        img = Image.new("RGBA", (bbox[2] - bbox[0] + 4, bbox[3] - bbox[1] + 4), (255, 255, 255, 0))
        d = ImageDraw.Draw(img)
        d.text((2, 2), math_tex, font=font, fill=color)
        _math_image_cache[key] = img
        return img

    content = math_tex
    if content.startswith("$$") and content.endswith("$$"):
        content = content[2:-2]
    elif content.startswith("$") and content.endswith("$"):
        content = content[1:-1]

    # Render math at a reasonably high DPI then scale to the target height
    base_dpi = 200
    fig = plt.figure(figsize=(0.01, 0.01), dpi=base_dpi)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    text_kwargs = dict(fontsize=font_size, color='#%02x%02x%02x' % color, ha='left', va='bottom')
    t = ax.text(0, 0, f"${content}$", **text_kwargs)
    fig.canvas.draw()
    bbox = t.get_window_extent(renderer=fig.canvas.get_renderer())
    width, height = int(bbox.width) + 6, int(bbox.height) + 6
    plt.close(fig)

    # Re-render at exact bbox size
    fig = plt.figure(figsize=(max(1e-3, width / base_dpi), max(1e-3, height / base_dpi)), dpi=base_dpi)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0, 0, f"${content}$", **text_kwargs)
    fig.canvas.draw()
    buf = fig.canvas.tostring_argb()
    w, h = fig.canvas.get_width_height()
    plt.close(fig)

    import numpy as _np
    arr = _np.frombuffer(buf, dtype=_np.uint8).reshape((h, w, 4))
    arr = arr[:, :, [1, 2, 3, 0]]
    pil_img = Image.fromarray(arr, mode="RGBA")
    bbox = pil_img.getbbox()
    if bbox:
        pil_img = pil_img.crop(bbox)

    # If caller requests a target height, scale the math image to match font height so it blends with handwriting
    if target_height is not None and pil_img.height > 0:
        scale = target_height / pil_img.height
        if scale != 1.0:
            new_w = max(1, int(pil_img.width * scale))
            new_h = max(1, int(pil_img.height * scale))
            pil_img = pil_img.resize((new_w, new_h), resample=Image.LANCZOS)

    _math_image_cache[key] = pil_img
    return pil_img

def split_text_preserving_math(text: str):
    tokens = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "$":
            if i + 1 < n and text[i+1] == "$":
                j = text.find("$$", i+2)
                if j == -1:
                    tokens.append(text[i:])
                    break
                tokens.append(text[i:j+2])
                i = j + 2
            else:
                j = text.find("$", i+1)
                if j == -1:
                    tokens.append(text[i:])
                    break
                tokens.append(text[i:j+1])
                i = j + 1
        else:
            j = text.find("$", i)
            if j == -1:
                tokens.append(text[i:])
                break
            tokens.append(text[i:j])
            i = j
    return tokens

def wrap_text_to_pixel_width(draw: ImageDraw.Draw, text: str, font: ImageFont.FreeTypeFont, max_width: int):
    tokens = split_text_preserving_math(text)
    lines = []
    current_line = []
    current_width = 0

    # correct space width calculation
    space_bbox = draw.textbbox((0, 0), " ", font=font)
    space_w = space_bbox[2] - space_bbox[0]

    for tok in tokens:
        if tok.startswith("$"):
            math_img = render_math_to_image(tok, font_size=int(font.size * 0.95), color=(0,0,0), target_height=int(font.size * 1.1))
            tok_width = math_img.width
            if current_width == 0:
                current_line.append(("math", tok))
                current_width = tok_width
            else:
                if current_width + space_w + tok_width <= max_width:
                    current_line.append(("text", " "))
                    current_line.append(("math", tok))
                    current_width += space_w + tok_width
                else:
                    lines.append(current_line)
                    current_line = [("math", tok)]
                    current_width = tok_width
        else:
            parts = tok.split("\n")
            for pi, part in enumerate(parts):
                words = part.split()
                for wi, w in enumerate(words):
                    w_bbox = draw.textbbox((0, 0), w, font=font)
                    w_width = w_bbox[2] - w_bbox[0]
                    if current_width == 0:
                        current_line.append(("text", w))
                        current_width = w_width
                    else:
                        if current_width + space_w + w_width <= max_width:
                            current_line.append(("text", " " + w))
                            current_width += space_w + w_width
                        else:
                            lines.append(current_line)
                            current_line = [("text", w)]
                            current_width = w_width
                if pi < len(parts) - 1:
                    lines.append(current_line)
                    current_line = []
                    current_width = 0
    if current_line:
        lines.append(current_line)
    return lines

def draw_ruled_lines(draw: ImageDraw.Draw, img_width: int, img_height: int, margin_left: int, margin_top: int, margin_right: int, line_spacing: int, line_color: tuple):
    y = margin_top
    while y < img_height - margin_top:
        draw.line([(margin_left - 20, y), (img_width - margin_right + 20, y)], fill=line_color, width=1)
        y += line_spacing

# --- improved per-character rendering with caching and stable sizing ---
def render_char_image(char: str, font: ImageFont.FreeTypeFont, ink_color: tuple, stroke_variation: float = 0.6):
    key = (char, getattr(font, "size", None), stroke_variation)
    if key in _char_image_cache:
        return _char_image_cache[key]

    # handle space by measuring width and returning transparent canvas of same height as font
    if char == " ":
        w = font.getmask(" ").getbbox()
        if w:
            w = w[2]
        else:
            # fallback
            w = font.size // 3
        canvas = Image.new("RGBA", (w, max(10, font.size)), (255,255,255,0))
        _char_image_cache[key] = canvas
        return canvas

    # use a sufficiently large temp canvas to draw glyph so we capture proper bbox
    tmp_size = max(64, font.size * 4)
    tmp = Image.new("RGBA", (tmp_size, tmp_size), (255,255,255,0))
    dt = ImageDraw.Draw(tmp)
    # draw character at offset so bbox is positive
    offset = (tmp_size // 8, tmp_size // 8)
    # layered strokes for pressure simulation
    layers = max(1, int(1 + stroke_variation * 2.5))
    for i in range(layers):
        ox = random.uniform(-0.5, 0.5) * (0.4 + stroke_variation)
        oy = random.uniform(-0.5, 0.5) * (0.4 + stroke_variation)
        alpha = int(255 * random.uniform(0.82, 1.0))
        fill = (ink_color[0], ink_color[1], ink_color[2], alpha)
        dt.text((offset[0] + ox, offset[1] + oy), char, font=font, fill=fill)
    # gentle blur but small radius to avoid fragments
    char_img = tmp.filter(ImageFilter.GaussianBlur(radius=0.25 + stroke_variation * 0.25))
    bbox = char_img.getbbox()
    if bbox:
        char_img = char_img.crop(bbox)
    # final ensure non-empty
    if char_img.size[0] == 0 or char_img.size[1] == 0:
        char_img = Image.new("RGBA", (max(1, font.size//2), max(1, font.size)), (255,255,255,0))

    _char_image_cache[key] = char_img
    return char_img

def transform_char_image(char_img: Image.Image, shear: float = 0.0, rotate: float = 0.0, scale: float = 1.0):
    w, h = char_img.size
    if scale != 1.0:
        char_img = char_img.resize((max(1, int(w * scale)), max(1, int(h * scale))), resample=Image.BICUBIC)
        w, h = char_img.size
    if abs(shear) > 1e-6:
        sh = math.tan(math.radians(shear))
        a, b, c, d, e, f = 1, sh, 0, 0, 1, 0
        new_w = int(w + abs(sh) * h) + 2
        char_img = char_img.transform((new_w, h), Image.AFFINE, (a, b, c, d, e, f), resample=Image.BICUBIC, fillcolor=(255,255,255,0))
    if abs(rotate) > 1e-6:
        char_img = char_img.rotate(rotate, resample=Image.BICUBIC, expand=True, fillcolor=(255,255,255,0))
    return char_img

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
    try:
        base = Image.new("RGB", (img_width, img_height), paper_color)
        noise = Image.effect_noise((img_width, img_height), 64).convert("L")
        noise = noise.point(lambda p: p * 0.07)
        noise_rgb = Image.merge("RGB", (noise, noise, noise))
        base = Image.blend(base, noise_rgb, alpha=0.12)
    except Exception:
        base = Image.new("RGB", (img_width, img_height), paper_color)

    text_layer = Image.new("RGBA", (img_width, img_height), (255,255,255,0))
    draw = ImageDraw.Draw(text_layer)
    max_text_width = img_width - margin_left - margin_left
    lines = wrap_text_to_pixel_width(draw, text, font_obj, max_text_width)

    if ruled:
        ruled_color = (180, 200, 215)
        draw_ruled_lines(draw, img_width, img_height, margin_left, margin_top, margin_left, int(font_obj.size * 1.9), ruled_color)

    if header:
        draw.text((margin_left, margin_top - int(font_obj.size * 1.6)), header, font=font_obj, fill=tuple(min(255, c + 20) for c in ink_color))

    # Handwriting parameters (tweakable)
    baseline_wave_amp = max(0.6, font_obj.size * 0.035)
    baseline_wave_freq = 180.0
    char_jitter_x = max(0.4, font_obj.size * 0.018)
    char_jitter_y = max(0.6, font_obj.size * 0.025)
    shear_range = 4.0
    rotate_range = rotation_jitter
    stroke_variation = 0.6

    x_start, y = margin_left, margin_top
    for line_idx, line in enumerate(lines):
        if y > img_height - margin_top - font_obj.size:
            break
        x = x_start
        line_offset = random.randint(-2, 2)
        for token_type, token_content in line:
            if token_type == "text":
                for ch in token_content:
                    if ch == " ":
                        sp_bbox = draw.textbbox((0,0)," ", font=font_obj)
                        sp = sp_bbox[2] - sp_bbox[0]
                        x += sp
                        continue

                    wave = baseline_wave_amp * math.sin((x + line_idx * 14) / baseline_wave_freq * 2 * math.pi)
                    jitter_x = random.gauss(0, char_jitter_x)
                    jitter_y = random.gauss(0, char_jitter_y)
                    rotation = random.uniform(-rotate_range, rotate_range) * 0.6
                    shear = random.uniform(-shear_range, shear_range) * 0.15
                    scale = random.uniform(0.98, 1.02)

                    char_img = render_char_image(ch, font_obj, ink_color, stroke_variation=stroke_variation)
                    char_img = transform_char_image(char_img, shear=shear, rotate=rotation, scale=scale)

                    px = int(x + jitter_x + line_offset)
                    py = int(y + jitter_y + wave)
                    # safe paste using alpha channel
                    text_layer.paste(char_img, (px, py), char_img)
                    # advance x: approximate using char_img width
                    x += char_img.width + random.uniform(0.6, 1.4)
            else:
                # math: render scaled to match font height so it doesn't dominate the line
                target_h = int(font_obj.size * 1.1)
                math_img = render_math_to_image(token_content, font_size=int(font_obj.size * 0.95), color=ink_color, target_height=target_h)
                # if math image still bigger, downscale
                if math_img.height > target_h:
                    s = target_h / math_img.height
                    math_img = math_img.resize((max(1, int(math_img.width * s)), target_h), resample=Image.LANCZOS)
                py = int(y - (math_img.height - font_obj.size) / 2 + random.uniform(-1, 1))
                px = int(x + random.uniform(-2, 2))
                text_layer.paste(math_img, (px, py), math_img)
                x += math_img.width + 4

        y += int(font_obj.size * 1.0) + line_spacing

    # small, gentle bleed (very small radius prevents fragments)
    text_layer = text_layer.filter(ImageFilter.GaussianBlur(radius=0.18))

    angle = random.uniform(-rotation_jitter, rotation_jitter) * 0.4
    rotated = text_layer.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(255,255,255,0))

    final = base.convert("RGBA")
    final = Image.alpha_composite(final, rotated).convert("RGB")
    final = ImageOps.autocontrast(final, cutoff=1)

    return final

# ... rest of the streamlit UI and flow unchanged (omitted here for brevity) ...

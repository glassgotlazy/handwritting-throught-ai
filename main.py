import streamlit as st
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps, ImageEnhance
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

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
    rcParams["mathtext.fontset"] = "dejavusans"
    rcParams['text.antialiased'] = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

# ==================== CONFIGURATION ====================
@dataclass
class HandwritingConfig:
    """Optimized configuration for natural dark handwriting"""
    char_rotation_range: float = 1.8
    char_scale_variance: float = 0.05
    char_shear_range: float = 3.0
    char_spacing_variance: float = 0.8
    baseline_wave_amplitude: float = 0.9
    baseline_wave_frequency: float = 180.0
    baseline_drift: float = 0.3
    ink_pressure_layers: int = 6  # Optimal for darkness
    ink_bleeding: float = 0.15  # Slight blur for natural look
    paper_noise_intensity: float = 0.08
    ligature_detection: bool = True
    word_spacing_natural: bool = True
    margin_irregularity: float = 6.0

client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))

# ==================== PAGE ESTIMATION ====================
def estimate_pages_needed(text: str, font_size: int = 32) -> Dict:
    """Estimate pages needed"""
    page_width, page_height, margin = 1240, 1754, 120
    avg_char_width = font_size * 0.6
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

# ==================== PHYSICS DIAGRAMS ====================
def generate_physics_diagram(diagram_type: str, ink_color: Tuple[int,int,int]) -> Image.Image:
    """Generate physics diagrams"""
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150, facecolor='none')
    fig.patch.set_alpha(0.0)
    ax.set_aspect('equal')
    ax.axis('off')
    
    ink_hex = f'#{ink_color[0]:02x}{ink_color[1]:02x}{ink_color[2]:02x}'
    
    try:
        if diagram_type == "free_body":
            box = Rectangle((4, 4), 2, 2, fill=False, edgecolor=ink_hex, linewidth=3)
            ax.add_patch(box)
            ax.text(5, 5, 'm', ha='center', va='center', fontsize=18, color=ink_hex, weight='bold')
            ax.arrow(5, 4, 0, -2, head_width=0.2, head_length=0.2, fc=ink_hex, ec=ink_hex, linewidth=3)
            ax.text(5.5, 2.5, '$F_g$', fontsize=14, color=ink_hex, weight='bold')
            ax.arrow(5, 6, 0, 2, head_width=0.2, head_length=0.2, fc=ink_hex, ec=ink_hex, linewidth=3)
            ax.text(5.5, 8.2, '$F_N$', fontsize=14, color=ink_hex, weight='bold')
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
        elif diagram_type == "circuit":
            ax.plot([2, 2], [3, 5], color=ink_hex, linewidth=4)
            ax.text(1.2, 4, '$V$', fontsize=16, color=ink_hex, weight='bold')
            ax.plot([2, 2, 8, 8, 2], [5, 7, 7, 3, 3], color=ink_hex, linewidth=3)
            ax.set_xlim(0, 10)
            ax.set_ylim(2, 8)
        else:
            ax.plot(5, 5, 'o', color=ink_hex, markersize=12)
            ax.arrow(5, 5, 2, 1, head_width=0.2, head_length=0.2, fc=ink_hex, ec=ink_hex, linewidth=3)
            ax.text(7.5, 6.2, r'$\vec{F}$', fontsize=14, color=ink_hex, weight='bold')
            ax.set_xlim(3, 9)
            ax.set_ylim(3, 9)
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0.2, dpi=150)
        plt.close(fig)
        
        buf.seek(0)
        pil_img = Image.open(buf).convert('RGBA')
        data = np.array(pil_img)
        white_areas = (data[:, :, 0] > 240) & (data[:, :, 1] > 240) & (data[:, :, 2] > 240)
        data[white_areas, 3] = 0
        return Image.fromarray(data, 'RGBA')
        
    except Exception:
        plt.close(fig)
        return None

# ==================== MATH OCR ====================
def read_math_from_image(image_data: bytes) -> str:
    """Extract math from images"""
    try:
        base64_image = base64.b64encode(image_data).decode('utf-8')
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all math formulas to LaTeX. Use $...$ for inline, $$...$$ for display."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }],
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OCR error: {e}")
        return ""

# ==================== AI GENERATION ====================
def generate_assignment_answer(question: str, pages: int, subject: str, include_diagrams: bool) -> str:
    """Generate answer"""
    target_words = pages * 180
    
    prompts = {
        "physics": "Physics student. Include LaTeX ($E=mc^2$) and [DIAGRAM:free_body] markers.",
        "mathematics": "Math student. Show solutions with LaTeX.",
        "science": "Science student. Include formulas in LaTeX.",
        "general": "Undergraduate student."
    }
    
    prompt = f"{prompts.get(subject, prompts['general'])}\n\nAnswer: {question}\n\nTarget: ~{target_words} words with clear paragraphs."
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=min(target_words * 2, 3000),
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"API error: {e}")
        return ""

# ==================== FONT ====================
def load_font_from_path(font_path: Union[str, Path, None], font_size: int):
    if font_path is None:
        return ImageFont.load_default()
    try:
        return ImageFont.truetype(str(font_path), font_size)
    except:
        return ImageFont.load_default()

# ==================== MATH RENDERING ====================
_math_cache = {}

def render_math_to_image(math_tex: str, font_size: int, color: Tuple[int,int,int]) -> Image.Image:
    """Render math with proper darkness"""
    cache_key = (math_tex, font_size, color)
    if cache_key in _math_cache:
        return _math_cache[cache_key]

    if not MATPLOTLIB_AVAILABLE:
        font = ImageFont.load_default()
        img = Image.new("RGBA", (len(math_tex) * 8, 20), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.text((2, 2), math_tex, font=font, fill=color)
        return img

    content = math_tex.strip()
    if content.startswith("$$") and content.endswith("$$"):
        content = content[2:-2]
    elif content.startswith("$") and content.endswith("$"):
        content = content[1:-1]

    try:
        fig = plt.figure(figsize=(0.1, 0.1), dpi=200, facecolor='none')
        fig.patch.set_alpha(0.0)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.patch.set_alpha(0.0)
        
        text_obj = ax.text(0, 0, f"${content}$", fontsize=font_size + 2,
                          color=f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}',
                          ha='left', va='baseline', weight='bold')
        
        fig.canvas.draw()
        bbox = text_obj.get_window_extent(renderer=fig.canvas.get_renderer())
        w_inch, h_inch = (bbox.width + 12) / fig.dpi, (bbox.height + 12) / fig.dpi
        plt.close(fig)
        
        fig = plt.figure(figsize=(w_inch, h_inch), dpi=200, facecolor='none')
        fig.patch.set_alpha(0.0)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.patch.set_alpha(0.0)
        
        ax.text(0.5, 0.5, f"${content}$", fontsize=font_size + 2,
               color=f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}',
               ha='center', va='center', transform=ax.transAxes, weight='bold')
        
        fig.canvas.draw()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        
        buf.seek(0)
        pil_img = Image.open(buf).convert('RGBA')
        data = np.array(pil_img)
        white_areas = (data[:, :, 0] > 240) & (data[:, :, 1] > 240) & (data[:, :, 2] > 240)
        data[white_areas, 3] = 0
        pil_img = Image.fromarray(data, 'RGBA')
        
        bbox = pil_img.getbbox()
        if bbox:
            pil_img = pil_img.crop(bbox)
        
        _math_cache[cache_key] = pil_img
        return pil_img
        
    except:
        return Image.new("RGBA", (50, 20), (0, 0, 0, 0))

# ==================== TEXT PROCESSING ====================
def split_text_preserving_all(text: str) -> List[str]:
    """Split text preserving math and diagrams"""
    tokens = []
    i, n = 0, len(text)
    
    while i < n:
        if text[i:i+9] == "[DIAGRAM:":
            j = text.find("]", i)
            if j != -1:
                tokens.append(text[i:j+1])
                i = j + 1
                continue
        
        if text[i] == "$":
            if i + 1 < n and text[i+1] == "$":
                j = text.find("$$", i+2)
                tokens.append(text[i:j+2] if j != -1 else text[i:])
                i = j + 2 if j != -1 else n
            else:
                j = text.find("$", i+1)
                tokens.append(text[i:j+1] if j != -1 else text[i:])
                i = j + 1 if j != -1 else n
        else:
            j1, j2 = text.find("$", i), text.find("[DIAGRAM:", i)
            if j1 == -1 and j2 == -1:
                tokens.append(text[i:])
                break
            j = min(x for x in [j1, j2] if x != -1)
            tokens.append(text[i:j])
            i = j
    
    return tokens

def wrap_text_to_pixel_width(draw: ImageDraw.Draw, text: str, font: ImageFont.FreeTypeFont, 
                             max_width: int, ink_color: Tuple[int,int,int]) -> List[List[Tuple]]:
    """Wrap text"""
    tokens = split_text_preserving_all(text)
    lines = []
    current_line = []
    current_width = 0
    
    try:
        space_w = draw.textbbox((0, 0), " ", font=font)[2]
    except:
        space_w = font.size // 3

    for tok in tokens:
        if tok.startswith("[DIAGRAM:"):
            parts = tok[9:-1].split(":", 1)
            diagram_img = generate_physics_diagram(parts[0] if parts else "vector", ink_color)
            if diagram_img:
                if current_line:
                    lines.append(current_line)
                lines.append([("diagram", tok, diagram_img)])
                current_line = []
                current_width = 0
            continue
            
        if tok.startswith("$"):
            math_img = render_math_to_image(tok, int(font.size * 0.95), ink_color)
            tok_width = math_img.width
            
            if current_width == 0:
                current_line.append(("math", tok, math_img))
                current_width = tok_width
            elif current_width + space_w + tok_width <= max_width:
                current_line.append(("text", " ", None))
                current_line.append(("math", tok, math_img))
                current_width += space_w + tok_width
            else:
                lines.append(current_line)
                current_line = [("math", tok, math_img)]
                current_width = tok_width
        else:
            for pi, part in enumerate(tok.split("\n")):
                for word in part.split():
                    try:
                        w_width = draw.textbbox((0, 0), word, font=font)[2]
                    except:
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
                
                if pi < len(tok.split("\n")) - 1:
                    lines.append(current_line)
                    current_line = []
                    current_width = 0
    
    if current_line:
        lines.append(current_line)
    return lines

# ==================== PAPER ====================
def create_textured_paper(width: int, height: int, base_color: Tuple[int,int,int], 
                         texture_intensity: float = 0.08) -> Image.Image:
    """Create paper"""
    base = Image.new("RGB", (width, height), base_color)
    try:
        noise = Image.effect_noise((width // 2, height // 2), 64).convert("L")
        noise = noise.resize((width, height), Image.BICUBIC)
        noise = noise.point(lambda p: int(p * texture_intensity))
        noise_rgb = Image.merge("RGB", (noise, noise, noise))
        base = Image.blend(base, noise_rgb, alpha=0.10)
    except:
        pass
    return base

# ==================== OPTIMIZED CHARACTER RENDERING ====================
def render_character_dark_natural(char: str, font: ImageFont.FreeTypeFont, 
                                 ink_color: Tuple[int,int,int], 
                                 config: HandwritingConfig) -> Image.Image:
    """
    Render character with optimal darkness and natural appearance
    """
    if char == " ":
        try:
            w = font.getbbox(" ")[2]
        except:
            w = font.size // 3
        return Image.new("RGBA", (max(w, 5), max(font.size, 10)), (255, 255, 255, 0))
    
    # Create canvas
    temp_img = Image.new("RGBA", (font.size * 4, font.size * 4), (255, 255, 255, 0))
    d = ImageDraw.Draw(temp_img)
    try:
        bbox = d.textbbox((0, 0), char, font=font)
        w = max(bbox[2] - bbox[0] + 14, 5)
        h = max(bbox[3] - bbox[1] + 14, 10)
    except:
        w, h = font.size * 2, font.size * 2
    
    char_img = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    cd = ImageDraw.Draw(char_img)
    
    # OPTIMIZED: 6 layers with high opacity and slight variation for natural look
    for i in range(config.ink_pressure_layers):
        # Minimal offset for natural ink flow
        offset_x = random.gauss(0, 0.2)
        offset_y = random.gauss(0, 0.2)
        
        # High opacity with slight natural variation
        alpha = int(255 * random.uniform(0.95, 1.0))
        fill_color = (ink_color[0], ink_color[1], ink_color[2], alpha)
        
        cd.text((7 + offset_x, 7 + offset_y), char, font=font, fill=fill_color)
    
    # OPTIMIZED: Slight blur for natural ink effect while maintaining darkness
    if config.ink_bleeding > 0:
        char_img = char_img.filter(ImageFilter.GaussianBlur(radius=config.ink_bleeding))
    
    # Trim
    bbox = char_img.getbbox()
    if bbox:
        char_img = char_img.crop(bbox)
    
    return char_img

def apply_char_transform(char_img: Image.Image, rotation: float, shear: float, scale: float) -> Image.Image:
    """Apply transformations"""
    w, h = char_img.size
    
    if abs(scale - 1.0) > 0.01:
        char_img = char_img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
        w, h = char_img.size
    
    if abs(shear) > 0.1:
        sh = math.tan(math.radians(shear))
        char_img = char_img.transform((int(w + abs(sh) * h) + 2, h), Image.AFFINE, 
                                      (1, sh, 0, 0, 1, 0), Image.BICUBIC, (255, 255, 255, 0))
    
    if abs(rotation) > 0.1:
        char_img = char_img.rotate(rotation, Image.BICUBIC, True, (255, 255, 255, 0))
    
    return char_img

def should_ligate(prev_char: str, curr_char: str) -> bool:
    """Check for natural letter connections"""
    ligature_pairs = [
        ('f', 'i'), ('f', 'l'), ('t', 'h'), ('c', 'h'),
        ('o', 'n'), ('i', 'n'), ('r', 'e'), ('t', 'o'),
        ('a', 'n'), ('e', 'r'), ('i', 't')
    ]
    return (prev_char.lower(), curr_char.lower()) in ligature_pairs

# ==================== MAIN RENDERING ====================
def render_handwritten_page(text: str, font_obj: ImageFont.ImageFont, config: HandwritingConfig,
                          img_width: int = 1240, img_height: int = 1754,
                          margin_left: int = 120, margin_top: int = 120, line_spacing: int = 12,
                          ink_color: Tuple[int,int,int] = (5, 5, 5),
                          paper_color: Tuple[int,int,int] = (245, 242, 230),
                          ruled: bool = False, page_number: int = 1, total_pages: int = 1,
                          header_text: str = None) -> Image.Image:
    """Render with optimal darkness and natural flow"""
    
    base = create_textured_paper(img_width, img_height, paper_color, config.paper_noise_intensity)
    text_layer = Image.new("RGBA", (img_width, img_height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(text_layer)
    
    max_text_width = img_width - 2 * margin_left
    lines = wrap_text_to_pixel_width(draw, text, font_obj, max_text_width, ink_color)
    
    if ruled:
        y = margin_top
        while y < img_height - margin_top:
            y_var = random.uniform(-0.5, 0.5)
            draw.line([(margin_left - 20, y + y_var), (img_width - margin_left + 20, y + y_var)], 
                     fill=(180, 200, 215), width=1)
            y += int(font_obj.size * 1.9)
    
    if header_text:
        draw.text((margin_left, margin_top - int(font_obj.size * 1.8)), 
                 header_text, font=font_obj, fill=ink_color)
    
    x_start = margin_left + random.uniform(-config.margin_irregularity, config.margin_irregularity)
    y = margin_top
    prev_char = None
    
    for line_idx, line in enumerate(lines):
        if y > img_height - margin_top - font_obj.size * 2:
            break
        
        x = x_start + random.uniform(-2, 2)
        baseline_offset = 0
        
        for token_type, token_content, extra_data in line:
            if token_type == "diagram":
                if extra_data:
                    px = (img_width - extra_data.width) // 2
                    py = int(y)
                    if py + extra_data.height < img_height:
                        text_layer.paste(extra_data, (px, py), extra_data)
                        y += extra_data.height + 30
                break
                
            elif token_type == "text":
                for char in token_content:
                    if char == " ":
                        try:
                            sp = draw.textbbox((0, 0), " ", font=font_obj)[2]
                        except:
                            sp = font_obj.size // 3
                        x += sp * random.uniform(0.95, 1.08)
                        prev_char = None
                        continue
                    
                    wave = config.baseline_wave_amplitude * math.sin(
                        (x + line_idx * 17) / config.baseline_wave_frequency * 2 * math.pi)
                    baseline_offset += config.baseline_drift * random.uniform(-0.5, 0.5)
                    
                    jitter_x = random.gauss(0, font_obj.size * 0.018)
                    jitter_y = random.gauss(0, font_obj.size * 0.025)
                    rotation = random.gauss(0, config.char_rotation_range) * 0.5
                    shear = random.gauss(0, config.char_shear_range) * 0.12
                    scale = random.gauss(1.0, config.char_scale_variance)
                    
                    char_img = render_character_dark_natural(char, font_obj, ink_color, config)
                    char_img = apply_char_transform(char_img, rotation, shear, scale)
                    
                    px = int(x + jitter_x)
                    py = int(y + jitter_y + wave + baseline_offset)
                    
                    if 0 <= px < img_width and 0 <= py < img_height:
                        text_layer.paste(char_img, (px, py), char_img)
                    
                    # Natural spacing with ligature adjustment
                    spacing = char_img.width + random.gauss(1.0, config.char_spacing_variance)
                    if config.ligature_detection and prev_char and should_ligate(prev_char, char):
                        spacing *= 0.7
                    
                    x += spacing
                    prev_char = char
                    
            else:  # Math
                if extra_data:
                    py = int(y - (extra_data.height - font_obj.size) / 2 + random.uniform(-1, 1))
                    px = int(x + random.uniform(-0.5, 0.5))
                    if 0 <= px < img_width and 0 <= py < img_height:
                        text_layer.paste(extra_data, (px, py), extra_data)
                    x += extra_data.width + 5
                prev_char = None
        
        y += int(font_obj.size * 1.08) + line_spacing + random.uniform(-0.5, 0.5)
    
    if total_pages > 1:
        footer_text = f"— {page_number} —"
        try:
            footer_w = draw.textbbox((0, 0), footer_text, font=font_obj)[2]
        except:
            footer_w = len(footer_text) * font_obj.size // 2
        draw.text(((img_width - footer_w) // 2, img_height - margin_top // 2), 
                 footer_text, font=font_obj, fill=ink_color)
    
    # Slight rotation for realism
    text_layer = text_layer.rotate(random.uniform(-0.5, 0.5), Image.BICUBIC, False, (255, 255, 255, 0))
    
    # Composite
    final = base.convert("RGBA")
    final = Image.alpha_composite(final, text_layer).convert("RGB")
    
    # Enhance for optimal darkness
    enhancer = ImageEnhance.Contrast(final)
    final = enhancer.enhance(1.15)
    
    return final

def split_text_into_pages(text: str, pages: int) -> List[str]:
    """Split text"""
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
    return [" ".join(words[p * per_page:(p + 1) * per_page if p < pages - 1 else len(words)]) 
            for p in range(pages)]

# ==================== STREAMLIT UI ====================
st.set_page_config(page_title="✍️ Perfect Handwriting Pro", layout="wide")

st.markdown("""
<style>
    .main-header {
        text-align: center; padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border-radius: 10px; margin-bottom: 2rem;
    }
    .feature-box {background-color: #f0f2f6; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;}
    .estimate-box {background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); 
                   padding: 1rem; border-radius: 8px; margin: 1rem 0; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>✍️ Perfect Handwriting Pro</h1>
    <p>✨ Dark, Natural & Beautiful Handwriting</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("🎨 Configuration")
    uploaded_font = st.file_uploader("Upload Font", type=["ttf", "otf"])
    font_size = st.slider("Font Size", 18, 72, 32)
    subject = st.selectbox("Subject", ["general", "physics", "mathematics", "science"])
    
    st.subheader("🔍 Math OCR")
    math_img = st.file_uploader("Upload math image", type=["jpg", "png"])
    if math_img and st.button("📖 Extract Math"):
        with st.spinner("Reading..."):
            extracted = read_math_from_image(math_img.read())
            if extracted:
                st.success("✅ Extracted!")
                st.code(extracted)
                st.session_state['math'] = extracted
    
    st.subheader("🎭 Style")
    style_preset = st.selectbox("Preset", [
        "Perfect Balance (Recommended)",
        "Extra Natural",
        "Neat & Clean"
    ])
    
    presets = {
        "Perfect Balance (Recommended)": HandwritingConfig(),
        "Extra Natural": HandwritingConfig(
            char_rotation_range=2.2,
            baseline_wave_amplitude=1.1,
            char_spacing_variance=1.0,
            ink_bleeding=0.2
        ),
        "Neat & Clean": HandwritingConfig(
            char_rotation_range=1.2,
            baseline_wave_amplitude=0.6,
            char_spacing_variance=0.6,
            ink_bleeding=0.1
        )
    }
    
    config = presets[style_preset]
    
    ink_choice = st.selectbox("Ink Color", ["Dark Black", "Blue Black", "Brown"])
    paper_choice = st.selectbox("Paper", ["Ivory", "White", "Aged"])
    ruled = st.checkbox("Add Ruled Lines")
    include_diagrams = st.checkbox("Generate Diagrams", subject=="physics")
    
    ink_colors = {
        "Dark Black": (8, 8, 8),
        "Blue Black": (10, 25, 65),
        "Brown": (45, 25, 10)
    }
    
    paper_colors = {
        "White": (255, 255, 255),
        "Ivory": (245, 242, 230),
        "Aged": (238, 230, 210)
    }
    
    st.success("✅ **Optimized for natural dark handwriting!**")

# Load font
try:
    repo_font_path = Path(__file__).parent / "fonts" / "handwriting.ttf"
except:
    repo_font_path = Path(os.getcwd()) / "fonts" / "handwriting.ttf"

font_path_to_use = None
if uploaded_font:
    tmp_font_path = Path(tempfile.gettempdir()) / uploaded_font.name
    with open(tmp_font_path, "wb") as f:
        f.write(uploaded_font.getbuffer())
    font_path_to_use = tmp_font_path
elif repo_font_path.exists():
    font_path_to_use = repo_font_path

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    question = st.text_area("📝 Enter Text or Question", height=250,
                           placeholder="Type or paste your text here...",
                           value=st.session_state.get('math', ''))
    
    if question.strip():
        estimate = estimate_pages_needed(question, int(font_size))
        st.markdown(f"""
        <div class="estimate-box">
        📊 <b>Estimation:</b> {estimate['estimated_pages']} pages needed | 
        {estimate['word_count']} words (~{estimate['words_per_page']}/page)
        </div>
        """, unsafe_allow_html=True)
        suggested_pages = estimate['estimated_pages']
    else:
        suggested_pages = 2
    
    pages = st.number_input("Pages", 1, 20, suggested_pages)

with col2:
    st.markdown("""
    <div class="feature-box">
    ✅ <b>Dark & Natural Ink</b><br>
    ✅ <b>Smart Page Estimation</b><br>
    ✅ <b>Math Formula OCR</b><br>
    ✅ <b>Physics Diagrams</b><br>
    ✅ <b>Realistic Flow</b><br>
    ✅ <b>PDF/ZIP Export</b>
    </div>
    """, unsafe_allow_html=True)

col_b1, col_b2 = st.columns(2)

with col_b1:
    gen_ai = st.button("🤖 AI Answer + Handwriting", type="primary", use_container_width=True)

with col_b2:
    gen_direct = st.button("✍️ Text to Handwriting", use_container_width=True)

if gen_ai or gen_direct:
    if not question.strip():
        st.warning("⚠️ Please enter text!")
    else:
        if gen_ai:
            with st.spinner("🤖 Generating AI answer..."):
                answer_text = generate_assignment_answer(question, int(pages), subject, include_diagrams)
        else:
            answer_text = question
        
        if answer_text:
            st.success(f"✅ Generated {len(answer_text.split())} words")
            
            with st.expander("📄 View Generated Text"):
                st.text_area("Content", answer_text, height=200)
            
            font_obj = load_font_from_path(font_path_to_use, int(font_size))
            chunks = split_text_into_pages(answer_text, int(pages))
            
            images = []
            progress = st.progress(0)
            status = st.empty()
            
            for idx, chunk in enumerate(chunks, start=1):
                status.text(f"✍️ Rendering page {idx}/{len(chunks)}...")
                
                img = render_handwritten_page(
                    chunk, font_obj, config,
                    ink_color=ink_colors[ink_choice],
                    paper_color=paper_colors[paper_choice],
                    ruled=ruled,
                    page_number=idx,
                    total_pages=len(chunks),
                    header_text=f"Page {idx}/{len(chunks)}" if len(chunks) > 1 else None
                )
                images.append(img)
                progress.progress(idx / len(chunks))
            
            status.text("✅ Rendering complete!")
            
            st.subheader("📄 Preview")
            cols = st.columns(min(3, len(images)))
            for i, img in enumerate(images):
                with cols[i % len(cols)]:
                    st.image(img.resize((300, int(300 * img.height / img.width))), 
                            caption=f"Page {i+1}", use_container_width=True)
            
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
                st.download_button("📦 Download ZIP", zip_buf, "handwritten.zip", 
                                  "application/zip", use_container_width=True)
            
            with col_d2:
                pdf_buf = io.BytesIO()
                rgb = [im.convert("RGB") for im in images]
                rgb[0].save(pdf_buf, "PDF", save_all=True, append_images=rgb[1:], quality=98)
                pdf_buf.seek(0)
                st.download_button("📄 Download PDF", pdf_buf, "handwritten.pdf", 
                                  "application/pdf", use_container_width=True)
            
            st.balloons()

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#666; padding:2rem;">
    <p><b>✍️ Perfect Handwriting Pro v5.0</b></p>
    <p>✨ Optimized for dark, natural, and beautiful handwriting generation</p>
</div>
""", unsafe_allow_html=True)

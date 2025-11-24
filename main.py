import streamlit as st
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps, ImageEnhance
import tempfile
import os
from pathlib import Path
import random
import io
import zipfile
from typing import List, Union, Tuple
import math
import numpy as np
from dataclasses import dataclass

# Advanced Math Rendering
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-GUI backend
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    MATPLOTLIB_AVAILABLE = True
    rcParams["mathtext.fontset"] = "dejavusans"
    rcParams['text.antialiased'] = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

# ==================== CONFIGURATION ====================
@dataclass
class HandwritingConfig:
    """Advanced handwriting synthesis configuration"""
    char_rotation_range: float = 2.5
    char_scale_variance: float = 0.08
    char_shear_range: float = 5.0
    char_spacing_variance: float = 1.8
    baseline_wave_amplitude: float = 1.2
    baseline_wave_frequency: float = 180.0
    baseline_drift: float = 0.4
    ink_pressure_layers: int = 3
    ink_bleeding: float = 0.4
    ink_fade_variance: float = 0.15
    stroke_width_variance: float = 0.3
    pen_pressure_simulation: bool = True
    paper_noise_intensity: float = 0.12
    paper_grain_size: int = 2
    fatigue_enabled: bool = False
    fatigue_rate: float = 0.0001
    ligature_detection: bool = True
    word_spacing_natural: bool = True
    margin_irregularity: float = 8.0

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))

# ==================== AI TEXT GENERATION ====================
def generate_assignment_answer(question: str, pages: int = 2, subject: str = "general") -> str:
    """Enhanced AI answer generation with subject-specific formatting"""
    approx_words_per_page = 180
    target_words = pages * approx_words_per_page

    subject_prompts = {
        "law": "You are writing as an Indian first-year LLB student. Use legal terminology appropriately.",
        "science": "You are a science student. Include formulas in LaTeX (e.g., $E=mc^2$).",
        "mathematics": "You are a mathematics student. Show solutions with LaTeX equations.",
        "general": "You are a first-year undergraduate student in India."
    }

    prompt = f"""
{subject_prompts.get(subject, subject_prompts["general"])}

Write a comprehensive answer to: {question}

Requirements:
- Target length: {target_words} words
- Use clear headings and structured paragraphs
- For math: use LaTeX with $...$ for inline or $$...$$ for display equations
- Write naturally with appropriate academic vocabulary
- No bullet points - use flowing paragraphs only
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=min(target_words * 2, 3000),
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return ""

# ==================== FONT MANAGEMENT ====================
def load_font_from_path(font_path: Union[str, Path, None], font_size: int):
    """Load font with fallback handling"""
    if font_path is None:
        st.warning("⚠️ No font provided. Using default font.")
        return ImageFont.load_default()

    try:
        return ImageFont.truetype(str(font_path), font_size)
    except Exception as e:
        st.warning(f"Could not load font: {e}")
        return ImageFont.load_default()

# ==================== IMPROVED MATH RENDERING ====================
_math_cache = {}

def render_math_to_image(math_tex: str, font_size: int = 28, color=(0, 0, 0)):
    """
    Render LaTeX math to TRANSPARENT image - FIXED VERSION
    No ugly background boxes!
    """
    cache_key = (math_tex, font_size, color)
    if cache_key in _math_cache:
        return _math_cache[cache_key]

    if not MATPLOTLIB_AVAILABLE:
        # Fallback to text rendering
        font = ImageFont.load_default()
        dummy = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
        d = ImageDraw.Draw(dummy)
        bbox = d.textbbox((0, 0), math_tex, font=font)
        w = max(bbox[2] - bbox[0] + 4, 5)
        h = max(bbox[3] - bbox[1] + 4, 5)
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.text((2, 2), math_tex, font=font, fill=color)
        _math_cache[cache_key] = img
        return img

    # Strip delimiters
    content = math_tex.strip()
    if content.startswith("$$") and content.endswith("$$"):
        content = content[2:-2]
    elif content.startswith("$") and content.endswith("$"):
        content = content[1:-1]

    try:
        # Create figure with TRANSPARENT background
        fig = plt.figure(figsize=(0.1, 0.1), dpi=150, facecolor='none')
        fig.patch.set_alpha(0.0)
        
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.patch.set_alpha(0.0)
        
        # Render math text
        text_obj = ax.text(
            0, 0, f"${content}$",
            fontsize=font_size,
            color=f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}',
            ha='left', va='baseline',
            transform=ax.transData
        )
        
        # Get tight bounding box
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = text_obj.get_window_extent(renderer=renderer)
        
        # Adjust figure size to content
        w_inch = (bbox.width + 10) / fig.dpi
        h_inch = (bbox.height + 10) / fig.dpi
        
        plt.close(fig)
        
        # Recreate with proper size
        fig = plt.figure(figsize=(w_inch, h_inch), dpi=150, facecolor='none')
        fig.patch.set_alpha(0.0)
        
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.patch.set_alpha(0.0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        ax.text(
            0.5, 0.5, f"${content}$",
            fontsize=font_size,
            color=f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}',
            ha='center', va='center',
            transform=ax.transAxes
        )
        
        fig.canvas.draw()
        
        # Convert to RGBA with transparency preserved
        buf = io.BytesIO()
        fig.savefig(buf, format='png', transparent=True, bbox_inches='tight', 
                   pad_inches=0.05, facecolor='none', edgecolor='none')
        plt.close(fig)
        
        buf.seek(0)
        pil_img = Image.open(buf).convert('RGBA')
        
        # Ensure no white background pixels
        data = np.array(pil_img)
        # Make white/near-white pixels transparent
        white_areas = (data[:, :, 0] > 240) & (data[:, :, 1] > 240) & (data[:, :, 2] > 240)
        data[white_areas, 3] = 0
        pil_img = Image.fromarray(data, 'RGBA')
        
        # Trim excess transparent space
        bbox = pil_img.getbbox()
        if bbox:
            pil_img = pil_img.crop(bbox)
        
        _math_cache[cache_key] = pil_img
        return pil_img
        
    except Exception as e:
        st.warning(f"Math rendering error: {e}")
        # Fallback: simple text
        font = ImageFont.load_default()
        img = Image.new("RGBA", (len(content) * 8, 20), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.text((2, 2), content, font=font, fill=color)
        return img

# ==================== TEXT PROCESSING ====================
def split_text_preserving_math(text: str) -> List[str]:
    """Split text into tokens, preserving math expressions"""
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

def wrap_text_to_pixel_width(draw: ImageDraw.Draw, text: str, 
                             font: ImageFont.FreeTypeFont, 
                             max_width: int) -> List[List[Tuple[str, str]]]:
    """Wrap text into lines based on pixel width"""
    tokens = split_text_preserving_math(text)
    lines = []
    current_line = []
    current_width = 0
    
    try:
        space_w = draw.textbbox((0, 0), " ", font=font)[2]
    except:
        space_w = font.size // 3

    for tok in tokens:
        if tok.startswith("$"):
            math_img = render_math_to_image(tok, font_size=int(font.size * 0.95))
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
                for word in words:
                    try:
                        w_bbox = draw.textbbox((0, 0), word, font=font)
                        w_width = w_bbox[2] - w_bbox[0]
                    except:
                        w_width = len(word) * font.size // 2
                    
                    if current_width == 0:
                        current_line.append(("text", word))
                        current_width = w_width
                    else:
                        if current_width + space_w + w_width <= max_width:
                            current_line.append(("text", " " + word))
                            current_width += space_w + w_width
                        else:
                            lines.append(current_line)
                            current_line = [("text", word)]
                            current_width = w_width
                
                if pi < len(parts) - 1:
                    lines.append(current_line)
                    current_line = []
                    current_width = 0
    
    if current_line:
        lines.append(current_line)
    
    return lines

# ==================== PAPER GENERATION ====================
def create_textured_paper(width: int, height: int, base_color: Tuple[int, int, int], 
                         texture_intensity: float = 0.12) -> Image.Image:
    """Create realistic paper texture"""
    base = Image.new("RGB", (width, height), base_color)
    
    try:
        noise = Image.effect_noise((width // 2, height // 2), 64).convert("L")
        noise = noise.resize((width, height), Image.BICUBIC)
        noise = noise.point(lambda p: int(p * texture_intensity))
        noise_rgb = Image.merge("RGB", (noise, noise, noise))
        base = Image.blend(base, noise_rgb, alpha=0.15)
    except Exception:
        pass
    
    return base

# ==================== CHARACTER RENDERING ====================
def render_character_with_pressure(char: str, font: ImageFont.FreeTypeFont, 
                                  ink_color: Tuple[int, int, int], 
                                  pressure: float = 1.0, 
                                  config: HandwritingConfig = None) -> Image.Image:
    """Render single character with pen pressure simulation"""
    if config is None:
        config = HandwritingConfig()
    
    if char == " ":
        try:
            w = font.getbbox(" ")[2]
        except:
            w = font.size // 3
        return Image.new("RGBA", (max(w, 5), max(font.size, 10)), (255, 255, 255, 0))
    
    temp_img = Image.new("RGBA", (font.size * 4, font.size * 4), (255, 255, 255, 0))
    d = ImageDraw.Draw(temp_img)
    try:
        bbox = d.textbbox((0, 0), char, font=font)
        w = max(bbox[2] - bbox[0] + 10, 5)
        h = max(bbox[3] - bbox[1] + 10, 10)
    except:
        w, h = font.size * 2, font.size * 2
    
    char_img = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    cd = ImageDraw.Draw(char_img)
    
    layers = max(1, int(config.ink_pressure_layers * pressure))
    for i in range(layers):
        offset_x = random.gauss(0, 0.6) * pressure
        offset_y = random.gauss(0, 0.6) * pressure
        alpha = int(255 * random.uniform(0.75 + pressure * 0.1, 0.95))
        fill = (ink_color[0], ink_color[1], ink_color[2], alpha)
        cd.text((5 + offset_x, 5 + offset_y), char, font=font, fill=fill)
    
    if config.ink_bleeding > 0:
        blur_radius = config.ink_bleeding * pressure
        char_img = char_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    bbox = char_img.getbbox()
    if bbox:
        char_img = char_img.crop(bbox)
    
    return char_img

def apply_character_transformation(char_img: Image.Image, 
                                  rotation: float = 0.0, 
                                  shear: float = 0.0, 
                                  scale: float = 1.0) -> Image.Image:
    """Apply geometric transformations"""
    w, h = char_img.size
    
    if abs(scale - 1.0) > 0.01:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        char_img = char_img.resize((new_w, new_h), resample=Image.BICUBIC)
        w, h = char_img.size
    
    if abs(shear) > 0.1:
        sh = math.tan(math.radians(shear))
        new_w = int(w + abs(sh) * h) + 2
        char_img = char_img.transform(
            (new_w, h), Image.AFFINE, (1, sh, 0, 0, 1, 0),
            resample=Image.BICUBIC, fillcolor=(255, 255, 255, 0)
        )
    
    if abs(rotation) > 0.1:
        char_img = char_img.rotate(
            rotation, resample=Image.BICUBIC, expand=True,
            fillcolor=(255, 255, 255, 0)
        )
    
    return char_img

def should_ligate(prev_char: str, curr_char: str) -> bool:
    """Determine if characters should connect"""
    ligature_pairs = [
        ('f', 'i'), ('f', 'l'), ('t', 'h'), ('c', 'h'),
        ('o', 'n'), ('i', 'n'), ('r', 'e'), ('t', 'o')
    ]
    return (prev_char.lower(), curr_char.lower()) in ligature_pairs

# ==================== MAIN RENDERING FUNCTION ====================
def render_handwritten_page(
    text: str,
    font_obj: ImageFont.ImageFont,
    config: HandwritingConfig,
    img_width: int = 1240,
    img_height: int = 1754,
    margin_left: int = 120,
    margin_top: int = 120,
    line_spacing: int = 12,
    ink_color: Tuple[int, int, int] = (10, 10, 10),
    paper_color: Tuple[int, int, int] = (245, 242, 230),
    ruled: bool = False,
    page_number: int = 1,
    total_pages: int = 1,
    header_text: str = None
) -> Image.Image:
    """Advanced handwriting rendering with transparent math"""
    
    # Create paper
    base = create_textured_paper(img_width, img_height, paper_color, config.paper_noise_intensity)
    
    # Text layer
    text_layer = Image.new("RGBA", (img_width, img_height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(text_layer)
    
    max_text_width = img_width - 2 * margin_left
    lines = wrap_text_to_pixel_width(draw, text, font_obj, max_text_width)
    
    # Ruled lines
    if ruled:
        ruled_color = (180, 200, 215, 80)
        y = margin_top
        line_height = int(font_obj.size * 1.9)
        while y < img_height - margin_top:
            y_offset = random.uniform(-0.5, 0.5)
            draw.line(
                [(margin_left - 20, y + y_offset), (img_width - margin_left + 20, y + y_offset)],
                fill=ruled_color, width=1
            )
            y += line_height
    
    # Header
    if header_text:
        header_y = margin_top - int(font_obj.size * 1.8)
        header_color = tuple(min(255, c + 30) for c in ink_color)
        draw.text((margin_left, header_y), header_text, font=font_obj, fill=header_color)
    
    # Rendering parameters
    baseline_amp = config.baseline_wave_amplitude
    baseline_freq = config.baseline_wave_frequency
    char_jitter_x = font_obj.size * 0.025
    char_jitter_y = font_obj.size * 0.035
    
    x_start = margin_left + random.uniform(-config.margin_irregularity, config.margin_irregularity)
    y = margin_top
    
    char_count = 0
    prev_char = None
    
    # Render lines
    for line_idx, line in enumerate(lines):
        if y > img_height - margin_top - font_obj.size * 2:
            break
        
        x = x_start + random.uniform(-3, 3)
        baseline_offset = 0
        
        for token_type, token_content in line:
            if token_type == "text":
                for char in token_content:
                    if char == " ":
                        try:
                            sp = draw.textbbox((0, 0), " ", font=font_obj)[2]
                        except:
                            sp = font_obj.size // 3
                        
                        if config.word_spacing_natural:
                            sp *= random.uniform(0.9, 1.2)
                        x += sp
                        prev_char = None
                        continue
                    
                    wave = baseline_amp * math.sin((x + line_idx * 17) / baseline_freq * 2 * math.pi)
                    baseline_offset += config.baseline_drift * random.uniform(-0.5, 0.5)
                    
                    jitter_x = random.gauss(0, char_jitter_x)
                    jitter_y = random.gauss(0, char_jitter_y)
                    rotation = random.gauss(0, config.char_rotation_range) * 0.6
                    shear = random.gauss(0, config.char_shear_range) * 0.15
                    scale = random.gauss(1.0, config.char_scale_variance)
                    pressure = random.uniform(0.85, 1.15)
                    
                    if config.fatigue_enabled:
                        fatigue = 1.0 + char_count * config.fatigue_rate
                        rotation *= fatigue
                        jitter_y += fatigue * 0.5
                    
                    char_img = render_character_with_pressure(char, font_obj, ink_color, pressure, config)
                    char_img = apply_character_transformation(char_img, rotation, shear, scale)
                    
                    px = int(x + jitter_x)
                    py = int(y + jitter_y + wave + baseline_offset)
                    
                    if 0 <= px < img_width and 0 <= py < img_height:
                        text_layer.paste(char_img, (px, py), char_img)
                    
                    spacing = char_img.width + random.gauss(1.2, config.char_spacing_variance)
                    
                    if config.ligature_detection and prev_char and should_ligate(prev_char, char):
                        spacing *= 0.75
                    
                    x += spacing
                    prev_char = char
                    char_count += 1
                    
            else:  # Math token - NOW WITH TRANSPARENT BACKGROUND!
                math_img = render_math_to_image(token_content, font_size=int(font_obj.size * 0.95), color=ink_color)
                
                # Center math vertically on baseline
                py = int(y - (math_img.height - font_obj.size) / 2 + random.uniform(-2, 2))
                px = int(x + random.uniform(-1, 1))
                
                # Paste with transparency preserved
                if 0 <= px < img_width and 0 <= py < img_height:
                    text_layer.paste(math_img, (px, py), math_img)
                
                x += math_img.width + 5
                prev_char = None
        
        y += int(font_obj.size * 1.05) + line_spacing + random.uniform(-1, 1)
    
    # Footer
    if total_pages > 1:
        footer_text = f"— {page_number} —"
        footer_y = img_height - margin_top // 2
        try:
            footer_bbox = draw.textbbox((0, 0), footer_text, font=font_obj)
            footer_w = footer_bbox[2] - footer_bbox[0]
        except:
            footer_w = len(footer_text) * font_obj.size // 2
        
        footer_x = (img_width - footer_w) // 2
        footer_color = tuple(min(255, c + 50) for c in ink_color)
        draw.text((footer_x, footer_y), footer_text, font=font_obj, fill=footer_color)
    
    # Subtle blur
    text_layer = text_layer.filter(ImageFilter.GaussianBlur(radius=0.25))
    
    # Slight rotation
    angle = random.uniform(-0.8, 0.8)
    text_layer = text_layer.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255, 0))
    
    # Composite
    final = base.convert("RGBA")
    final = Image.alpha_composite(final, text_layer).convert("RGB")
    final = ImageOps.autocontrast(final, cutoff=0.5)
    
    return final

# ==================== TEXT SPLITTING ====================
def split_text_into_pages(text: str, pages: int) -> List[str]:
    """Split text into pages"""
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
    chunks = []
    for p in range(pages):
        start = p * per_page
        end = start + per_page if p < pages - 1 else len(words)
        chunk = " ".join(words[start:end])
        chunks.append(chunk if chunk else " ")
    
    return chunks

# ==================== STREAMLIT UI ====================
st.set_page_config(
    page_title="✍️ AI Handwriting Generator Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>✍️ AI Handwriting Generator Pro</h1>
    <p>Transform text into ultra-realistic handwritten documents with transparent math formulas</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🎨 Configuration")
    
    st.subheader("📝 Font Settings")
    uploaded_font = st.file_uploader("Upload TTF/OTF font", type=["ttf", "otf"])
    font_size = st.slider("Font size", 18, 72, 32, 1)
    
    st.subheader("📄 Document Settings")
    pages = st.number_input("Pages", 1, 15, 2)
    subject = st.selectbox("Subject", ["general", "law", "science", "mathematics"])
    
    st.subheader("🎭 Style Presets")
    style_preset = st.selectbox("Preset", [
        "Balanced (Recommended)",
        "Neat & Tidy",
        "Natural & Casual",
        "Slightly Messy",
        "Exam Rush Mode"
    ])
    
    preset_configs = {
        "Balanced (Recommended)": HandwritingConfig(),
        "Neat & Tidy": HandwritingConfig(
            char_rotation_range=1.0,
            baseline_wave_amplitude=0.6,
            char_spacing_variance=0.8,
            ink_bleeding=0.2
        ),
        "Natural & Casual": HandwritingConfig(
            char_rotation_range=3.0,
            baseline_wave_amplitude=1.5,
            char_spacing_variance=2.2,
            baseline_drift=0.8
        ),
        "Slightly Messy": HandwritingConfig(
            char_rotation_range=4.0,
            baseline_wave_amplitude=2.0,
            char_spacing_variance=2.5,
            baseline_drift=1.2,
            margin_irregularity=15.0
        ),
        "Exam Rush Mode": HandwritingConfig(
            char_rotation_range=3.5,
            baseline_wave_amplitude=1.8,
            char_spacing_variance=3.0,
            fatigue_enabled=True,
            fatigue_rate=0.0002,
            ink_pressure_layers=2
        )
    }
    
    config = preset_configs[style_preset]
    
    with st.expander("⚙️ Advanced"):
        config.char_rotation_range = st.slider("Rotation", 0.0, 6.0, config.char_rotation_range, 0.1)
        config.baseline_wave_amplitude = st.slider("Wobble", 0.0, 3.0, config.baseline_wave_amplitude, 0.1)
        config.ink_bleeding = st.slider("Ink bleeding", 0.0, 1.5, config.ink_bleeding, 0.1)
    
    st.subheader("🎨 Appearance")
    ink_color_choice = st.selectbox("Ink", ["Black", "Dark Blue", "Brown", "Gray", "Green"])
    paper_color_choice = st.selectbox("Paper", ["Ivory", "White", "Aged (Beige)", "Light Blue"])
    ruled = st.checkbox("Ruled lines", False)
    
    ink_colors = {
        "Black": (20, 20, 20),
        "Dark Blue": (8, 35, 86),
        "Brown": (60, 30, 10),
        "Gray": (60, 60, 60),
        "Green": (15, 70, 30)
    }
    
    paper_colors = {
        "White": (255, 255, 255),
        "Ivory": (245, 242, 230),
        "Aged (Beige)": (238, 230, 210),
        "Light Blue": (240, 248, 255)
    }
    
    st.success("✅ Math formulas now render with **transparent backgrounds**!")

# Font loading
try:
    repo_font_path = Path(__file__).parent / "fonts" / "handwriting.ttf"
except NameError:
    repo_font_path = Path(os.getcwd()) / "fonts" / "handwriting.ttf"

font_path_to_use = None
if uploaded_font:
    try:
        tmp_font_path = Path(tempfile.gettempdir()) / uploaded_font.name
        with open(tmp_font_path, "wb") as f:
            f.write(uploaded_font.getbuffer())
        font_path_to_use = tmp_font_path
    except Exception as e:
        st.error(f"Font error: {e}")
elif repo_font_path.exists():
    font_path_to_use = repo_font_path

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    question = st.text_area(
        "📝 Enter question or text",
        height=200,
        placeholder="Example: Explain Newton's laws. Include: $F = ma$ and $E = mc^2$"
    )

with col2:
    st.markdown("### 🚀 Features")
    st.markdown("""
    <div class="feature-box">
    ✅ AI answer generation<br>
    ✅ <b>Transparent math formulas</b> (FIXED!)<br>
    ✅ Ultra-realistic handwriting<br>
    ✅ Custom fonts<br>
    ✅ Multiple presets<br>
    ✅ PDF & ZIP export
    </div>
    """, unsafe_allow_html=True)

# Generate
if st.button("✨ Generate", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("⚠️ Enter a question!")
    else:
        with st.spinner("🤖 Generating answer..."):
            answer_text = generate_assignment_answer(question, int(pages), subject)
        
        if not answer_text:
            st.error("❌ Generation failed")
        else:
            word_count = len(answer_text.split())
            st.success(f"✅ Generated {word_count} words")
            
            font_obj = load_font_from_path(font_path_to_use, int(font_size))
            chunks = split_text_into_pages(answer_text, int(pages))
            
            images = []
            progress = st.progress(0)
            status = st.empty()
            
            for idx, chunk in enumerate(chunks, start=1):
                status.text(f"✍️ Rendering page {idx}/{len(chunks)}...")
                
                ink_base = ink_colors[ink_color_choice]
                ink_varied = tuple(max(0, min(255, c + random.randint(-12, 12))) for c in ink_base)
                header = f"Page {idx} of {len(chunks)}" if len(chunks) > 1 else None
                
                img = render_handwritten_page(
                    chunk, font_obj, config,
                    img_width=1240, img_height=1754,
                    margin_left=100, margin_top=120,
                    line_spacing=int(font_obj.size * 0.45),
                    ink_color=ink_varied,
                    paper_color=paper_colors[paper_color_choice],
                    ruled=ruled,
                    page_number=idx,
                    total_pages=len(chunks),
                    header_text=header
                )
                
                images.append(img)
                progress.progress(idx / len(chunks))
            
            status.text("✅ Complete!")
            
            # Preview
            st.subheader("📄 Preview")
            cols = st.columns(min(3, len(images)))
            for i, img in enumerate(images):
                with cols[i % len(cols)]:
                    st.image(img.resize((300, int(300 * img.height / img.width))), 
                            caption=f"Page {i+1}", use_container_width=True)
            
            # Downloads
            st.subheader("⬇️ Download")
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for i, img in enumerate(images, start=1):
                        img_bytes = io.BytesIO()
                        img.save(img_bytes, format="PNG", quality=95)
                        img_bytes.seek(0)
                        zf.writestr(f"page_{i:02d}.png", img_bytes.read())
                zip_buffer.seek(0)
                
                st.download_button(
                    "📦 ZIP", zip_buffer,
                    "handwritten.zip", "application/zip",
                    use_container_width=True
                )
            
            with col_dl2:
                pdf_bytes = io.BytesIO()
                rgb_images = [im.convert("RGB") for im in images]
                rgb_images[0].save(pdf_bytes, format="PDF", 
                                  save_all=True, append_images=rgb_images[1:], quality=95)
                pdf_bytes.seek(0)
                
                st.download_button(
                    "📄 PDF", pdf_bytes,
                    "handwritten.pdf", "application/pdf",
                    use_container_width=True
                )
            
            st.balloons()

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><b>✍️ AI Handwriting Generator Pro</b> | Math formulas now render transparently!</p>
</div>
""", unsafe_allow_html=True)

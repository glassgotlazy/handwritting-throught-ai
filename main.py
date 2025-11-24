import streamlit as st
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps, ImageEnhance
import textwrap
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
import json

# Advanced Math Rendering
try:
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    MATPLOTLIB_AVAILABLE = True
    rcParams["mathtext.fontset"] = "dejavusans"
except Exception:
    MATPLOTLIB_AVAILABLE = False

# ==================== CONFIGURATION ====================
@dataclass
class HandwritingConfig:
    """Advanced handwriting synthesis configuration"""
    # Character-level variations
    char_rotation_range: float = 2.5
    char_scale_variance: float = 0.08
    char_shear_range: float = 5.0
    char_spacing_variance: float = 1.8
    
    # Baseline dynamics
    baseline_wave_amplitude: float = 1.2
    baseline_wave_frequency: float = 180.0
    baseline_drift: float = 0.4  # gradual y-axis drift
    
    # Ink simulation
    ink_pressure_layers: int = 3
    ink_bleeding: float = 0.4
    ink_fade_variance: float = 0.15
    
    # Pen dynamics
    stroke_width_variance: float = 0.3
    pen_pressure_simulation: bool = True
    
    # Paper texture
    paper_noise_intensity: float = 0.12
    paper_grain_size: int = 2
    
    # Writing fatigue simulation
    fatigue_enabled: bool = False
    fatigue_rate: float = 0.0001  # progressive degradation
    
    # Advanced features
    ligature_detection: bool = True  # connect certain letter pairs
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
        "law": "You are writing as an Indian first-year LLB student. Use legal terminology appropriately and cite relevant articles/sections where needed.",
        "science": "You are a science student. Include formulas in LaTeX (e.g., $E=mc^2$) and explain concepts clearly with examples.",
        "mathematics": "You are a mathematics student. Show step-by-step solutions with proper LaTeX formatting for all equations.",
        "general": "You are a first-year undergraduate student in India writing an exam answer."
    }

    prompt = f"""
{subject_prompts.get(subject, subject_prompts["general"])}

Write a comprehensive answer to: {question}

Requirements:
- Target length: {target_words} words
- Use clear headings and structured paragraphs
- For math: use LaTeX with $...$ for inline or $$...$$ for display equations
- Write naturally with appropriate academic vocabulary
- Include relevant examples where helpful
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
        st.warning("⚠️ No font provided. Using default font (limited realism).")
        return ImageFont.load_default()

    try:
        return ImageFont.truetype(str(font_path), font_size)
    except Exception as e:
        st.warning(f"Could not load font: {e}")
        return ImageFont.load_default()

# ==================== MATH RENDERING ====================
_math_cache = {}

def render_math_to_image(math_tex: str, font_size: int = 28, color=(0, 0, 0)):
    """Render LaTeX math to image with caching"""
    cache_key = (math_tex, font_size, color)
    if cache_key in _math_cache:
        return _math_cache[cache_key]

    if not MATPLOTLIB_AVAILABLE:
        # Fallback to text rendering
        font = ImageFont.load_default()
        dummy = Image.new("RGBA", (10, 10), (255, 255, 255, 0))
        d = ImageDraw.Draw(dummy)
        bbox = d.textbbox((0, 0), math_tex, font=font)
        img = Image.new("RGBA", (bbox[2] - bbox[0] + 4, bbox[3] - bbox[1] + 4), (255, 255, 255, 0))
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
        fig = plt.figure(figsize=(0.01, 0.01), dpi=200)
        fig.patch.set_alpha(0.0)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        
        text_obj = ax.text(0, 0, f"${content}$", 
                          fontsize=font_size, 
                          color='#%02x%02x%02x' % color, 
                          ha='left', va='bottom')
        
        fig.canvas.draw()
        bbox = text_obj.get_window_extent(renderer=fig.canvas.get_renderer())
        width, height = int(bbox.width) + 8, int(bbox.height) + 8
        
        plt.close(fig)
        
        fig = plt.figure(figsize=(width / 200.0, height / 200.0), dpi=200)
        fig.patch.set_alpha(0.0)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.text(0, 0, f"${content}$", fontsize=font_size, 
                color='#%02x%02x%02x' % color, ha='left', va='bottom')
        
        fig.canvas.draw()
        buf = fig.canvas.tostring_argb()
        w, h = fig.canvas.get_width_height()
        plt.close(fig)
        
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
        arr = arr[:, :, [1, 2, 3, 0]]  # ARGB -> RGBA
        pil_img = Image.fromarray(arr, mode="RGBA")
        
        bbox = pil_img.getbbox()
        if bbox:
            pil_img = pil_img.crop(bbox)
        
        _math_cache[cache_key] = pil_img
        return pil_img
    except Exception as e:
        # Fallback
        return Image.new("RGBA", (50, 20), (255, 255, 255, 0))

# ==================== TEXT PROCESSING ====================
def split_text_preserving_math(text: str) -> List[str]:
    """Split text into tokens, preserving math expressions"""
    tokens = []
    i = 0
    n = len(text)
    
    while i < n:
        if text[i] == "$":
            if i + 1 < n and text[i+1] == "$":
                # Display math $$...$$
                j = text.find("$$", i+2)
                if j == -1:
                    tokens.append(text[i:])
                    break
                tokens.append(text[i:j+2])
                i = j + 2
            else:
                # Inline math $...$
                j = text.find("$", i+1)
                if j == -1:
                    tokens.append(text[i:])
                    break
                tokens.append(text[i:j+1])
                i = j + 1
        else:
            # Regular text
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
    """Wrap text into lines based on pixel width, preserving math"""
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
            # Math token
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
            # Plain text
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
                
                # Handle newlines
                if pi < len(parts) - 1:
                    lines.append(current_line)
                    current_line = []
                    current_width = 0
    
    if current_line:
        lines.append(current_line)
    
    return lines

# ==================== PAPER GENERATION ====================
def create_textured_paper(width: int, height: int, base_color: Tuple[int, int, int], 
                         texture_intensity: float = 0.12, grain_size: int = 2) -> Image.Image:
    """Create realistic paper texture with grain and subtle variations"""
    base = Image.new("RGB", (width, height), base_color)
    
    # Add Perlin-like noise for paper texture
    try:
        noise = Image.effect_noise((width // grain_size, height // grain_size), 64).convert("L")
        noise = noise.resize((width, height), Image.BICUBIC)
        noise = noise.point(lambda p: int(p * texture_intensity))
        noise_rgb = Image.merge("RGB", (noise, noise, noise))
        base = Image.blend(base, noise_rgb, alpha=0.15)
        
        # Add subtle color variations (paper imperfections)
        color_noise = Image.effect_noise((width // 8, height // 8), 32).convert("L")
        color_noise = color_noise.resize((width, height), Image.BICUBIC)
        color_noise = color_noise.point(lambda p: int((p - 128) * 0.05))
        color_rgb = Image.merge("RGB", (color_noise, color_noise, color_noise))
        base = Image.blend(base, color_rgb, alpha=0.08)
    except Exception:
        pass
    
    return base

# ==================== ADVANCED HANDWRITING RENDERING ====================
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
    
    # Estimate size
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
    
    # Simulate pen pressure with multiple layers
    layers = max(1, int(config.ink_pressure_layers * pressure))
    for i in range(layers):
        offset_x = random.gauss(0, 0.6) * pressure
        offset_y = random.gauss(0, 0.6) * pressure
        
        # Vary opacity based on pressure
        alpha = int(255 * random.uniform(0.75 + pressure * 0.1, 0.95))
        fill = (ink_color[0], ink_color[1], ink_color[2], alpha)
        
        cd.text((5 + offset_x, 5 + offset_y), char, font=font, fill=fill)
    
    # Ink bleeding effect
    if config.ink_bleeding > 0:
        blur_radius = config.ink_bleeding * pressure
        char_img = char_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # Trim transparent borders
    bbox = char_img.getbbox()
    if bbox:
        char_img = char_img.crop(bbox)
    
    return char_img

def apply_character_transformation(char_img: Image.Image, 
                                  rotation: float = 0.0, 
                                  shear: float = 0.0, 
                                  scale: float = 1.0) -> Image.Image:
    """Apply geometric transformations to character"""
    w, h = char_img.size
    
    # Scale
    if abs(scale - 1.0) > 0.01:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        char_img = char_img.resize((new_w, new_h), resample=Image.BICUBIC)
        w, h = char_img.size
    
    # Shear (horizontal)
    if abs(shear) > 0.1:
        sh = math.tan(math.radians(shear))
        new_w = int(w + abs(sh) * h) + 2
        char_img = char_img.transform(
            (new_w, h), 
            Image.AFFINE, 
            (1, sh, 0, 0, 1, 0), 
            resample=Image.BICUBIC, 
            fillcolor=(255, 255, 255, 0)
        )
    
    # Rotation
    if abs(rotation) > 0.1:
        char_img = char_img.rotate(
            rotation, 
            resample=Image.BICUBIC, 
            expand=True, 
            fillcolor=(255, 255, 255, 0)
        )
    
    return char_img

def should_ligate(prev_char: str, curr_char: str) -> bool:
    """Determine if two characters should be connected (ligatures)"""
    # Common letter pairs that naturally connect in cursive
    ligature_pairs = [
        ('f', 'i'), ('f', 'l'), ('f', 'f'),
        ('t', 'h'), ('c', 'h'), ('s', 'h'),
        ('o', 'n'), ('i', 'n'), ('i', 'o'),
        ('r', 'e'), ('t', 'o'), ('a', 'n')
    ]
    return (prev_char.lower(), curr_char.lower()) in ligature_pairs

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
    """
    Advanced handwriting rendering with human-like characteristics
    """
    # Create textured paper
    base = create_textured_paper(img_width, img_height, paper_color, 
                                 config.paper_noise_intensity, config.paper_grain_size)
    
    # Create transparent layer for text
    text_layer = Image.new("RGBA", (img_width, img_height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(text_layer)
    
    max_text_width = img_width - 2 * margin_left
    
    # Wrap text
    lines = wrap_text_to_pixel_width(draw, text, font_obj, max_text_width)
    
    # Draw ruled lines if enabled
    if ruled:
        ruled_color = (180, 200, 215, 80)
        y = margin_top
        line_height = int(font_obj.size * 1.9)
        while y < img_height - margin_top:
            # Add slight irregularity to ruled lines
            y_offset = random.uniform(-0.5, 0.5)
            draw.line(
                [(margin_left - 20, y + y_offset), (img_width - margin_left + 20, y + y_offset)],
                fill=ruled_color, width=1
            )
            y += line_height
    
    # Draw header
    if header_text:
        header_y = margin_top - int(font_obj.size * 1.8)
        header_color = tuple(min(255, c + 30) for c in ink_color)
        draw.text((margin_left, header_y), header_text, font=font_obj, fill=header_color)
    
    # Handwriting parameters
    baseline_amp = config.baseline_wave_amplitude
    baseline_freq = config.baseline_wave_frequency
    char_jitter_x = font_obj.size * 0.025
    char_jitter_y = font_obj.size * 0.035
    
    x_start = margin_left + random.uniform(-config.margin_irregularity, config.margin_irregularity)
    y = margin_top
    
    char_count = 0  # For fatigue simulation
    prev_char = None
    
    # Render each line
    for line_idx, line in enumerate(lines):
        if y > img_height - margin_top - font_obj.size * 2:
            break
        
        x = x_start + random.uniform(-3, 3)  # Line-start irregularity
        baseline_offset = 0
        
        for token_type, token_content in line:
            if token_type == "text":
                for char in token_content:
                    if char == " ":
                        # Natural space width with variation
                        try:
                            sp = draw.textbbox((0, 0), " ", font=font_obj)[2]
                        except:
                            sp = font_obj.size // 3
                        
                        if config.word_spacing_natural:
                            sp *= random.uniform(0.9, 1.2)
                        x += sp
                        prev_char = None
                        continue
                    
                    # Baseline wave + drift
                    wave = baseline_amp * math.sin((x + line_idx * 17) / baseline_freq * 2 * math.pi)
                    baseline_offset += config.baseline_drift * random.uniform(-0.5, 0.5)
                    
                    # Character-level randomization
                    jitter_x = random.gauss(0, char_jitter_x)
                    jitter_y = random.gauss(0, char_jitter_y)
                    
                    rotation = random.gauss(0, config.char_rotation_range) * 0.6
                    shear = random.gauss(0, config.char_shear_range) * 0.15
                    scale = random.gauss(1.0, config.char_scale_variance)
                    
                    # Pen pressure variation (natural writing has varying pressure)
                    pressure = random.uniform(0.85, 1.15)
                    
                    # Fatigue simulation (handwriting degrades over time)
                    if config.fatigue_enabled:
                        fatigue_factor = 1.0 + char_count * config.fatigue_rate
                        rotation *= fatigue_factor
                        jitter_y += fatigue_factor * 0.5
                    
                    # Render character
                    char_img = render_character_with_pressure(char, font_obj, ink_color, pressure, config)
                    char_img = apply_character_transformation(char_img, rotation, shear, scale)
                    
                    # Calculate position
                    px = int(x + jitter_x)
                    py = int(y + jitter_y + wave + baseline_offset)
                    
                    # Paste character
                    if 0 <= px < img_width and 0 <= py < img_height:
                        text_layer.paste(char_img, (px, py), char_img)
                    
                    # Advance x position
                    spacing = char_img.width + random.gauss(1.2, config.char_spacing_variance)
                    
                    # Ligature adjustment (closer spacing for connected letters)
                    if config.ligature_detection and prev_char and should_ligate(prev_char, char):
                        spacing *= 0.75
                    
                    x += spacing
                    prev_char = char
                    char_count += 1
                    
            else:  # Math token
                math_img = render_math_to_image(token_content, font_size=int(font_obj.size * 0.95), color=ink_color)
                py = int(y - (math_img.height - font_obj.size) / 2 + random.uniform(-2, 2))
                px = int(x + random.uniform(-1, 1))
                
                if 0 <= px < img_width and 0 <= py < img_height:
                    text_layer.paste(math_img, (px, py), math_img)
                
                x += math_img.width + 5
                prev_char = None
        
        # Move to next line
        y += int(font_obj.size * 1.05) + line_spacing + random.uniform(-1, 1)
    
    # Add footer (page number)
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
    
    # Apply subtle blur for ink realism
    text_layer = text_layer.filter(ImageFilter.GaussianBlur(radius=0.25))
    
    # Slight page rotation for realism
    angle = random.uniform(-0.8, 0.8)
    text_layer = text_layer.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255, 0))
    
    # Composite onto paper
    final = base.convert("RGBA")
    final = Image.alpha_composite(final, text_layer).convert("RGB")
    
    # Subtle post-processing
    final = ImageOps.autocontrast(final, cutoff=0.5)
    
    # Add slight vignette effect
    enhancer = ImageEnhance.Brightness(final)
    final = enhancer.enhance(0.98)
    
    return final

# ==================== TEXT SPLITTING ====================
def split_text_into_pages(text: str, pages: int) -> List[str]:
    """Intelligently split text into pages"""
    words = text.split()
    if pages <= 1:
        return [text]
    
    # Split by paragraphs if possible
    paragraphs = text.split("\n\n")
    if len(paragraphs) >= pages:
        # Distribute paragraphs across pages
        chunks = [[] for _ in range(pages)]
        for i, para in enumerate(paragraphs):
            chunks[i % pages].append(para)
        return ["\n\n".join(chunk) for chunk in chunks]
    
    # Otherwise split by words
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

# Custom CSS
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
    <p>Transform typed text into ultra-realistic handwritten documents</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("🎨 Configuration")
    
    # Font upload
    st.subheader("📝 Font Settings")
    uploaded_font = st.file_uploader("Upload TTF/OTF font (optional)", type=["ttf", "otf"])
    font_size = st.slider("Font size", 18, 72, 32, 1)
    
    # Document settings
    st.subheader("📄 Document Settings")
    pages = st.number_input("Number of pages", 1, 15, 2)
    subject = st.selectbox("Subject type", ["general", "law", "science", "mathematics"])
    
    # Style presets
    st.subheader("🎭 Style Presets")
    style_preset = st.selectbox("Choose preset", [
        "Balanced (Recommended)",
        "Neat & Tidy",
        "Natural & Casual",
        "Slightly Messy",
        "Exam Rush Mode"
    ])
    
    # Map presets to config
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
    
    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        config.char_rotation_range = st.slider("Character rotation", 0.0, 6.0, config.char_rotation_range, 0.1)
        config.baseline_wave_amplitude = st.slider("Baseline wobble", 0.0, 3.0, config.baseline_wave_amplitude, 0.1)
        config.char_spacing_variance = st.slider("Spacing variation", 0.0, 4.0, config.char_spacing_variance, 0.1)
        config.ink_bleeding = st.slider("Ink bleeding", 0.0, 1.5, config.ink_bleeding, 0.1)
        config.ink_pressure_layers = st.slider("Pressure layers", 1, 5, config.ink_pressure_layers)
        config.fatigue_enabled = st.checkbox("Enable writing fatigue", config.fatigue_enabled)
        config.ligature_detection = st.checkbox("Letter connection (ligatures)", config.ligature_detection)
    
    # Appearance
    st.subheader("🎨 Appearance")
    ink_color_choice = st.selectbox("Ink color", ["Black", "Dark Blue", "Brown", "Gray", "Green"])
    paper_color_choice = st.selectbox("Paper color", ["Ivory", "White", "Aged (Beige)", "Light Blue"])
    ruled = st.checkbox("Add ruled lines", False)
    
    # Color mappings
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
    
    # Info
    st.info("💡 **Tip**: Upload a cursive or handwriting-style font for best results!")
    
    if not MATPLOTLIB_AVAILABLE:
        st.warning("⚠️ Install matplotlib for proper math rendering: `pip install matplotlib`")

# Load font
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
        st.error(f"Font upload error: {e}")
elif repo_font_path.exists():
    font_path_to_use = repo_font_path

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    question = st.text_area(
        "📝 Enter your assignment question or text",
        height=200,
        placeholder="Example: Explain the doctrine of separation of powers in Indian Constitution.\n\nYou can include math like: The quadratic formula is $x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}$"
    )

with col2:
    st.markdown("### 🚀 Features")
    st.markdown("""
    <div class="feature-box">
    ✅ <b>AI-powered answer generation</b><br>
    ✅ <b>Ultra-realistic handwriting</b><br>
    ✅ <b>Math formula support (LaTeX)</b><br>
    ✅ <b>Multiple style presets</b><br>
    ✅ <b>Custom font support</b><br>
    ✅ <b>Writing fatigue simulation</b><br>
    ✅ <b>Natural letter connections</b><br>
    ✅ <b>PDF & ZIP export</b>
    </div>
    """, unsafe_allow_html=True)

# Generate button
if st.button("✨ Generate Handwritten Assignment", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("⚠️ Please enter a question or text!")
    else:
        # Generate AI answer
        with st.spinner("🤖 Generating AI answer..."):
            answer_text = generate_assignment_answer(question, int(pages), subject)
        
        if not answer_text:
            st.error("❌ Failed to generate answer. Check API key.")
        else:
            # Show stats
            word_count = len(answer_text.split())
            st.success(f"✅ Generated {word_count} words across ~{pages} page(s)")
            
            # Load font
            font_obj = load_font_from_path(font_path_to_use, int(font_size))
            
            # Split into pages
            chunks = split_text_into_pages(answer_text, int(pages))
            
            # Render pages
            images = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, chunk in enumerate(chunks, start=1):
                status_text.text(f"✍️ Rendering page {idx}/{len(chunks)}...")
                
                # Vary ink slightly per page
                ink_base = ink_colors[ink_color_choice]
                ink_varied = tuple(max(0, min(255, c + random.randint(-12, 12))) for c in ink_base)
                
                header = f"Page {idx} of {len(chunks)}" if len(chunks) > 1 else None
                
                img = render_handwritten_page(
                    chunk,
                    font_obj,
                    config,
                    img_width=1240,
                    img_height=1754,
                    margin_left=100,
                    margin_top=120,
                    line_spacing=int(font_obj.size * 0.45),
                    ink_color=ink_varied,
                    paper_color=paper_colors[paper_color_choice],
                    ruled=ruled,
                    page_number=idx,
                    total_pages=len(chunks),
                    header_text=header
                )
                
                images.append(img)
                progress_bar.progress(idx / len(chunks))
            
            status_text.text("✅ Rendering complete!")
            
            # Display preview
            st.subheader("📄 Preview")
            cols = st.columns(min(3, len(images)))
            for i, img in enumerate(images):
                with cols[i % len(cols)]:
                    st.image(
                        img.resize((300, int(300 * img.height / img.width))),
                        caption=f"Page {i+1}",
                        use_container_width=True
                    )
            
            # Download options
            st.subheader("⬇️ Download")
            
            col_dl1, col_dl2 = st.columns(2)
            
            # ZIP download
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
                    label="📦 Download ZIP (All Images)",
                    data=zip_buffer,
                    file_name="handwritten_assignment.zip",
                    mime="application/zip",
                    use_container_width=True
                )
            
            # PDF download
            with col_dl2:
                if images:
                    pdf_bytes = io.BytesIO()
                    rgb_images = [im.convert("RGB") for im in images]
                    rgb_images[0].save(
                        pdf_bytes,
                        format="PDF",
                        save_all=True,
                        append_images=rgb_images[1:],
                        quality=95
                    )
                    pdf_bytes.seek(0)
                    
                    st.download_button(
                        label="📄 Download PDF (Combined)",
                        data=pdf_bytes,
                        file_name="handwritten_assignment.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            
            st.balloons()
            st.success("🎉 Your handwritten assignment is ready!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><b>✍️ AI Handwriting Generator Pro</b> | Built with Streamlit + OpenAI + PIL</p>
    <p>💡 For best results: Use cursive fonts, adjust style presets, and enable advanced features</p>
</div>
""", unsafe_allow_html=True)

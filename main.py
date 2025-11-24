import os
import textwrap
from dotenv import load_dotenv
import openai
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from openai import OpenAI

# Access key from secrets
api_key = st.secrets["OPENAI_API_KEY"]

# Initialize client
client = OpenAI(api_key=api_key)

# Generate text with OpenAI GPT-4
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

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": prompt}
    ],
    max_tokens=target_words * 4,
    temperature=0.7
)

    return response.choices[0].message.content.strip()

# Render handwritten text as image
def render_handwritten_image(
    text: str,
    font_path: str = "fonts/handwriting.ttf",
    font_size: int = 32,
    img_width: int = 1240,
    img_height: int = 1754,
    margin_left: int = 120,
    margin_top: int = 120,
    line_spacing: int = 10,
):
    image = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, font_size)

    max_text_width = img_width - 2 * margin_left
    lines = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        wrapped = textwrap.wrap(paragraph, width=60)
        lines.extend(wrapped)

    x, y = margin_left, margin_top
    for line in lines:
        if y > img_height - margin_top:
            break
        draw.text((x, y), line, font=font, fill="black")
        _, line_height = draw.textsize(line, font=font)
        y += line_height + line_spacing

    return image

# Streamlit UI
st.title("AI Handwritten Assignment Generator ✍️")

question = st.text_area("Enter your assignment question", height=150)
pages = st.number_input("Approximate pages (handwritten)", min_value=1, max_value=10, value=2)

if st.button("Generate Handwritten Assignment"):
    if not question.strip():
        st.warning("Please enter an assignment question.")
    else:
        with st.spinner("Generating answer with OpenAI GPT-4..."):
            answer_text = generate_assignment_answer(question, pages)

        with st.spinner("Rendering handwritten image..."):
            img = render_handwritten_image(answer_text)

        st.image(img, caption="Generated handwritten assignment page", use_column_width=True)

        img.save("assignment_page.png")
        st.success("Rendered page saved as assignment_page.png")
        st.download_button(
            label="Download Handwritten Assignment Image",
            data=open("assignment_page.png", "rb").read(),
            file_name="assignment_page.png",
            mime="image/png",
        )

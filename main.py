import os
import textwrap
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
import openai

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Generate assignment answer with OpenAI GPT-4
def generate_assignment_answer(question: str, pages: int = 2) -> str:
    approx_words_per_page = 180  # adjust as needed
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

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=target_words * 4,  # approx tokens = words * 4/3, + buffer
        temperature=0.7,
    )

    answer = response.choices[0].message.content.strip()
    return answer

# Render handwritten style image using Pillow
def render_handwritten_page(
    text: str,
    output_path: str = "assignment_page.png",
    font_path: str = "fonts/handwriting.ttf",
    font_size: int = 32,
):
    img_width, img_height = 1240, 1754  # roughly half A4 at 300 dpi
    margin_left, margin_top = 120, 120
    line_spacing = 10

    # Create white canvas
    image = Image.new("RGB", (img_width, img_height), color="white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, font_size)

    max_text_width = img_width - 2 * margin_left
    lines = []
    # Wrap text for each paragraph
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        wrapped = textwrap.wrap(paragraph, width=60)  # tweak this for your font/size
        lines.extend(wrapped)

    x, y = margin_left, margin_top
    for line in lines:
        if y > img_height - margin_top:
            break  # stop if page full
        draw.text((x, y), line, font=font, fill="black")
        _, line_height = draw.textsize(line, font=font)
        y += line_height + line_spacing

    image.save(output_path)
    print(f"Saved handwritten page to {output_path}")

def main():
    question = input("Enter your assignment question: ")
    pages = int(input("Approx pages (handwritten): ") or "2")

    print("\nGenerating answer with OpenAI GPT-4...")
    answer_text = generate_assignment_answer(question, pages)
    print("Done. Rendering handwriting...")

    render_handwritten_page(answer_text, output_path="assignment_page.png")
    print("Complete! See assignment_page.png")

if __name__ == "__main__":
    main()

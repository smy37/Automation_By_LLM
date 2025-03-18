import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import openai
from PIL import Image
import variable
import io
import base64
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL_NAME_DEFAULT = "text-embedding-3-large"
def ask_gpt(content: str, prompt: str, output_format = None, max_tokens = 10000):
    if not output_format:
        completion = openai.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "developer",
                 "content": prompt},
                {"role": "user", "content": content}
            ],
            max_tokens=max_tokens
        )
        gpt_result = completion.choices[0].message.content
    else:
        completion = openai.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "developer",
                 "content": prompt},
                {"role": "user", "content": content}
            ],
            response_format=output_format,
            max_tokens=max_tokens
        )
        gpt_result = completion.choices[0].message.parsed
    return gpt_result

def base64_encode(image_path):
    image = Image.open(image_path)
    num_pixel = image.width*image.height
    if num_pixel > variable.MAX_PIXELS:
        scale_f = (variable.MAX_PIXELS / num_pixel) ** 0.5
        new_width = int(image.width*scale_f)
        new_height = int(image.height*scale_f)
        image = image.resize((new_width, new_height))
    image_format = image.format if image.format else 'PNG'
    buffered = io.BytesIO()
    image.save(buffered, format=image_format)
    print(image.width, image.height, image_format)
    image_bytes = buffered.getvalue()
    return base64.b64encode(image_bytes).decode('utf-8')

def ask_gpt_vision(img_path_l, prompt, model_n = variable.DEFAULT_LLM_MODEL):
    base64_l = []
    for img_p in img_path_l:
        print(img_p)
        base64_img = base64_encode(img_p)
        base64_l.append(base64_img)

    content = [{"type": "text", "text": prompt}]
    for b_img in base64_l:
        content.append({"type":"image_url", "image_url": {"url": f"data:image/png;base64,{b_img}"}})

    response = openai.chat.completions.create(
        model=model_n,
        messages=[
            {
                "role": "user",
                "content": content
            }
        ],
    )
    res_data = response.choices[0].message.content
    return res_data

def embedding_openAI_batch(q_texts:list, model_name: str = MODEL_NAME_DEFAULT, dimension: int = 1536) -> list:
    openai_api_key = os.getenv("OPEN_AI_API_KEY")
    client = OpenAI(api_key=openai_api_key)
    return client.embeddings.create(input = q_texts, model = model_name, dimensions=dimension).data
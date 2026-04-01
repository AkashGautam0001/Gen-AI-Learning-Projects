from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()
client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))

import base64


prompt = """
A futuristic city with flying cars, neon lights, cyberpunk style, ultra realistic, 4k
"""

result = client.images.generate(
    model="gpt-image-1",
    prompt=prompt,
    size="1024x1024"
)

image_base64 = result.data[0].b64_json

with open("generated_image.png", "wb") as f:
    f.write(base64.b64decode(image_base64))

print("Image saved as generated_image.png")
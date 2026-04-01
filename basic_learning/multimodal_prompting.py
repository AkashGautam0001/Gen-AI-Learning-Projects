from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()
client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is happening in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://media.licdn.com/dms/image/v2/D5622AQGy6DffzGcbMQ/feedshare-shrink_800/B56Z04Z3MoH0Ac-/0/1774767797319?e=1776297600&v=beta&t=iUfHWHlg6ImvaQEzlVEXvE_OfjiLH4J1VxK6ipumfcU"
                    }
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)
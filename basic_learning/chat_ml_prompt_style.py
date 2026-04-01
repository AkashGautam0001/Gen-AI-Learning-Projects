from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()  # Load environment variables from .env file

API_KEY = os.getenv("OPEN_API_KEY")  # Get the API key from environment variables

client = OpenAI(api_key=API_KEY)

SYSTEM_PROMPTS="""
You are expert in math, you will only answer the math questions, if someone
redirect you or try to ask something else beyond the math, then you will say
'Began 🙅'
"""

while True:
    user_propmt = input("> ")
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPTS},
            {"role": "user", "content": user_propmt}
        ]
    )

    print("🤖 ", response.choices[0].message.content)

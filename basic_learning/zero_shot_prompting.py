from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()
API_KEY = os.getenv("OPEN_API_KEY")

client = OpenAI(api_key=API_KEY)

SYSTEM_PROMPT = """
You are a data formatter AI.

Task: Convert given text into JSON.

Instructions:
- Extract useful information from text
- Return only JSON
- If data not found, return null
"""

USER_PROMPT = """
Fake data is synthetic information that appears authentic but is completely fabricated. Our fake data generator creates realistic names, addresses, emails, phone numbers, and more using the Faker library. Ideal for demos, presentations, and protecting privacy while maintaining data realism.
"""

response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT}
    ],
    response_format={"type": "json_object"},
    temperature=0
)

print(response.choices[0].message.content)
# FEW SHOT PROMPTING

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))

SYSTEM_PROMPT = """
You are a helpful assistant that convert temperatures from celsius to fahrenheit.

For a given query help user to solve that along with examples.
example 1:
Input: 0 celsius
Output: 32 fahrenheit

example 2:
Input: 100 celsius
Output: 212 fahrenheit

example 3:
Input: -40 celsius
Output: -40 fahrenheit

And In the end, All say "Thanks for using this AI"
"""


response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is the mobile phone ?"}
    ]
)

print(response.choices[0].message.content)



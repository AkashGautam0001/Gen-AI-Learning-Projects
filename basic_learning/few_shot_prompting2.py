# FEW SHOT PROMPTING

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))

few_shot_examples = [
    {"role": "user", "content": "What is the weather like in New York?"},
    {"role": "assistant", "content": "It is 32 degrees in New York."},
    {"role": "user", "content": "What is the weather like in Paris?"},
    {"role": "assistant", "content": "It is 68 degrees in Paris."}
]

main_query = {"role": "user", "content": "What is the weather like in Indai?"}

messages = [
    {"role": "system", "content": "You are a helpful assistant that convert temperatures from celsius to fahrenheit."}
]

response = client.chat.completions.create(
    model="gpt-4.1",
    messages=messages + few_shot_examples + [main_query]
)

print(response.choices[0].message.content)
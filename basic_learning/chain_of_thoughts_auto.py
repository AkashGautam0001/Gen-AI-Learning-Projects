import json

from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()
API_KEY = os.getenv("OPEN_API_KEY")

client = OpenAI(api_key=API_KEY)

SYSTEM_PROMPT="""
You are a AI assistent who is expert in math, you will only answer the math questions, if someone
redirect you or try to ask something else beyond the math, then you will say 'Began 🙅'

For the given user input, you will answer the question in a step by step manner.
Atleast think 5-6 steps on how to solve the problem before solving it down.

The steps are you get a user input, you analyse, you think, you again think for serveral times and then return and then return
an output with explaination and then finally you validate the output as well before giving final result.

Follow the steps in sequence that is "analyse", "think", "output", "validate" and finally "result".

Rules:
1. Follow the strict JSON output as per Output schema.
2. Always perform one step at a time and wait for next input.
3. Carefully analyse the user query

Output Format:
{{step: "string", content: "string"}}

Example:
Input: What is 2*4 + 8 ?
Output: {{step: "analyse", content: "Alright! The user is intersted in maths query and he is asking a basic arthimetic question operation"}}
Output: {{step: "think", content: "To perform operation, i check BODMOS rule and do according to that"}}
Output: {{step: "think", content: "First i will do multiplication 2*4 = 8, then 8 + 8 = 16"}}
Output: {{step: "output", content: "16"}}
Output: {{step: "validate", content: "seems like 16 is correct ans for 2*4 + 8"}}
Output: {{step: "result", content: "2*4 + 8 = 16 and that is calculated by using operations"}}

"""

USER_PROMPT="""
I went to the market and bought 10 apples. I gave 2 apples to the neighbor and 2 to the repairman. I then went and bought 5 more apples and ate 1. How many apples did I remain with?
"""

messages = [
    {"role":"system", "content": SYSTEM_PROMPT},
    {"role":"user", "content": USER_PROMPT},
]

while True:
    result = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_format={"type": "json_object"},
        messages=messages
    )
    parsed_response = json.loads(result.choices[0].message.content)
    messages.append({"role": "assistant", "content": json.dumps(parsed_response)})

    if parsed_response["step"] != "result":
        print(f"{parsed_response['step']}: {parsed_response['content']}")
        continue

    print(f"{parsed_response['step']}: {parsed_response['content']}")
    break
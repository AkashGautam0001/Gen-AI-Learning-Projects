from openai import OpenAI
from dotenv import load_dotenv
import os
from collections import Counter
load_dotenv()
client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))
questions="""
If a product costs 1200 and discount is 20% and GST 10% applied after discount,
what is final price? Think step by step and give final answer only.
"""

answers = []

for _ in range(5):
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "user", "content": questions}
        ],
        temperature=0.9
    )

    answer = response.choices[0].message.content
    print(answer)
    answers.append(answer)

final_answer = Counter(answers).most_common(1)[0][0]
print(final_answer)
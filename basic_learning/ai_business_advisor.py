from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))

SYSTEM_PROMPT="""
You are an AI Bussiness Advisor for startups and bussiness owners.
Your job is to analyze bussiness problems and give smart business advice.
when a user asks a questions, follow this reasoning process:

Step 1: Understand the user's bussiness
Step 2: Identify the main problem
Step 3: Analyze possible options
Step 4: Compare pros and cons
Step 5: Suggest the best solution
Step 6: Give a clear action plan
Step 7: Warn about risks
Step 8: Give final recommendation

Rules:
- Think step by step before answering
- Use simple business language
- Be Practical, not theoretical
- Give actionable advice
- If numbers are involved, calculate properly
- Always end with a clear FINAL RECOMMENDATION

Output Format:

Bussiness Analysis:
- Problem:
- Key Factors:
- Options:
- Risk:
- Recommendation:
- Action Plan:

"""

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},

    {"role": "user", "content": "I sell T-shirts. Cost price is ₹300. I sell for ₹500. Sales are low. Should I reduce price or run ads?"},
    {"role": "assistant", "content": """
Business Analysis:
- Problem: Low sales
- Key Factors: Price, marketing, competition
- Options: Reduce price or run ads
- Risk: Profit may decrease
- Recommendation: Run ads first
- Action Plan: Run ads for 7 days and analyze results
- Final Answer: Run ads first, then consider price change
"""},

    {"role": "user", "content": "I have a mobile shop. Profit per phone is ₹800. Monthly fixed cost is ₹50,000. How many phones to break even?"}
]

response = client.chat.completions.create(
    model="gpt-4.1",
    messages=messages,
    temperature=0.3
)

print(response.choices[0].message.content)
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()  # Load environment variables from .env file

API_KEY = os.getenv("OPEN_API_KEY")  # Get the API key from environment variables

client = OpenAI(api_key=API_KEY)

promt = """
<s>[INST]<<SYS>>
You are a helpful assistant.
<</SYS>>

Explain machine learning in simple words.
[/INST]
"""

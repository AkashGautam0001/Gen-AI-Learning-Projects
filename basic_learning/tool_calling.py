import json
import requests
from dotenv import load_dotenv
from openai import OpenAI
import os
load_dotenv()

client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))

def run_command(command):
    result = os.system(command=command)
    return result

def get_weather(city:str):
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}."
    return "Something went wrong"


def add(x, y):
    print("🔨 Tool Called: add", x, y)
    return x + y

avaia

system_prompt = """
You are a helpful AI assistent who is specialized in resolving user query.
you work on start, plan, action, observe mode.
For the given user query and available tools, plan the step by step execution, based on the planning, select the relevant tool from the avaiable tools, and based on the tool selection you perfrom an action to call the tool.
wait for the observation and based on the observation from the tool call resolve the user query.

Rules:
- Follow the output JSON format.
- Always perform one step at a time and wait for next input
- carefully analyse the user query

Output JSON Format:
{{
   "step": "string",
   "content" : "string",
   "function" : "The name of function if the step is action",
   "input" : "The input parameter for the function,
}}

Avaiable Tools:
- get_weather: Takes a city name as an input and returns the current weather for the city
- run_command: Takes a command as input to execute on system and returns output

Example:
    User Query: What is the weather of new york?
    Output: {{ "step": "plan", "content": "The user is interseted in weather data of new york" }}
    Output: {{ "step": "plan", "content": "From the available tools I should call get_weather" }}
    Output: {{ "step": "action", "function": "get_weather", "input": "new york" }}
    Output: {{ "step": "observe", "output": "12 Degree Cel" }}
    Output: {{ "step": "output", "content": "The weather for new york seems to be 12 degrees." }}
"""
import json
import requests
import subprocess
import os
import time
import logging
from typing import Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

# =======================
# Setup
# =======================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

MAX_STEPS = 10
MODEL_NAME = "gpt-4o"
REQUEST_TIMEOUT = 10


# =======================
# Safe Tool Implementations
# =======================

def get_weather(city: str) -> str:
    try:
        url = f"https://wttr.in/{city}?format=%C+%t"
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            return f"The weather in {city} is {response.text}"
        return "Weather API failed"
    except Exception as e:
        return f"Weather error: {str(e)}"


SAFE_COMMANDS = {
    "date": ["date"],
    "uptime": ["uptime"],
    "whoami": ["whoami"]
}

def run_command(command_name: str) -> Dict[str, Any]:
    if command_name not in SAFE_COMMANDS:
        return {"error": "Command not allowed"}

    try:
        result = subprocess.run(
            SAFE_COMMANDS[command_name],
            capture_output=True,
            text=True,
            timeout=5
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except Exception as e:
        return {"error": str(e)}


available_tools = {
    "get_weather": {
        "fn": get_weather,
        "description": "Get weather of a city"
    },
    "run_command": {
        "fn": run_command,
        "description": "Run a safe system command (date, uptime, whoami)"
    }
}


# =======================
# System Prompt
# =======================

system_prompt = """
You are an AI agent that works in steps:

Steps:
- plan
- action
- observe
- output

Rules:
- Output must be JSON
- One step at a time
- Only call available tools
- Stop when final answer is ready

JSON Format:
{
  "step": "plan | action | observe | output",
  "content": "text",
  "function": "tool name",
  "input": "tool input"
}
"""


# =======================
# LLM Call with Retry
# =======================

def call_llm(messages):
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                response_format={"type": "json_object"},
                messages=messages,
                timeout=REQUEST_TIMEOUT
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logging.error(f"LLM Error: {e}")
            time.sleep(2)

    return {"step": "output", "content": "LLM failed after retries"}


# =======================
# Validate Model Output
# =======================

def validate_step(data):
    if "step" not in data:
        return False
    if data["step"] not in ["plan", "action", "observe", "output"]:
        return False
    return True


# =======================
# Agent Loop
# =======================

def run_agent():
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    while True:
        user_query = input("\n> ")
        messages.append({"role": "user", "content": user_query})

        step_count = 0

        while step_count < MAX_STEPS:
            step_count += 1

            parsed_output = call_llm(messages)
            logging.info(f"Agent Step: {parsed_output}")

            if not validate_step(parsed_output):
                messages.append({
                    "role": "assistant",
                    "content": json.dumps({
                        "step": "output",
                        "content": "Invalid response format"
                    })
                })
                break

            messages.append({
                "role": "assistant",
                "content": json.dumps(parsed_output)
            })

            if parsed_output["step"] == "plan":
                continue

            if parsed_output["step"] == "action":
                tool_name = parsed_output.get("function")
                tool_input = parsed_output.get("input")

                if tool_name not in available_tools:
                    messages.append({
                        "role": "assistant",
                        "content": json.dumps({
                            "step": "observe",
                            "output": "Tool not found"
                        })
                    })
                    continue

                try:
                    tool_output = available_tools[tool_name]["fn"](tool_input)
                except Exception as e:
                    tool_output = str(e)

                messages.append({
                    "role": "assistant",
                    "content": json.dumps({
                        "step": "observe",
                        "output": tool_output
                    })
                })
                continue

            if parsed_output["step"] == "output":
                print("\nFinal Answer:", parsed_output["content"])
                break

        if step_count >= MAX_STEPS:
            print("Max steps reached. Stopping agent.")


# =======================
# Start Agent
# =======================

if __name__ == "__main__":
    run_agent()
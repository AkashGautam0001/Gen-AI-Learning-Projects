"""
LLM interaction utilities
"""
import logging
from .config import CHAT_MODEL


def get_ai_response(openai_client, user_question, context):
    """Get response from LLM based on context"""
    system_prompt = f"""
    You are a helpful AI assistant.
    Answer the question ONLY using the provided context.
    If the answer is not in the context, say: "Answer not found in document".

    Context:
    {context}

    Question: {user_question}
    Answer:
    """
    logging.info("Sending query to GPT...")
    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers from provided PDF context only."},
            {"role": "user", "content": system_prompt}
        ]
    )
    return response.choices[0].message.content

"""
Text processing utilities
"""
import logging
import ast


def clean_llm_output(text):
    """Clean markdown code blocks from LLM output"""
    text = text.strip()
    text = text.replace("```python", "").replace("```", "")
    return text.strip()


def generate_queries(openai_client, user_query):
    """Generate multiple search queries for a user question"""
    logging.info(f"Generating multiple queries for: {user_query}")
    prompt = f"""
    Generate 5 different search queries for the question:
    "{user_query}"
    Make them diverse and cover different perspectives.

    Return ONLY a Python list.
    Example: ["query1", "query2", "query3"]
    """
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    try:
        raw_output = response.choices[0].message.content
        clean_output = clean_llm_output(raw_output)
        queries = ast.literal_eval(clean_output)
        logging.info(f"Successfully parsed {len(queries)} queries")
    except Exception as e:
        logging.warning(f"Failed to parse queries, using original query: {e}")
        queries = [user_query]
    return queries


def deduplicate_chunks(chunks):
    """Remove duplicate chunks based on text content"""
    logging.info(f"Deduplicating {len(chunks)} chunks...")
    seen = set()
    unique = []
    for chunk in chunks:
        if chunk["text"] not in seen:
            seen.add(chunk["text"])
            unique.append(chunk)
    
    logging.info(f"Deduplicated to {len(unique)} unique chunks")
    return unique

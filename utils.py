import os
# import time
import requests
import copy
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE")
MAX_TOKENS = int(os.getenv("MAX_TOKENS"))
TEMPERATURE = float(os.getenv("TEMPERATURE"))
DEBUG = int(os.environ.get("DEBUG", "0"))

def generate_together(model, messages, max_tokens=MAX_TOKENS, temperature=TEMPERATURE):
    logger.info(
        f"Input data: model={model}, messages={messages}, max_tokens={max_tokens}, temperature={temperature}"
    )

    output = None
    endpoint = f"{API_BASE}/chat/completions"

    logger.info(f"Sending request to {endpoint}")

    try:
        res = requests.post(
            endpoint,
            json={
                "model": model,
                "max_tokens": max_tokens,
                "temperature": (temperature if temperature > 1e-4 else 0),
                "messages": messages,
            },
            headers={
                "Authorization": f"Bearer {API_KEY}",
            },
        )

        response_json = res.json()
        logger.info(f"Response: {response_json}")

        if "error" in response_json:
            logger.error(response_json)
            return None

        output = response_json["choices"][0]["message"]["content"]

    except Exception as e:
        logger.error(e)
        return None

    if output is None:
        return output

    output = output.strip()
    logger.info(f"Output: `{output[:20]}...`.")
    return output

def inject_references_to_messages(messages, references):
    messages = copy.deepcopy(messages)

    system_message = "You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability. \n\nResponses from models:"

    for i, reference in enumerate(references):
        system_message += f"\n{i+1}. {reference}"

    if messages[0]["role"] == "system":
        messages[0]["content"] += "\n\n" + system_message
    else:
        messages = [{"role": "system", "content": system_message}] + messages

    return messages

def generate_with_references(model, messages, references=[], max_tokens=MAX_TOKENS, temperature=TEMPERATURE, generate_fn=generate_together):
    if len(references) > 0:
        messages = inject_references_to_messages(messages, references)

    return generate_fn(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
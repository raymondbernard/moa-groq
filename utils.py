import os
import requests
from loguru import logger
from dotenv import load_dotenv
from time import sleep
import copy

load_dotenv()

# Load environment variables
API_BASE_1 = os.getenv("API_BASE_1")
API_KEY_1 = os.getenv("API_KEY_1")
API_BASE_2 = os.getenv("API_BASE_2")
API_KEY_2 = os.getenv("API_KEY_2")

MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "4096"))
TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.9"))

# Load model configurations from environment variables
MODEL_AGGREGATE = os.getenv("MODEL_AGGREGATE")
MODEL_AGGREGATE_API_BASE = os.getenv("MODEL_AGGREGATE_API_BASE")
MODEL_AGGREGATE_API_KEY = os.getenv("MODEL_AGGREGATE_API_KEY")

MODEL_REFERENCE_1 = os.getenv("MODEL_REFERENCE_1")
MODEL_REFERENCE_1_API_BASE = os.getenv("MODEL_REFERENCE_1_API_BASE")
MODEL_REFERENCE_1_API_KEY = os.getenv("MODEL_REFERENCE_1_API_KEY")

MODEL_REFERENCE_2 = os.getenv("MODEL_REFERENCE_2")
MODEL_REFERENCE_2_API_BASE = os.getenv("MODEL_REFERENCE_2_API_BASE")
MODEL_REFERENCE_2_API_KEY = os.getenv("MODEL_REFERENCE_2_API_KEY")

MODEL_REFERENCE_3 = os.getenv("MODEL_REFERENCE_3")
MODEL_REFERENCE_3_API_BASE = os.getenv("MODEL_REFERENCE_3_API_BASE")
MODEL_REFERENCE_3_API_KEY = os.getenv("MODEL_REFERENCE_3_API_KEY")

MODELS = {
    MODEL_AGGREGATE: {
        "api_base": MODEL_AGGREGATE_API_BASE,
        "api_key": MODEL_AGGREGATE_API_KEY
    },
    MODEL_REFERENCE_1: {
        "api_base": MODEL_REFERENCE_1_API_BASE,
        "api_key": MODEL_REFERENCE_1_API_KEY
    },
    MODEL_REFERENCE_2: {
        "api_base": MODEL_REFERENCE_2_API_BASE,
        "api_key": MODEL_REFERENCE_2_API_KEY
    },
    MODEL_REFERENCE_3: {
        "api_base": MODEL_REFERENCE_3_API_BASE,
        "api_key": MODEL_REFERENCE_3_API_KEY
    }
}

def make_api_call(url, headers, data):
    try:
        logger.info(f">>>> Making API call to {url} with data: {data}")
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Response received: {result}")
        return result
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return None

def generate_together(model_name, messages, max_tokens=MAX_TOKENS, temperature=TEMPERATURE):
    model_info = MODELS.get(model_name)
    if not model_info:
        raise ValueError(f"Model {model_name} not found in configuration.")

    api_base = model_info["api_base"]
    api_key = model_info["api_key"]
    url = f"{api_base}/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    data = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    while True:
        logger.info(f"Sending request to {url} for model {model_name}")
        response = make_api_call(url, headers, data)
        if response:
            if 'error' in response:
                error_code = response['error'].get('code', '')
                if error_code == 'rate_limit_exceeded':
                    wait_time = int(response['error'].get('retry-after', 30))
                    logger.warning(f"Rate limit exceeded for {model_name}. Retrying in {wait_time} seconds...")
                    sleep(wait_time)
                    continue
            else:
                return response["choices"][0]["message"]["content"]
        break

    return None

def generate_layered_response(model_name, messages, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, layers=4, agents_per_layer=3):
    layer_outputs = messages
    final_output = ""
    for layer in range(1, layers + 1):
        logger.info(f"++++++ Processing Layer {layer} for model {model_name}")
        layer_responses = []
        for agent in range(1, agents_per_layer + 1):
            agent_id = f"A{layer},{agent}"
            logger.info(f"*** Generating response for agent {agent_id}")
            output = generate_together(model_name, layer_outputs, max_tokens, temperature)
            if output:
                layer_responses.append({"role": "assistant", "content": output})
        if layer_responses:
            layer_outputs = layer_responses
        else:
            logger.error(f"No responses for Layer {layer} for model {model_name}")
            break

    if layer_responses:
        final_output = layer_responses[0]["content"]
    else:
        logger.error(f"No final output generated for model {model_name}")
    
    return final_output

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

    logger.info(f"Generating with references for model {model}")
    return generate_fn(model, messages=messages, temperature=temperature, max_tokens=max_tokens)

import os
import copy
import requests
from loguru import logger
from dotenv import load_dotenv
from utils import generate_together, inject_references_to_messages, generate_with_references

load_dotenv()

API_KEY = os.getenv("API_KEY_1")
API_BASE = os.getenv("API_BASE")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 2048))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
DEBUG = int(os.getenv("DEBUG", 0))

def test_generate_together():
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    model = "llama3-8b-8192"
    response = generate_together(model, messages, max_tokens=100, temperature=0.5)
    assert response is not None, "generate_together failed to return a response"
    print("generate_together passed")

def test_inject_references_to_messages():
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    references = ["Paris is the capital of France.", "The capital of France is Paris."]
    updated_messages = inject_references_to_messages(messages, references)
    assert updated_messages[0]["content"].startswith("You have been provided with a set of responses"), "inject_references_to_messages did not add the references correctly"
    print("inject_references_to_messages passed")

def test_generate_with_references():
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    references = ["Paris is the capital of France.", "The capital of France is Paris."]
    model = "llama3-8b-8192"
    response = generate_with_references(model, messages, references, max_tokens=100, temperature=0.5)
    assert response is not None, "generate_with_references failed to return a response"
    print("generate_with_references passed")

if __name__ == "__main__":
    test_generate_together()
    test_inject_references_to_messages()
    test_generate_with_references()
    print("All tests passed.")

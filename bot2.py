import datasets
from functools import partial
from loguru import logger
import os
from time import sleep
from dotenv import load_dotenv
from utils import generate_layered_response, generate_together, generate_with_references
import requests 
from dotenv import load_dotenv

load_dotenv()


# Set default values
DEFAULT_MAX_TOKENS = os.getenv("DEFAULT_MAX_TOKENS", "4096")
DEFAULT_TEMPERATURE = os.getenv("DEFAULT_TEMPERATURE", "0.9")
DEFAULT_ROUNDS = os.getenv("DEFAULT_ROUNDS", "1")

MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", DEFAULT_MAX_TOKENS))
TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", DEFAULT_TEMPERATURE))
ROUNDS = int(os.getenv("DEFAULT_ROUNDS", DEFAULT_ROUNDS))
MULTITURN = os.getenv("MULTITURN") == "True"

MODEL_AGGREGATE = os.getenv("MODEL_AGGREGATE")
MODEL_REFERENCE_1 = os.getenv("MODEL_REFERENCE_1")
MODEL_REFERENCE_2 = os.getenv("MODEL_REFERENCE_2")
MODEL_REFERENCE_3 = os.getenv("MODEL_REFERENCE_3")

LAYERS = int(os.getenv("LAYERS"))
AGENTS_PER_LAYER = int(os.getenv("AGENTS_PER_LAYER"))

default_reference_models = [
    MODEL_REFERENCE_1,
    MODEL_REFERENCE_2,
    MODEL_REFERENCE_3,
]

# logger.info(f"Loaded configuration: MAX_TOKENS={MAX_TOKENS}, TEMPERATURE={TEMPERATURE}, ROUNDS={ROUNDS}, MULTITURN={MULTITURN}")
# logger.info(f"Models: AGGREGATE={MODEL_AGGREGATE}, REFERENCE_1={MODEL_REFERENCE_1}, REFERENCE_2={MODEL_REFERENCE_2}, REFERENCE_3={MODEL_REFERENCE_3}")

def process_fn(item, temperature=TEMPERATURE, max_tokens=MAX_TOKENS):
    # references = item.get("references", [])
    model = item["model"]
    messages = item["instruction"]

    # logger.info(f"Processing model {model} with instruction {messages}")

    while True:
        try:
            output = generate_layered_response(
                model_name=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                layers=LAYERS,  # Adjust based on your architecture
                agents_per_layer=AGENTS_PER_LAYER,  # Adjust based on your architecture
            )
            break  # Exit loop if request is successful
        except requests.exceptions.RequestException as e:
            response = e.response
            if response and response.status_code == 429:  # Rate limit error
                retry_after = int(response.headers.get('retry-after', 30))
                logger.warning(f"Rate limit exceeded for {model}. Retrying in {retry_after} seconds...")
                sleep(retry_after)
            else:
                raise e

    logger.info(f"!! Finished querying !!! == {model}. Output: {output[:20]}")

    return {"output": output}

def main(
    model: str = MODEL_AGGREGATE,
    reference_models: list[str] = default_reference_models,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    rounds: int = ROUNDS,
    multi_turn=MULTITURN,
):
    # sleep(0.75)

    data = {
        "instruction": [[] for _ in range(len(reference_models))],
        "references": [""] * len(reference_models),
        "model": reference_models,
    }

    num_proc = len(reference_models)


    instruction = input("Prompt >>")

    if multi_turn:
        for i in range(len(reference_models)):
            data["instruction"][i].append({"role": "user", "content": instruction})
            data["references"] = [""] * len(reference_models)
    else:
        data = {
            "instruction": [[{"role": "user", "content": instruction}]] * len(reference_models),
            "references": [""] * len(reference_models),
            "model": reference_models,
        }

    eval_set = datasets.Dataset.from_dict(data)

    for i_round in range(rounds):
        logger.info(f"Starting round {i_round + 1} of processing.")
        eval_set = eval_set.map(
            partial(
                process_fn,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            batched=False,
            num_proc=num_proc,  # Ensure no parallel processing
        )
        references = [item["output"] for item in eval_set]
        data["references"] = references
        eval_set = datasets.Dataset.from_dict(data)
        # sleep(0.75)

        # logger.info(f"Data Structure After Round {i_round + 1}:")
        # logger.info(data)

    logger.info("\n Aggregating results & querying the aggregate model...")

    output = generate_with_references(
        model_name=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=data["instruction"][0],
        references=references,
        generate_fn=generate_together,
    )

    # all_output = output
    logger.info(f"\n ## Final answer from {model}")

    logger.info("Output received from generate_with_references:")
    logger.info(output)

    # logger.info(f"\n Debug model: {model}, instruction: {data['instruction'][0]}, output: {all_output[:20]}")

    # if multi_turn:
    #     for i in range(len(reference_models)):
    #         data["instruction"][i].append({"role": "assistant", "content": all_output})

if __name__ == "__main__":
    main()

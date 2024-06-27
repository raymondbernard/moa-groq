import datasets
from functools import partial
from loguru import logger
from utils import (
    generate_together,
    generate_with_references
)
import typer
import os
from rich import print
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from datasets.utils.logging import disable_progress_bar
from time import sleep
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

disable_progress_bar()

console = Console()

welcome_message = (
    """
# MoA (Mixture-of-Agents)

Mixture of Agents (MoA) is a novel approach that leverages the collective strengths of multiple LLMs to enhance performance, achieving state-of-the-art results. By employing a layered architecture where each layer comprises several LLM agents, MoA can significantly outperform GPT-4 Omni's 57.5% on AlpacaEval 2.0 with a score of 65.1%, using open-source models!

The following LLMs as reference models, then passes the results to the aggregate model for the final response:
- """
    + MODEL_AGGREGATE
    + """   <--- Aggregate model
- """
    + MODEL_REFERENCE_1
    + """   <--- Reference model 1
- """
    + MODEL_REFERENCE_2
    + """   <--- Reference model 2
- """
    + MODEL_REFERENCE_3
    + """   <--- Reference model 3

"""
)

def process_fn(
    item,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
):
    references = item.get("references", [])
    model_name = item["model"]
    messages = item["instruction"]

    while True:
        try:
            output = generate_with_references(
                model_name=model_name,
                messages=messages,
                references=references,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            break  # Exit loop if request is successful
        except requests.exceptions.RequestException as e:
            response = e.response
            if response and response.status_code == 429:  # Rate limit error
                retry_after = int(response.headers.get('retry-after', 30))
                print(f"Rate limit exceeded. Retrying in {retry_after} seconds...")
                sleep(retry_after)
            else:
                raise e

    print(f"\nFinished querying [bold]{model_name}.[/bold]")

    return {"output": output}

def main(
    model: str = MODEL_AGGREGATE,
    reference_models: list[str] = default_reference_models,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    rounds: int = ROUNDS,
    multi_turn=MULTITURN,
):
    md = Markdown(welcome_message)
    console.print(md)
    sleep(0.75)
    console.print(
        "\n[bold]To use this demo, answer the questions below to get started [cyan](press enter to use the defaults)[/cyan][/bold]:"
    )

    data = {
        "instruction": [[] for _ in range(len(reference_models))],
        "references": [""] * len(reference_models),
        "model": reference_models,
    }

    num_proc = len(reference_models)

    model = Prompt.ask(
        "\n1. What main model do you want to use?",
        default=MODEL_AGGREGATE,
    )
    console.print(f"Selected {model}.", style="yellow italic")
    temperature = float(
        Prompt.ask(
            "2. What temperature do you want to use?",
            default=str(TEMPERATURE),
            show_default=True,
        )
    )
    console.print(f"Selected {temperature}.", style="yellow italic")
    max_tokens = int(
        Prompt.ask(
            "3. What max tokens do you want to use?",
            default=str(MAX_TOKENS),
            show_default=True,
        )
    )
    console.print(f"Selected {max_tokens}.", style="yellow italic")

    while True:

        try:
            instruction = Prompt.ask(
                "\n[cyan bold]Prompt >>[/cyan bold] ",
                default="Top things to do in NYC",
                show_default=True,
            )
        except EOFError:
            break

        if instruction == "exit" or instruction == "quit":
            print("Goodbye!")
            break
        if multi_turn:
            for i in range(len(reference_models)):
                data["instruction"][i].append({"role": "user", "content": instruction})
                data["references"] = [""] * len(reference_models)
        else:
            data = {
                "instruction": [[{"role": "user", "content": instruction}]]
                * len(reference_models),
                "references": [""] * len(reference_models),
                "model": reference_models,
            }

        eval_set = datasets.Dataset.from_dict(data)

        with console.status("[bold green]Querying all the models...") as status:
            for _ in range(rounds):
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

        console.print(
            "[cyan bold]Aggregating results & querying the aggregate model...[/cyan bold]"
        )

        output = generate_with_references(
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=data["instruction"][0],
            references=references,
            generate_fn=generate_together,
        )

        all_output = ""
        print("\n")
        console.log(Markdown(f"## Final answer from {model}"))

        print("Output received from generate_with_references:")
        print(output)

        if multi_turn:
            for i in range(len(reference_models)):
                data["instruction"][i].append(
                    {"role": "assistant", "content": all_output}
                )

if __name__ == "__main__":
    typer.run(main)

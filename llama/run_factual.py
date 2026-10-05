import torch
import os
import random
import re
import json
import gc
from datetime import datetime

# === CONFIGURATION ===
positional_path = "/zfsstore/user/s4251938/llama_checkpoints/positional/"
centrality_path = "/zfsstore/user/s4251938/llama_checkpoints/centrality/"
cache_path = "/home/s4251938/landformer/hf-cache/"
results_dir = "/home/s4251938/landformer/passkey_results/"
use_flash = False
top_k = 5
dtype = torch.bfloat16
num_tests = 50
n_values = [0, 100, 500, 1000, 5000, 8000, 10000, 12000, 14000, 18000, 20000, 25000, 38000]

os.makedirs(results_dir, exist_ok=True)

GARBAGE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
GARBAGE_INF = " ".join([GARBAGE] * 2000)
TASK_DESC = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."

# Made-up facts: (entity, attribute, value, query_template, info_template)
FACTS = [
    ("Zarathia", "capital", "Mondeville", "What is the capital of {entity}? The capital of {entity} is", "The capital of {entity} is {value}. Remember it. {value} is the capital of {entity}."),
    ("Brentoria", "capital", "Halcyondale", "What is the capital of {entity}? The capital of {entity} is", "The capital of {entity} is {value}. Remember it. {value} is the capital of {entity}."),
    ("Velmorath", "capital", "Crystalline", "What is the capital of {entity}? The capital of {entity} is", "The capital of {entity} is {value}. Remember it. {value} is the capital of {entity}."),
    ("Dravinia", "king", "Aldenmere", "Who is the king of {entity}? The king of {entity} is", "The king of {entity} is {value}. Remember it. {value} is the king of {entity}."),
    ("Pentholus", "king", "Wrothstone", "Who is the king of {entity}? The king of {entity} is", "The king of {entity} is {value}. Remember it. {value} is the king of {entity}."),
    ("Mythara", "king", "Thornbrook", "Who is the king of {entity}? The king of {entity} is", "The king of {entity} is {value}. Remember it. {value} is the king of {entity}."),
    ("Corvenhal", "population", "Seventy thousand", "What is the population of {entity}? The population of {entity} is", "The population of {entity} is {value}. Remember it. {value} is the population of {entity}."),
    ("Ithranox", "population", "Forty three thousand", "What is the population of {entity}? The population of {entity} is", "The population of {entity} is {value}. Remember it. {value} is the population of {entity}."),
    ("Elyndra", "color", "crimson and gold", "What is the color of {entity}? The color of {entity} is", "The color of {entity} is {value}. Remember it. {value} is the color of {entity}."),
    ("Vandross", "color", "silver and violet", "What is the color of {entity}? The color of {entity} is", "The color of {entity} is {value}. Remember it. {value} is the color of {entity}."),
]

def load_model(checkpoint_path):
    from llama_mem import LlamaForCausalLM
    import transformers
    from transformers import pipeline

    print(f"Loading model from {checkpoint_path}...")
    model = LlamaForCausalLM.from_pretrained(
        checkpoint_path, cache_dir=cache_path, torch_dtype=dtype
    )
    model.to("cuda:0")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        checkpoint_path, cache_dir=cache_path,
        model_max_length=model.config.train_context_length,
        padding_side="right", use_fast=False,
    )
    mem_id = tokenizer.convert_tokens_to_ids("<landmark>")
    model.set_mem_id(mem_id)

    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        device=model.device, offload_cache_to_cpu=use_flash,
        use_flash=use_flash, cache_top_k=top_k,
    )
    print(f"Model loaded. mem_id={mem_id}")
    return pipe

def unload_model(pipe):
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    print("Model unloaded.")

def generate_fact_prompt(n_garbage):
    fact = random.choice(FACTS)
    entity, attr, value, query_tmpl, info_tmpl = fact

    info_line = info_tmpl.format(entity=entity, value=value)
    query_line = query_tmpl.format(entity=entity)

    n_prefix = random.randint(0, n_garbage)
    n_suffix = n_garbage - n_prefix

    prompt = "\n".join([
        TASK_DESC,
        GARBAGE_INF[:n_prefix],
        info_line,
        GARBAGE_INF[:n_suffix],
        query_line
    ])
    return prompt, value

def check_answer(response, expected_value):
    response_clean = response.strip().lower()
    expected_clean = expected_value.strip().lower()
    # Check if the first word(s) of response match the expected value
    first_words = expected_clean.split()
    response_words = response_clean.split()
    if len(response_words) == 0:
        return False
    # Check if first word matches
    if response_words[0] == first_words[0]:
        return True
    # Check if full value appears
    if expected_clean in response_clean:
        return True
    return False

def run_factual_test(pipe, n_values, num_tests):
    print("\n=== FACTUAL RETRIEVAL TEST ===")
    results = {}
    for n in n_values:
        correct = 0
        details = []
        for i in range(num_tests):
            prompt, expected = generate_fact_prompt(n)
            response = pipe(prompt, num_return_sequences=1, max_new_tokens=10)[0]["generated_text"][len(prompt):]
            is_correct = check_answer(response, expected)
            if is_correct:
                correct += 1
            details.append({
                "expected": expected,
                "response": response.strip()[:50],
                "correct": is_correct
            })
            if (i + 1) % 10 == 0:
                print(f"  n={n}, test {i+1}/{num_tests}, accuracy: {correct/(i+1)*100:.1f}%")
        accuracy = correct / num_tests * 100
        results[n] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": num_tests,
            "details": details
        }
        print(f"  n={n}: {accuracy:.1f}%")
    return results

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    models = {
        "positional": positional_path,
        "centrality": centrality_path,
    }

    all_results = {}

    for model_name, model_path in models.items():
        if not os.path.exists(model_path):
            print(f"Skipping {model_name}")
            continue

        print(f"\n{'='*60}")
        print(f"TESTING: {model_name}")
        print(f"{'='*60}")

        pipe = load_model(model_path)
        all_results[model_name] = run_factual_test(pipe, n_values, num_tests)
        unload_model(pipe)

        save_path = os.path.join(results_dir, f"factual_results_{timestamp}.json")
        with open(save_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved to {save_path}")

    # Print comparison
    print(f"\n{'='*60}")
    print("FACTUAL RETRIEVAL COMPARISON")
    print(f"{'='*60}")
    print(f"{'n':>8}  {'Positional':>12}  {'Centrality':>12}  {'Diff':>8}")
    print("-" * 50)
    for n in n_values:
        p = all_results.get("positional", {}).get(str(n), all_results.get("positional", {}).get(n, {})).get("accuracy", 0)
        c = all_results.get("centrality", {}).get(str(n), all_results.get("centrality", {}).get(n, {})).get("accuracy", 0)
        d = c - p
        marker = " <<<" if d > 0 else ""
        print(f"{n:>8}  {p:>11.1f}%  {c:>11.1f}%  {d:>+7.1f}%{marker}")

    print(f"\nResults saved to {save_path}")

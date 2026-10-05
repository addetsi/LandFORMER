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
SEED = 42

os.makedirs(results_dir, exist_ok=True)

GARBAGE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
GARBAGE_INF = " ".join([GARBAGE] * 2000)
TASK_DESC = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."

# === MODEL LOADING ===
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

# === PRE-GENERATE ALL PROMPTS WITH FIXED SEED ===
def generate_all_standard_prompts(n_values, num_tests, seed):
    random.seed(seed)
    prompts = {}
    for n in n_values:
        prompts[n] = []
        for _ in range(num_tests):
            n_prefix = random.randint(0, n)
            n_suffix = n - n_prefix
            pass_key = random.randint(1, 50000)
            prompt = "\n".join([
                TASK_DESC,
                GARBAGE_INF[:n_prefix],
                f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key.",
                GARBAGE_INF[:n_suffix],
                "What is the pass key? The pass key is"
            ])
            prompts[n].append((prompt, pass_key))
    return prompts

def generate_all_split_digit_prompts(n_values, num_tests, seed):
    random.seed(seed)
    ordinals = ["first", "second", "third", "fourth"]
    prompts = {}
    for n in n_values:
        prompts[n] = []
        for _ in range(num_tests):
            pass_key = random.randint(1000, 9999)
            digits = list(str(pass_key))
            section_len = n // 5
            sections = [GARBAGE_INF[:section_len] for _ in range(5)]
            parts = [TASK_DESC]
            for i in range(4):
                parts.append(sections[i])
                parts.append(f"The {ordinals[i]} digit of the pass key is {digits[i]}. Remember it. {digits[i]} is the {ordinals[i]} digit.")
            parts.append(sections[4])
            body = "\n".join(parts)
            queries = []
            for i in range(4):
                query = f"What is the {ordinals[i]} digit of the pass key? The {ordinals[i]} digit of the pass key is"
                queries.append((body + "\n" + query, int(digits[i])))
            prompts[n].append((queries, pass_key))
    return prompts

def generate_all_split_integer_prompts(n_values, num_tests, seed):
    random.seed(seed)
    ordinals = ["first", "second", "third", "fourth"]
    prompts = {}
    for n in n_values:
        prompts[n] = []
        for _ in range(num_tests):
            keys = [random.randint(1, 50000) for _ in range(4)]
            section_len = n // 5
            sections = [GARBAGE_INF[:section_len] for _ in range(5)]
            parts = [TASK_DESC]
            for i in range(4):
                parts.append(sections[i])
                parts.append(f"The {ordinals[i]} pass key is {keys[i]}. Remember it. {keys[i]} is the {ordinals[i]} pass key.")
            parts.append(sections[4])
            body = "\n".join(parts)
            queries = []
            for i in range(4):
                query = f"What is the {ordinals[i]} pass key? The {ordinals[i]} pass key is"
                queries.append((body + "\n" + query, keys[i]))
            prompts[n].append((queries, keys))
    return prompts

# === TEST RUNNERS (use pre-generated prompts) ===
def extract_number(response):
    try:
        return int(re.search(r'\d+', response).group())
    except:
        return None

def run_standard_test(pipe, prompts):
    print("\n=== STANDARD PASSKEY TEST ===")
    results = {}
    for n in sorted(prompts.keys()):
        correct = 0
        for i, (prompt, pass_key) in enumerate(prompts[n]):
            response = pipe(prompt, num_return_sequences=1, max_new_tokens=10)[0]["generated_text"][len(prompt):]
            output = extract_number(response)
            if output == pass_key:
                correct += 1
            if (i + 1) % 10 == 0:
                print(f"  n={n}, test {i+1}/{len(prompts[n])}, running accuracy: {correct/(i+1)*100:.1f}%")
        accuracy = correct / len(prompts[n]) * 100
        results[n] = {"accuracy": accuracy, "correct": correct, "total": len(prompts[n])}
        print(f"  n={n}: {accuracy:.1f}%")
    return results

def run_split_digit_test(pipe, prompts):
    print("\n=== SPLIT-DIGIT PASSKEY TEST ===")
    results = {}
    for n in sorted(prompts.keys()):
        per_digit_correct = [0, 0, 0, 0]
        all_correct = 0
        for i, (queries, full_key) in enumerate(prompts[n]):
            trial_correct = 0
            for d, (prompt, expected) in enumerate(queries):
                response = pipe(prompt, num_return_sequences=1, max_new_tokens=5)[0]["generated_text"][len(prompt):]
                output = extract_number(response)
                if output == expected:
                    per_digit_correct[d] += 1
                    trial_correct += 1
            if trial_correct == 4:
                all_correct += 1
            if (i + 1) % 10 == 0:
                print(f"  n={n}, test {i+1}/{len(prompts[n])}, all-correct: {all_correct/(i+1)*100:.1f}%")
        total = len(prompts[n])
        results[n] = {
            "all_correct_accuracy": all_correct / total * 100,
            "per_digit_accuracy": [c / total * 100 for c in per_digit_correct],
            "all_correct": all_correct, "total": total
        }
        print(f"  n={n}: all_correct={results[n]['all_correct_accuracy']:.1f}%")
    return results

def run_split_integer_test(pipe, prompts):
    print("\n=== SPLIT-INTEGER PASSKEY TEST ===")
    results = {}
    for n in sorted(prompts.keys()):
        per_key_correct = [0, 0, 0, 0]
        all_correct = 0
        for i, (queries, full_keys) in enumerate(prompts[n]):
            trial_correct = 0
            for k, (prompt, expected) in enumerate(queries):
                response = pipe(prompt, num_return_sequences=1, max_new_tokens=10)[0]["generated_text"][len(prompt):]
                output = extract_number(response)
                if output == expected:
                    per_key_correct[k] += 1
                    trial_correct += 1
            if trial_correct == 4:
                all_correct += 1
            if (i + 1) % 10 == 0:
                print(f"  n={n}, test {i+1}/{len(prompts[n])}, all-correct: {all_correct/(i+1)*100:.1f}%")
        total = len(prompts[n])
        results[n] = {
            "all_correct_accuracy": all_correct / total * 100,
            "per_key_accuracy": [c / total * 100 for c in per_key_correct],
            "all_correct": all_correct, "total": total
        }
        print(f"  n={n}: all_correct={results[n]['all_correct_accuracy']:.1f}%")
    return results

# === MAIN ===
if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Pre-generate ALL prompts with fixed seed
    print("Pre-generating prompts with seed=42...")
    standard_prompts = generate_all_standard_prompts(n_values, num_tests, SEED)
    split_digit_prompts = generate_all_split_digit_prompts(n_values, num_tests, SEED + 1)
    split_integer_prompts = generate_all_split_integer_prompts(n_values, num_tests, SEED + 2)
    print(f"Generated {sum(len(v) for v in standard_prompts.values())} standard prompts")
    print(f"Generated {sum(len(v) for v in split_digit_prompts.values())} split-digit prompts")
    print(f"Generated {sum(len(v) for v in split_integer_prompts.values())} split-integer prompts")

    models = {
        "positional": positional_path,
        "centrality": centrality_path,
    }

    all_results = {}

    for model_name, model_path in models.items():
        if not os.path.exists(model_path):
            print(f"Skipping {model_name} - path not found: {model_path}")
            continue

        print(f"\n{'='*60}")
        print(f"TESTING: {model_name}")
        print(f"{'='*60}")

        pipe = load_model(model_path)

        all_results[model_name] = {
            "standard": run_standard_test(pipe, standard_prompts),
            "split_digit": run_split_digit_test(pipe, split_digit_prompts),
            "split_integer": run_split_integer_test(pipe, split_integer_prompts),
        }

        unload_model(pipe)

        # Save after each model
        save_path = os.path.join(results_dir, f"passkey_matched_{timestamp}.json")
        with open(save_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {save_path}")

    # Print comparison
    print(f"\n{'='*60}")
    print("MATCHED-PROMPT COMPARISON")
    print(f"Seed: {SEED} | Tests per n: {num_tests}")
    print(f"{'='*60}")

    for test_type in ["standard", "split_digit", "split_integer"]:
        metric = "accuracy" if test_type == "standard" else "all_correct_accuracy"
        print(f"\n  {test_type.upper()}")
        print(f"{'n':>8}  {'Positional':>12}  {'Centrality':>12}  {'Diff':>8}")
        print("-" * 50)
        for n in n_values:
            p = all_results.get("positional", {}).get(test_type, {}).get(n, {}).get(metric, 0)
            c = all_results.get("centrality", {}).get(test_type, {}).get(n, {}).get(metric, 0)
            diff = c - p
            marker = " <<<" if diff > 0 else ""
            print(f"{n:>8}  {p:>11.1f}%  {c:>11.1f}%  {diff:>+7.1f}%{marker}")

    print(f"\nResults saved to {save_path}")

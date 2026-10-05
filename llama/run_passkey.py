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

# === PROMPT GENERATORS ===
def generate_standard_prompt(n_garbage):
    n_prefix = random.randint(0, n_garbage)
    n_suffix = n_garbage - n_prefix
    pass_key = random.randint(1, 50000)
    prompt = "\n".join([
        TASK_DESC,
        GARBAGE_INF[:n_prefix],
        f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key.",
        GARBAGE_INF[:n_suffix],
        "What is the pass key? The pass key is"
    ])
    return prompt, pass_key

def generate_split_digit_prompt(n_garbage):
    pass_key = random.randint(1000, 9999)  # always 4 digits
    digits = list(str(pass_key))
    ordinals = ["first", "second", "third", "fourth"]

    # Split garbage into 5 sections for 4 digit insertions
    section_len = n_garbage // 5
    sections = [GARBAGE_INF[:section_len] for _ in range(5)]

    # Build the prompt body (without query)
    parts = [TASK_DESC]
    positions = list(range(4))
    random.shuffle(positions)  # randomize insertion order in text
    # Actually, keep them in order so ordinals match position
    positions = list(range(4))

    for i in range(4):
        parts.append(sections[i])
        parts.append(f"The {ordinals[i]} digit of the pass key is {digits[i]}. Remember it. {digits[i]} is the {ordinals[i]} digit.")
    parts.append(sections[4])

    body = "\n".join(parts)

    # Generate 4 separate queries
    queries = []
    for i in range(4):
        query = f"What is the {ordinals[i]} digit of the pass key? The {ordinals[i]} digit of the pass key is"
        queries.append((body + "\n" + query, int(digits[i])))

    return queries, pass_key

def generate_split_integer_prompt(n_garbage):
    keys = [random.randint(1, 50000) for _ in range(4)]
    ordinals = ["first", "second", "third", "fourth"]

    section_len = n_garbage // 5
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

    return queries, keys

# === TEST RUNNERS ===
def extract_number(response):
    try:
        return int(re.search(r'\d+', response).group())
    except:
        return None

def run_standard_test(pipe, n_values, num_tests):
    print("\n=== STANDARD PASSKEY TEST ===")
    results = {}
    for n in n_values:
        correct = 0
        for i in range(num_tests):
            prompt, pass_key = generate_standard_prompt(n)
            num_tokens = len(pipe.tokenizer.encode(prompt))
            response = pipe(prompt, num_return_sequences=1, max_new_tokens=10)[0]["generated_text"][len(prompt):]
            output = extract_number(response)
            if output == pass_key:
                correct += 1
            if (i + 1) % 10 == 0:
                print(f"  n={n}, test {i+1}/{num_tests}, running accuracy: {correct/(i+1)*100:.1f}%")
        accuracy = correct / num_tests * 100
        results[n] = {"accuracy": accuracy, "correct": correct, "total": num_tests}
        print(f"  n={n}: {accuracy:.1f}%")
    return results

def run_split_digit_test(pipe, n_values, num_tests):
    print("\n=== SPLIT-DIGIT PASSKEY TEST ===")
    results = {}
    for n in n_values:
        per_digit_correct = [0, 0, 0, 0]
        all_correct = 0
        for i in range(num_tests):
            queries, full_key = generate_split_digit_prompt(n)
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
                print(f"  n={n}, test {i+1}/{num_tests}, all-correct rate: {all_correct/(i+1)*100:.1f}%")
        results[n] = {
            "all_correct_accuracy": all_correct / num_tests * 100,
            "per_digit_accuracy": [c / num_tests * 100 for c in per_digit_correct],
            "all_correct": all_correct,
            "total": num_tests
        }
        print(f"  n={n}: all_correct={results[n]['all_correct_accuracy']:.1f}%, per_digit={[f'{a:.0f}%' for a in results[n]['per_digit_accuracy']]}")
    return results

def run_split_integer_test(pipe, n_values, num_tests):
    print("\n=== SPLIT-INTEGER PASSKEY TEST ===")
    results = {}
    for n in n_values:
        per_key_correct = [0, 0, 0, 0]
        all_correct = 0
        for i in range(num_tests):
            queries, full_keys = generate_split_integer_prompt(n)
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
                print(f"  n={n}, test {i+1}/{num_tests}, all-correct rate: {all_correct/(i+1)*100:.1f}%")
        results[n] = {
            "all_correct_accuracy": all_correct / num_tests * 100,
            "per_key_accuracy": [c / num_tests * 100 for c in per_key_correct],
            "all_correct": all_correct,
            "total": num_tests
        }
        print(f"  n={n}: all_correct={results[n]['all_correct_accuracy']:.1f}%, per_key={[f'{a:.0f}%' for a in results[n]['per_key_accuracy']]}")
    return results

# === MAIN ===
if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    models = {
        # "positional": positional_path,
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
            "standard": run_standard_test(pipe, n_values, num_tests),
            "split_digit": run_split_digit_test(pipe, n_values, num_tests),
            "split_integer": run_split_integer_test(pipe, n_values, num_tests),
        }

        unload_model(pipe)

        # Save after each model in case of crash
        save_path = os.path.join(results_dir, f"passkey_results_{timestamp}.json")
        with open(save_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {save_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY - STANDARD PASSKEY")
    print(f"{'='*60}")
    print(f"{'n':>8}", end="")
    for model_name in all_results:
        print(f"  {model_name:>12}", end="")
    print()
    for n in n_values:
        print(f"{n:>8}", end="")
        for model_name in all_results:
            acc = all_results[model_name]["standard"].get(str(n), all_results[model_name]["standard"].get(n, {})).get("accuracy", "N/A")
            print(f"  {acc:>11.1f}%", end="")
        print()

    print(f"\nAll results saved to {save_path}")

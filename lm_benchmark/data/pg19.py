import os
import numpy as np
from .utils import add_mem_tokens

PG19_ORIGINAL_PATH = os.path.expanduser("~/landformer/LandFORMER/data")

def get_path(config):
    base = os.path.expanduser("~/landformer/LandFORMER/data/pg19_processed")
    return os.path.join(base, f"mem={config.mem_freq}")

def prepare_pg19_data(config):
    DATA_PATH = get_path(config)
    print(f"Output path: {DATA_PATH}", flush=True)
    os.makedirs(DATA_PATH, exist_ok=True)
    
    train_out = os.path.join(DATA_PATH, 'train.bin')
    if not os.path.exists(train_out):
        print("Adding landmarks to training data...", flush=True)
        train_data = np.memmap(os.path.join(PG19_ORIGINAL_PATH, 'train.bin'), dtype=np.uint16, mode='r')
        add_mem_tokens(config.landmark_id, train_data, config.mem_freq, output_file=train_out)
        print(f"Saved train.bin with landmarks", flush=True)
    
    val_out = os.path.join(DATA_PATH, 'val.bin')
    if not os.path.exists(val_out):
        print("Adding landmarks to validation data...", flush=True)
        val_data = np.memmap(os.path.join(PG19_ORIGINAL_PATH, 'validation.bin'), dtype=np.uint16, mode='r')
        raw_tokenized_eval = add_mem_tokens(config.landmark_id, val_data, config.mem_freq)
        eval_tokenized = np.array(raw_tokenized_eval, dtype=np.uint16)
        eval_tokenized.tofile(val_out)
        print(f"Saved val.bin with landmarks", flush=True)
    
    print("Landmark tokens added successfully!", flush=True)
    return get_pg19_data(config)

def get_pg19_data(config):
    DATA_PATH = get_path(config)
    
    use_cache = getattr(config, 'use_cache', False)
    landmark_selection = getattr(config, 'landmark_selection', 'positional')
    
    # Use raw data (no landmarks) when:
    # 1. Cache is enabled (cache inserts landmarks itself)
    # 2. Content-aware selection is used (model selects its own landmarks)
    if use_cache or landmark_selection != 'positional':
        print(f"Using raw data (no pre-inserted landmarks) [cache={use_cache}, selection={landmark_selection}]", flush=True)
        train_data = np.memmap(os.path.join(PG19_ORIGINAL_PATH, 'train.bin'), dtype=np.uint16, mode='r')
        val_data = np.memmap(os.path.join(PG19_ORIGINAL_PATH, 'validation.bin'), dtype=np.uint16, mode='r')
    else:
        train_data = np.memmap(os.path.join(DATA_PATH, 'train.bin'), dtype=np.uint16, mode='r')
        val_data = np.memmap(os.path.join(DATA_PATH, 'val.bin'), dtype=np.uint16, mode='r')
    
    return {'train': train_data, 'val': val_data}

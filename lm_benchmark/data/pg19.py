# Copyright 2023 Amirkeivan Mohtashami, Martin Jaggi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
    train_data = np.memmap(os.path.join(DATA_PATH, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(DATA_PATH, 'val.bin'), dtype=np.uint16, mode='r')
    return {'train': train_data, 'val': val_data}

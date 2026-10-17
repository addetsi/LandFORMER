# LandFORMER

**Content-Aware Landmark Selection for Efficient Long-Context Language Modeling**

Master's Thesis | Leiden University (LIACS) | 2025

---

## Overview

LandFORMER extends the landmark attention mechanism ([Mohtashami & Jaggi, NeurIPS 2023](https://arxiv.org/abs/2305.16300)) with a content-aware landmark selection strategy called **copy-and-append**. Instead of using randomly initialised special tokens as landmarks, this method replaces each landmark embedding with a copy of the most representative token in the corresponding block. The modification introduces no additional parameters, requires no changes to the attention mechanism or cache, and can be applied to any model that uses landmark attention.

Two selection criteria are evaluated:

- **Centrality-based** — selects the token whose embedding is closest to the block centroid (cosine similarity)
- **Norm-based** — selects the token with the highest embedding magnitude (L2 norm)

---

## Key Findings

The experiments reveal a consistent **perplexity–retrieval trade-off**:

| Metric | Positional (baseline) | Centrality (ours) |
|---|---|---|
| Perplexity gap | — | +0.3–0.4 points |
| Standard passkey (long context) | 33–45% | 90–98% |
| Factual retrieval advantage | — | +38 pp average |
| BLiMP linguistic competence | Equivalent | Equivalent |

- The positional baseline maintains a small perplexity advantage across all context lengths
- Content-aware landmarks dramatically outperform on targeted extraction (~50 percentage points on passkey retrieval)
- The advantage extends to split-digit, split-integer, factual extraction, and needle-in-a-haystack tasks at context lengths up to 72,000 characters
- Post-hoc analysis traces the advantage to more discriminative landmark key vectors

---

## Experiments

Experiments are conducted at two model scales:

### Small scale (150M parameters)
- GPT-2 architecture trained on PG-19 for 240,000 iterations
- Evaluations: perplexity (512–4,096 tokens), convergence dynamics, test set generalisation, BLiMP linguistic competence

### Large scale (7B parameters)
- LLaMA-2 7B fine-tuned with landmark attention
- Evaluations: passkey retrieval (5 variants), factual extraction, needle-in-a-haystack on PG-19 books, context lengths up to 72,000 characters

---

## Setup

### Prerequisites

```bash
Python 3.9+
PyTorch 2.0+
CUDA 11.8+ (for GPU training)
```

### Data Preparation

**PG-19:**
```bash
# Download PG-19 from https://github.com/deepmind/pg19
python data/pg19/prepare.py
```

---

## Usage

### Training

```bash
# Train with centrality-based landmark selection
python main.py \
  --config_format rotary \
  --model base_rotary \
  --dataset pg19 \
  --n_layer 6 --n_head 8 --n_embd 512 \
  --sequence_length 512 \
  --mem_freq 50 \
  --landmark_selection centrality \
  --iterations 240000 \
  --batch_size 4 \
  --device cuda:0
```

### Evaluation

```bash
# Evaluate perplexity
python eval.py \
  --checkpoint ./exps/[CHECKPOINT_PATH] \
  --eval_seq_length 4096 \
  --use_cache \
  --device cuda:0
```

---

## Repository Structure

```
landformer/
├── lm_benchmark/
│   ├── models/
│   │   ├── landmark.py          # Main model with copy-and-append
│   │   ├── caches/
│   │   │   ├── mem_cache.py     # Block-based memory cache
│   │   │   └── cache.py         # Cache base classes
│   │   └── positional_encoders/
│   │       └── rotary.py        # RoPE positional encoding
│   ├── data/
│   │   └── pg19/                # PG-19 dataset utilities
│   └── config/
│       └── rotary.py            # Configuration and arguments
├── scripts/                     # Training and evaluation scripts
├── main.py                      # Training entry point
└── eval.py                      # Evaluation entry point
```

---

## Baseline

This work builds on the landmark attention approach:

- **Landmark Attention** (Mohtashami & Jaggi, NeurIPS 2023)
  - Inserts special landmark tokens at fixed intervals to organise memory into blocks
  - Uses grouped softmax for block-level attention gating
  - Supports long-context inference through block-based cache retrieval
  - Paper: [https://arxiv.org/abs/2305.16300](https://arxiv.org/abs/2305.16300)
  - Code: [https://github.com/epfml/landmark-attention](https://github.com/epfml/landmark-attention)

---

## Requirements

```
torch>=2.0.0
tiktoken>=0.5.0
numpy>=1.24.0
tqdm>=4.65.0
wandb>=0.15.0 (optional, for experiment tracking)
```

See `requirements.txt` for complete dependencies.

---

## Citation

If you use this code, please cite the original landmark attention paper:

```bibtex
@inproceedings{mohtashami2023landmark,
  title={{Random-Access Infinite Context Length for Transformers}},
  author={Mohtashami, Amirkeivan and Jaggi, Martin},
  booktitle={Advances in Neural Information Processing Systems 36 (NeurIPS 2023)},
  year={2023}
}
```

---

## Acknowledgements

Thesis supervision: Prof. Dr. Suzan Verberne, Leiden University (LIACS).

This project builds upon the landmark attention implementation by Mohtashami & Jaggi (2023).

---

## License

This project is developed as part of a master's thesis at Leiden University. The baseline landmark attention code is licensed under Apache 2.0. See `LICENSE` for details.

---

## Contact

**Student:** Godwin Addetsi
**Institution:** Leiden Institute of Advanced Computer Science (LIACS), Leiden University
**Program:** MSc Computer Science
**Email:** godwinaddetsi12@gmail.com
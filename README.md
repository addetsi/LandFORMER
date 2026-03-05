# Landformer

**Efficient Long-Context Language Modeling with Enhanced Landmark Attention**

Master's Thesis Project | February - August 2025

---

## Overview

Landformer extends the landmark attention mechanism ([Mohtashami & Jaggi, NeurIPS 2023](https://arxiv.org/abs/2305.16300)) to improve efficiency and retrieval quality for long-context language modeling. This repository contains the implementation and experimental evaluation.

**Status:** 🚧 Work in progress - implementation and experimentation phase

---

## Project Goals

This work investigates improvements to landmark-based block retrieval for efficient transformer attention:

1. **Inference optimization** - exploring computational efficiency improvements during generation
2. **Adaptive landmark selection** - investigating content-aware strategies for landmark placement
3. **Architectural variants** - comparing different approaches to landmark token integration

The project evaluates these improvements on standard language modeling benchmarks.

---

## Setup

### Prerequisites

```bash
Python 3.9+
PyTorch 2.0+
CUDA 11.8+ (for GPU training)
```



### Data Preparation

**WikiText-2** (for development and testing):
```bash
python scripts/prepare_wikitext.py
```

**PG-19** (for final evaluation):
```bash
# Download PG-19 from https://github.com/deepmind/pg19
python data/pg19/prepare.py
```

---

## Usage

### Training

```bash
# Train baseline model on WikiText-2
python main.py \
  --config_format rotary \
  --model base_rotary \
  --dataset wikitext2 \
  --n_layer 6 --n_head 8 --n_embd 512 \
  --sequence_length 512 \
  --mem_freq 50 \
  --iterations 1000 \
  --batch_size 4 \
  --device cuda:0
```

### Evaluation

```bash
# Evaluate on test set
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
│   │   ├── landmark.py          # Main model implementation
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

## Baseline Model

This work builds on the landmark attention approach:

**Landmark Attention** (Mohtashami & Jaggi, 2023)
- Uses special landmark tokens to organize memory into blocks
- Achieves O(n) complexity through block-based retrieval
- Supports context lengths up to 32K tokens
- Paper: [https://arxiv.org/abs/2305.16300](https://arxiv.org/abs/2305.16300)
- Code: [https://github.com/epfml/landmark-attention](https://github.com/epfml/landmark-attention)

---

## Evaluation Metrics

**Primary:**
- Perplexity on PG-19 test set (target baseline: ~14.72)
- Inference speed (tokens/second)

**Secondary:**
- Peak memory usage (GB)
- Context length scaling

**Baseline Comparison:**
- Landmark attention (Mohtashami & Jaggi, 2023)
- Transformer-XL (Dai et al., 2019)

---

## Implementation Notes

### Development Workflow

1. **Phase 1:** WikiText-2 testing and validation
   - Fast iteration cycles (training in minutes)
   - Debugging and verification of modifications
   - Ensure correctness before scaling to PG-19

2. **Phase 2:** PG-19 final evaluation  
   - Full-scale training (hours to days)
   - Final results for thesis

### Experiment Tracking

All experiments are logged with:
- Configuration parameters
- Training curves (loss, perplexity)
- Evaluation metrics (perplexity, speed, memory)
- Results saved to `./results/` as JSON

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
  title={Landmark Attention: Random-Access Infinite Context Length for Transformers},
  author={Mohtashami, Amirkeivan and Jaggi, Martin},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```

---

## Acknowledgments

This project builds upon the landmark attention implementation by Mohtashami & Jaggi (2023). 

Thesis supervision: Suzan Verbane, Leiden University

---

## License

This project is developed as part of a master's thesis at Leiden University. The baseline landmark attention code is licensed under Apache 2.0. See `LICENSE` for details.

---

## Contact

**Student:** Godwin Addetsi 
**Institution:** Leiden Institute of Advanced Computer Science (LIACS), Leiden University  
**Program:** MSc Computer Science  
**Email:** godwinaddetsi12@gmail.com
**Expected Completion:** August 2025

---




*This is a research project in active development. Code and documentation are subject to change.*

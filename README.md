# User Model Extraction via Belief Self-Distillation

This repository contains the code and data for the paper **"User Model Extraction via Belief Self-Distillation"**.

## Abstract

Models can silently infer who they are talking to and adapt accordingly, yet these inferred beliefs are neither inspectable nor editable. No external annotation can establish what a model believes about a user, so we ask the model itself. To achieve this, we propose Belief Self-Distillation (BSD), a method that elicits a model's beliefs through structured queries and uses its own answers as a training signal. BSD compresses conversational activations into a compact belief vector using the frozen model as both teacher and decoder; only a linear projector is trained (≈0.01% of parameters), so the model, given the vector alone, answers questions about the user as it would given the full conversation.

This yields a read-write interface on *perceived* rather than actual user attributes. Across three models, reading is faithful: belief vectors match probes on full hidden states within 4% macro-F1 despite 32× compression. Writing is stronger: injecting a modified vector changes stated beliefs up to 3.7× more often than the same intervention in the raw hidden-state space. Beliefs are causally relevant for safety: the user-safety axis both predicts refusal and, when shifted, lowers it from 98% to 62% on harmful requests with the prompt unchanged. Finally, one global rotation transfers vectors between independently trained models, carrying the encoded belief in 53–54% of cases.

## Project Structure

* `src/user_distillation/`: Source code for the `user_distillation` package.
  * `data/`: Attribute schema, MCQ probe layouts, dataset loading, and belief extraction (`extraction/`).
  * `training/`: Low-rank projector `BA`, distillation trainer, and checkpointing.
  * `probes/`: Linear probes fit on the user vector and on raw hidden states.
  * `steering/`: CAA steering vectors, activation injection, and random-direction / random-projector controls.
  * `config/`: Training configuration and per-model architecture configs.
* `data/`: Train/validation/test conversation splits, probe layouts, and the explicit persona bank.
* `projectors/`: Trained projectors (`A`, `B`) for the three evaluated models.
* `bsd-env.yml`: Conda environment specification.
* `pyproject.toml`: Installation script for the `user_distillation` package.

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/holmov1/bsd-user-models.git
cd bsd-user-models
```

2. **Create and activate the Conda environment** (this also installs the package in editable mode):
```bash
conda env create -f bsd-env.yml
conda activate eml-usm
```

Without Conda, any Python ≥3.10 works:
```bash
pip install -e ".[dev]"
```

## Citation

Under review at ICLR 2027.

```bibtex

```

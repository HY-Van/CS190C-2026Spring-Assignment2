# CS190C Assignment 2 Starter Repository

This repository is the starting point for HW2.

The repository is intentionally centered on:

- `transformers` for model construction,
- `accelerate` for single-GPU and multi-GPU training.

You are **not** expected to implement Transformer layers from scratch in this homework. We provide the model family through `transformers.LlamaForCausalLM`. Your job is to complete the training pipeline around it.

**It is recommended to use the `cs190c_hw2` image on the cluster.**

## What you should learn here

This starter is designed to help you practice:

- initializing `Accelerator`,
- building tokenized datasets and dataloaders,
- preparing objects with `accelerator.prepare(...)`,
- running training and validation,
- logging to TensorBoard,
- comparing pilot runs before scaling up.

## Dataset note

The required dataset for the assignment is `TinyStories`.

Why we use it:

- it is much easier to train from scratch than noisy web data,
- it is large enough to show meaningful scaling trends,
- small models can still learn visible structure from it.

We pair it with the official tokenizer from `roneneldan/TinyStories-33M`, so students can train immediately without building a tokenizer first.

The assignment uses the dataset's official `train` and `validation` splits only. There is no separate test split in the required setup.

Tokenizer facts:

- tokenizer name: `roneneldan/TinyStories-33M`
- vocabulary size: `50257`
- `bos_token_id = eos_token_id = 50256`
- there is no dedicated pad token, so the starter code reuses EOS as padding

## Model note

The provided model family is a LLaMA-style decoder-only language model.

Important parameters:

- `hidden_size`: model width
- `num_hidden_layers`: number of Transformer blocks
- `num_attention_heads`: number of attention heads
- `intermediate_size`: feed-forward width
- `vocab_size`: tokenizer vocabulary size
- `max_position_embeddings`: sequence length

We provide several model config files in `configs/models/`.

Because the official tokenizer uses a 50k vocabulary, the starter model configs set `tie_word_embeddings = true` to keep the pilot models at a reasonable size.

## Repository layout

```text
CS190C_Assignment2_Starter/
  accelerate_configs/
  configs/
    experiments/
    models/
  scripts/
    train.py
    evaluate.py
  src/
    hw2/
      common.py
      data.py
```

## Files you should look at first

- `scripts/train.py`
- `scripts/evaluate.py`
- `configs/experiments/pilot_s.yaml`
- `configs/models/s.json`

## TODO guide

The main missing pieces are marked with `TODO(student)` in English.

You are expected to complete items such as:

- creating the `Accelerator`,
- building the dataloaders,
- calling `accelerator.prepare(...)`,
- implementing the training step,
- implementing validation,
- logging metrics.

## Example commands

Single-GPU pilot:

```bash
accelerate launch --config_file accelerate_configs/single_gpu.yaml scripts/train.py \
  --experiment-config configs/experiments/pilot_s.yaml \
  --model-config configs/models/s.json
```

Two-GPU scale-up:

```bash
accelerate launch --config_file accelerate_configs/two_gpu_ddp.yaml scripts/train.py \
  --experiment-config configs/experiments/scaleup_m.yaml \
  --model-config configs/models/m.json
```

Evaluation:

```bash
accelerate launch --config_file accelerate_configs/single_gpu.yaml scripts/evaluate.py \
  --experiment-config configs/experiments/pilot_s.yaml \
  --model-config configs/models/s.json \
  --checkpoint-path path/to/checkpoint
```

## Suggested workflow

1. Read the assignment handout.
2. Fill in the `TODO(student)` sections in `scripts/train.py`.
3. Finish `scripts/evaluate.py`.
4. Run a smoke test with a tiny configuration.
5. Run the required pilot study.
6. Choose and justify a final 2-GPU run.

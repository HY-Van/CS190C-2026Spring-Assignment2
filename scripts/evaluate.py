from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, default_data_collator

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hw2.common import ensure_dir, format_metrics, load_json, load_yaml
from hw2.data import build_language_modeling_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CS190C HW2 evaluation script")
    parser.add_argument("--experiment-config", type=str, required=True)
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    return parser.parse_args()


def create_accelerator() -> "Accelerator":
    """
    TODO(student):
    1. Import `Accelerator` from `accelerate`.
    2. Initialize a basic accelerator for evaluation.
    3. Return the accelerator object.
    """
    raise NotImplementedError("TODO(student): initialize Accelerator for evaluation.")


def build_eval_dataloader(exp_config: dict, tokenizer) -> DataLoader:
    """
    TODO(student):
    1. Reuse `build_language_modeling_splits(...)`.
    2. Select the validation split.
    3. Create a DataLoader with `default_data_collator`.
    4. Do not shuffle the evaluation dataloader.
    """
    raise NotImplementedError("TODO(student): create the evaluation dataloader.")


@torch.no_grad()
def evaluate(accelerator, model, dataloader) -> dict[str, float]:
    """
    TODO(student):
    1. Put the model in eval mode.
    2. Iterate over the validation dataloader.
    3. Compute the language modeling loss.
    4. Gather losses across processes when needed.
    5. Return:
       - `val_loss`
       - `val_perplexity`
    """
    raise NotImplementedError("TODO(student): implement the evaluation loop.")


def main() -> None:
    args = parse_args()
    exp_config = load_yaml(args.experiment_config)
    model_config_dict = load_json(args.model_config)

    ensure_dir(Path(exp_config["output_dir"]) / "eval")

    accelerator = create_accelerator()

    tokenizer = AutoTokenizer.from_pretrained(exp_config["tokenizer_name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_config_dict["vocab_size"] = len(tokenizer)
    if tokenizer.bos_token_id is not None:
        model_config_dict["bos_token_id"] = tokenizer.bos_token_id
    if tokenizer.eos_token_id is not None:
        model_config_dict["eos_token_id"] = tokenizer.eos_token_id
    if tokenizer.pad_token_id is not None:
        model_config_dict["pad_token_id"] = tokenizer.pad_token_id

    model_config = LlamaConfig(**model_config_dict)
    model = LlamaForCausalLM(model_config)

    """
    TODO(student):
    Load the checkpoint weights into the model.
    You may use either:
    - `accelerator.load_state(...)` if you save full training state, or
    - `model.load_state_dict(...)` if you save only model weights.
    """
    raise NotImplementedError("TODO(student): load checkpoint weights before evaluation.")

    eval_dataloader = build_eval_dataloader(exp_config, tokenizer)

    """
    TODO(student):
    Prepare the model and dataloader with `accelerator.prepare(...)`.
    """
    raise NotImplementedError("TODO(student): prepare the model and dataloader for evaluation.")

    metrics = evaluate(accelerator, model, eval_dataloader)

    if accelerator.is_main_process:
        print(format_metrics(metrics))


if __name__ == "__main__":
    main()

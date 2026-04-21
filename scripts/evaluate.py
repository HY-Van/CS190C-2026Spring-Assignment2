from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
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
    return Accelerator()


def build_eval_dataloader(exp_config: dict, tokenizer) -> DataLoader:
    """
    TODO(student):
    1. Reuse `build_language_modeling_splits(...)`.
    2. Select the validation split.
    3. Create a DataLoader with `default_data_collator`.
    4. Do not shuffle the evaluation dataloader.
    """
    # TODO(student): keep evaluation on the validation split for apples-to-apples comparison.
    dataset_splits = build_language_modeling_splits(
        dataset_name=exp_config["dataset_name"],
        dataset_config_name=exp_config["dataset_config_name"],
        tokenizer=tokenizer,
        block_size=exp_config["block_size"],
        num_preprocessing_workers=exp_config["num_preprocessing_workers"],
    )

    max_eval_examples = exp_config.get("max_eval_examples")
    if max_eval_examples is not None:
        val_size = min(max_eval_examples, len(dataset_splits["validation"]))
        dataset_splits["validation"] = dataset_splits["validation"].select(range(val_size))

    return DataLoader(
        dataset_splits["validation"],
        batch_size=exp_config["per_device_eval_batch_size"],
        shuffle=False,
        collate_fn=default_data_collator,
    )


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
    model.eval()
    gathered_losses = []

    for batch in dataloader:
        outputs = model(**batch)
        loss = outputs.loss.detach()
        batch_size = batch["input_ids"].size(0)
        gathered_loss = accelerator.gather_for_metrics(loss.repeat(batch_size))
        gathered_losses.append(gathered_loss)

    if not gathered_losses:
        return {"val_loss": float("nan"), "val_perplexity": float("nan")}

    val_loss = torch.cat(gathered_losses).mean().item()
    try:
        val_perplexity = math.exp(val_loss)
    except OverflowError:
        val_perplexity = float("inf")

    return {
        "val_loss": val_loss,
        "val_perplexity": val_perplexity,
    }


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

    eval_dataloader = build_eval_dataloader(exp_config, tokenizer)

    """
    TODO(student):
    Prepare the model and dataloader with `accelerator.prepare(...)`.
    """
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

    """
    TODO(student):
    Load the checkpoint weights into the model.
    You may use either:
    - `accelerator.load_state(...)` if you save full training state, or
    - `model.load_state_dict(...)` if you save only model weights.
    """
    accelerator.load_state(args.checkpoint_path)

    metrics = evaluate(accelerator, model, eval_dataloader)

    if accelerator.is_main_process:
        print(format_metrics(metrics))


if __name__ == "__main__":
    main()

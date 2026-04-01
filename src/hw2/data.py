from __future__ import annotations

from typing import Any

from datasets import DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase


def _group_texts(examples: dict[str, list[list[int]]], block_size: int) -> dict[str, list[list[int]]]:
    concatenated = []
    for input_ids in examples["input_ids"]:
        concatenated.extend(input_ids)

    total_length = (len(concatenated) // block_size) * block_size
    concatenated = concatenated[:total_length]

    blocks = [
        concatenated[i : i + block_size]
        for i in range(0, total_length, block_size)
    ]

    return {
        "input_ids": blocks,
        "labels": [block[:] for block in blocks],
    }


def build_language_modeling_splits(
    dataset_name: str,
    dataset_config_name: str | None,
    tokenizer: PreTrainedTokenizerBase,
    block_size: int,
    num_preprocessing_workers: int = 1,
) -> DatasetDict:
    raw = load_dataset(dataset_name, dataset_config_name)

    if "validation" not in raw:
        raise ValueError("Expected the dataset to contain a validation split.")

    split_datasets = DatasetDict(
        train=raw["train"],
        validation=raw["validation"],
    )

    column_names = split_datasets["train"].column_names
    text_column = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
        return tokenizer(examples[text_column], add_special_tokens=True, truncation=False)

    tokenized = split_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=num_preprocessing_workers,
        remove_columns=column_names,
        desc="Tokenizing dataset",
    )

    grouped = tokenized.map(
        lambda x: _group_texts(x, block_size),
        batched=True,
        num_proc=num_preprocessing_workers,
        remove_columns=tokenized["train"].column_names,
        desc=f"Packing tokens into blocks of {block_size}",
    )

    return grouped

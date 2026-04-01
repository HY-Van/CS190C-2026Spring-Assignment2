# SHANGHAITECH UNIVERSITY, CS190C

## Assignment 2 - Scaling from Pilot Runs to Multi-GPU Training

### Overview

In HW1, you implemented core tokenizer and Transformer components from scratch. In HW2, you will move closer to a real language model training workflow with Hugging Face `transformers` and `accelerate`.

The central idea of this assignment is simple:

1. run small pilot experiments on a single GPU,
2. study the resulting trends,
3. use those trends to justify a larger training run on 2 GPUs.

This is one of the main practical uses of scaling laws. Instead of treating large training runs as guesswork, you will use smaller runs as a decision tool.

For this assignment, `TinyStories` is used with its official `train` and `validation` splits only. We do not define a separate test split. Use the validation split consistently for model selection and final reporting.

### Why this dataset?

We use `TinyStories` for the required track.

`TinyStories` is an English story corpus designed for training small language models from scratch. It is much cleaner and easier to train on than noisy web-scale corpora, while still being large enough to reveal meaningful training trends. This makes it a good teaching dataset for:

- pilot experiments under tight compute budgets,
- controlled comparisons between model sizes,
- visible improvements in loss and generation quality.

In short, it is large enough to be interesting and small enough to be practical.

For tokenization, we use the tokenizer released with `roneneldan/TinyStories-33M`. This gives everyone a consistent, ready-to-use tokenizer and avoids turning HW2 into another tokenizer engineering assignment.

### Why this model family?

We use a LLaMA-style decoder-only language model built with:

- `transformers.LlamaConfig`
- `transformers.LlamaForCausalLM`

This model family is a good fit for HW2 because:

- it is closer to modern LLM practice than older GPT-2 style baselines,
- it matches the components you already studied in HW1,
- students do not need to reimplement model internals,
- it still exposes the important training-system decisions around batch size, learning rate, sequence length, and distributed training.

### Important model parameters

You do not need to write model code in this homework, but you should understand the main configuration parameters:

- `vocab_size`: size of the tokenizer vocabulary. In this homework it should match the `TinyStories-33M` tokenizer, whose vocabulary size is `50257`.
- `hidden_size`: width of the Transformer hidden states.
- `num_hidden_layers`: number of Transformer blocks.
- `num_attention_heads`: number of attention heads.
- `intermediate_size`: width of the feed-forward network inside each block.
- `max_position_embeddings`: maximum sequence length.

Very roughly:

- larger `hidden_size` and more layers usually improve model capacity,
- more parameters usually require more training tokens and more memory,
- longer sequence length increases memory cost,
- larger batch size may improve throughput, but only up to a point.

The `TinyStories-33M` tokenizer is GPT-2/GPT-Neo style:

- vocabulary size: `50257`
- special token: `<|endoftext|>`
- `bos_token_id = eos_token_id = 50256`
- there is no separate pad token, so in the starter code we reuse EOS as padding

To keep the model sizes manageable under this larger vocabulary, the starter configs use `tie_word_embeddings = true`.

### Recommended size ladder

Assume the `TinyStories-33M` tokenizer, `vocab_size = 50257`, `tie_word_embeddings = true`, and `max_position_embeddings = 512`.
We suggest a initial setting `S` for begining.

| Name | Hidden size | Layers | Heads | Intermediate size | Approx. params |
| --- | --- | --- | --- | --- | --- |
| S  | 512 | 12 | 8 | 1536 | about 67M |


### Software requirements

Mandatory:

- `torch`
- `transformers`
- `accelerate`
- `tensorboard`

Recommended:

- `datasets`
- `pyyaml`
- `pandas`
- `matplotlib`

You must use `accelerate` directly. Do not use `transformers.Trainer` as the main training abstraction.

### Hardware target

Pilot stage:

- single GPU only
- no more than 30 GB GPU memory
- each run should finish within about 2 hour

Final stage:

- 2 GPUs
- total GPU memory up to 90 GB
- up to 12 hours for training

### What is provided

We provide a starter repository centered on `transformers` and `accelerate`.

You are **not** expected to:

- implement the Transformer architecture,
- write attention / MLP / normalization modules from scratch.

You **are** expected to:

- initialize `Accelerator`,
- build dataloaders,
- call `accelerator.prepare(...)`,
- write the training and validation logic,
- log metrics to TensorBoard,
- analyze pilot results and justify your scale-up.

For detailed guidelines, please refer to [Guide](./Guide.md).

### Assignment tasks

#### Part A. Complete the training pipeline

Use the starter code to build a reproducible pretraining pipeline with:

- a model created from `LlamaConfig`,
- dataloaders,
- training loop,
- validation loop,
- checkpointing,
- TensorBoard logging.

The starter repo intentionally contains several `TODO(student)` markers. Read each instruction carefully and complete the missing pieces.

#### Part B. Single-GPU pilot study

Run at least 6 pilot experiments total:

1. learning-rate sweep on model `S`: 3 runs
2. model-size sweep using the best learning rate: 3 runs

Recommended pilot budget:

- sequence length `512`
- `10M` to `20M` training tokens per run

Keep the following fixed during the size sweep:

- tokenizer
- sequence length
- token budget
- optimizer family
- learning-rate schedule type

#### Part C. Scale up to 2 GPUs

Choose one final configuration based on your pilot study and train it on 2 GPUs.

Compared with your main pilot run, the final run must increase at least two of the following:

- number of GPUs,
- model size,
- total training tokens,
- effective batch size.

You must explain why your final choice is reasonable. Please ensure that the duration of each training session is kept within 12 hours.

#### Part D. **Optional bonus**

Optionally try `accelerate` with:

- FSDP
- DeepSpeed

Compare against the default multi-GPU setup and report:

- maximum memory per GPU,
- tokens per second,
- whether the backend allowed a larger model or batch size.

### Submission

Submit the following:

1. code,
2. pilot results table,
3. TensorBoard logs for **Part C**,
4. a short report.

### Report requirements

Recommended length: 2 to 3 pages.

Your report should answer:

1. What pilot experiments did you run?
2. What trends did you observe?
3. How did those trends affect your final scale-up decision?
4. What were your final validation results?
5. Did the large run behave as expected?
6. If you tried FSDP or DeepSpeed, how much memory did they save?

### Grading emphasis

This homework is not graded purely by the lowest loss. We care about:

- whether your pipeline works,
- whether your experiments are controlled and reproducible,
- whether your scale-up is justified,
- whether your report shows clear reasoning.

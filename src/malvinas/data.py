import json
from collections.abc import Iterable, Iterator

import torch
from datasets import load_dataset


def pack_tokens(
    documents: Iterable[list[int]], block_size: int, separator_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate tokenized documents with a separator between each, then
    slice into fixed-length (input, target) blocks -- standard pretraining
    packing (plan 00 §6). Any final stretch too short for a full
    block+target is dropped."""
    stream: list[int] = []
    for doc in documents:
        stream.extend(doc)
        stream.append(separator_id)

    num_blocks = max(0, (len(stream) - 1) // block_size)
    x = torch.tensor([stream[i * block_size : i * block_size + block_size] for i in range(num_blocks)])
    y = torch.tensor(
        [stream[i * block_size + 1 : i * block_size + block_size + 1] for i in range(num_blocks)]
    )
    return x, y


def dolma_quality_score(metadata_json: str) -> float:
    """Ai2's own quality classifier score (probability of class "1" = high
    quality) from a Dolma 3 Mix row's `metadata` field. Missing/malformed
    scores are treated as 0.0 (fail the filter), not skipped -- this is the
    filter docs/plans/00 §5 says must be applied, not optional metadata."""
    try:
        metadata = json.loads(metadata_json)
        return float(metadata["dolma2_qc"]["1"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return 0.0


def stream_pretrain_documents(
    repo_id: str, min_quality_score: float = 0.5, max_documents: int | None = None
) -> Iterator[str]:
    """Stream `text` from a Dolma-3-Mix-shaped HF dataset, keeping only rows
    whose dolma2_qc score clears `min_quality_score`."""
    ds = load_dataset(repo_id, split="train", streaming=True)
    count = 0
    for row in ds:
        if max_documents is not None and count >= max_documents:
            break
        if dolma_quality_score(row["metadata"]) >= min_quality_score:
            yield row["text"]
        count += 1


def build_sft_example(messages: list[dict], tokenizer) -> tuple[list[int], list[bool]]:
    """Format a chat conversation (SmolTalk-shaped: list of {role, content})
    into (input_ids, loss_mask) -- loss_mask is True only for the
    assistant's own tokens (plan 00 §7), so `compute_loss` can ignore the
    user's prompt entirely."""
    input_ids: list[int] = []
    loss_mask: list[bool] = []
    for message in messages:
        turn_text = f"<|{message['role']}|>\n{message['content']}\n"
        turn_ids = tokenizer.encode(turn_text)
        input_ids.extend(turn_ids)
        loss_mask.extend([message["role"] == "assistant"] * len(turn_ids))
    return input_ids, loss_mask


def stream_sft_examples(
    repo_id: str, config_name: str, tokenizer, max_examples: int | None = None
) -> Iterator[tuple[list[int], list[bool]]]:
    """Stream a SmolTalk-shaped ({messages: [{role, content}]}) HF dataset
    and format each conversation via build_sft_example (plan 00 §7)."""
    ds = load_dataset(repo_id, config_name, split="train", streaming=True)
    for i, row in enumerate(ds):
        if max_examples is not None and i >= max_examples:
            break
        yield build_sft_example(row["messages"], tokenizer)

import json
from collections.abc import Iterable, Iterator

import torch
from datasets import load_dataset


def pack_tokens(
    documents: Iterable[list[int]], block_size: int, separator_id: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Concatenate tokenized documents with a separator between each, then
    yield fixed-length (input, target) blocks incrementally. Memory remains
    bounded by the current document plus one incomplete block; any final
    stretch too short for a full block+target is dropped."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    buffer: list[int] = []
    for doc in documents:
        buffer.extend(doc)
        buffer.append(separator_id)
        consumed = 0
        while len(buffer) - consumed >= block_size + 1:
            block = torch.tensor(
                buffer[consumed : consumed + block_size + 1], dtype=torch.long
            )
            yield block[:-1], block[1:]
            consumed += block_size
        if consumed:
            del buffer[:consumed]


def pack_sft_tokens(
    examples: Iterable[tuple[list[int], list[bool]]],
    block_size: int,
    separator_id: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Pack tokenized conversations and align the assistant-only mask with
    next-token targets. Separators never contribute to the SFT loss."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    token_buffer: list[int] = []
    mask_buffer: list[bool] = []
    for input_ids, loss_mask in examples:
        if len(input_ids) != len(loss_mask):
            raise ValueError("input_ids and loss_mask must have the same length")
        token_buffer.extend(input_ids)
        token_buffer.append(separator_id)
        mask_buffer.extend(loss_mask)
        mask_buffer.append(False)

        consumed = 0
        while len(token_buffer) - consumed >= block_size + 1:
            tokens = torch.tensor(
                token_buffer[consumed : consumed + block_size + 1], dtype=torch.long
            )
            target_mask = torch.tensor(
                mask_buffer[consumed + 1 : consumed + block_size + 1], dtype=torch.bool
            )
            yield tokens[:-1], tokens[1:], target_mask
            consumed += block_size
        if consumed:
            del token_buffer[:consumed]
            del mask_buffer[:consumed]


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


def xlam_to_messages(row: dict) -> list[dict]:
    """Adapt an xLAM-shaped row ({query, tools, answers}, all JSON strings
    except query) into the {role, content} shape build_sft_example expects
    (plan 00 §9): system turn lists the available tools, user turn is the
    query, assistant turn is the tool call, marked with <|tool_call|>."""
    return [
        {"role": "system", "content": f"Available tools:\n{row['tools']}"},
        {"role": "user", "content": row["query"]},
        {"role": "assistant", "content": f"<|tool_call|>{row['answers']}"},
    ]


def stream_sft_examples(
    repo_id: str,
    tokenizer,
    config_name: str | None = None,
    max_examples: int | None = None,
    row_to_messages=lambda row: row["messages"],
) -> Iterator[tuple[list[int], list[bool]]]:
    """Stream a chat-shaped HF dataset and format each conversation via
    build_sft_example (plan 00 §7). Defaults to the {messages: [...]} shape
    (SmolTalk, Dolci-Think-SFT); pass `row_to_messages` (e.g.
    xlam_to_messages) to adapt a differently-shaped dataset first."""
    ds = load_dataset(repo_id, config_name, split="train", streaming=True)
    for i, row in enumerate(ds):
        if max_examples is not None and i >= max_examples:
            break
        yield build_sft_example(row_to_messages(row), tokenizer)

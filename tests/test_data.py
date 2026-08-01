import json

import torch

from malvinas.data import (
    dolma_quality_score,
    pack_sft_tokens,
    pack_tokens,
    stream_pretrain_documents,
)

DOLMA3_MIX_150B = "allenai/dolma3_mix-150B-1025"


def test_pack_tokens_concatenates_with_separator_and_slices_into_blocks():
    separator_id = 999
    docs = [[1, 2, 3], [4, 5]]  # doc A, doc B
    block_size = 3

    blocks = list(pack_tokens(docs, block_size=block_size, separator_id=separator_id))
    x = torch.stack([block_x for block_x, _ in blocks])
    y = torch.stack([block_y for _, block_y in blocks])

    # concatenated stream: [1,2,3, 999, 4,5, 999] (sep after each doc)
    # sliced into block_size=3 chunks for x, shifted by one for y, dropping
    # any final incomplete block
    expected_stream = [1, 2, 3, 999, 4, 5, 999]
    expected_x = torch.tensor([expected_stream[i : i + block_size] for i in range(0, 4, block_size)])
    expected_y = torch.tensor(
        [expected_stream[i + 1 : i + block_size + 1] for i in range(0, 4, block_size)]
    )

    assert torch.equal(x, expected_x)
    assert torch.equal(y, expected_y)
    assert x.shape == y.shape


def test_pack_tokens_drops_final_incomplete_block():
    docs = [[1, 2]]
    block_size = 5  # concatenated stream [1,2,999] has only 3 tokens, can't fill one block+target

    blocks = list(pack_tokens(docs, block_size=block_size, separator_id=999))

    assert blocks == []


def test_pack_tokens_yields_before_consuming_the_document_stream():
    consumed = []

    def documents():
        consumed.append(0)
        yield [1, 2, 3, 4]
        consumed.append(1)
        yield [5, 6, 7, 8]

    blocks = pack_tokens(documents(), block_size=3, separator_id=999)
    x, y = next(blocks)

    assert consumed == [0]
    assert torch.equal(x, torch.tensor([1, 2, 3]))
    assert torch.equal(y, torch.tensor([2, 3, 4]))


def test_pack_sft_tokens_aligns_targets_and_masks_separator():
    examples = [([10, 11, 12], [False, True, True]), ([20, 21], [False, True])]

    blocks = list(pack_sft_tokens(examples, block_size=3, separator_id=999))

    assert len(blocks) == 2
    first_x, first_y, first_mask = blocks[0]
    assert torch.equal(first_x, torch.tensor([10, 11, 12]))
    assert torch.equal(first_y, torch.tensor([11, 12, 999]))
    assert torch.equal(first_mask, torch.tensor([True, True, False]))
    second_x, second_y, second_mask = blocks[1]
    assert torch.equal(second_x, torch.tensor([999, 20, 21]))
    assert torch.equal(second_y, torch.tensor([20, 21, 999]))
    assert torch.equal(second_mask, torch.tensor([False, True, False]))


def test_pack_sft_tokens_rejects_misaligned_loss_mask():
    examples = [([1, 2], [True])]

    try:
        next(pack_sft_tokens(examples, block_size=1, separator_id=999))
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("expected a ValueError")


def test_dolma_quality_score_reads_class_1_probability():
    metadata = json.dumps({"dolma2_qc": {"0": 0.999, "1": 0.001}})
    assert abs(dolma_quality_score(metadata) - 0.001) < 1e-9


def test_dolma_quality_score_handles_missing_field_as_zero():
    metadata = json.dumps({"some_other_field": True})
    assert dolma_quality_score(metadata) == 0.0


def test_stream_pretrain_documents_actually_applies_the_quality_filter():
    """Integration test against the real Dolma 3 Mix stream (network):
    a strict threshold must not admit more documents than a permissive one
    over the same scanned window. Doesn't inspect document content, only
    counts and types -- the filter mechanism is what's under test, not the
    corpus itself."""
    permissive = list(
        stream_pretrain_documents(DOLMA3_MIX_150B, min_quality_score=0.0, max_documents=20)
    )
    strict = list(
        stream_pretrain_documents(DOLMA3_MIX_150B, min_quality_score=0.999, max_documents=20)
    )

    assert len(permissive) == 20  # threshold 0.0 admits every scanned row
    assert len(strict) < len(permissive)  # strict: measured 0/20 on this window
    assert all(isinstance(doc, str) for doc in permissive)

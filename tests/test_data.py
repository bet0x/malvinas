import json

import torch

from malvinas.data import dolma_quality_score, pack_tokens, stream_pretrain_documents

DOLMA3_MIX_150B = "allenai/dolma3_mix-150B-1025"


def test_pack_tokens_concatenates_with_separator_and_slices_into_blocks():
    separator_id = 999
    docs = [[1, 2, 3], [4, 5]]  # doc A, doc B
    block_size = 3

    x, y = pack_tokens(docs, block_size=block_size, separator_id=separator_id)

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

    x, y = pack_tokens(docs, block_size=block_size, separator_id=999)

    assert x.shape[0] == 0
    assert y.shape[0] == 0


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

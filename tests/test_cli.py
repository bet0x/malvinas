import pytest
import torch

import malvinas.cli as cli
from malvinas.checkpoint import load_checkpoint
from malvinas.cli import _next_batch, _validate_args, build_parser


def test_cli_exposes_pretrain_sft_presets_and_checkpoint_options():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--mode",
            "sft",
            "--preset",
            "0.5b",
            "--init-from",
            "pretrain.pt",
        ]
    )

    assert args.mode == "sft"
    assert args.preset == "0.5b"
    assert str(args.init_from) == "pretrain.pt"


def test_cli_rejects_resume_and_init_from_together():
    args = build_parser().parse_args(
        ["--mode", "pretrain", "--resume", "latest", "--init-from", "base.pt"]
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        _validate_args(args)


def test_next_batch_stacks_sft_masks():
    blocks = iter(
        [
            (torch.tensor([1, 2]), torch.tensor([2, 3]), torch.tensor([False, True])),
            (torch.tensor([4, 5]), torch.tensor([5, 6]), torch.tensor([True, True])),
        ]
    )

    batch, consumed = _next_batch(blocks, batch_size=2)

    assert consumed == 2
    assert batch[0].shape == (2, 2)
    assert batch[2].tolist() == [[False, True], [True, True]]


def test_training_can_resume_and_initialize_a_new_stage(tmp_path, monkeypatch):
    class FakeTokenizer:
        vocab_size = 32

        def __init__(self, _repo_id):
            pass

        def token_to_id(self, _token):
            return 0

    def fake_blocks(run_config, _tokenizer):
        for offset in range(8):
            inputs = torch.arange(run_config["block_size"]) % 32
            targets = (inputs + offset + 1) % 32
            mask = (
                torch.ones(run_config["block_size"], dtype=torch.bool)
                if run_config["mode"] == "sft"
                else None
            )
            yield inputs, targets, mask

    monkeypatch.setattr(cli, "Tokenizer", FakeTokenizer)
    monkeypatch.setattr(cli, "_block_stream", fake_blocks)
    parser = build_parser()

    pretrain = parser.parse_args(
        [
            "--mode",
            "pretrain",
            "--block-size",
            "4",
            "--max-steps",
            "2",
            "--checkpoint-dir",
            str(tmp_path / "pretrain"),
            "--save-every",
            "2",
        ]
    )
    pretrain_path = cli.run_training(pretrain)
    assert load_checkpoint(pretrain_path)["step"] == 2

    resumed = parser.parse_args(
        [
            "--mode",
            "pretrain",
            "--block-size",
            "4",
            "--max-steps",
            "3",
            "--checkpoint-dir",
            str(tmp_path / "pretrain"),
            "--resume",
            "latest",
        ]
    )
    resumed_path = cli.run_training(resumed)
    resumed_payload = load_checkpoint(resumed_path)
    assert resumed_payload["step"] == 3
    assert resumed_payload["blocks_consumed"] == 3

    sft = parser.parse_args(
        [
            "--mode",
            "sft",
            "--block-size",
            "4",
            "--max-steps",
            "1",
            "--checkpoint-dir",
            str(tmp_path / "sft"),
            "--init-from",
            str(resumed_path),
        ]
    )
    sft_path = cli.run_training(sft)
    sft_payload = load_checkpoint(sft_path)
    assert sft_payload["mode"] == "sft"
    assert sft_payload["step"] == 1

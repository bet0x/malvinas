import json

import pytest
import torch

import malvinas.cli as cli
from malvinas.checkpoint import load_checkpoint
from malvinas.cli import (
    _next_batch,
    _evaluate,
    _resolve_autocast_dtype,
    _validate_args,
    build_parser,
)


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
    assert args.models_dir.name == "models"
    assert args.checkpoint_dir is None


def test_cli_rejects_resume_and_init_from_together():
    args = build_parser().parse_args(
        ["--mode", "pretrain", "--resume", "latest", "--init-from", "base.pt"]
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        _validate_args(args)


def test_cli_rejects_model_name_with_path_components():
    args = build_parser().parse_args(
        ["--mode", "pretrain", "--model-name", "../outside"]
    )

    with pytest.raises(ValueError, match="--model-name"):
        _validate_args(args)


def test_cli_validates_token_accumulation_and_float16_device():
    args = build_parser().parse_args(
        [
            "--mode",
            "pretrain",
            "--block-size",
            "8",
            "--batch-size",
            "2",
            "--tokens-per-update",
            "24",
        ]
    )
    with pytest.raises(ValueError, match="divisible"):
        _validate_args(args)
    with pytest.raises(ValueError, match="requires CUDA"):
        _resolve_autocast_dtype("float16", torch.device("cpu"))


def test_cli_validates_global_token_batch_for_distributed_workers():
    args = build_parser().parse_args(
        [
            "--mode",
            "pretrain",
            "--block-size",
            "8",
            "--batch-size",
            "2",
            "--tokens-per-update",
            "32",
        ]
    )

    _validate_args(args, world_size=2)
    args.tokens_per_update = 48
    with pytest.raises(ValueError, match="distributed world size"):
        _validate_args(args, world_size=2)


def test_sft_name_does_not_repeat_stage_suffix():
    assert cli._stage_model_name("malvinas-tiny", "sft") == "malvinas-tiny-sft"
    assert cli._stage_model_name("malvinas-tiny-sft", "sft") == "malvinas-tiny-sft"


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


def test_evaluate_returns_weighted_reproducible_metrics():
    model = cli.model_config_from_preset("tiny", vocab_size=32, max_seq_len=4).build()
    blocks = iter(
        [
            (
                torch.tensor([0, 1, 2, 3]),
                torch.tensor([1, 2, 3, 4]),
                torch.tensor([True, True, True, True]),
            ),
            (
                torch.tensor([4, 5, 6, 7]),
                torch.tensor([5, 6, 7, 8]),
                torch.tensor([False, True, False, True]),
            ),
        ]
    )

    metrics = _evaluate(
        model,
        blocks,
        batch_size=1,
        max_batches=2,
        device=torch.device("cpu"),
        autocast_dtype=None,
    )

    assert metrics["tokens"] == 6
    assert metrics["batches"] == 2
    assert metrics["loss"] > 0
    assert metrics["perplexity"] > 1


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
            "--models-dir",
            str(tmp_path / "models"),
            "--save-every",
            "2",
        ]
    )
    pretrain_path = cli.run_training(pretrain)
    assert load_checkpoint(pretrain_path)["step"] == 2
    assert pretrain_path.parent == tmp_path / "models" / "malvinas-tiny" / "checkpoints"
    assert (tmp_path / "models" / "malvinas-tiny" / "model.pt").exists()

    resumed = parser.parse_args(
        [
            "--mode",
            "pretrain",
            "--block-size",
            "4",
            "--max-steps",
            "3",
            "--models-dir",
            str(tmp_path / "models"),
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
            "--models-dir",
            str(tmp_path / "models"),
            "--init-from",
            str(resumed_path),
        ]
    )
    sft_path = cli.run_training(sft)
    sft_payload = load_checkpoint(sft_path)
    assert sft_payload["mode"] == "sft"
    assert sft_payload["step"] == 1
    assert sft_path.parent == tmp_path / "models" / "malvinas-tiny-sft" / "checkpoints"
    assert (tmp_path / "models" / "malvinas-tiny-sft" / "model.pt").exists()


def test_training_accumulates_to_requested_token_batch(tmp_path, monkeypatch):
    class FakeTokenizer:
        vocab_size = 32

        def __init__(self, _repo_id):
            pass

        def token_to_id(self, _token):
            return 0

    def fake_blocks(run_config, _tokenizer):
        # Seven blocks make the last micro-batch contain one of two blocks.
        for offset in range(7):
            inputs = torch.arange(run_config["block_size"]) % 32
            yield inputs, (inputs + offset + 1) % 32, None

    monkeypatch.setattr(cli, "Tokenizer", FakeTokenizer)
    monkeypatch.setattr(cli, "_block_stream", fake_blocks)
    args = build_parser().parse_args(
        [
            "--mode",
            "pretrain",
            "--block-size",
            "4",
            "--batch-size",
            "2",
            "--tokens-per-update",
            "16",
            "--max-steps",
            "2",
            "--models-dir",
            str(tmp_path / "models"),
        ]
    )

    payload = load_checkpoint(cli.run_training(args))

    assert payload["step"] == 2
    assert payload["blocks_consumed"] == 7
    assert payload["run_config"]["training"]["tokens_per_update"] == 16
    assert payload["scheduler_state_dict"]["step_num"] == 2


def test_training_writes_metrics_best_checkpoint_and_preserved_milestone(
    tmp_path, monkeypatch
):
    class FakeTokenizer:
        vocab_size = 32

        def __init__(self, _repo_id):
            pass

        def token_to_id(self, _token):
            return 0

    def fake_blocks(run_config, _tokenizer):
        for offset in range(4):
            inputs = torch.arange(run_config["block_size"]) % 32
            targets = (inputs + offset + 1) % 32
            yield inputs, targets, None

    monkeypatch.setattr(cli, "Tokenizer", FakeTokenizer)
    monkeypatch.setattr(cli, "_block_stream", fake_blocks)
    args = build_parser().parse_args(
        [
            "--mode",
            "pretrain",
            "--block-size",
            "4",
            "--max-steps",
            "2",
            "--models-dir",
            str(tmp_path / "models"),
            "--save-every",
            "1",
            "--keep-checkpoints",
            "1",
            "--milestone-every",
            "2",
            "--validation-dataset",
            "validation",
            "--validate-every",
            "1",
            "--validation-steps",
            "1",
        ]
    )

    cli.run_training(args)

    model_dir = tmp_path / "models" / "malvinas-tiny"
    checkpoint_dir = model_dir / "checkpoints"
    records = [
        json.loads(line)
        for line in (model_dir / "metrics.jsonl").read_text().splitlines()
    ]
    training = [record for record in records if record["kind"] == "train"]
    validation = [record for record in records if record["kind"] == "validation"]
    assert len(training) == 2
    assert len(validation) == 2
    assert {
        "tokens_per_second",
        "eta_seconds",
        "data_wait_seconds",
        "forward_backward_seconds",
        "optimizer_seconds",
        "expert_load_cv",
    } <= training[-1].keys()
    assert {"loss", "perplexity", "tokens", "batches"} <= validation[-1].keys()
    assert (checkpoint_dir / "best.pt").exists()
    assert (checkpoint_dir / "pretrain-milestone-00000002.pt").exists()
    assert [path.name for path in checkpoint_dir.glob("pretrain-step-*.pt")] == [
        "pretrain-step-00000002.pt"
    ]

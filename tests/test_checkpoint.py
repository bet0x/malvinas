from pathlib import Path

import torch

from malvinas.checkpoint import (
    latest_checkpoint,
    load_checkpoint,
    restore_checkpoint,
    save_checkpoint,
    save_model,
)
from malvinas.config import model_config_from_preset


def make_model_and_optimizer():
    model = model_config_from_preset("tiny", vocab_size=32, max_seq_len=8).build()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer


def test_checkpoint_restores_model_optimizer_and_progress(tmp_path: Path):
    torch.manual_seed(7)
    config = model_config_from_preset("tiny", 32, 8)
    model, optimizer = make_model_and_optimizer()
    loss = model(torch.randint(0, 32, (1, 4))).sum()
    loss.backward()
    optimizer.step()
    expected = {name: tensor.clone() for name, tensor in model.state_dict().items()}

    path = save_checkpoint(
        tmp_path,
        model,
        optimizer,
        step=12,
        blocks_consumed=24,
        mode="pretrain",
        model_config=config.to_dict(),
        run_config={"tokenizer": "test", "block_size": 8},
    )
    restored_model, restored_optimizer = make_model_and_optimizer()
    payload = load_checkpoint(path)
    restore_checkpoint(payload, restored_model, restored_optimizer, torch.device("cpu"))

    assert payload["step"] == 12
    assert payload["blocks_consumed"] == 24
    assert not path.with_suffix(".pt.tmp").exists()
    assert restored_optimizer.state
    for name, tensor in restored_model.state_dict().items():
        assert torch.equal(tensor, expected[name])


def test_latest_checkpoint_filters_by_training_mode(tmp_path: Path):
    model, optimizer = make_model_and_optimizer()
    config = model_config_from_preset("tiny", 32, 8).to_dict()
    for mode, step in (("pretrain", 2), ("sft", 99), ("pretrain", 10)):
        save_checkpoint(
            tmp_path,
            model,
            optimizer,
            step=step,
            blocks_consumed=step,
            mode=mode,
            model_config=config,
            run_config={"tokenizer": "test", "block_size": 8},
        )

    assert latest_checkpoint(tmp_path, "pretrain").name == "pretrain-step-00000010.pt"
    assert latest_checkpoint(tmp_path, "sft").name == "sft-step-00000099.pt"


def test_save_model_writes_inference_artifact_without_optimizer(tmp_path: Path):
    model, _ = make_model_and_optimizer()
    config = model_config_from_preset("tiny", 32, 8).to_dict()

    path = save_model(
        tmp_path / "malvinas-tiny",
        model,
        step=12,
        mode="pretrain",
        model_name="malvinas-tiny",
        model_config=config,
        run_config={"model_name": "malvinas-tiny"},
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert path == tmp_path / "malvinas-tiny" / "model.pt"
    assert payload["model_name"] == "malvinas-tiny"
    assert "model_state_dict" in payload
    assert "optimizer_state_dict" not in payload

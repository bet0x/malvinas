import pytest
import torch

from malvinas.benchmark_moe import _parse_args, _resolve_kernels


def test_cpu_defaults_to_the_portable_reference_kernel():
    args = _parse_args(["--device", "cpu"])

    assert _resolve_kernels(args.kernels, torch.device("cpu")) == ("eager_mm",)


def test_cpu_rejects_an_explicit_grouped_kernel():
    with pytest.raises(SystemExit, match="require a CUDA device"):
        _resolve_kernels(["grouped_mm"], torch.device("cpu"))

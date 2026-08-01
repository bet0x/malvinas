import argparse
import time

import torch

from malvinas.moe import MoEFeedForward


KERNELS = ("eager_mm", "grouped_mm", "grouped_mm_fast")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the Malvinas MoE kernels")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--expert-dim", type=int, default=512)
    parser.add_argument("--num-experts", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--kernels", nargs="+", choices=KERNELS)
    return parser.parse_args(argv)


def _resolve_kernels(
    requested: list[str] | None, device: torch.device
) -> tuple[str, ...]:
    if requested is not None:
        unsupported = [kernel for kernel in requested if kernel != "eager_mm"]
        if device.type != "cuda" and unsupported:
            names = ", ".join(unsupported)
            raise SystemExit(f"{names} require a CUDA device")
        return tuple(requested)
    return KERNELS if device.type == "cuda" else ("eager_mm",)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_kernel(
    layer: MoEFeedForward,
    x: torch.Tensor,
    *,
    warmup: int,
    iterations: int,
) -> tuple[float, torch.Tensor]:
    for _ in range(warmup):
        output = layer(x)
    _synchronize(x.device)

    started = time.perf_counter()
    for _ in range(iterations):
        output = layer(x)
    _synchronize(x.device)
    elapsed = time.perf_counter() - started
    return elapsed / iterations, output


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.iterations <= 0 or args.warmup < 0:
        raise SystemExit("iterations must be positive and warmup cannot be negative")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    kernels = _resolve_kernels(args.kernels, device)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    if device.type == "cpu" and dtype == torch.bfloat16:
        raise SystemExit("use --dtype float32 for the CPU benchmark")

    torch.manual_seed(0)
    x = torch.randn(
        args.batch_size,
        args.sequence_length,
        args.d_model,
        device=device,
        dtype=dtype,
    )
    reference_state = MoEFeedForward(
        args.d_model,
        args.num_experts,
        args.top_k,
        args.expert_dim,
    ).state_dict()
    reference_output: torch.Tensor | None = None

    print("kernel\tms/iteration\ttokens/s\tmax_abs_diff")
    with torch.inference_mode():
        for kernel in kernels:
            layer = MoEFeedForward(
                args.d_model,
                args.num_experts,
                args.top_k,
                args.expert_dim,
                kernel=kernel,
            ).to(device=device, dtype=dtype)
            layer.load_state_dict(reference_state)
            seconds, output = benchmark_kernel(
                layer,
                x,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            if reference_output is None:
                reference_output = output
                max_abs_diff = 0.0
            else:
                max_abs_diff = (output - reference_output).abs().max().item()
            tokens_per_second = args.batch_size * args.sequence_length / seconds
            print(
                f"{kernel}\t{seconds * 1000:.3f}\t"
                f"{tokens_per_second:.0f}\t{max_abs_diff:.6g}"
            )


if __name__ == "__main__":
    main()

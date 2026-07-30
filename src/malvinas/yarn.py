import math

import torch


def _find_correction_dim(num_rotations: float, dim: int, theta: float, original_len: int) -> float:
    return (dim * math.log(original_len / (num_rotations * 2 * math.pi))) / (2 * math.log(theta))


def _find_correction_range(
    low_rot: float, high_rot: float, dim: int, theta: float, original_len: int
) -> tuple[int, int]:
    low = math.floor(_find_correction_dim(low_rot, dim, theta, original_len))
    high = math.ceil(_find_correction_dim(high_rot, dim, theta, original_len))
    return max(low, 0), min(high, dim - 1)


def _linear_ramp_factor(min_val: float, max_val: float, size: int) -> torch.Tensor:
    if min_val == max_val:
        max_val += 0.001
    linear = (torch.arange(size, dtype=torch.float) - min_val) / (max_val - min_val)
    return torch.clamp(linear, 0, 1)


def yarn_inv_freq(
    dim: int,
    theta: float,
    original_len: int,
    target_len: int,
    alpha: float = 1.0,
    beta: float = 32.0,
) -> torch.Tensor:
    """YaRN-rescaled RoPE inverse frequencies (arXiv:2309.00071): NTK-by-parts
    interpolation. High-frequency (short-wavelength) dims are left
    extrapolated (unscaled, already seen many full cycles in `original_len`);
    low-frequency (long-wavelength) dims are linearly interpolated by
    `target_len / original_len`; a ramp between `alpha`/`beta`-controlled
    correction dims blends the two smoothly."""
    scale = target_len / original_len
    freq_indices = torch.arange(0, dim, 2, dtype=torch.float)
    pos_freqs = theta ** (freq_indices / dim)

    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (scale * pos_freqs)

    low, high = _find_correction_range(beta, alpha, dim, theta, original_len)
    extrapolation_factor = 1 - _linear_ramp_factor(low, high, dim // 2)

    return (
        inv_freq_interpolation * (1 - extrapolation_factor)
        + inv_freq_extrapolation * extrapolation_factor
    )

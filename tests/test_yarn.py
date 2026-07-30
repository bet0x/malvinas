import torch

from malvinas.rope import freqs_cis_from_inv_freq
from malvinas.yarn import yarn_inv_freq


def test_scale_one_is_a_no_op():
    """target_len == original_len -> scale=1 -> YaRN's inv_freq must equal
    the plain (unscaled) RoPE inv_freq, regardless of the interpolation ramp,
    since interpolated and extrapolated frequencies coincide at scale=1."""
    dim, theta = 32, 10000.0
    original_len = 4096

    freq_indices = torch.arange(0, dim, 2, dtype=torch.float)
    plain_inv_freq = 1.0 / (theta ** (freq_indices / dim))

    yarn_freq = yarn_inv_freq(dim, theta, original_len, target_len=original_len)

    assert torch.allclose(yarn_freq, plain_inv_freq, atol=1e-6)


def test_highest_frequency_dim_is_left_unscaled():
    """The very first (highest-frequency, shortest-wavelength) dim should be
    left essentially untouched by a 32x extension -- short wavelengths were
    already seen many times during the original context, no need to
    interpolate them."""
    dim, theta = 32, 10000.0
    original_len, target_len = 4096, 4096 * 32

    freq_indices = torch.arange(0, dim, 2, dtype=torch.float)
    plain_inv_freq = 1.0 / (theta ** (freq_indices / dim))

    yarn_freq = yarn_inv_freq(dim, theta, original_len, target_len)

    assert torch.isclose(yarn_freq[0], plain_inv_freq[0], rtol=1e-3)


def test_lowest_frequency_dim_is_scaled_down_by_the_extension_factor():
    """The lowest-frequency (longest-wavelength) dim never completed even one
    cycle in the original context -- it should be fully interpolated, i.e.
    divided by the scale factor, same as naive position interpolation."""
    dim, theta = 32, 10000.0
    original_len, target_len = 4096, 4096 * 32
    scale = target_len / original_len

    freq_indices = torch.arange(0, dim, 2, dtype=torch.float)
    plain_inv_freq = 1.0 / (theta ** (freq_indices / dim))

    yarn_freq = yarn_inv_freq(dim, theta, original_len, target_len)

    assert torch.isclose(yarn_freq[-1], plain_inv_freq[-1] / scale, rtol=1e-3)


def test_yarn_inv_freq_plugs_into_precompute_freqs_cis():
    dim, theta = 32, 10000.0
    original_len, target_len = 64, 256

    yarn_freq = yarn_inv_freq(dim, theta, original_len, target_len)
    freqs_cis = freqs_cis_from_inv_freq(yarn_freq, max_seq_len=target_len)

    assert freqs_cis.shape == (target_len, dim // 2)

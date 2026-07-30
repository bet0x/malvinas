import torch

from malvinas.kda import KimiDeltaAttention


def make_kda(d_model=16, n_heads=4, d_head=4):
    return KimiDeltaAttention(d_model=d_model, n_heads=n_heads, d_head=d_head)


def test_output_shape_matches_input():
    torch.manual_seed(0)
    d_model, seq_len = 16, 5
    kda = make_kda(d_model=d_model)

    x = torch.randn(1, seq_len, d_model)
    out = kda(x)

    assert out.shape == x.shape


def test_causal_recurrence_ignores_future_tokens():
    """The state at time t only ever depends on tokens <= t by construction
    of the recurrence -- changing a later token must not change earlier
    outputs."""
    torch.manual_seed(0)
    d_model, seq_len = 16, 5
    kda = make_kda(d_model=d_model)

    x = torch.randn(1, seq_len, d_model)
    out_a = kda(x)

    x_changed_future = x.clone()
    x_changed_future[0, -1] = torch.randn(d_model)
    out_b = kda(x_changed_future)

    assert torch.allclose(out_a[0, :-1], out_b[0, :-1], atol=1e-6)
    assert not torch.allclose(out_a[0, -1], out_b[0, -1], atol=1e-6)


def test_first_token_matches_eq1_with_zero_initial_state():
    """Kimi Linear tech report, eq. 1: S_t = (I - beta_t k_t k_t^T)
    Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T, o_t = S_t^T q_t. At t=1 with
    S_0 = 0, the first term vanishes entirely regardless of the
    (I - beta k k^T) factor, so S_1 = beta_1 k_1 v_1^T exactly, and
    o_1 = beta_1 * (k_1 . q_1) * v_1 -- a precise, checkable closed form."""
    torch.manual_seed(0)
    d_model, n_heads, d_head = 16, 4, 4
    kda = make_kda(d_model=d_model, n_heads=n_heads, d_head=d_head)

    x = torch.randn(1, 1, d_model)
    out = kda(x)

    with torch.no_grad():
        q = torch.nn.functional.normalize(torch.nn.functional.silu(kda.W_q(x)), dim=-1)
        k = torch.nn.functional.normalize(torch.nn.functional.silu(kda.W_k(x)), dim=-1)
        v = torch.nn.functional.silu(kda.W_v(x))
        beta = torch.sigmoid(kda.W_beta(x))  # (1,1,n_heads)

        q = q.view(1, 1, n_heads, d_head)[0, 0]  # (n_heads, d_head)
        k = k.view(1, 1, n_heads, d_head)[0, 0]
        v = v.view(1, 1, n_heads, d_head)[0, 0]
        beta = beta[0, 0]  # (n_heads,)

        dot_kq = (k * q).sum(-1)  # (n_heads,)
        expected_o = beta.unsqueeze(-1) * dot_kq.unsqueeze(-1) * v  # (n_heads, d_head)
        expected_o = expected_o.reshape(1, 1, n_heads * d_head)

        gate = torch.sigmoid(kda.W_gate_up(kda.W_gate_down(x)))
        expected_out = kda.out_proj(gate * kda.out_norm(expected_o))

    assert torch.allclose(out, expected_out, atol=1e-5)

import torch
from torch import nn
from torch.nn import functional as F

from malvinas.norm import RMSNorm


class KimiDeltaAttention(nn.Module):
    """Kimi Delta Attention (Kimi Team, "Kimi Linear: An Expressive,
    Efficient Attention Architecture", arXiv:2510.26692, Nov 2025) --
    plan 11. Implemented directly against the paper's eq. 1 (fetched, not
    from memory):

        S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
        o_t = S_t^T q_t

    This is the sequential recurrence -- semantically identical to the
    paper's "chunkwise" formulation (eq. 2-9), which is the same function
    reformulated for hardware efficiency, not a different definition. This
    implementation is the reference-correct sequential form, looped over
    time, not the hardware-optimized chunked kernel.

    Two documented simplifications versus the paper's full neural
    parameterization (eq. 10), neither touching the recurrence itself:
    plain linear q/k/v projections instead of ShortConv-then-Swish (a
    standard, low-risk auxiliary component being skipped, not the delta
    rule), and sigmoid as a stand-in for the unspecified decay function
    f(.) that produces alpha_t in [0,1] (the paper names GDN/Mamba-style
    decay functions but the exact formula wasn't visible in the fetched
    pages).
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head

        self.W_q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_beta = nn.Linear(d_model, n_heads, bias=False)

        # low-rank decay projection, rank = d_head, per the paper's eq. 10
        self.W_alpha_down = nn.Linear(d_model, d_head, bias=False)
        self.W_alpha_up = nn.Linear(d_head, n_heads * d_head, bias=False)

        self.out_norm = RMSNorm(n_heads * d_head)
        self.W_gate_down = nn.Linear(d_model, d_head, bias=False)
        self.W_gate_up = nn.Linear(d_head, n_heads * d_head, bias=False)
        self.out_proj = nn.Linear(n_heads * d_head, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        n_h, d_h = self.n_heads, self.d_head

        q = F.normalize(F.silu(self.W_q(x)), dim=-1).view(B, T, n_h, d_h)
        k = F.normalize(F.silu(self.W_k(x)), dim=-1).view(B, T, n_h, d_h)
        v = F.silu(self.W_v(x)).view(B, T, n_h, d_h)
        alpha = torch.sigmoid(self.W_alpha_up(self.W_alpha_down(x))).view(B, T, n_h, d_h)
        beta = torch.sigmoid(self.W_beta(x))  # (B, T, n_h)

        state = torch.zeros(B, n_h, d_h, d_h, device=x.device, dtype=x.dtype)  # S_0 = 0
        outputs = []
        for t in range(T):
            k_t, v_t, q_t = k[:, t], v[:, t], q[:, t]  # (B, n_h, d_h)
            alpha_t, beta_t = alpha[:, t], beta[:, t]  # (B,n_h,d_h), (B,n_h)

            decayed_state = alpha_t.unsqueeze(-1) * state  # Diag(alpha_t) @ S_{t-1}
            kk = k_t.unsqueeze(-1) @ k_t.unsqueeze(-2)  # k_t k_t^T, (B, n_h, d_h, d_h)
            beta_expanded = beta_t.view(B, n_h, 1, 1)
            state = (
                decayed_state
                - beta_expanded * (kk @ decayed_state)
                + beta_expanded * (k_t.unsqueeze(-1) @ v_t.unsqueeze(-2))
            )
            o_t = (state.transpose(-2, -1) @ q_t.unsqueeze(-1)).squeeze(-1)  # S_t^T q_t
            outputs.append(o_t)

        out = torch.stack(outputs, dim=1).reshape(B, T, n_h * d_h)
        gate = torch.sigmoid(self.W_gate_up(self.W_gate_down(x)))
        out = gate * self.out_norm(out)
        return self.out_proj(out)

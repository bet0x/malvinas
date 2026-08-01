import torch
from torch import nn


class SwiGLU(nn.Module):
    """Gate/up/down MLP shared by the routed experts and the shared expert."""

    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class MoEFeedForward(nn.Module):
    """Top-k routed experts + one always-on shared expert (DeepSeekMoE-style)."""

    def __init__(self, d_model: int, num_experts: int, top_k: int, expert_dim: int):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = nn.Linear(d_model, num_experts, bias=False)

        self.expert_gate_up_proj = nn.Parameter(
            torch.empty(num_experts, d_model, 2 * expert_dim).normal_(std=0.02)
        )
        self.expert_down_proj = nn.Parameter(
            torch.empty(num_experts, expert_dim, d_model).normal_(std=0.02)
        )
        self.act = nn.SiLU()

        self.shared_expert = SwiGLU(d_model, expert_dim)

        self.register_buffer("expert_bias", torch.zeros(num_experts))
        self.last_selected_experts = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x_flat = x.reshape(-1, C)

        router_logits = self.router(x_flat)
        # expert_bias only steers *selection*; the gate value uses the
        # unbiased affinity (DeepSeek-V3 auxiliary-loss-free balancing).
        biased_logits = router_logits + self.expert_bias
        _, selected_experts = torch.topk(biased_logits, self.top_k, dim=-1)
        self.last_selected_experts = selected_experts.reshape(-1).detach()
        routing_weights = torch.sigmoid(torch.gather(router_logits, -1, selected_experts))

        token_idx = torch.arange(B * T, device=x.device).repeat_interleave(self.top_k)
        expert_idx = selected_experts.reshape(-1)
        routing_weights_flat = routing_weights.reshape(-1)

        expert_inputs = x_flat[token_idx]
        gate_up_w = self.expert_gate_up_proj[expert_idx]
        down_w = self.expert_down_proj[expert_idx]

        gate_up = torch.bmm(expert_inputs.unsqueeze(1), gate_up_w)
        gate, up = gate_up.chunk(2, dim=-1)
        activated = self.act(gate) * up
        expert_out = torch.bmm(activated, down_w).squeeze(1)
        expert_out = expert_out * routing_weights_flat.unsqueeze(-1)

        combined = torch.zeros_like(x_flat)
        combined.scatter_add_(0, token_idx.unsqueeze(-1).expand(-1, C), expert_out)

        out = combined.view(B, T, C) + self.shared_expert(x)
        return out

    @torch.no_grad()
    def update_expert_bias(self, update_rate: float) -> None:
        """DeepSeek-V3-style auxiliary-loss-free balancing: nudge each
        expert's bias down if it took more than its uniform share of the
        last forward's selections, up if it took less. No gradient."""
        counts = torch.bincount(self.last_selected_experts, minlength=self.num_experts).float()
        target = counts.sum() / self.num_experts
        self.expert_bias -= update_rate * torch.sign(counts - target)

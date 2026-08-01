import torch
from torch import nn
from torch.nn import functional as F


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

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int,
        expert_dim: int,
        kernel: str = "auto",
    ):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        valid_kernels = {"auto", "eager_mm", "grouped_mm", "grouped_mm_fast"}
        if kernel not in valid_kernels:
            raise ValueError(
                f"kernel must be one of {sorted(valid_kernels)}, got {kernel!r}"
            )
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.kernel = kernel

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

        kernel = self._resolved_kernel(x_flat)
        if kernel == "eager_mm":
            combined = self._eager_experts_forward(
                x_flat,
                selected_experts,
                routing_weights,
            )
        else:
            combined = self._grouped_experts_forward(
                x_flat,
                selected_experts,
                routing_weights,
                fast=kernel == "grouped_mm_fast",
            )

        out = combined.view(B, T, C) + self.shared_expert(x)
        return out

    def _resolved_kernel(self, x: torch.Tensor) -> str:
        if self.kernel != "auto":
            return self.kernel
        if x.is_cuda and hasattr(F, "grouped_mm"):
            return "grouped_mm"
        return "eager_mm"

    def _sorted_assignments(
        self,
        x_flat: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        expert_idx = selected_experts.reshape(-1)
        weights = routing_weights.reshape(-1)
        token_idx = torch.arange(x_flat.shape[0], device=x_flat.device).repeat_interleave(
            self.top_k
        )
        sort_idx = torch.argsort(expert_idx, stable=True)
        inverse_idx = torch.empty_like(sort_idx)
        inverse_idx[sort_idx] = torch.arange(sort_idx.numel(), device=sort_idx.device)
        sorted_token_idx = token_idx.index_select(0, sort_idx)
        return (
            x_flat.index_select(0, sorted_token_idx),
            expert_idx.index_select(0, sort_idx),
            weights.index_select(0, sort_idx),
            inverse_idx,
            sorted_token_idx,
        )

    def _eager_experts_forward(
        self,
        x_flat: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        expert_inputs, sorted_experts, sorted_weights, inverse_idx, _ = (
            self._sorted_assignments(x_flat, selected_experts, routing_weights)
        )
        counts = torch.bincount(sorted_experts, minlength=self.num_experts).tolist()
        expert_outputs: list[torch.Tensor] = []
        offset = 0
        for expert_id, count in enumerate(counts):
            if count == 0:
                continue
            expert_input = expert_inputs[offset : offset + count]
            gate_up = expert_input @ self.expert_gate_up_proj[expert_id]
            gate, up = gate_up.chunk(2, dim=-1)
            expert_outputs.append((self.act(gate) * up) @ self.expert_down_proj[expert_id])
            offset += count

        sorted_output = torch.cat(expert_outputs, dim=0) * sorted_weights.unsqueeze(-1)
        assignment_output = sorted_output.index_select(0, inverse_idx)
        return assignment_output.reshape(
            x_flat.shape[0],
            self.top_k,
            self.d_model,
        ).sum(dim=1)

    def _grouped_experts_forward(
        self,
        x_flat: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        *,
        fast: bool,
    ) -> torch.Tensor:
        if not hasattr(F, "grouped_mm"):
            raise RuntimeError(
                "grouped_mm requires a PyTorch build with torch.nn.functional.grouped_mm"
            )
        if not x_flat.is_cuda:
            raise RuntimeError("grouped_mm kernels require a CUDA device")

        expert_inputs, sorted_experts, sorted_weights, inverse_idx, sorted_token_idx = (
            self._sorted_assignments(x_flat, selected_experts, routing_weights)
        )
        offsets = torch.bincount(sorted_experts, minlength=self.num_experts).cumsum(
            0,
            dtype=torch.int32,
        )
        gate_up = F.grouped_mm(expert_inputs, self.expert_gate_up_proj, offs=offsets)
        gate, up = gate_up.chunk(2, dim=-1)
        sorted_output = F.grouped_mm(
            self.act(gate) * up,
            self.expert_down_proj,
            offs=offsets,
        )
        sorted_output = sorted_output * sorted_weights.unsqueeze(-1)

        if fast:
            combined = torch.zeros_like(x_flat)
            combined.index_add_(0, sorted_token_idx, sorted_output.to(combined.dtype))
            return combined

        assignment_output = sorted_output.index_select(0, inverse_idx)
        return assignment_output.reshape(
            x_flat.shape[0],
            self.top_k,
            self.d_model,
        ).sum(dim=1)

    @torch.no_grad()
    def update_expert_bias(
        self,
        update_rate: float,
        counts: torch.Tensor | None = None,
    ) -> None:
        """DeepSeek-V3-style auxiliary-loss-free balancing: nudge each
        expert's bias down if it took more than its uniform share, up if it
        took less. No gradient."""
        if counts is None:
            if self.last_selected_experts is None:
                return
            counts = torch.bincount(
                self.last_selected_experts,
                minlength=self.num_experts,
            ).float()
        target = counts.sum() / self.num_experts
        self.expert_bias -= update_rate * torch.sign(counts - target)

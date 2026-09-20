"""HCMT event-time transformer for sparse perioperative trajectories."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class EventHCMTOutput:
    logits: Optional[torch.Tensor]
    family_logits: Optional[torch.Tensor] = None
    log_total_rate: Optional[torch.Tensor] = None
    time_mu: Optional[torch.Tensor] = None
    time_log_sigma: Optional[torch.Tensor] = None
    family_time_mu: Optional[torch.Tensor] = None
    family_time_log_sigma: Optional[torch.Tensor] = None
    trajectory_logits: Optional[torch.Tensor] = None
    token_logits: Optional[torch.Tensor] = None
    value_prediction: Optional[torch.Tensor] = None
    hidden_state: Optional[torch.Tensor] = None


class ContinuousTimeEncoding(nn.Module):
    def __init__(self, hidden_dim: int, enhanced: bool = False):
        super().__init__()
        half = hidden_dim // 2
        frequencies = torch.exp(torch.linspace(math.log(1 / 5), math.log(1 / 43200), half))
        self.register_buffer("frequencies", frequencies)
        self.projection = nn.Linear(half * 4, hidden_dim, bias=False)
        self.scale_projection = (nn.Sequential(
            nn.Linear(4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim, bias=False)
        ) if enhanced else None)

    def forward(self, absolute_minutes: torch.Tensor, gap_minutes: torch.Tensor) -> torch.Tensor:
        absolute = absolute_minutes[..., None] * self.frequencies
        gap = gap_minutes[..., None] * self.frequencies
        features = torch.cat((absolute.sin(), absolute.cos(), gap.sin(), gap.cos()), dim=-1)
        periodic = self.projection(features)
        if self.scale_projection is None:
            return periodic
        scale = math.log1p(43200.0)
        monotonic = torch.stack((
            torch.log1p(absolute_minutes.clamp_min(0)) / scale,
            torch.log1p(gap_minutes.clamp_min(0)) / scale,
            (gap_minutes > 0).to(absolute_minutes.dtype),
            (absolute_minutes / 43200.0).clamp(0, 2),
        ), dim=-1)
        return periodic + self.scale_projection(monotonic)


class EventHCMT(nn.Module):
    """Causal transformer predicting the next outcome set and waiting time.

    Output logits are interpreted as log cause-specific rates.  Their softmax
    predicts event identity; their log-sum-exp predicts the total event rate,
    following Delphi's competing-exponentials formulation.
    """

    def __init__(self, num_tokens: int, num_outcomes: int, num_static: int = 2,
                 hidden_dim: int = 256, num_layers: int = 8, num_heads: int = 8,
                 ffn_dim: int = 1024, dropout: float = 0.1,
                 initial_event_interval_hours: float = 24.0,
                 decoupled_time_head: bool = False,
                 enhanced_time_encoding: bool = False,
                 num_trajectory_horizons: int = 3,
                 lognormal_time_head: bool = False,
                 outcome_family_ids: Optional[torch.Tensor] = None,
                 same_time_block_causal: bool = False,
                 relative_time_attention: bool = False,
                 event_conditioned_time_head: bool = False,
                 phase_memory: bool = False,
                 observation_intensity: bool = False,
                 value_reconstruction: bool = False):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_outcomes = int(num_outcomes)
        self.num_heads = int(num_heads)
        self.decoupled_time_head = bool(decoupled_time_head)
        self.lognormal_time_head = bool(lognormal_time_head)
        self.same_time_block_causal = bool(same_time_block_causal)
        self.relative_time_attention = bool(relative_time_attention)
        self.event_conditioned_time_head = bool(event_conditioned_time_head)
        self.phase_memory = bool(phase_memory)
        self.observation_intensity = bool(observation_intensity)
        self.value_reconstruction = bool(value_reconstruction)
        if outcome_family_ids is None:
            family_ids = torch.empty(0, dtype=torch.long)
            self.num_event_families = 0
        else:
            family_ids = torch.as_tensor(outcome_family_ids, dtype=torch.long)
            if family_ids.shape != (self.num_outcomes,):
                raise ValueError("outcome_family_ids must contain one ID per outcome")
            if family_ids.min().item() < 0:
                raise ValueError("outcome family IDs must be non-negative")
            self.num_event_families = int(family_ids.max().item()) + 1
        if self.event_conditioned_time_head and not self.num_event_families:
            raise ValueError("event-conditioned time requires outcome_family_ids")
        self.register_buffer("outcome_family_ids", family_ids, persistent=False)
        self.num_trajectory_horizons = int(num_trajectory_horizons)
        self.token_embedding = nn.Embedding(num_tokens, hidden_dim, padding_idx=0)
        self.kind_embedding = nn.Embedding(16, hidden_dim, padding_idx=0)
        self.phase_embedding = (nn.Embedding(8, hidden_dim) if self.phase_memory else None)
        self.value_projection = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.observation_projection = (nn.Sequential(
            nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim, bias=False)
        ) if self.observation_intensity else None)
        self.static_projection = nn.Sequential(
            nn.Linear(num_static, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.history_projection = (nn.Sequential(
            nn.Linear(self.num_event_families, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        ) if self.num_event_families and self.phase_memory else None)
        self.time_encoding = ContinuousTimeEncoding(hidden_dim, enhanced_time_encoding)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            bias=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                             norm=nn.LayerNorm(hidden_dim))
        self.outcome_head = nn.Linear(hidden_dim, num_outcomes, bias=True)
        self.family_head = (nn.Linear(hidden_dim, self.num_event_families, bias=True)
                            if self.num_event_families else None)
        self.relative_time_bias = (nn.Sequential(
            nn.Linear(2, max(16, num_heads * 2)), nn.GELU(),
            nn.Linear(max(16, num_heads * 2), num_heads, bias=False),
        ) if self.relative_time_attention else None)
        self.time_head = (nn.Linear(hidden_dim, 1, bias=True)
                          if self.decoupled_time_head and not self.lognormal_time_head else None)
        time_output_dim = 2 * (self.num_event_families
                               if self.event_conditioned_time_head else 1)
        self.time_distribution_head = (nn.Linear(hidden_dim, time_output_dim, bias=True)
                                       if self.lognormal_time_head else None)
        self.trajectory_head = nn.Linear(
            hidden_dim, self.num_trajectory_horizons * self.num_outcomes, bias=True)
        self.masked_token_bias = nn.Parameter(torch.zeros(num_tokens))
        self.value_head = (nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)) if self.value_reconstruction else None)
        self.apply(self._init_weights)
        # Initialise the aggregate rate to a clinically plausible sparse-event
        # interval.  A low starting rate prevents long event-free episodes from
        # producing an explosive waiting-time likelihood at step zero.
        interval = max(float(initial_event_interval_hours), 1.0 / 60.0)
        if self.lognormal_time_head:
            nn.init.zeros_(self.outcome_head.bias)
            nn.init.zeros_(self.time_distribution_head.weight)
            with torch.no_grad():
                parameters = self.time_distribution_head.bias.view(-1, 2)
                parameters[:, 0] = math.log(interval)
                parameters[:, 1] = 0.0
        elif self.decoupled_time_head:
            nn.init.zeros_(self.outcome_head.bias)
            nn.init.constant_(self.time_head.bias, math.log(1.0 / interval))
        else:
            nn.init.constant_(self.outcome_head.bias, math.log(1.0 / (interval * num_outcomes)))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward(self, batch: Dict[str, torch.Tensor], *, causal: bool = True,
                return_token_logits: bool = False,
                masked_only: bool = False) -> EventHCMTOutput:
        x = self.token_embedding(batch["token_id"])
        x = x + self.kind_embedding(batch["token_kind"])
        # Raw clinical values span several orders of magnitude.  A signed-log
        # transform prevents EBL/enzymes from dominating bounded severity and
        # binary signals while preserving direction and ordering.
        robust_value = torch.sign(batch["value"]) * torch.log1p(batch["value"].abs())
        robust_value = robust_value.clamp(-12.0, 12.0)
        value_features = torch.stack((robust_value, batch["has_value"]), dim=-1)
        x = x + self.value_projection(value_features)
        if self.observation_projection is not None and "observation_features" in batch:
            x = x + self.observation_projection(batch["observation_features"])
        x = x + self.time_encoding(batch["time_min"], batch["gap_min"])
        x = x + self.static_projection(batch["static"])[:, None, :]
        if self.phase_embedding is not None and "phase_id" in batch:
            x = x + self.phase_embedding(batch["phase_id"])
        if (self.history_projection is not None and "history_family_counts" in batch and
                batch["history_family_counts"].shape[-1] == self.num_event_families):
            x = x + self.history_projection(batch["history_family_counts"])[:, None, :]

        length = x.shape[1]
        causal_mask = self._attention_mask(batch["time_min"], causal)
        padding = ~batch["attention_mask"].bool()
        if causal_mask is not None and causal_mask.dtype.is_floating_point:
            float_padding = torch.zeros_like(padding, dtype=causal_mask.dtype)
            padding = float_padding.masked_fill(padding, -torch.inf)
        encoded = self.encoder(x, mask=causal_mask, src_key_padding_mask=padding)
        token_logits = (F.linear(encoded.float(), self.token_embedding.weight.float(),
                                 self.masked_token_bias.float())
                        if return_token_logits else None)
        value_prediction = (self.value_head(encoded).squeeze(-1).float()
                            if self.value_head is not None else None)
        if masked_only:
            return EventHCMTOutput(
                logits=None, token_logits=token_logits,
                value_prediction=value_prediction, hidden_state=encoded)
        log_total_rate = (self.time_head(encoded).squeeze(-1).float()
                          if self.time_head is not None else None)
        event_logits = self.outcome_head(encoded).float()
        family_logits = (self.family_head(encoded).float()
                         if self.family_head is not None else None)
        if family_logits is not None:
            event_logits = event_logits + family_logits[..., self.outcome_family_ids]
        time_mu = time_log_sigma = None
        family_time_mu = family_time_log_sigma = None
        if self.time_distribution_head is not None:
            time_parameters = self.time_distribution_head(encoded).float()
            if self.event_conditioned_time_head:
                time_parameters = time_parameters.reshape(
                    *encoded.shape[:2], self.num_event_families, 2)
                family_time_mu = time_parameters[..., 0]
                family_time_log_sigma = time_parameters[..., 1].clamp(-3.0, 2.0)
                family_probability = torch.softmax(family_logits, dim=-1)
                time_mu = (family_probability * family_time_mu).sum(-1)
                time_log_sigma = (family_probability * family_time_log_sigma).sum(-1)
            else:
                time_mu = time_parameters[..., 0]
                time_log_sigma = time_parameters[..., 1].clamp(-3.0, 2.0)
        trajectory_logits = self.trajectory_head(encoded).float().reshape(
            *encoded.shape[:2], self.num_trajectory_horizons, self.num_outcomes)
        return EventHCMTOutput(
            logits=event_logits, family_logits=family_logits,
            log_total_rate=log_total_rate,
            time_mu=time_mu, time_log_sigma=time_log_sigma,
            family_time_mu=family_time_mu,
            family_time_log_sigma=family_time_log_sigma,
            trajectory_logits=trajectory_logits, token_logits=token_logits,
            value_prediction=value_prediction, hidden_state=encoded)

    def _attention_mask(self, time_min: torch.Tensor,
                        causal: bool) -> Optional[torch.Tensor]:
        """Build a temporal bias and optionally hide all concurrent siblings."""
        batch_size, length = time_min.shape
        if not causal and self.relative_time_bias is None:
            return None
        query_time = time_min[:, :, None]
        key_time = time_min[:, None, :]
        delta = query_time - key_time
        mask = torch.zeros(
            batch_size, self.num_heads, length, length,
            device=time_min.device, dtype=torch.float32)
        if self.relative_time_bias is not None:
            scale = math.log1p(43200.0)
            relative_features = torch.stack((
                torch.sign(delta) * torch.log1p(delta.abs()) / scale,
                (delta == 0).to(delta.dtype),
            ), dim=-1)
            bias = self.relative_time_bias(relative_features).permute(0, 3, 1, 2)
            mask = mask + bias.float()
        if causal:
            positions = torch.arange(length, device=time_min.device)
            future = positions[None, :] > positions[:, None]
            forbidden = future[None, :, :].expand(batch_size, -1, -1)
            if self.same_time_block_causal:
                same_time = query_time == key_time
                diagonal = torch.eye(length, dtype=torch.bool, device=time_min.device)
                forbidden = forbidden | (same_time & ~diagonal[None, :, :])
            mask = mask.masked_fill(forbidden[:, None, :, :], -torch.inf)
        return mask.reshape(batch_size * self.num_heads, length, length)


def event_time_loss(logits: torch.Tensor, target_set: torch.Tensor,
                    target_dt_hours: torch.Tensor, loss_mask: torch.Tensor,
                    time_mask: torch.Tensor | None = None, time_weight: float = 1.0,
                    max_wait_hours: float = 24.0 * 38,
                    stable_index: int = -1,
                    log_total_rate: torch.Tensor | None = None,
                    time_mu: torch.Tensor | None = None,
                    time_log_sigma: torch.Tensor | None = None,
                    trajectory_logits: torch.Tensor | None = None,
                    trajectory_target: torch.Tensor | None = None,
                    trajectory_mask: torch.Tensor | None = None,
                    trajectory_weight: float = 0.0,
                    class_weights: torch.Tensor | None = None,
                    family_logits: torch.Tensor | None = None,
                    outcome_family_ids: torch.Tensor | None = None,
                    family_weight: float = 0.0,
                    family_time_mu: torch.Tensor | None = None,
                    family_time_log_sigma: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
    """Event CE plus competing-exponential event/censor waiting-time NLL."""
    event_valid = loss_mask.bool() & (target_set.sum(-1) > 0)
    time_valid = event_valid if time_mask is None else time_mask.bool()
    has_trajectory = (trajectory_logits is not None and trajectory_target is not None and
                      trajectory_mask is not None and trajectory_mask.any())
    if not event_valid.any() and not time_valid.any() and not has_trajectory:
        zero = logits.sum() * 0.0
        if log_total_rate is not None:
            zero = zero + log_total_rate.sum() * 0.0
        return {"loss": zero, "event_loss": zero, "family_loss": zero,
                "time_loss": zero,
                "trajectory_loss": zero,
                "event_accuracy": zero.detach(), "time_mae_hours": zero.detach(),
                "nonstable_accuracy": zero.detach(),
                "stable_prediction_fraction": zero.detach(),
                "num_targets": event_valid.sum().detach(),
                "num_time_targets": time_valid.sum().detach(),
                "num_time_mae_targets": event_valid.sum().detach(),
                "num_nonstable_targets": event_valid.sum().detach()}

    zero = logits.sum() * 0.0
    if log_total_rate is not None:
        zero = zero + log_total_rate.sum() * 0.0
    if event_valid.any():
        selected_logits = logits[event_valid]
        selected_targets = target_set[event_valid]
        # Probability mass of the complete tied event set.  Unlike averaging
        # independent cross-entropies, this does not force simultaneous events
        # to compete as if only one could be correct.
        positive_logits = selected_logits.masked_fill(selected_targets <= 0, -torch.inf)
        event_nll = torch.logsumexp(selected_logits, dim=-1) - torch.logsumexp(
            positive_logits, dim=-1)
        if class_weights is not None:
            sample_weight = (selected_targets * class_weights).sum(-1) / selected_targets.sum(-1).clamp_min(1)
            event_nll = event_nll * sample_weight
        event_loss = event_nll.mean()
    else:
        selected_logits = logits.new_zeros((0, logits.shape[-1]))
        selected_targets = target_set.new_zeros((0, target_set.shape[-1]))
        event_loss = zero

    family_loss = zero
    if family_logits is not None and outcome_family_ids is not None and event_valid.any():
        num_families = family_logits.shape[-1]
        membership = F.one_hot(
            outcome_family_ids.to(logits.device), num_classes=num_families).to(target_set.dtype)
        family_targets = ((target_set[event_valid] @ membership) > 0).to(target_set.dtype)
        selected_family_logits = family_logits[event_valid]
        positive_family_logits = selected_family_logits.masked_fill(
            family_targets <= 0, -torch.inf)
        family_loss = (
            torch.logsumexp(selected_family_logits, dim=-1)
            - torch.logsumexp(positive_family_logits, dim=-1)
        ).mean()

    dt = target_dt_hours[time_valid].float().clamp(
        min=1.0 / 60.0, max=float(max_wait_hours))
    is_event = event_valid[time_valid]
    if time_mu is not None and time_log_sigma is not None:
        mu = time_mu[time_valid].float()
        log_sigma = time_log_sigma[time_valid].float().clamp(-3.0, 2.0)
        if (family_time_mu is not None and family_time_log_sigma is not None and
                outcome_family_ids is not None and is_event.any()):
            membership = F.one_hot(
                outcome_family_ids.to(logits.device),
                num_classes=family_time_mu.shape[-1]).to(target_set.dtype)
            time_family_targets = (target_set[time_valid] @ membership).clamp_max(1.0)
            weights = time_family_targets / time_family_targets.sum(-1, keepdim=True).clamp_min(1.0)
            conditional_mu = (family_time_mu[time_valid].float() * weights).sum(-1)
            conditional_log_sigma = (
                family_time_log_sigma[time_valid].float() * weights).sum(-1).clamp(-3.0, 2.0)
            mu = torch.where(is_event, conditional_mu, mu)
            log_sigma = torch.where(is_event, conditional_log_sigma, log_sigma)
        sigma = log_sigma.exp()
        log_dt = dt.log()
        z = (log_dt - mu) / sigma
        event_time_nll = (0.5 * z.square() + log_sigma + log_dt +
                          0.5 * math.log(2 * math.pi))
        survival = (0.5 * torch.erfc(z / math.sqrt(2.0))).clamp_min(1e-7)
        time_nll = torch.where(is_event, event_time_nll, -survival.log())
        log_rate = None
    elif log_total_rate is None:
        time_logits = logits[time_valid]
        log_rate = torch.logsumexp(time_logits, dim=-1)
    else:
        log_rate = log_total_rate[time_valid]
    if log_rate is not None:
        log_rate = log_rate.clamp(min=-16.0, max=8.0)
        time_nll = torch.exp(log_rate) * dt - log_rate * is_event.float()
    time_loss = time_nll.mean() if time_nll.numel() else zero

    trajectory_loss = zero
    if (trajectory_logits is not None and trajectory_target is not None and
            trajectory_mask is not None and trajectory_mask.any()):
        valid = trajectory_mask.bool().unsqueeze(-1).expand_as(trajectory_target)
        raw = F.binary_cross_entropy_with_logits(
            trajectory_logits[valid], trajectory_target[valid], reduction="none")
        probability = torch.sigmoid(trajectory_logits[valid])
        target = trajectory_target[valid]
        pt = torch.where(target > 0, probability, 1.0 - probability)
        alpha = torch.where(target > 0, 0.75, 0.25)
        trajectory_loss = (alpha * (1.0 - pt).square() * raw).mean()
    total = (event_loss + float(family_weight) * family_loss +
             float(time_weight) * time_loss +
             float(trajectory_weight) * trajectory_loss)

    top = selected_logits.argmax(-1) if len(selected_logits) else torch.empty(
        0, dtype=torch.long, device=logits.device)
    hit = (selected_targets.gather(1, top[:, None]).squeeze(1) > 0
           if len(selected_logits) else torch.empty(0, dtype=torch.bool, device=logits.device))
    if stable_index >= 0:
        nonstable = selected_targets.sum(-1).bool() & ~selected_targets[:, stable_index].bool()
        stable_fraction = (top == stable_index).float().mean() if len(top) else zero.detach()
    else:
        nonstable = torch.ones(len(selected_targets), dtype=torch.bool, device=logits.device)
        stable_fraction = zero.detach()
    nonstable_accuracy = hit[nonstable].float().mean() if nonstable.any() else hit.float().sum() * 0.0
    expected_wait = (torch.exp(mu + 0.5 * sigma.square()).clamp_max(max_wait_hours)
                     if time_mu is not None and time_log_sigma is not None
                     else torch.exp(-log_rate))
    event_wait_mae = ((expected_wait[is_event] - dt[is_event]).abs().mean()
                      if is_event.any() else zero.detach())
    return {
        "loss": total,
        "event_loss": event_loss,
        "family_loss": family_loss,
        "time_loss": time_loss,
        "trajectory_loss": trajectory_loss,
        "event_accuracy": hit.float().mean().detach() if len(hit) else zero.detach(),
        "nonstable_accuracy": nonstable_accuracy.detach(),
        "stable_prediction_fraction": stable_fraction.detach(),
        "time_mae_hours": event_wait_mae.detach(),
        "num_targets": event_valid.sum().detach(),
        "num_time_targets": time_valid.sum().detach(),
        "num_time_mae_targets": is_event.sum().detach(),
        "num_nonstable_targets": nonstable.sum().detach(),
    }


def masked_event_loss(token_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross entropy over positions selected for bidirectional event masking."""
    valid = labels >= 0
    if not valid.any():
        return token_logits.sum() * 0.0
    return F.cross_entropy(token_logits[valid], labels[valid])


def masked_value_loss(value_prediction: torch.Tensor, labels: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
    """Robustly reconstruct signed-log clinical values hidden from the encoder."""
    if not mask.any():
        return value_prediction.sum() * 0.0
    target = torch.sign(labels[mask]) * torch.log1p(labels[mask].abs())
    target = target.clamp(-12.0, 12.0)
    return F.smooth_l1_loss(value_prediction[mask], target)

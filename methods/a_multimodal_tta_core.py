from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import logging

import torch
import torch.nn as nn


DEFAULT_UPDATE_PATTERNS = [
    "prompt_learner",
    "adapter",
    "gate",
    "ln",
    "bn",
    "norm",
]


def safe_cfg_get(cfg, node_name: str, key: str, default):
    if hasattr(cfg, node_name):
        node = getattr(cfg, node_name)
        if hasattr(node, key):
            return getattr(node, key)
    return default


def safe_cfg_get_list(cfg, node_name: str, key: str, default: Sequence[str]) -> List[str]:
    value = safe_cfg_get(cfg, node_name, key, None)
    if value is None:
        return list(default)
    return list(value)


def forward_logits(model: nn.Module, imgs: torch.Tensor) -> torch.Tensor:
    outputs = model(imgs)
    if isinstance(outputs, dict):
        if "logits" in outputs:
            return outputs["logits"]
        if "output" in outputs:
            return outputs["output"]
        raise ValueError("Model returned dict but no 'logits' / 'output' key was found.")
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


def configure_model_for_tta(model: nn.Module) -> None:
    model.eval()
    model.requires_grad_(False)
    found_norm = False
    for m in model.modules():
        cls_name = m.__class__.__name__
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
            found_norm = True
        elif isinstance(m, nn.BatchNorm1d):
            m.train()
            found_norm = True
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            # make sure layernorm/groupnorm are left in a sensible state
            try:
                m.train()
            except Exception:
                pass
            found_norm = True
        else:
            # fallback: some models use custom norm implementations not matching torch types
            if cls_name.endswith('Norm') or 'LayerNorm' in cls_name or 'GroupNorm' in cls_name or 'BatchNorm' in cls_name:
                try:
                    if hasattr(m, 'track_running_stats'):
                        m.track_running_stats = False
                    if hasattr(m, 'running_mean'):
                        m.running_mean = None
                    if hasattr(m, 'running_var'):
                        m.running_var = None
                    try:
                        m.train()
                    except Exception:
                        pass
                except Exception:
                    pass
                found_norm = True

    if not found_norm:
        logging.getLogger(__name__).warning(
            "configure_model_for_tta: no standard normalization layers found; if using custom Norms, consider adding class-name fallback"
        )


def collect_adapt_params(
    model: nn.Module,
    update_patterns: Optional[Sequence[str]] = None,
) -> Tuple[List[nn.Parameter], List[str]]:
    params: List[nn.Parameter] = []
    names: List[str] = []

    patterns = [p.lower() for p in (update_patterns or []) if p]

    for _, p in model.named_parameters():
        p.requires_grad_(False)

    if patterns:
        for name, p in model.named_parameters():
            lname = name.lower()
            if any(key in lname for key in patterns):
                p.requires_grad_(True)
                params.append(p)
                names.append(name)

    if len(params) == 0:
        # try standard norm classes first
        for nm, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters(recurse=False):
                    if np in ["weight", "bias"]:
                        p.requires_grad_(True)
                        params.append(p)
                        names.append(f"{nm}.{np}")

    if len(params) == 0:
        # fallback: heuristic based on module class name (handles custom norm layers)
        logging.getLogger(__name__).warning(
            "collect_adapt_params: no standard norm params found; falling back to class-name heuristic"
        )
        for nm, m in model.named_modules():
            cls_name = m.__class__.__name__
            if cls_name.endswith('Norm') or 'LayerNorm' in cls_name or 'GroupNorm' in cls_name or 'BatchNorm' in cls_name:
                for np, p in m.named_parameters(recurse=False):
                    if np in ["weight", "bias"]:
                        p.requires_grad_(True)
                        params.append(p)
                        names.append(f"{nm}.{np}")

    return params, names


def softmax_entropy_from_logits(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return -(probs * (probs.clamp_min(eps).log())).sum(dim=-1)


def batch_diversity_loss_from_logits(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    marginal = torch.softmax(logits, dim=-1).mean(dim=0, keepdim=True)
    return (marginal * marginal.clamp_min(eps).log()).sum(dim=-1).mean()


@dataclass
class AdaptResult:
    logits: torch.Tensor
    probs: torch.Tensor
    entropy: torch.Tensor
    alpha: torch.Tensor
    rho_v: torch.Tensor
    rho_t: torch.Tensor
    conflict: torch.Tensor
    js_div: torch.Tensor
    rank_disagreement: torch.Tensor
    prior: torch.Tensor
    z_v: torch.Tensor
    z_t: torch.Tensor
    z_fuse: torch.Tensor


class Multimodal_TTA_Core(nn.Module):
    def __init__(
        self,
        num_classes: int,
        gate_hidden_dim: int = 8,
        gate_granularity: str = "batch",
        fixed_alpha: float = -1.0,
        anchor_conf_threshold: float = 0.7,
        anchor_momentum: float = 0.9,
        reliability_tau: float = 5.0,
        kappa_rank_weight: float = 0.5,
        reliability_conflict_weight: float = 0.0,
        use_entropy_feature: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.gate_hidden_dim = int(gate_hidden_dim)
        self.gate_granularity = str(gate_granularity)
        self.fixed_alpha = float(fixed_alpha)
        self.anchor_conf_threshold = float(anchor_conf_threshold)
        self.anchor_momentum = float(anchor_momentum)
        self.reliability_tau = float(reliability_tau)
        self.kappa_rank_weight = float(kappa_rank_weight)
        self.reliability_conflict_weight = float(reliability_conflict_weight)
        self.use_entropy_feature = bool(use_entropy_feature)

        if self.gate_granularity not in {"batch", "sample"}:
            raise ValueError("gate_granularity must be one of {'batch', 'sample'}")

        gate_in_dim = 2 + (2 if self.use_entropy_feature else 0)
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, self.gate_hidden_dim),
            nn.GELU(),
            nn.Linear(self.gate_hidden_dim, 1),
        )

        uniform = 1.0 / max(self.num_classes, 1)
        self.register_buffer("anchor_v", torch.full((self.num_classes,), uniform))
        self.register_buffer("anchor_t", torch.full((self.num_classes,), uniform))
        self.register_buffer("anchor_v_ready", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("anchor_t_ready", torch.tensor(False, dtype=torch.bool))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.gate_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    @torch.no_grad()
    def reset_state(self) -> None:
        uniform = 1.0 / max(self.num_classes, 1)
        self.anchor_v.fill_(uniform)
        self.anchor_t.fill_(uniform)
        self.anchor_v_ready.fill_(False)
        self.anchor_t_ready.fill_(False)

    @staticmethod
    def _entropy_from_probs(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return -(probs * probs.clamp_min(eps).log()).sum(dim=-1)

    @staticmethod
    def _js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        m = 0.5 * (p + q)

        def _kl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            a = a.clamp_min(eps)
            b = b.clamp_min(eps)
            return (a * (a.log() - b.log())).sum(dim=-1)

        return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

    @staticmethod
    def _rank_disagreement(
        p: torch.Tensor, q: torch.Tensor, chunk_size: int = 128
    ) -> torch.Tensor:
        """Eq. (16): R = 2 * N_disc / (K * (K - 1)).

        N_disc counts unordered class pairs whose ordering differs between the
        two modalities.  Chunking keeps the exact definition practical for K=1000.
        """
        batch_size, num_classes = p.shape
        if num_classes <= 1:
            return torch.zeros(batch_size, device=p.device, dtype=p.dtype)

        n_disc = torch.zeros(batch_size, device=p.device, dtype=p.dtype)
        all_idx = torch.arange(num_classes, device=p.device)
        p_all = p.unsqueeze(1)
        q_all = q.unsqueeze(1)

        for i0 in range(0, num_classes, chunk_size):
            i1 = min(i0 + chunk_size, num_classes)
            pi = p[:, i0:i1].unsqueeze(-1)
            qi = q[:, i0:i1].unsqueeze(-1)
            discord = ((pi > p_all) & (qi < q_all)) | ((pi < p_all) & (qi > q_all))

            ii = torch.arange(i0, i1, device=p.device).view(-1, 1)
            jj = all_idx.view(1, -1)
            upper_triangle = (ii < jj).unsqueeze(0)
            n_disc += (discord & upper_triangle).sum(dim=(1, 2)).to(p.dtype)

        return 2.0 * n_disc / float(num_classes * (num_classes - 1))

    @staticmethod
    def _sorted_l1_to_anchor(probs: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        # Eq. (10): sorted L1 deviation from the running modality anchor.
        ps = torch.sort(probs, dim=-1, descending=True).values
        qs = torch.sort(anchor.detach(), dim=-1, descending=True).values.unsqueeze(0).expand_as(ps)
        return (ps - qs).abs().sum(dim=-1)

    @torch.no_grad()
    def _update_single_anchor(
        self,
        probs: torch.Tensor,
        anchor: torch.Tensor,
        ready_flag: torch.Tensor,
    ) -> None:
        # Eqs. (11)-(13): confidence filter -> batch posterior mean -> EMA anchor.
        conf = probs.max(dim=-1).values
        mask = conf >= self.anchor_conf_threshold
        if not bool(mask.any()):
            return

        batch_anchor = probs[mask].mean(dim=0)
        batch_anchor = batch_anchor / batch_anchor.sum().clamp_min(1e-8)

        if not bool(ready_flag.item()):
            anchor.copy_(batch_anchor)
            ready_flag.fill_(True)
        else:
            new_anchor = self.anchor_momentum * anchor + (1.0 - self.anchor_momentum) * batch_anchor
            new_anchor = new_anchor / new_anchor.sum().clamp_min(1e-8)
            anchor.copy_(new_anchor)

    @torch.no_grad()
    def update_anchors(self, z_v: torch.Tensor, z_t: torch.Tensor) -> None:
        p_v = torch.softmax(z_v, dim=-1)
        p_t = torch.softmax(z_t, dim=-1)
        self._update_single_anchor(p_v, self.anchor_v, self.anchor_v_ready)
        self._update_single_anchor(p_t, self.anchor_t, self.anchor_t_ready)

    # Eqs. (18)-(20): reliability-aware gate prior.
    def _compute_prior(
        self,
        rho_v: torch.Tensor,
        rho_t: torch.Tensor,
        kappa: torch.Tensor,
    ) -> torch.Tensor:
        logits_v = -self.reliability_tau * rho_v
        logits_t = -self.reliability_tau * rho_t

        if self.reliability_conflict_weight > 0.0:
            direction = torch.sign(rho_v - rho_t)
            logits_v = logits_v - self.reliability_conflict_weight * kappa * direction
            logits_t = logits_t + self.reliability_conflict_weight * kappa * direction

        return torch.softmax(torch.stack([logits_v, logits_t], dim=-1), dim=-1)

    def _compute_alpha(
        self,
        rho_v: torch.Tensor,
        rho_t: torch.Tensor,
        kappa: torch.Tensor,
        p_v: torch.Tensor,
        p_t: torch.Tensor,
    ) -> torch.Tensor:
        if self.fixed_alpha >= 0.0:
            return torch.full_like(rho_v, fill_value=self.fixed_alpha)

        features = [
            (rho_t - rho_v).detach().unsqueeze(-1),
            kappa.detach().unsqueeze(-1),
        ]
        if self.use_entropy_feature:
            features.append(self._entropy_from_probs(p_v).detach().unsqueeze(-1))
            features.append(self._entropy_from_probs(p_t).detach().unsqueeze(-1))
        gate_in = torch.cat(features, dim=-1)

        if self.gate_granularity == "batch":
            gate_logits = self.gate_mlp(gate_in.mean(dim=0, keepdim=True)).squeeze(-1)
            alpha = torch.sigmoid(gate_logits)
            return alpha.expand_as(rho_v)

        gate_logits = self.gate_mlp(gate_in).squeeze(-1)
        return torch.sigmoid(gate_logits)

    def forward(self, z_v: torch.Tensor, z_t: torch.Tensor) -> AdaptResult:
        p_v = torch.softmax(z_v, dim=-1)
        p_t = torch.softmax(z_t, dim=-1)

        rho_v = self._sorted_l1_to_anchor(p_v, self.anchor_v)
        rho_t = self._sorted_l1_to_anchor(p_t, self.anchor_t)

        js_div = self._js_divergence(p_v, p_t)
        rank_dis = self._rank_disagreement(p_v, p_t)
        # Eq. (17): JS probability disagreement + weighted ranking disagreement.
        kappa = js_div + self.kappa_rank_weight * rank_dis

        prior = self._compute_prior(rho_v, rho_t, kappa)
        alpha = self._compute_alpha(rho_v, rho_t, kappa, p_v, p_t)

        # Eqs. (2)-(3): deployed logit-level fusion followed by softmax.
        z_fuse = alpha.unsqueeze(-1) * z_v + (1.0 - alpha.unsqueeze(-1)) * z_t
        p_fuse = torch.softmax(z_fuse, dim=-1)
        ent = self._entropy_from_probs(p_fuse)

        return AdaptResult(
            logits=z_fuse,
            probs=p_fuse,
            entropy=ent,
            alpha=alpha,
            rho_v=rho_v,
            rho_t=rho_t,
            conflict=kappa,
            js_div=js_div,
            rank_disagreement=rank_dis,
            prior=prior,
            z_v=z_v,
            z_t=z_t,
            z_fuse=z_fuse,
        )

    # Eqs. (21)-(23): entropy + gate KL + batch-marginal diversity.
    def compute_mgmtta_loss(
        self,
        result: AdaptResult,
        lambda_gate: float,
        lambda_div: float,
    ) -> Tuple[torch.Tensor, dict]:
        loss_ent = result.entropy.mean()

        gate_probs = torch.stack([result.alpha, 1.0 - result.alpha], dim=-1)
        loss_gate = (
            gate_probs * (
                (gate_probs.clamp_min(1e-8).log()) -
                (result.prior.clamp_min(1e-8).log())
            )
        ).sum(dim=-1).mean()

        marginal = result.probs.mean(dim=0, keepdim=True)
        loss_div = (marginal * marginal.clamp_min(1e-8).log()).sum(dim=-1).mean()

        total = loss_ent + float(lambda_gate) * loss_gate + float(lambda_div) * loss_div
        return total, {
            "loss_ent": float(loss_ent.detach().item()),
            "loss_gate": float(loss_gate.detach().item()),
            "loss_div": float(loss_div.detach().item()),
        }

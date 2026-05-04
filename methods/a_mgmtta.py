from copy import deepcopy

import logging
import torch

from methods.base import TTAMethod


# module-level logger
logger = logging.getLogger(__name__)
from methods.a_multimodal_tta_core import (
    DEFAULT_UPDATE_PATTERNS,
    Multimodal_TTA_Core,
    collect_adapt_params,
    configure_model_for_tta,
    safe_cfg_get,
    safe_cfg_get_list,
)
from utils.registry import ADAPTATION_REGISTRY


@ADAPTATION_REGISTRY.register()
class MG_MTTA(TTAMethod):
    """
    Majorization-Guided Multimodal Test-Time Adaptation.

    The text-shift generation stays OUTSIDE this file. This method only consumes:
      - clean text bank from model.get_clean_text_bank()
      - shifted text bank from model.get_shifted_text_bank()
    """
    def __init__(self, cfg, model, num_classes):
        self.update_patterns = safe_cfg_get_list(
            cfg, "MGMTTA", "UPDATE_PATTERNS", DEFAULT_UPDATE_PATTERNS
        )
        self.lambda_gate = float(safe_cfg_get(cfg, "MGMTTA", "LAMBDA_GATE", 0.1))
        self.lambda_div = float(safe_cfg_get(cfg, "MGMTTA", "LAMBDA_DIV", 0.01))
        self.gate_hidden_dim = int(safe_cfg_get(cfg, "MGMTTA", "GATE_HIDDEN_DIM", 8))
        self.gate_granularity = str(safe_cfg_get(cfg, "MGMTTA", "GATE_GRANULARITY", "batch"))
        self.fixed_alpha = float(safe_cfg_get(cfg, "MGMTTA", "FIXED_ALPHA", -1.0))
        self.anchor_conf_threshold = float(safe_cfg_get(cfg, "MGMTTA", "ANCHOR_CONF_THRESHOLD", 0.7))
        self.anchor_momentum = float(safe_cfg_get(cfg, "MGMTTA", "ANCHOR_MOMENTUM", 0.9))
        self.reliability_tau = float(safe_cfg_get(cfg, "MGMTTA", "RELIABILITY_TAU", 5.0))
        self.kappa_rank_weight = float(safe_cfg_get(cfg, "MGMTTA", "KAPPA_RANK_WEIGHT", 0.5))
        self.reliability_conflict_weight = float(
            safe_cfg_get(cfg, "MGMTTA", "RELIABILITY_CONFLICT_WEIGHT", 0.0)
        )
        self.use_entropy_feature = bool(safe_cfg_get(cfg, "MGMTTA", "USE_ENTROPY_FEATURE", True))
        super().__init__(cfg, model, num_classes)
        self._prepare_text_banks()

    def configure_model(self):
        configure_model_for_tta(self.model)
        self.mm_core = Multimodal_TTA_Core(
            num_classes=self.num_classes,
            gate_hidden_dim=self.gate_hidden_dim,
            gate_granularity=self.gate_granularity,
            fixed_alpha=self.fixed_alpha,
            anchor_conf_threshold=self.anchor_conf_threshold,
            anchor_momentum=self.anchor_momentum,
            reliability_tau=self.reliability_tau,
            kappa_rank_weight=self.kappa_rank_weight,
            reliability_conflict_weight=self.reliability_conflict_weight,
            use_entropy_feature=self.use_entropy_feature,
        ).to(self.device)

    def collect_params(self):
        model_params, model_names = collect_adapt_params(self.model, self.update_patterns)
        core_params = list(self.mm_core.parameters())
        core_names = [f"mm_core.{name}" for name, _ in self.mm_core.named_parameters()]
        # keep separate lists for creating parameter groups (e.g., higher LR for core)
        self.model_params = model_params
        self.core_params = core_params
        return model_params + core_params, model_names + core_names

    def setup_optimizer(self):
        # create optimizer with separate parameter groups: model_params @ base LR, core_params @ scaled LR
        base_lr = float(self.cfg.OPTIM.LR)
        method = getattr(self.cfg.OPTIM, "METHOD", "Adam")
        core_lr_mult = float(safe_cfg_get(self.cfg, "MGMTTA", "CORE_LR_MULT", 10.0))
        core_lr = base_lr * core_lr_mult

        param_groups = []
        if hasattr(self, "model_params") and len(self.model_params) > 0:
            param_groups.append({"params": self.model_params, "lr": base_lr})
        if hasattr(self, "core_params") and len(self.core_params) > 0:
            param_groups.append({"params": self.core_params, "lr": core_lr})

        if len(param_groups) == 0:
            return None

        if method == 'Adam':
            return torch.optim.Adam(param_groups, lr=base_lr, betas=(self.cfg.OPTIM.BETA, 0.999), weight_decay=self.cfg.OPTIM.WD)
        elif method == 'AdamW':
            return torch.optim.AdamW(param_groups, lr=base_lr, betas=(self.cfg.OPTIM.BETA, 0.999), weight_decay=self.cfg.OPTIM.WD)
        elif method == 'SGD':
            return torch.optim.SGD(param_groups, lr=base_lr, momentum=self.cfg.OPTIM.MOMENTUM, dampening=self.cfg.OPTIM.DAMPENING, weight_decay=self.cfg.OPTIM.WD, nesterov=self.cfg.OPTIM.NESTEROV)
        else:
            raise NotImplementedError

    def copy_model_and_optimizer(self):
        model_states = {
            "model": deepcopy(self.model.state_dict()),
            "mm_core": deepcopy(self.mm_core.state_dict()),
        }
        optimizer_state = deepcopy(self.optimizer.state_dict()) if self.optimizer is not None else None
        return model_states, optimizer_state

    def load_model_and_optimizer(self):
        self.model.load_state_dict(self.model_states["model"], strict=True)
        self.mm_core.load_state_dict(self.model_states["mm_core"], strict=True)
        if self.optimizer is not None and self.optimizer_state is not None:
            self.optimizer.load_state_dict(self.optimizer_state)

    def reset(self):
        self.load_model_and_optimizer()
        self._prepare_text_banks()

    def _prepare_text_banks(self):
        if not hasattr(self.model, "get_clean_text_bank"):
            raise RuntimeError(
                "MG_MTTA requires ZeroShotCLIP to expose get_clean_text_bank()/get_shifted_text_bank()."
            )
        # Keep references to the model's text banks. Do not force a .detach() here
        # so that downstream code can choose a gradient-enabled logits path when needed.
        self.text_features_v = self.model.get_clean_text_bank()
        self.text_features_t = self.model.get_shifted_text_bank()

    def _extract_stream_logits(self, imgs_test):
        if not hasattr(self.model, "logits_with_text_bank"):
            raise RuntimeError(
                "MG_MTTA requires ZeroShotCLIP.logits_with_text_bank(). Please update models/model.py first."
            )
        # Use the gradient-capable logits helper when available so adaptation
        # components (e.g., mm_core) can receive gradients.
        if hasattr(self.model, "logits_with_text_bank_grad"):
            z_v, _, _ = self.model.logits_with_text_bank_grad(imgs_test, self.text_features_v)
            z_t, _, _ = self.model.logits_with_text_bank_grad(imgs_test, self.text_features_t)
        else:
            z_v, _, _ = self.model.logits_with_text_bank(imgs_test, self.text_features_v)
            z_t, _, _ = self.model.logits_with_text_bank(imgs_test, self.text_features_t)
        return z_v, z_t

    def loss_calculation(self, x):
        imgs_test = x[0]
        z_v, z_t = self._extract_stream_logits(imgs_test)
        result = self.mm_core(z_v, z_t)
        loss, loss_dict = self.mm_core.compute_mgmtta_loss(
            result, lambda_gate=self.lambda_gate, lambda_div=self.lambda_div
        )
        return result, loss, loss_dict

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        # imgs_test for re-evaluating features after the parameter update
        imgs_test = x[0] if isinstance(x, list) else x

        verbose_diag = bool(getattr(getattr(self.cfg, "LOG", None), "VERBOSE_DIAG", False))

        # diagnostics: inspect a few parameter norms
        num_check = min(5, len(self.params)) if hasattr(self, "params") else 0
        if verbose_diag:
            pre_param_norms = [float(p.detach().norm().item()) for p in (self.params[:num_check] if num_check > 0 else [])]
            logger.info(f"[MG_MTTA][DIAG] pre_param_norms={pre_param_norms}")

        if self.mixed_precision and self.device == "cuda":
            with torch.cuda.amp.autocast():
                result, loss, _ = self.loss_calculation(x)
            self.scaler.scale(loss).backward()
            if verbose_diag:
                grad_norms = [float(p.grad.detach().norm().item()) if (p.grad is not None) else 0.0 for p in (self.params[:num_check] if num_check > 0 else [])]
                logger.info(f"[MG_MTTA][DIAG] grad_norms_after_backward={grad_norms}")
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        else:
            result, loss, _ = self.loss_calculation(x)
            loss.backward()
            if verbose_diag:
                grad_norms = [float(p.grad.detach().norm().item()) if (p.grad is not None) else 0.0 for p in (self.params[:num_check] if num_check > 0 else [])]
                logger.info(f"[MG_MTTA][DIAG] grad_norms_after_backward={grad_norms}")
            self.optimizer.step()
            self.optimizer.zero_grad()

        if verbose_diag:
            post_param_norms = [float(p.detach().norm().item()) for p in (self.params[:num_check] if num_check > 0 else [])]
            logger.info(f"[MG_MTTA][DIAG] post_param_norms={post_param_norms}")

        # Recompute post-update stream features and mm_core result, then update anchors.
        # Keep returning pre-update logits to align pre/post metric semantics with other methods.
        with torch.no_grad():
            try:
                z_v_post, z_t_post = self._extract_stream_logits(imgs_test)
                result_post = self.mm_core(z_v_post, z_t_post)
                self.mm_core.update_anchors(z_v_post.detach(), z_t_post.detach())
            except Exception:
                logger.exception("[MG_MTTA][DIAG] failed to compute post-update stream logits; falling back to pre-update anchors")
                result_post = result
                self.mm_core.update_anchors(result.z_v.detach(), result.z_t.detach())

        # preserve both pre/post results for diagnostics
        self.last_result = result
        self.last_result_post = result_post
        return result.logits.detach()

    @torch.no_grad()
    def get_last_result(self):
        return getattr(self, "last_result", None)

    @torch.no_grad()
    def infer_post_result(self, imgs_test):
        z_v, z_t = self._extract_stream_logits(imgs_test)
        return self.mm_core(z_v, z_t)

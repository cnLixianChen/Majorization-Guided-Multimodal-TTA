from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


RUN_CONFIG_REQUIRED_FIELDS = [
    "run_id",
    "timestamp",
    "mode",
    "method",
    "dataset",
    "arch",
    "weights",
    "prompt_mode",
    "prompt_template",
    "text_shift_enabled",
    "text_shift_family",
    "text_shift_level",
    "text_shift_protocol",
    "batch_size",
    "lr",
    "steps",
    "wd",
    "seed",
    "device",
    "num_classes",
    "num_samples_total",
    "num_batches_total",
    "output_dir",
]

BATCH_SCHEMA_FIELDS = [
    "run_id",
    "mode",
    "method",
    "dataset",
    "arch",
    "shift_id",
    "shift_index",
    "corruption_name",
    "severity",
    "visual_degradation_tag",
    "is_text_shift",
    "text_shift_family",
    "text_shift_level",
    "text_shift_protocol",
    "batch_idx",
    "num_batches_in_shift",
    "global_sample_start",
    "global_sample_end",
    "batch_size_actual",
    "batch_acc",
    "entropy_pre_mean",
    "entropy_post_mean",
    "delta_entropy_mean",
    "top1_conf_pre_mean",
    "top1_conf_post_mean",
    "delta_top1_conf_mean",
    "pred_flip_rate",
    "wrong_to_right_rate",
    "right_to_wrong_rate",
    "majorization_ratio_post_over_pre",
    "majorization_improve_count",
    "majorization_valid_count",
    "closer_to_clean_partialsum_ratio",
    "dist_to_perm_pre_mean",
    "dist_to_perm_post_mean",
    "delta_dist_to_perm_mean",
    "batch_marginal_entropy",
    "batch_class_imbalance",
    "top1_class_ratio",
    "alpha_mean",
    "alpha_std",
    "alpha_min",
    "alpha_max",
    "alpha_gt_0_5_ratio",
    "alpha_lt_0_5_ratio",
    "rho_v_mean",
    "rho_t_mean",
    "rho_gap_mean",
    "conflict_mean",
    "js_div_mean",
    "rank_disagreement_mean",
    "prior_v_mean",
    "prior_t_mean",
    "gate_prior_kl_mean",
    "dominant_modality_rate",
    "low_entropy_wrong_count",
    "high_conflict_wrong_count",
    "alpha_threshold_cross_count",
    "right_to_wrong_count",
]

SHIFT_SCHEMA_FIELDS = [
    "run_id",
    "mode",
    "method",
    "dataset",
    "arch",
    "shift_id",
    "shift_index",
    "corruption_name",
    "severity",
    "visual_degradation_tag",
    "is_text_shift",
    "text_shift_family",
    "text_shift_level",
    "text_shift_protocol",
    "num_prompts_per_class",
    "clean_prompt_template",
    "shifted_prompt_template_preview",
    "acc",
    "err",
    "num_samples",
    "num_batches",
    "entropy_pre_mean_global",
    "entropy_post_mean_global",
    "delta_entropy_mean_global",
    "top1_conf_pre_mean_global",
    "top1_conf_post_mean_global",
    "delta_top1_conf_mean_global",
    "pred_flip_rate_global",
    "wrong_to_right_rate_global",
    "right_to_wrong_rate_global",
    "majorization_ratio_post_over_pre_global",
    "closer_to_clean_partialsum_ratio",
    "dist_to_perm_pre_mean_global",
    "dist_to_perm_post_mean_global",
    "delta_dist_to_perm_mean_global",
    "collapse_rate",
    "mean_batch_marginal_entropy",
    "alpha_mean_global",
    "alpha_std_global",
    "alpha_gt_0_5_ratio_global",
    "alpha_lt_0_5_ratio_global",
    "rho_v_mean_global",
    "rho_t_mean_global",
    "rho_gap_mean_global",
    "conflict_mean_global",
    "js_div_mean_global",
    "rank_disagreement_mean_global",
    "prior_v_mean_global",
    "prior_t_mean_global",
    "gate_prior_kl_mean_global",
    "dominant_modality_rate",
    "low_entropy_wrong_rate",
    "high_conflict_wrong_rate",
    "alpha_threshold_cross_rate",
    "right_to_wrong_cases_topk",
]

PREDICTION_REQUIRED_KEYS = [
    "sample_ids",
    "image_path",
    "gt_prompt",
    "shifted_prompt",
    "y_true",
    "pred_pre",
    "pred_post",
    "conf_pre",
    "conf_post",
    "entropy_pre",
    "entropy_post",
    "majorization_flag",
    "dist_to_perm_pre",
    "dist_to_perm_post",
    "alpha",
    "rho_v",
    "rho_t",
    "conflict",
    "js_div",
    "rank_disagreement",
    "q_pre_top3_idx",
    "q_pre_top3_val",
    "q_post_top3_idx",
    "q_post_top3_val",
    "p_v_top3_idx",
    "p_v_top3_val",
    "p_t_top3_idx",
    "p_t_top3_val",
]


WEIGHTED_SHIFT_KEYS = [
    "entropy_pre_mean",
    "entropy_post_mean",
    "delta_entropy_mean",
    "top1_conf_pre_mean",
    "top1_conf_post_mean",
    "delta_top1_conf_mean",
    "pred_flip_rate",
    "wrong_to_right_rate",
    "right_to_wrong_rate",
    "dist_to_perm_pre_mean",
    "dist_to_perm_post_mean",
    "delta_dist_to_perm_mean",
    "alpha_mean",
    "alpha_std",
    "alpha_gt_0_5_ratio",
    "alpha_lt_0_5_ratio",
    "rho_v_mean",
    "rho_t_mean",
    "rho_gap_mean",
    "conflict_mean",
    "js_div_mean",
    "rank_disagreement_mean",
    "prior_v_mean",
    "prior_t_mean",
    "gate_prior_kl_mean",
    "dominant_modality_rate",
]


def _json_safe_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if hasattr(value, "item"):
        try:
            return _json_safe_scalar(value.item())
        except Exception:
            return None
    if isinstance(value, (list, tuple)):
        return [_json_safe_scalar(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe_scalar(v) for k, v in value.items()}
    return str(value)


def _fill_schema(record: Dict[str, Any], schema: Iterable[str]) -> Dict[str, Any]:
    return {k: _json_safe_scalar(record.get(k, None)) for k in schema}


def _mean_or_none(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _weighted_average(sum_value: float, denom: float) -> Optional[float]:
    if denom <= 0:
        return None
    return float(sum_value / denom)


class UnifiedExperimentLogger:
    def __init__(
        self,
        cfg,
        mode: str,
        method: str,
        dataset: str,
        arch: str,
        weights: str,
        num_classes: int,
        device: str,
        trainable_params: Optional[int] = None,
    ) -> None:
        self.cfg = cfg
        self.mode = str(mode)
        self.method = str(method)
        self.dataset = str(dataset)
        self.arch = str(arch)
        self.weights = str(weights)
        self.num_classes = int(num_classes)
        self.device = str(device)
        self.trainable_params = trainable_params

        ts = getattr(cfg, "LOG_TIME", "") or datetime.now().strftime("%y%m%d_%H%M%S")
        self.run_id = f"{ts}_{self.mode}_{self.method}_{self.dataset}"
        self.timestamp = datetime.now().isoformat(timespec="seconds")

        output_root = getattr(getattr(cfg, "EXPERIMENT_OUTPUT", None), "ROOT", "./outputs")
        self.output_dir = os.path.join(output_root, self.dataset, self.mode, self.method, self.run_id)
        os.makedirs(self.output_dir, exist_ok=True)

        self.run_config_path = os.path.join(self.output_dir, "run_config.json")
        self.batch_jsonl_path = os.path.join(self.output_dir, "per_batch_metrics.jsonl")
        self.shift_csv_path = os.path.join(self.output_dir, "per_shift_summary.csv")
        self.shift_json_path = os.path.join(self.output_dir, "per_shift_summary.json")
        self.run_summary_path = os.path.join(self.output_dir, "run_summary.json")
        self.prompt_bank_path = os.path.join(self.output_dir, "prompt_bank.json")
        self.failure_cases_path = os.path.join(self.output_dir, "failure_cases.json")
        self.pred_npz_path = os.path.join(self.output_dir, f"predictions_{self.mode}_{self.method}.npz")

        self.shift_summaries: List[Dict[str, Any]] = []
        self.prompt_bank_payload: Optional[Dict[str, Any]] = None
        self.failure_cases: List[Dict[str, Any]] = []

        self._pred_store: Dict[str, List[np.ndarray]] = {k: [] for k in PREDICTION_REQUIRED_KEYS}
        self._current_shift_meta: Optional[Dict[str, Any]] = None
        self._current_shift_agg: Optional[Dict[str, Any]] = None

        with open(self.batch_jsonl_path, "w", encoding="utf-8"):
            pass

        self._run_weighted_sums = {k: 0.0 for k in WEIGHTED_SHIFT_KEYS}
        self._run_weighted_denoms = {k: 0.0 for k in WEIGHTED_SHIFT_KEYS}
        self._run_total_samples = 0
        self._run_total_batches = 0
        self._run_total_correct = 0.0
        self._run_total_majorization_improve = 0.0
        self._run_total_majorization_valid = 0.0
        self._run_total_collapse_batches = 0.0

        self._dump_run_config(num_samples_total=None, num_batches_total=None)

    def _dump_run_config(self, num_samples_total: Optional[int], num_batches_total: Optional[int]) -> None:
        collapse_top1_thr = getattr(getattr(self.cfg, "EXPERIMENT_OUTPUT", None), "COLLAPSE_TOP1_RATIO_THRESHOLD", 0.8)
        collapse_entropy_thr = getattr(getattr(self.cfg, "EXPERIMENT_OUTPUT", None), "COLLAPSE_MARGINAL_ENTROPY_THRESHOLD", -1.0)

        payload = {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "mode": self.mode,
            "method": self.method,
            "dataset": self.dataset,
            "arch": self.arch,
            "weights": self.weights,
            "prompt_mode": getattr(getattr(self.cfg, "CLIP", None), "PROMPT_MODE", None),
            "prompt_template": list(getattr(getattr(self.cfg, "CLIP", None), "PROMPT_TEMPLATE", [])),
            "text_shift_enabled": bool(getattr(getattr(self.cfg, "TEXT_SHIFT", None), "ENABLED", False)),
            "text_shift_family": getattr(getattr(self.cfg, "TEXT_SHIFT", None), "FAMILY", None),
            "text_shift_level": getattr(getattr(self.cfg, "TEXT_SHIFT", None), "LEVEL", None),
            "text_shift_protocol": getattr(getattr(self.cfg, "TEXT_SHIFT", None), "PROTOCOL", None),
            "batch_size": int(getattr(getattr(self.cfg, "TEST", None), "BATCH_SIZE", 0)),
            "lr": float(getattr(getattr(self.cfg, "OPTIM", None), "LR", 0.0)),
            "steps": int(getattr(getattr(self.cfg, "OPTIM", None), "STEPS", 1)),
            "wd": float(getattr(getattr(self.cfg, "OPTIM", None), "WD", 0.0)),
            "seed": int(getattr(self.cfg, "RNG_SEED", -1)),
            "device": self.device,
            "num_classes": self.num_classes,
            "num_samples_total": num_samples_total,
            "num_batches_total": num_batches_total,
            "output_dir": self.output_dir,
            "collapse_threshold_definition": {
                "top1_class_ratio_ge": float(collapse_top1_thr),
                "batch_marginal_entropy_le": float(collapse_entropy_thr),
            },
            "trainable_params": self.trainable_params,
        }
        payload = _fill_schema(payload, list(dict.fromkeys(RUN_CONFIG_REQUIRED_FIELDS + list(payload.keys()))))
        with open(self.run_config_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def set_prompt_bank(
        self,
        clean_prompt_bank: Optional[Dict[str, List[str]]],
        shifted_prompt_bank: Optional[Dict[str, List[str]]],
        class_names: Optional[List[str]] = None,
    ) -> None:
        if clean_prompt_bank is None and shifted_prompt_bank is None:
            return

        example_prompts = {}
        names = class_names or []
        if not names and clean_prompt_bank:
            names = list(clean_prompt_bank.keys())[:5]
        for name in names[:5]:
            clean_examples = (clean_prompt_bank or {}).get(name, [])[:2]
            shifted_examples = (shifted_prompt_bank or {}).get(name, [])[:2]
            example_prompts[name] = {
                "clean": clean_examples,
                "shifted": shifted_examples,
            }

        self.prompt_bank_payload = {
            "run_id": self.run_id,
            "mode": self.mode,
            "method": self.method,
            "dataset": self.dataset,
            "text_shift_family": getattr(getattr(self.cfg, "TEXT_SHIFT", None), "FAMILY", None),
            "text_shift_level": getattr(getattr(self.cfg, "TEXT_SHIFT", None), "LEVEL", None),
            "text_shift_protocol": getattr(getattr(self.cfg, "TEXT_SHIFT", None), "PROTOCOL", None),
            "class_names": class_names,
            "clean_prompt_bank": clean_prompt_bank,
            "shifted_prompt_bank": shifted_prompt_bank,
            "example_prompts_per_class": example_prompts,
        }

    def start_shift(self, shift_meta: Dict[str, Any], num_batches_in_shift: int) -> None:
        self._current_shift_meta = dict(shift_meta)
        self._current_shift_agg = {
            "num_samples": 0,
            "num_batches": 0,
            "num_correct": 0.0,
            "majorization_improve": 0.0,
            "majorization_valid": 0.0,
            "collapse_batches": 0.0,
            "batch_marginal_entropy_values": [],
            "failure_low_entropy_wrong": 0.0,
            "failure_high_conflict_wrong": 0.0,
            "failure_alpha_cross": 0.0,
            "failure_right_to_wrong": 0.0,
            "weighted_sums": {k: 0.0 for k in WEIGHTED_SHIFT_KEYS},
            "weighted_denoms": {k: 0.0 for k in WEIGHTED_SHIFT_KEYS},
            "num_batches_in_shift": int(num_batches_in_shift),
        }

    def _update_shift_weighted(self, key: str, value: Optional[float], weight: float) -> None:
        if self._current_shift_agg is None:
            return
        if value is None:
            return
        self._current_shift_agg["weighted_sums"][key] += float(value) * float(weight)
        self._current_shift_agg["weighted_denoms"][key] += float(weight)

    def log_batch_metrics(self, batch_record: Dict[str, Any], pred_artifacts: Optional[Dict[str, np.ndarray]] = None) -> None:
        rec = _fill_schema(batch_record, BATCH_SCHEMA_FIELDS)
        with open(self.batch_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if self._current_shift_agg is None:
            return

        bs = float(rec.get("batch_size_actual") or 0.0)
        self._current_shift_agg["num_samples"] += int(bs)
        self._current_shift_agg["num_batches"] += 1

        batch_acc = rec.get("batch_acc")
        if batch_acc is not None:
            self._current_shift_agg["num_correct"] += float(batch_acc) * bs

        for key in WEIGHTED_SHIFT_KEYS:
            self._update_shift_weighted(key, rec.get(key), bs)

        self._current_shift_agg["majorization_improve"] += float(rec.get("majorization_improve_count") or 0.0)
        self._current_shift_agg["majorization_valid"] += float(rec.get("majorization_valid_count") or 0.0)

        top1_ratio = rec.get("top1_class_ratio")
        bme = rec.get("batch_marginal_entropy")
        collapse = False
        if top1_ratio is not None and top1_ratio >= self._collapse_top1_threshold():
            collapse = True
        ent_thr = self._collapse_entropy_threshold()
        if bme is not None and ent_thr >= 0.0 and bme <= ent_thr:
            collapse = True
        self._current_shift_agg["collapse_batches"] += 1.0 if collapse else 0.0

        if bme is not None:
            self._current_shift_agg["batch_marginal_entropy_values"].append(float(bme))

        self._current_shift_agg["failure_low_entropy_wrong"] += float(rec.get("low_entropy_wrong_count") or 0.0)
        self._current_shift_agg["failure_high_conflict_wrong"] += float(rec.get("high_conflict_wrong_count") or 0.0)
        self._current_shift_agg["failure_alpha_cross"] += float(rec.get("alpha_threshold_cross_count") or 0.0)
        self._current_shift_agg["failure_right_to_wrong"] += float(rec.get("right_to_wrong_count") or 0.0)

        if pred_artifacts is not None:
            for k in PREDICTION_REQUIRED_KEYS:
                arr = pred_artifacts.get(k)
                if arr is None:
                    continue
                self._pred_store[k].append(np.asarray(arr))

    def _collapse_top1_threshold(self) -> float:
        return float(getattr(getattr(self.cfg, "EXPERIMENT_OUTPUT", None), "COLLAPSE_TOP1_RATIO_THRESHOLD", 0.8))

    def _collapse_entropy_threshold(self) -> float:
        return float(getattr(getattr(self.cfg, "EXPERIMENT_OUTPUT", None), "COLLAPSE_MARGINAL_ENTROPY_THRESHOLD", -1.0))

    def add_failure_cases(self, cases: List[Dict[str, Any]]) -> None:
        if not cases:
            return
        for c in cases:
            self.failure_cases.append(_json_safe_scalar(c))

    def finalize_shift_summary(self, right_to_wrong_cases_topk: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        if self._current_shift_meta is None or self._current_shift_agg is None:
            raise RuntimeError("No active shift to finalize.")

        agg = self._current_shift_agg
        n = float(agg["num_samples"])
        b = float(agg["num_batches"])

        summary = {
            **self._current_shift_meta,
            "run_id": self.run_id,
            "mode": self.mode,
            "method": self.method,
            "dataset": self.dataset,
            "arch": self.arch,
            "acc": _weighted_average(agg["num_correct"], n),
            "err": None,
            "num_samples": int(agg["num_samples"]),
            "num_batches": int(agg["num_batches"]),
            "majorization_ratio_post_over_pre_global": _weighted_average(agg["majorization_improve"], agg["majorization_valid"]),
            "closer_to_clean_partialsum_ratio": None,
            "collapse_rate": _weighted_average(agg["collapse_batches"], b),
            "mean_batch_marginal_entropy": _mean_or_none(agg["batch_marginal_entropy_values"]),
            "right_to_wrong_cases_topk": right_to_wrong_cases_topk if right_to_wrong_cases_topk is not None else [],
            "low_entropy_wrong_rate": _weighted_average(agg["failure_low_entropy_wrong"], n),
            "high_conflict_wrong_rate": _weighted_average(agg["failure_high_conflict_wrong"], n),
            "alpha_threshold_cross_rate": _weighted_average(agg["failure_alpha_cross"], n),
        }

        for key in WEIGHTED_SHIFT_KEYS:
            summary[f"{key}_global"] = _weighted_average(agg["weighted_sums"][key], agg["weighted_denoms"][key])

        summary["entropy_pre_mean_global"] = summary.pop("entropy_pre_mean_global", summary.get("entropy_pre_mean_global"))
        summary["entropy_post_mean_global"] = summary.pop("entropy_post_mean_global", summary.get("entropy_post_mean_global"))
        summary["delta_entropy_mean_global"] = summary.pop("delta_entropy_mean_global", summary.get("delta_entropy_mean_global"))
        summary["top1_conf_pre_mean_global"] = summary.pop("top1_conf_pre_mean_global", summary.get("top1_conf_pre_mean_global"))
        summary["top1_conf_post_mean_global"] = summary.pop("top1_conf_post_mean_global", summary.get("top1_conf_post_mean_global"))
        summary["delta_top1_conf_mean_global"] = summary.pop("delta_top1_conf_mean_global", summary.get("delta_top1_conf_mean_global"))
        summary["pred_flip_rate_global"] = summary.pop("pred_flip_rate_global", summary.get("pred_flip_rate_global"))
        summary["wrong_to_right_rate_global"] = summary.pop("wrong_to_right_rate_global", summary.get("wrong_to_right_rate_global"))
        summary["right_to_wrong_rate_global"] = summary.pop("right_to_wrong_rate_global", summary.get("right_to_wrong_rate_global"))
        summary["dist_to_perm_pre_mean_global"] = summary.pop("dist_to_perm_pre_mean_global", summary.get("dist_to_perm_pre_mean_global"))
        summary["dist_to_perm_post_mean_global"] = summary.pop("dist_to_perm_post_mean_global", summary.get("dist_to_perm_post_mean_global"))
        summary["delta_dist_to_perm_mean_global"] = summary.pop("delta_dist_to_perm_mean_global", summary.get("delta_dist_to_perm_mean_global"))

        summary["alpha_mean_global"] = summary.pop("alpha_mean_global", summary.get("alpha_mean_global"))
        summary["alpha_std_global"] = summary.pop("alpha_std_global", summary.get("alpha_std_global"))
        summary["alpha_gt_0_5_ratio_global"] = summary.pop("alpha_gt_0_5_ratio_global", summary.get("alpha_gt_0_5_ratio_global"))
        summary["alpha_lt_0_5_ratio_global"] = summary.pop("alpha_lt_0_5_ratio_global", summary.get("alpha_lt_0_5_ratio_global"))
        summary["rho_v_mean_global"] = summary.pop("rho_v_mean_global", summary.get("rho_v_mean_global"))
        summary["rho_t_mean_global"] = summary.pop("rho_t_mean_global", summary.get("rho_t_mean_global"))
        summary["rho_gap_mean_global"] = summary.pop("rho_gap_mean_global", summary.get("rho_gap_mean_global"))
        summary["conflict_mean_global"] = summary.pop("conflict_mean_global", summary.get("conflict_mean_global"))
        summary["js_div_mean_global"] = summary.pop("js_div_mean_global", summary.get("js_div_mean_global"))
        summary["rank_disagreement_mean_global"] = summary.pop("rank_disagreement_mean_global", summary.get("rank_disagreement_mean_global"))
        summary["prior_v_mean_global"] = summary.pop("prior_v_mean_global", summary.get("prior_v_mean_global"))
        summary["prior_t_mean_global"] = summary.pop("prior_t_mean_global", summary.get("prior_t_mean_global"))
        summary["gate_prior_kl_mean_global"] = summary.pop("gate_prior_kl_mean_global", summary.get("gate_prior_kl_mean_global"))
        summary["dominant_modality_rate"] = summary.get("dominant_modality_rate_global", summary.get("dominant_modality_rate"))

        if summary["acc"] is not None:
            summary["err"] = float(1.0 - summary["acc"])

        summary = _fill_schema(summary, SHIFT_SCHEMA_FIELDS)
        self.shift_summaries.append(summary)
        self._write_shift_summaries()

        self._run_total_samples += int(summary.get("num_samples") or 0)
        self._run_total_batches += int(summary.get("num_batches") or 0)
        if summary.get("acc") is not None and summary.get("num_samples") is not None:
            self._run_total_correct += float(summary["acc"]) * float(summary["num_samples"])

        self._run_total_majorization_improve += float(agg["majorization_improve"])
        self._run_total_majorization_valid += float(agg["majorization_valid"])
        self._run_total_collapse_batches += float(agg["collapse_batches"])

        for key in WEIGHTED_SHIFT_KEYS:
            denom = agg["weighted_denoms"][key]
            if denom > 0:
                self._run_weighted_sums[key] += float(agg["weighted_sums"][key])
                self._run_weighted_denoms[key] += float(denom)

        self._current_shift_meta = None
        self._current_shift_agg = None
        return summary

    def _write_shift_summaries(self) -> None:
        with open(self.shift_json_path, "w", encoding="utf-8") as f:
            json.dump(self.shift_summaries, f, indent=2, ensure_ascii=False)

        with open(self.shift_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SHIFT_SCHEMA_FIELDS)
            writer.writeheader()
            for rec in self.shift_summaries:
                row = {}
                for k in SHIFT_SCHEMA_FIELDS:
                    v = rec.get(k, None)
                    if isinstance(v, list):
                        row[k] = json.dumps(v, ensure_ascii=False)
                    elif v is None:
                        row[k] = ""
                    else:
                        row[k] = v
                writer.writerow(row)

    def save_prediction_artifacts(self) -> None:
        payload = {}
        empty_dtype = {
            "sample_ids": np.int64,
            "image_path": np.str_,
            "gt_prompt": np.str_,
            "shifted_prompt": np.str_,
            "y_true": np.int64,
            "pred_pre": np.int64,
            "pred_post": np.int64,
            "majorization_flag": np.int32,
            "q_pre_top3_idx": np.int64,
            "q_post_top3_idx": np.int64,
            "p_v_top3_idx": np.int64,
            "p_t_top3_idx": np.int64,
        }
        for k in PREDICTION_REQUIRED_KEYS:
            chunks = self._pred_store.get(k, [])
            if len(chunks) == 0:
                payload[k] = np.array([], dtype=empty_dtype.get(k, np.float32))
                continue
            payload[k] = np.concatenate(chunks, axis=0)
        np.savez_compressed(self.pred_npz_path, **payload)

    def finalize_run_summary(self, top_k: int = 5) -> Dict[str, Any]:
        self.save_prediction_artifacts()

        if self.prompt_bank_payload is not None and self.mode in {"text_shift", "joint_shift", "failure"}:
            with open(self.prompt_bank_path, "w", encoding="utf-8") as f:
                json.dump(_json_safe_scalar(self.prompt_bank_payload), f, indent=2, ensure_ascii=False)

        if self.mode == "failure":
            with open(self.failure_cases_path, "w", encoding="utf-8") as f:
                json.dump(_json_safe_scalar(self.failure_cases), f, indent=2, ensure_ascii=False)

        global_mean_acc = _weighted_average(self._run_total_correct, float(self._run_total_samples))
        global_mean_err = None if global_mean_acc is None else float(1.0 - global_mean_acc)

        per_shift_type = {
            "by_corruption": {},
            "by_text_family_level": {},
        }
        by_corr = {}
        by_text = {}
        for rec in self.shift_summaries:
            corr = str(rec.get("corruption_name") or "none")
            if corr not in by_corr:
                by_corr[corr] = {"sum": 0.0, "n": 0.0}
            if rec.get("acc") is not None and rec.get("num_samples") is not None:
                by_corr[corr]["sum"] += float(rec["acc"]) * float(rec["num_samples"])
                by_corr[corr]["n"] += float(rec["num_samples"])

            tfam = str(rec.get("text_shift_family") or "none")
            tlev = str(rec.get("text_shift_level") if rec.get("text_shift_level") is not None else "none")
            tk = f"{tfam}:{tlev}"
            if tk not in by_text:
                by_text[tk] = {"sum": 0.0, "n": 0.0}
            if rec.get("acc") is not None and rec.get("num_samples") is not None:
                by_text[tk]["sum"] += float(rec["acc"]) * float(rec["num_samples"])
                by_text[tk]["n"] += float(rec["num_samples"])

        for k, v in by_corr.items():
            per_shift_type["by_corruption"][k] = _weighted_average(v["sum"], v["n"])
        for k, v in by_text.items():
            per_shift_type["by_text_family_level"][k] = _weighted_average(v["sum"], v["n"])

        sorted_shifts = sorted(
            self.shift_summaries,
            key=lambda r: -1.0 if r.get("acc") is None else float(r["acc"]),
            reverse=True,
        )
        top_best = [
            {
                "shift_id": rec.get("shift_id"),
                "acc": rec.get("acc"),
                "num_samples": rec.get("num_samples"),
            }
            for rec in sorted_shifts[:top_k]
        ]
        top_worst = [
            {
                "shift_id": rec.get("shift_id"),
                "acc": rec.get("acc"),
                "num_samples": rec.get("num_samples"),
            }
            for rec in sorted_shifts[-top_k:]
        ]

        run_summary = {
            "run_id": self.run_id,
            "mode": self.mode,
            "method": self.method,
            "dataset": self.dataset,
            "global_mean_acc": global_mean_acc,
            "global_mean_err": global_mean_err,
            "mean_entropy_delta": _weighted_average(
                self._run_weighted_sums["delta_entropy_mean"],
                self._run_weighted_denoms["delta_entropy_mean"],
            ),
            "mean_majorization_ratio": _weighted_average(
                self._run_total_majorization_improve,
                self._run_total_majorization_valid,
            ),
            "mean_dist_to_perm_delta": _weighted_average(
                self._run_weighted_sums["delta_dist_to_perm_mean"],
                self._run_weighted_denoms["delta_dist_to_perm_mean"],
            ),
            "mean_collapse_rate": _weighted_average(
                self._run_total_collapse_batches,
                float(self._run_total_batches),
            ),
            "per-shift-type means": _json_safe_scalar(per_shift_type),
            "top_k_best_shifts": _json_safe_scalar(top_best),
            "top_k_worst_shifts": _json_safe_scalar(top_worst),
            "total_num_samples": int(self._run_total_samples),
            "total_num_batches": int(self._run_total_batches),
        }

        with open(self.run_summary_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe_scalar(run_summary), f, indent=2, ensure_ascii=False)

        self._dump_run_config(
            num_samples_total=int(self._run_total_samples),
            num_batches_total=int(self._run_total_batches),
        )
        return run_summary


def init_run_logger(cfg, mode: str, method: str, dataset: str, arch: str, weights: str, num_classes: int, device: str, trainable_params: Optional[int] = None) -> UnifiedExperimentLogger:
    return UnifiedExperimentLogger(
        cfg=cfg,
        mode=mode,
        method=method,
        dataset=dataset,
        arch=arch,
        weights=weights,
        num_classes=num_classes,
        device=device,
        trainable_params=trainable_params,
    )


def log_batch_metrics(run_logger: UnifiedExperimentLogger, batch_record: Dict[str, Any], pred_artifacts: Optional[Dict[str, np.ndarray]] = None) -> None:
    run_logger.log_batch_metrics(batch_record=batch_record, pred_artifacts=pred_artifacts)


def finalize_shift_summary(run_logger: UnifiedExperimentLogger, right_to_wrong_cases_topk: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    return run_logger.finalize_shift_summary(right_to_wrong_cases_topk=right_to_wrong_cases_topk)


def save_prediction_artifacts(run_logger: UnifiedExperimentLogger) -> None:
    run_logger.save_prediction_artifacts()


def finalize_run_summary(run_logger: UnifiedExperimentLogger, top_k: int = 5) -> Dict[str, Any]:
    return run_logger.finalize_run_summary(top_k=top_k)

import torch
import logging
import numpy as np
import torch.nn.functional as F
import os
import json
from typing import Any, Dict, List, Optional, Tuple
from typing import Union
from mydatasets.imagenet_subsets import IMAGENET_D_MAPPING
from utils.experiment_logger import log_batch_metrics, finalize_shift_summary


logger = logging.getLogger(__name__)


def _parse_target_sample_ids_from_env() -> List[int]:
    raw = os.environ.get("FIG5_TARGET_SAMPLE_IDS", "").strip()
    if not raw:
        return []
    out: List[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except Exception:
            continue
    return sorted(list(dict.fromkeys(out)))


def _build_case_record(
    *,
    sample_id: int,
    local_idx: int,
    shift_meta: Dict[str, Any],
    mode: str,
    labels: torch.Tensor,
    preds_pre: torch.Tensor,
    preds_post: torch.Tensor,
    conf_pre: torch.Tensor,
    conf_post: torch.Tensor,
    entropy_pre: torch.Tensor,
    entropy_post: torch.Tensor,
    alpha: Optional[np.ndarray],
    rho_v: Optional[np.ndarray],
    rho_t: Optional[np.ndarray],
    conflict: Optional[np.ndarray],
    gt_prompt_arr: Optional[np.ndarray],
    shifted_prompt_arr: Optional[np.ndarray],
    img_paths: Optional[List[str]],
    logits_pre: torch.Tensor,
    logits_post: torch.Tensor,
    probs_pre: torch.Tensor,
    probs_post: torch.Tensor,
    z_v: Optional[torch.Tensor],
    z_t: Optional[torch.Tensor],
    p_v: Optional[torch.Tensor],
    p_t: Optional[torch.Tensor],
) -> Dict[str, Any]:
    rec = {
        "sample_id": int(sample_id),
        "shift_type": mode,
        "shift_id": (shift_meta or {}).get("shift_id"),
        "level": (shift_meta or {}).get("text_shift_level"),
        "y_true": int(labels[local_idx].item()),
        "image_path": None if img_paths is None else str(img_paths[local_idx]),
        "clean_prompt": None if gt_prompt_arr is None else str(gt_prompt_arr[local_idx]),
        "shifted_prompt": None if shifted_prompt_arr is None else str(shifted_prompt_arr[local_idx]),
        "pred_pre": int(preds_pre[local_idx].item()),
        "pred_post": int(preds_post[local_idx].item()),
        "conf_pre": float(conf_pre[local_idx].item()),
        "conf_post": float(conf_post[local_idx].item()),
        "entropy_pre": float(entropy_pre[local_idx].item()),
        "entropy_post": float(entropy_post[local_idx].item()),
        "alpha": None if alpha is None else float(alpha[local_idx]),
        "rho_v": None if rho_v is None else float(rho_v[local_idx]),
        "rho_t": None if rho_t is None else float(rho_t[local_idx]),
        "conflict": None if conflict is None else float(conflict[local_idx]),
        "logits_pre": logits_pre[local_idx].detach().float().cpu().tolist(),
        "logits_post": logits_post[local_idx].detach().float().cpu().tolist(),
        "q_pre": probs_pre[local_idx].detach().float().cpu().tolist(),
        "q_post": probs_post[local_idx].detach().float().cpu().tolist(),
        "logits_v": None,
        "logits_t": None,
        "p_v": None,
        "p_t": None,
    }

    if z_v is not None and z_t is not None and p_v is not None and p_t is not None:
        rec["logits_v"] = z_v[local_idx].detach().float().cpu().tolist()
        rec["logits_t"] = z_t[local_idx].detach().float().cpu().tolist()
        rec["p_v"] = p_v[local_idx].detach().float().cpu().tolist()
        rec["p_t"] = p_t[local_idx].detach().float().cpu().tolist()

    return rec


def _flush_case_capture(
    capture_path: str,
    run_id: str,
    mode: str,
    method_name: str,
    target_ids: List[int],
    captured: Dict[int, Dict[str, Any]],
):
    payload = {
        "run_id": run_id,
        "mode": mode,
        "method": method_name,
        "target_sample_ids": [int(x) for x in target_ids],
        "captured_sample_ids": [int(x) for x in sorted(captured.keys())],
        "records": [captured[k] for k in sorted(captured.keys())],
    }
    os.makedirs(os.path.dirname(capture_path), exist_ok=True)
    with open(capture_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _extract_logits(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        if "logits" in output:
            return output["logits"]
        if "output" in output:
            return output["output"]
        raise ValueError("Model returned dict output but no 'logits' / 'output' keys were found.")
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _entropy_from_probs(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return -(probs * probs.clamp_min(eps).log()).sum(dim=-1)


def _js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    m = 0.5 * (p + q)

    def _kl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a.clamp_min(eps)
        b = b.clamp_min(eps)
        return (a * (a.log() - b.log())).sum(dim=-1)

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _rank_disagreement(
    p: torch.Tensor, q: torch.Tensor, chunk_size: int = 128
) -> torch.Tensor:
    """Paper Eq. (16): normalized discordant unordered class-pair count."""
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


def _dist_to_perm(probs: torch.Tensor) -> torch.Tensor:
    max_prob = probs.max(dim=-1).values
    return 2.0 * (1.0 - max_prob)


def _majorization_flag(probs_pre: torch.Tensor, probs_post: torch.Tensor, tol: float = 1e-8) -> torch.Tensor:
    sp = torch.sort(probs_pre, dim=-1, descending=True).values
    sq = torch.sort(probs_post, dim=-1, descending=True).values
    csp = sp.cumsum(dim=-1)
    csq = sq.cumsum(dim=-1)
    return (csq + tol >= csp).all(dim=-1)


def _forward_post_logits(tta_model: torch.nn.Module, imgs_eval: torch.Tensor) -> torch.Tensor:
    if hasattr(tta_model, "infer_post_result"):
        result = tta_model.infer_post_result(imgs_eval)
        if hasattr(result, "logits"):
            return result.logits
        return _extract_logits(result)

    model_for_eval = tta_model.model if hasattr(tta_model, "model") else tta_model
    output = model_for_eval(imgs_eval)
    return _extract_logits(output)


def _to_numpy_optional(x: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    if x is None:
        return None
    return x.detach().float().cpu().numpy()


def _extract_method_diagnostics(tta_model: torch.nn.Module, imgs_eval: torch.Tensor) -> Dict[str, Optional[np.ndarray]]:
    diag = {
        "alpha_pre": None,
        "alpha_post": None,
        "rho_v_pre": None,
        "rho_v_post": None,
        "rho_t_pre": None,
        "rho_t_post": None,
        "conflict_pre": None,
        "conflict_post": None,
        "js_div_pre": None,
        "js_div_post": None,
        "rank_disagreement_pre": None,
        "rank_disagreement_post": None,
        "prior_pre": None,
        "prior_post": None,
    }

    if hasattr(tta_model, "get_last_result"):
        r = tta_model.get_last_result()
    else:
        r = getattr(tta_model, "last_result", None)

    if r is not None:
        diag["alpha_pre"] = _to_numpy_optional(getattr(r, "alpha", None))
        diag["rho_v_pre"] = _to_numpy_optional(getattr(r, "rho_v", None))
        diag["rho_t_pre"] = _to_numpy_optional(getattr(r, "rho_t", None))
        diag["conflict_pre"] = _to_numpy_optional(getattr(r, "conflict", None))
        diag["js_div_pre"] = _to_numpy_optional(getattr(r, "js_div", None))
        diag["rank_disagreement_pre"] = _to_numpy_optional(getattr(r, "rank_disagreement", None))
        prior = getattr(r, "prior", None)
        if prior is not None:
            diag["prior_pre"] = prior.detach().float().cpu().numpy()

    if hasattr(tta_model, "infer_post_result"):
        with torch.no_grad():
            r_post = tta_model.infer_post_result(imgs_eval)
    elif hasattr(tta_model, "mm_core") and hasattr(tta_model, "_extract_stream_logits"):
        with torch.no_grad():
            z_v, z_t = tta_model._extract_stream_logits(imgs_eval)
            r_post = tta_model.mm_core(z_v, z_t)
    else:
        r_post = None

    if r_post is not None:
        diag["alpha_post"] = _to_numpy_optional(getattr(r_post, "alpha", None))
        diag["rho_v_post"] = _to_numpy_optional(getattr(r_post, "rho_v", None))
        diag["rho_t_post"] = _to_numpy_optional(getattr(r_post, "rho_t", None))
        diag["conflict_post"] = _to_numpy_optional(getattr(r_post, "conflict", None))
        diag["js_div_post"] = _to_numpy_optional(getattr(r_post, "js_div", None))
        diag["rank_disagreement_post"] = _to_numpy_optional(getattr(r_post, "rank_disagreement", None))
        prior = getattr(r_post, "prior", None)
        if prior is not None:
            diag["prior_post"] = prior.detach().float().cpu().numpy()

    return diag


def _safe_mean(x: Optional[np.ndarray]) -> Optional[float]:
    if x is None or x.size == 0:
        return None
    return float(np.mean(x))


def _safe_std(x: Optional[np.ndarray]) -> Optional[float]:
    if x is None or x.size == 0:
        return None
    return float(np.std(x))


def _safe_min(x: Optional[np.ndarray]) -> Optional[float]:
    if x is None or x.size == 0:
        return None
    return float(np.min(x))


def _safe_max(x: Optional[np.ndarray]) -> Optional[float]:
    if x is None or x.size == 0:
        return None
    return float(np.max(x))


def _safe_ratio(mask: Optional[np.ndarray]) -> Optional[float]:
    if mask is None or mask.size == 0:
        return None
    return float(np.mean(mask.astype(np.float32)))


def _gate_prior_kl(alpha: Optional[np.ndarray], prior: Optional[np.ndarray], eps: float = 1e-8) -> Optional[np.ndarray]:
    if alpha is None or prior is None:
        return None
    gate = np.stack([alpha, 1.0 - alpha], axis=-1)
    gate = np.clip(gate, eps, 1.0)
    prior = np.clip(prior, eps, 1.0)
    return np.sum(gate * (np.log(gate) - np.log(prior)), axis=-1)


def split_results_by_domain(domain_dict: dict, data: list, predictions: torch.tensor):
    """
    Separates the label prediction pairs by domain
    Input:
        domain_dict: Dictionary, where the keys are the domain names and the values are lists with pairs [[label1, prediction1], ...]
        data: List containing [images, labels, domains, ...]
        predictions: Tensor containing the predictions of the model
    Returns:
        domain_dict: Updated dictionary containing the domain seperated label prediction pairs
    """

    labels, domains = data[1], data[2]
    assert predictions.shape[0] == labels.shape[0], "The batch size of predictions and labels does not match!"

    for i in range(labels.shape[0]):
        if domains[i] in domain_dict.keys():
            domain_dict[domains[i]].append([labels[i].item(), predictions[i].item()])
        else:
            domain_dict[domains[i]] = [[labels[i].item(), predictions[i].item()]]

    return domain_dict


def eval_domain_dict(domain_dict: dict, domain_seq: list):
    """
    Print detailed results for each domain. This is useful for settings where the domains are mixed
    Input:
        domain_dict: Dictionary containing the labels and predictions for each domain
        domain_seq: Order to print the results (if all domains are contained in the domain dict)
    """
    correct = []
    num_samples = []
    avg_error_domains = []
    domain_names = domain_seq if all([dname in domain_seq for dname in domain_dict.keys()]) else domain_dict.keys()
    logger.info(f"Splitting the results by domain...")
    for key in domain_names:
        label_prediction_arr = np.array(domain_dict[key])  # rows: samples, cols: (label, prediction)
        correct.append((label_prediction_arr[:, 0] == label_prediction_arr[:, 1]).sum())
        num_samples.append(label_prediction_arr.shape[0])
        accuracy = correct[-1] / num_samples[-1]
        error = 1 - accuracy
        avg_error_domains.append(error)
        logger.info(f"{key:<20} error: {error:.2%}")
    logger.info(f"Average error across all domains: {sum(avg_error_domains) / len(avg_error_domains):.2%}")
    # The error across all samples differs if each domain contains different amounts of samples
    logger.info(f"Error over all samples: {1 - sum(correct) / sum(num_samples):.2%}")


def get_accuracy(model: torch.nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 dataset_name: str,
                 domain_name: str,
                 setting: str,
                 domain_dict: dict,
                 print_every: int,
                 device: Union[str, torch.device],
                 mode: str = "visual_shift",
                 method_name: str = "source",
                 shift_meta: Optional[Dict[str, Any]] = None,
                 shift_index: int = 0,
                 global_sample_offset: int = 0,
                 run_logger=None,
                 num_classes: Optional[int] = None):

    num_correct = 0.
    num_samples = 0
    num_batches_in_shift = len(data_loader) if hasattr(data_loader, "__len__") else -1
    batch_metrics_stdout = False
    if run_logger is not None and hasattr(run_logger, "cfg"):
        batch_metrics_stdout = bool(getattr(getattr(run_logger.cfg, "LOG", None), "BATCH_METRICS_STDOUT", False))
    if batch_metrics_stdout:
        print_every_eff = print_every if print_every and print_every > 0 else 10
        if num_batches_in_shift > 0 and num_batches_in_shift < 10:
            print_every_eff = 1
    else:
        print_every_eff = -1

    failure_entropy_thr = 0.5
    failure_conflict_thr = 0.5
    failure_topk = 50
    if run_logger is not None and hasattr(run_logger, "cfg") and hasattr(run_logger.cfg, "EXPERIMENT_OUTPUT"):
        failure_entropy_thr = float(getattr(run_logger.cfg.EXPERIMENT_OUTPUT, "FAILURE_LOW_ENTROPY_THRESHOLD", 0.5))
        failure_conflict_thr = float(getattr(run_logger.cfg.EXPERIMENT_OUTPUT, "FAILURE_HIGH_CONFLICT_THRESHOLD", 0.5))
        failure_topk = int(getattr(run_logger.cfg.EXPERIMENT_OUTPUT, "FAILURE_TOPK", 50))

    if run_logger is not None:
        run_logger.start_shift(shift_meta=shift_meta or {}, num_batches_in_shift=max(0, num_batches_in_shift))

    # Optional env-driven case-level capture for targeted Fig.5 reruns.
    fig5_target_ids = _parse_target_sample_ids_from_env()
    fig5_capture_out = os.environ.get("FIG5_CAPTURE_OUT", "").strip()
    fig5_capture_enabled = len(fig5_target_ids) > 0 and len(fig5_capture_out) > 0
    fig5_early_stop = str(os.environ.get("FIG5_EARLY_STOP", "1")).strip().lower() in {"1", "true", "yes"}
    fig5_captured: Dict[int, Dict[str, Any]] = {}
    if fig5_capture_enabled and run_logger is not None:
        prev = getattr(run_logger, "fig5_capture_records", None)
        if isinstance(prev, dict):
            fig5_captured = dict(prev)
    fig5_done = False

    prompt_class_names = None
    clean_prompt_bank = None
    shifted_prompt_bank = None
    if run_logger is not None and getattr(run_logger, "prompt_bank_payload", None) is not None:
        pb = run_logger.prompt_bank_payload
        if isinstance(pb, dict):
            cns = pb.get("class_names")
            if isinstance(cns, list) and len(cns) > 0:
                prompt_class_names = cns
            cpb = pb.get("clean_prompt_bank")
            spb = pb.get("shifted_prompt_bank")
            if isinstance(cpb, dict):
                clean_prompt_bank = cpb
            if isinstance(spb, dict):
                shifted_prompt_bank = spb

    right_to_wrong_cases_all: List[Dict[str, Any]] = []
    failure_cases_all: List[Dict[str, Any]] = []

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            imgs, labels = data[0], data[1]
            img_paths = None
            if isinstance(data, (list, tuple)) and len(data) > 3:
                maybe_paths = data[3]
                if isinstance(maybe_paths, (list, tuple)):
                    img_paths = [str(p) for p in maybe_paths]
            imgs_for_adapt = [img.to(device) for img in imgs] if isinstance(imgs, list) else imgs.to(device)
            imgs_eval = imgs[0].to(device) if isinstance(imgs, list) else imgs.to(device)

            output_pre = model(imgs_for_adapt)
            logits_pre = _extract_logits(output_pre)
            logits_post = _forward_post_logits(model, imgs_eval)

            probs_pre = F.softmax(logits_pre, dim=-1)
            probs_post = F.softmax(logits_post, dim=-1)

            q_pre_top3_val, q_pre_top3_idx = torch.topk(probs_pre, k=min(3, probs_pre.shape[1]), dim=-1)
            q_post_top3_val, q_post_top3_idx = torch.topk(probs_post, k=min(3, probs_post.shape[1]), dim=-1)

            p_v_top3_idx = None
            p_v_top3_val = None
            p_t_top3_idx = None
            p_t_top3_val = None
            z_v = None
            z_t = None
            p_v = None
            p_t = None
            if hasattr(model, "_extract_stream_logits"):
                try:
                    z_v, z_t = model._extract_stream_logits(imgs_eval)
                    p_v = F.softmax(z_v, dim=-1)
                    p_t = F.softmax(z_t, dim=-1)
                    p_v_top3_val, p_v_top3_idx = torch.topk(p_v, k=min(3, p_v.shape[1]), dim=-1)
                    p_t_top3_val, p_t_top3_idx = torch.topk(p_t, k=min(3, p_t.shape[1]), dim=-1)
                except Exception:
                    p_v_top3_idx = None
                    p_v_top3_val = None
                    p_t_top3_idx = None
                    p_t_top3_val = None
                    z_v = None
                    z_t = None
                    p_v = None
                    p_t = None

            predictions_pre = probs_pre.argmax(1)
            predictions_post = probs_post.argmax(1)

            if dataset_name == "imagenet_d" and domain_name != "none":
                mapping_vector = list(IMAGENET_D_MAPPING.values())
                predictions_pre = torch.tensor([mapping_vector[pred] for pred in predictions_pre], device=device)
                predictions_post = torch.tensor([mapping_vector[pred] for pred in predictions_post], device=device)

            labels_device = labels.to(device)
            num_correct += (predictions_post == labels_device).float().sum()

            entropy_pre = _entropy_from_probs(probs_pre)
            entropy_post = _entropy_from_probs(probs_post)
            conf_pre = probs_pre.max(dim=-1).values
            conf_post = probs_post.max(dim=-1).values

            pred_flip = predictions_pre != predictions_post
            wrong_to_right = (predictions_pre != labels_device) & (predictions_post == labels_device)
            right_to_wrong = (predictions_pre == labels_device) & (predictions_post != labels_device)

            majorization_flag = _majorization_flag(probs_pre, probs_post)
            dist_pre = _dist_to_perm(probs_pre)
            dist_post = _dist_to_perm(probs_post)

            if num_classes is None:
                num_classes_eff = probs_post.shape[1]
            else:
                num_classes_eff = int(num_classes)

            pred_hist = torch.bincount(predictions_post, minlength=num_classes_eff).float()
            pred_hist = pred_hist / max(1, predictions_post.shape[0])
            batch_marginal = probs_post.mean(dim=0)
            batch_marginal_entropy = float(_entropy_from_probs(batch_marginal.unsqueeze(0)).item())
            top1_class_ratio = float(pred_hist.max().item())
            class_imbalance = 1.0 - batch_marginal_entropy / max(np.log(max(num_classes_eff, 2)), 1e-8)

            diag = _extract_method_diagnostics(model, imgs_eval)
            alpha = diag["alpha_post"] if diag["alpha_post"] is not None else diag["alpha_pre"]
            rho_v = diag["rho_v_post"] if diag["rho_v_post"] is not None else diag["rho_v_pre"]
            rho_t = diag["rho_t_post"] if diag["rho_t_post"] is not None else diag["rho_t_pre"]
            conflict = diag["conflict_post"] if diag["conflict_post"] is not None else diag["conflict_pre"]
            js_div = diag["js_div_post"] if diag["js_div_post"] is not None else diag["js_div_pre"]
            rank_dis = diag["rank_disagreement_post"] if diag["rank_disagreement_post"] is not None else diag["rank_disagreement_pre"]
            prior = diag["prior_post"] if diag["prior_post"] is not None else diag["prior_pre"]

            gate_prior_kl = _gate_prior_kl(alpha, prior)
            rho_gap = None
            if rho_v is not None and rho_t is not None:
                rho_gap = rho_v - rho_t

            alpha_pre = diag["alpha_pre"]
            alpha_post = diag["alpha_post"]
            alpha_cross = None
            if alpha_pre is not None and alpha_post is not None and alpha_pre.shape == alpha_post.shape:
                alpha_cross = ((alpha_pre - 0.5) * (alpha_post - 0.5) < 0.0)

            conflict_arr = conflict
            low_entropy_wrong_mask = ((entropy_post < failure_entropy_thr) & (predictions_post != labels_device))
            high_conflict_wrong_mask = None
            if conflict_arr is not None:
                conflict_t = torch.from_numpy(conflict_arr).to(device)
                high_conflict_wrong_mask = (conflict_t > failure_conflict_thr) & (predictions_post != labels_device)

            bs = labels.shape[0]
            global_start = global_sample_offset + num_samples
            global_end = global_start + bs - 1
            sample_ids = np.arange(global_start, global_end + 1, dtype=np.int64)

            gt_prompt_arr = None
            shifted_prompt_arr = None
            if prompt_class_names is not None and clean_prompt_bank is not None:
                y_np = labels.detach().cpu().numpy().astype(np.int64)
                gt_prompts = []
                shifted_prompts = []
                for y in y_np:
                    cls_name = prompt_class_names[y] if 0 <= y < len(prompt_class_names) else None
                    cands_clean = clean_prompt_bank.get(cls_name, []) if cls_name is not None else []
                    cands_shift = shifted_prompt_bank.get(cls_name, []) if (cls_name is not None and shifted_prompt_bank is not None) else []
                    gt_prompts.append(cands_clean[0] if isinstance(cands_clean, list) and len(cands_clean) > 0 else "")
                    shifted_prompts.append(cands_shift[0] if isinstance(cands_shift, list) and len(cands_shift) > 0 else "")
                gt_prompt_arr = np.asarray(gt_prompts, dtype=np.str_)
                shifted_prompt_arr = np.asarray(shifted_prompts, dtype=np.str_)

            batch_record = {
                "run_id": run_logger.run_id if run_logger is not None else None,
                "mode": mode,
                "method": method_name,
                "dataset": dataset_name,
                "arch": getattr(getattr(run_logger.cfg, "MODEL", None), "ARCH", None) if run_logger is not None else None,
                "shift_id": (shift_meta or {}).get("shift_id") if shift_meta is not None else None,
                "shift_index": shift_index,
                "corruption_name": (shift_meta or {}).get("corruption_name") if shift_meta is not None else None,
                "severity": (shift_meta or {}).get("severity") if shift_meta is not None else None,
                "visual_degradation_tag": (shift_meta or {}).get("visual_degradation_tag") if shift_meta is not None else None,
                "is_text_shift": (shift_meta or {}).get("is_text_shift") if shift_meta is not None else None,
                "text_shift_family": (shift_meta or {}).get("text_shift_family") if shift_meta is not None else None,
                "text_shift_level": (shift_meta or {}).get("text_shift_level") if shift_meta is not None else None,
                "text_shift_protocol": (shift_meta or {}).get("text_shift_protocol") if shift_meta is not None else None,
                "batch_idx": i + 1,
                "num_batches_in_shift": num_batches_in_shift,
                "global_sample_start": global_start,
                "global_sample_end": global_end,
                "batch_size_actual": bs,
                "batch_acc": float((predictions_post == labels_device).float().mean().item()),
                "entropy_pre_mean": float(entropy_pre.mean().item()),
                "entropy_post_mean": float(entropy_post.mean().item()),
                "delta_entropy_mean": float((entropy_post - entropy_pre).mean().item()),
                "top1_conf_pre_mean": float(conf_pre.mean().item()),
                "top1_conf_post_mean": float(conf_post.mean().item()),
                "delta_top1_conf_mean": float((conf_post - conf_pre).mean().item()),
                "pred_flip_rate": float(pred_flip.float().mean().item()),
                "wrong_to_right_rate": float(wrong_to_right.float().mean().item()),
                "right_to_wrong_rate": float(right_to_wrong.float().mean().item()),
                "majorization_ratio_post_over_pre": float(majorization_flag.float().mean().item()),
                "majorization_improve_count": float(majorization_flag.float().sum().item()),
                "majorization_valid_count": float(bs),
                "closer_to_clean_partialsum_ratio": None,
                "dist_to_perm_pre_mean": float(dist_pre.mean().item()),
                "dist_to_perm_post_mean": float(dist_post.mean().item()),
                "delta_dist_to_perm_mean": float((dist_post - dist_pre).mean().item()),
                "batch_marginal_entropy": batch_marginal_entropy,
                "batch_class_imbalance": float(class_imbalance),
                "top1_class_ratio": float(top1_class_ratio),
                "alpha_mean": _safe_mean(alpha),
                "alpha_std": _safe_std(alpha),
                "alpha_min": _safe_min(alpha),
                "alpha_max": _safe_max(alpha),
                "alpha_gt_0_5_ratio": _safe_ratio(None if alpha is None else (alpha > 0.5)),
                "alpha_lt_0_5_ratio": _safe_ratio(None if alpha is None else (alpha < 0.5)),
                "rho_v_mean": _safe_mean(rho_v),
                "rho_t_mean": _safe_mean(rho_t),
                "rho_gap_mean": _safe_mean(rho_gap),
                "conflict_mean": _safe_mean(conflict),
                "js_div_mean": _safe_mean(js_div),
                "rank_disagreement_mean": _safe_mean(rank_dis),
                "prior_v_mean": _safe_mean(None if prior is None else prior[:, 0]),
                "prior_t_mean": _safe_mean(None if prior is None else prior[:, 1]),
                "gate_prior_kl_mean": _safe_mean(gate_prior_kl),
                "dominant_modality_rate": _safe_ratio(None if alpha is None else (alpha > 0.5)),
                "low_entropy_wrong_count": float(low_entropy_wrong_mask.float().sum().item()),
                "high_conflict_wrong_count": float(high_conflict_wrong_mask.float().sum().item()) if high_conflict_wrong_mask is not None else None,
                "alpha_threshold_cross_count": float(np.sum(alpha_cross)) if alpha_cross is not None else None,
                "right_to_wrong_count": float(right_to_wrong.float().sum().item()),
            }

            pred_artifacts = {
                "sample_ids": sample_ids,
                "image_path": None if img_paths is None else np.asarray(img_paths),
                "gt_prompt": gt_prompt_arr,
                "shifted_prompt": shifted_prompt_arr,
                "y_true": labels.cpu().numpy().astype(np.int64),
                "pred_pre": predictions_pre.detach().cpu().numpy().astype(np.int64),
                "pred_post": predictions_post.detach().cpu().numpy().astype(np.int64),
                "conf_pre": conf_pre.detach().cpu().numpy().astype(np.float32),
                "conf_post": conf_post.detach().cpu().numpy().astype(np.float32),
                "entropy_pre": entropy_pre.detach().cpu().numpy().astype(np.float32),
                "entropy_post": entropy_post.detach().cpu().numpy().astype(np.float32),
                "majorization_flag": majorization_flag.detach().cpu().numpy().astype(np.int32),
                "dist_to_perm_pre": dist_pre.detach().cpu().numpy().astype(np.float32),
                "dist_to_perm_post": dist_post.detach().cpu().numpy().astype(np.float32),
                "alpha": None if alpha is None else alpha.astype(np.float32),
                "rho_v": None if rho_v is None else rho_v.astype(np.float32),
                "rho_t": None if rho_t is None else rho_t.astype(np.float32),
                "conflict": None if conflict is None else conflict.astype(np.float32),
                "js_div": None if js_div is None else js_div.astype(np.float32),
                "rank_disagreement": None if rank_dis is None else rank_dis.astype(np.float32),
                "q_pre_top3_idx": q_pre_top3_idx.detach().cpu().numpy().astype(np.int64),
                "q_pre_top3_val": q_pre_top3_val.detach().cpu().numpy().astype(np.float32),
                "q_post_top3_idx": q_post_top3_idx.detach().cpu().numpy().astype(np.int64),
                "q_post_top3_val": q_post_top3_val.detach().cpu().numpy().astype(np.float32),
                "p_v_top3_idx": None if p_v_top3_idx is None else p_v_top3_idx.detach().cpu().numpy().astype(np.int64),
                "p_v_top3_val": None if p_v_top3_val is None else p_v_top3_val.detach().cpu().numpy().astype(np.float32),
                "p_t_top3_idx": None if p_t_top3_idx is None else p_t_top3_idx.detach().cpu().numpy().astype(np.int64),
                "p_t_top3_val": None if p_t_top3_val is None else p_t_top3_val.detach().cpu().numpy().astype(np.float32),
            }

            if run_logger is not None:
                log_batch_metrics(run_logger, batch_record=batch_record, pred_artifacts=pred_artifacts)

            if fig5_capture_enabled:
                sid_to_local = {int(sample_ids[j]): j for j in range(len(sample_ids))}
                for sid in fig5_target_ids:
                    if sid in fig5_captured:
                        continue
                    if sid not in sid_to_local:
                        continue
                    local_idx = int(sid_to_local[sid])
                    fig5_captured[sid] = _build_case_record(
                        sample_id=sid,
                        local_idx=local_idx,
                        shift_meta=shift_meta or {},
                        mode=mode,
                        labels=labels,
                        preds_pre=predictions_pre,
                        preds_post=predictions_post,
                        conf_pre=conf_pre,
                        conf_post=conf_post,
                        entropy_pre=entropy_pre,
                        entropy_post=entropy_post,
                        alpha=alpha,
                        rho_v=rho_v,
                        rho_t=rho_t,
                        conflict=conflict,
                        gt_prompt_arr=gt_prompt_arr,
                        shifted_prompt_arr=shifted_prompt_arr,
                        img_paths=img_paths,
                        logits_pre=logits_pre,
                        logits_post=logits_post,
                        probs_pre=probs_pre,
                        probs_post=probs_post,
                        z_v=z_v,
                        z_t=z_t,
                        p_v=p_v,
                        p_t=p_t,
                    )

                if run_logger is not None:
                    setattr(run_logger, "fig5_capture_records", fig5_captured)

                _flush_case_capture(
                    capture_path=fig5_capture_out,
                    run_id=(run_logger.run_id if run_logger is not None else "unknown"),
                    mode=mode,
                    method_name=method_name,
                    target_ids=fig5_target_ids,
                    captured=fig5_captured,
                )

                if fig5_early_stop and all([(sid in fig5_captured) for sid in fig5_target_ids]):
                    fig5_done = True
                    if run_logger is not None:
                        setattr(run_logger, "fig5_capture_done", True)
                    logger.info(
                        f"[FIG5-CAPTURE] collected all targets {fig5_target_ids}; early stopping current run loop."
                    )
                    break

            wrong_idx = torch.where(right_to_wrong)[0].detach().cpu().numpy().tolist()
            for local_idx in wrong_idx:
                right_to_wrong_cases_all.append({
                    "sample_id": int(sample_ids[local_idx]),
                    "true_label": int(labels[local_idx].item()),
                    "pred_pre": int(predictions_pre[local_idx].item()),
                    "pred_post": int(predictions_post[local_idx].item()),
                    "entropy_pre": float(entropy_pre[local_idx].item()),
                    "entropy_post": float(entropy_post[local_idx].item()),
                    "conf_pre": float(conf_pre[local_idx].item()),
                    "conf_post": float(conf_post[local_idx].item()),
                    "alpha": None if alpha is None else float(alpha[local_idx]),
                    "rho_v": None if rho_v is None else float(rho_v[local_idx]),
                    "rho_t": None if rho_t is None else float(rho_t[local_idx]),
                    "conflict": None if conflict is None else float(conflict[local_idx]),
                    "js_div": None if js_div is None else float(js_div[local_idx]),
                    "rank_disagreement": None if rank_dis is None else float(rank_dis[local_idx]),
                    "corruption_name": (shift_meta or {}).get("corruption_name"),
                    "severity": (shift_meta or {}).get("severity"),
                    "text_shift_family": (shift_meta or {}).get("text_shift_family"),
                    "text_shift_level": (shift_meta or {}).get("text_shift_level"),
                    "text_shift_protocol": (shift_meta or {}).get("text_shift_protocol"),
                })

            if mode == "failure":
                failure_mask = low_entropy_wrong_mask.clone()
                if high_conflict_wrong_mask is not None:
                    failure_mask = failure_mask | high_conflict_wrong_mask
                if alpha_cross is not None and alpha_cross.shape[0] == bs:
                    failure_mask = failure_mask | torch.from_numpy(alpha_cross).to(device)

                failure_idx = torch.where(failure_mask)[0].detach().cpu().numpy().tolist()
                for local_idx in failure_idx:
                    failure_cases_all.append({
                        "sample_id": int(sample_ids[local_idx]),
                        "true_label": int(labels[local_idx].item()),
                        "pred_pre": int(predictions_pre[local_idx].item()),
                        "pred_post": int(predictions_post[local_idx].item()),
                        "entropy_pre": float(entropy_pre[local_idx].item()),
                        "entropy_post": float(entropy_post[local_idx].item()),
                        "conf_pre": float(conf_pre[local_idx].item()),
                        "conf_post": float(conf_post[local_idx].item()),
                        "alpha": None if alpha is None else float(alpha[local_idx]),
                        "rho_v": None if rho_v is None else float(rho_v[local_idx]),
                        "rho_t": None if rho_t is None else float(rho_t[local_idx]),
                        "conflict": None if conflict is None else float(conflict[local_idx]),
                        "js_div": None if js_div is None else float(js_div[local_idx]),
                        "rank_disagreement": None if rank_dis is None else float(rank_dis[local_idx]),
                        "corruption_name": (shift_meta or {}).get("corruption_name"),
                        "severity": (shift_meta or {}).get("severity"),
                        "text_shift_family": (shift_meta or {}).get("text_shift_family"),
                        "text_shift_level": (shift_meta or {}).get("text_shift_level"),
                        "text_shift_protocol": (shift_meta or {}).get("text_shift_protocol"),
                    })

            if "mixed_domains" in setting and len(data) >= 3:
                domain_dict = split_results_by_domain(domain_dict, data, predictions_post)

            # track progress
            num_samples += imgs[0].shape[0] if isinstance(imgs, list) else imgs.shape[0]
            if print_every_eff > 0 and (i + 1) % print_every_eff == 0:
                shift_tag = (shift_meta or {}).get("shift_id", f"{domain_name}-sev{(shift_meta or {}).get('severity', 'na')}")
                logger.info(
                    f"[mode={mode}][method={method_name}][shift={shift_tag}][batch {i+1}/{num_batches_in_shift}]"
                )
                logger.info(
                    f"acc={batch_record['batch_acc']:.4f} "
                    f"ent_pre={batch_record['entropy_pre_mean']:.4f} ent_post={batch_record['entropy_post_mean']:.4f} "
                    f"d_ent={batch_record['delta_entropy_mean']:.4f}"
                )
                logger.info(
                    f"conf_pre={batch_record['top1_conf_pre_mean']:.4f} "
                    f"conf_post={batch_record['top1_conf_post_mean']:.4f} "
                    f"d_conf={batch_record['delta_top1_conf_mean']:.4f}"
                )
                logger.info(
                    f"flip={batch_record['pred_flip_rate']:.4f} "
                    f"w2r={batch_record['wrong_to_right_rate']:.4f} "
                    f"r2w={batch_record['right_to_wrong_rate']:.4f}"
                )
                logger.info(
                    f"maj_ratio={batch_record['majorization_ratio_post_over_pre']:.4f} "
                    f"maj_cnt={int(batch_record['majorization_improve_count'])}/{int(batch_record['majorization_valid_count'])} "
                    f"distperm_pre={batch_record['dist_to_perm_pre_mean']:.4f} "
                    f"distperm_post={batch_record['dist_to_perm_post_mean']:.4f} "
                    f"d_distperm={batch_record['delta_dist_to_perm_mean']:.4f}"
                )
                logger.info(
                    f"collapse_inputs: top1_ratio={batch_record['top1_class_ratio']:.4f} "
                    f"batch_marginal_entropy={batch_record['batch_marginal_entropy']:.4f} "
                    f"class_imbalance={batch_record['batch_class_imbalance']:.4f}"
                )
                if batch_record["alpha_mean"] is None:
                    logger.info("alpha/rho/conflict=N/A")
                else:
                    logger.info(
                        f"alpha={batch_record['alpha_mean']:.4f}±{batch_record['alpha_std']:.4f} "
                        f"rho_v={batch_record['rho_v_mean']:.4f} "
                        f"rho_t={batch_record['rho_t_mean']:.4f} "
                        f"conflict={batch_record['conflict_mean']:.4f}"
                    )

            if dataset_name == "ccc" and num_samples >= 7500000:
                break

            if fig5_done:
                break

    accuracy = num_correct.item() / num_samples

    shift_summary = None
    if run_logger is not None:
        right_to_wrong_cases_all = sorted(
            right_to_wrong_cases_all,
            key=lambda x: x.get("conf_pre", 0.0),
            reverse=True,
        )
        topk_cases = right_to_wrong_cases_all[:failure_topk]
        shift_summary = finalize_shift_summary(run_logger, right_to_wrong_cases_topk=topk_cases)
        if mode == "failure" and len(failure_cases_all) > 0:
            run_logger.add_failure_cases(failure_cases_all)

    return accuracy, domain_dict, num_samples, shift_summary

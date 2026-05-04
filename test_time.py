import os
import torch
import logging
import numpy as np
import methods

from models.model import get_model
from utils.misc import print_memory_info
from utils.eval_utils import get_accuracy, eval_domain_dict
from utils.experiment_logger import init_run_logger, finalize_run_summary
from utils.registry import ADAPTATION_REGISTRY
from mydatasets.data_loading import get_test_loader
from conf import cfg, load_cfg_from_args, get_num_classes, ckpt_path_to_domain_seq

logger = logging.getLogger(__name__)


def _normalize_corruption_types(cfg_):
    types = list(getattr(cfg_.CORRUPTION, "TYPE", []))
    return [str(t).strip().lower() for t in types]


def _is_visual_clean_input(cfg_):
    types = _normalize_corruption_types(cfg_)
    if len(types) == 0:
        return True
    clean_tokens = {"", "none", "clean", "original"}
    return all(t in clean_tokens for t in types)


def _is_visual_dirty_input(cfg_):
    return not _is_visual_clean_input(cfg_)


def validate_shift_protocol(cfg_):
    """
    Enforce protocol semantics at startup.

    visual-only:
      - visual input must be dirty
      - text input must stay clean (TEXT_SHIFT.ENABLED=False)

    text-only:
      - visual input must be clean
      - text shift must be enabled
      - text family must be tokenization_stress with valid level

    joint:
      - visual input must be dirty
      - text shift must be enabled
      - text family must be tokenization_stress with valid level
    """
    regime = str(getattr(cfg_.TEXT_SHIFT, "REGIME", "clean")).strip().lower()
    text_enabled = bool(getattr(cfg_.TEXT_SHIFT, "ENABLED", False))
    family = str(getattr(cfg_.TEXT_SHIFT, "FAMILY", "")).strip().lower()
    level = int(getattr(cfg_.TEXT_SHIFT, "LEVEL", 1))
    allow_clean_text_baseline = str(os.environ.get("ALLOW_CLEAN_TEXT_BASELINE", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
    }

    if regime in {"visual", "visual_shift"}:
        if text_enabled:
            raise ValueError(
                "Protocol violation (visual-only): TEXT_SHIFT.ENABLED must be False so text input stays clean."
            )
        if not _is_visual_dirty_input(cfg_):
            raise ValueError(
                "Protocol violation (visual-only): CORRUPTION.TYPE must be dirty (not ['none']/clean)."
            )
        return

    if regime in {"text_only", "text", "text_shift"}:
        if (not text_enabled) and allow_clean_text_baseline:
            if not _is_visual_clean_input(cfg_):
                raise ValueError(
                    "Protocol violation (clean-text baseline): visual input must be clean (e.g., CORRUPTION.TYPE=['none'])."
                )
            return
        if not text_enabled:
            raise ValueError(
                "Protocol violation (text-only): TEXT_SHIFT.ENABLED must be True."
            )
        if not _is_visual_clean_input(cfg_):
            raise ValueError(
                "Protocol violation (text-only): visual input must be clean (e.g., CORRUPTION.TYPE=['none'])."
            )
        if family != "tokenization_stress":
            raise ValueError(
                "Protocol violation (text-only): TEXT_SHIFT.FAMILY must be 'tokenization_stress'."
            )
        if not (1 <= level <= 5):
            raise ValueError(
                "Protocol violation (text-only): TEXT_SHIFT.LEVEL must be in [1, 5]."
            )
        return

    if regime == "joint":
        if not text_enabled:
            raise ValueError(
                "Protocol violation (joint): TEXT_SHIFT.ENABLED must be True."
            )
        if not _is_visual_dirty_input(cfg_):
            raise ValueError(
                "Protocol violation (joint): visual input must be dirty (not ['none']/clean)."
            )
        if family != "tokenization_stress":
            raise ValueError(
                "Protocol violation (joint): TEXT_SHIFT.FAMILY must be 'tokenization_stress'."
            )
        if not (1 <= level <= 5):
            raise ValueError(
                "Protocol violation (joint): TEXT_SHIFT.LEVEL must be in [1, 5]."
            )
        return

    raise ValueError(
        "Unsupported TEXT_SHIFT.REGIME for strict protocol. "
        "Use one of: visual_shift, text_only, joint."
    )


def infer_experiment_mode(cfg_):
    text_enabled = bool(getattr(cfg_.TEXT_SHIFT, "ENABLED", False))
    regime = str(getattr(cfg_.TEXT_SHIFT, "REGIME", "clean")).lower()

    if (not text_enabled) or regime in {"clean", "visual", "visual_shift"}:
        return "visual_shift"
    if regime in {"text_only", "text", "text_shift"}:
        return "text_shift"
    if regime == "joint":
        return "joint_shift"
    if regime == "failure":
        return "failure"
    return "visual_shift"


def build_shift_meta(cfg_, mode, domain_name, severity, shift_index, model_wrapper):
    text_family = getattr(cfg_.TEXT_SHIFT, "FAMILY", None) if mode in {"text_shift", "joint_shift", "failure"} else None
    text_level = getattr(cfg_.TEXT_SHIFT, "LEVEL", None) if mode in {"text_shift", "joint_shift", "failure"} else None
    text_protocol = getattr(cfg_.TEXT_SHIFT, "PROTOCOL", None) if mode in {"text_shift", "joint_shift", "failure"} else None

    if mode == "visual_shift":
        shift_id = f"{domain_name}-sev{severity}"
    elif mode == "text_shift":
        shift_id = f"{text_family}-lvl{text_level}"
    elif mode == "joint_shift":
        shift_id = f"{domain_name}-sev{severity}+{text_family}-lvl{text_level}"
    else:
        shift_id = f"failure-{domain_name}-sev{severity}+{text_family}-lvl{text_level}"

    num_prompts_per_class = None
    clean_prompt_template = None
    shifted_prompt_template_preview = None

    base_model = model_wrapper.model if hasattr(model_wrapper, "model") else model_wrapper
    clean_prompt_bank = base_model.get_clean_prompt_bank() if hasattr(base_model, "get_clean_prompt_bank") else getattr(base_model, "clean_prompt_bank", None)
    shifted_prompt_bank = base_model.get_shifted_prompt_bank() if hasattr(base_model, "get_shifted_prompt_bank") else getattr(base_model, "shift_prompt_bank", None)

    if isinstance(clean_prompt_bank, dict) and len(clean_prompt_bank) > 0:
        first_class = next(iter(clean_prompt_bank.keys()))
        num_prompts_per_class = len(clean_prompt_bank[first_class])
        clean_prompt_template = list(getattr(base_model, "clean_prompt_templates", []))

    if isinstance(shifted_prompt_bank, dict) and len(shifted_prompt_bank) > 0:
        first_class = next(iter(shifted_prompt_bank.keys()))
        shifted_prompt_template_preview = list(shifted_prompt_bank[first_class][:3])

    return {
        "shift_id": shift_id,
        "shift_index": int(shift_index),
        "corruption_name": domain_name,
        "severity": severity,
        "visual_degradation_tag": domain_name,
        "is_text_shift": mode in {"text_shift", "joint_shift", "failure"},
        "text_shift_family": text_family,
        "text_shift_level": text_level,
        "text_shift_protocol": text_protocol,
        "num_prompts_per_class": num_prompts_per_class,
        "clean_prompt_template": clean_prompt_template,
        "shifted_prompt_template_preview": shifted_prompt_template_preview,
    }


def evaluate(description):
    load_cfg_from_args(description)
    validate_shift_protocol(cfg)
    valid_settings = ["reset_each_shift",           # reset the model state after the adaptation to a domain
                      "continual",                  # train on sequence of domain shifts without knowing when a shift occurs
                      "gradual",                    # sequence of gradually increasing / decreasing domain shifts
                      "mixed_domains",              # consecutive test samples are likely to originate from different domains
                      "correlated",                 # sorted by class label
                      "mixed_domains_correlated",   # mixed domains + sorted by class label
                      "gradual_correlated",         # gradual domain shifts + sorted by class label
                      "reset_each_shift_correlated"
                      ]
    assert cfg.SETTING in valid_settings, f"The setting '{cfg.SETTING}' is not supported! Choose from: {valid_settings}"

    if torch.cuda.is_available():
        cuda_index = int(os.environ.get("BATCLIP_CUDA_INDEX", "0"))
        torch.cuda.set_device(cuda_index)
        device = f"cuda:{cuda_index}"
        logger.info(
            "[SANITY][device] "
            f"using device={device} "
            f"name={torch.cuda.get_device_name(cuda_index)}"
        )
    else:
        device = "cpu"
        logger.warning("[SANITY][device] CUDA unavailable, falling back to CPU")
    num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)

    # get the base model and its corresponding input pre-processing (if available)
    base_model, model_preprocess = get_model(cfg, num_classes, device)
    try:
        model_device = next(base_model.parameters()).device
    except StopIteration:
        model_device = "<no-parameters>"
    logger.info(f"[SANITY][device] base_model_parameter_device={model_device}")

    # append the input pre-processing to the base model
    base_model.model_preprocess = model_preprocess

    mode = infer_experiment_mode(cfg)
    run_logger = init_run_logger(
        cfg=cfg,
        mode=mode,
        method=cfg.MODEL.ADAPTATION,
        dataset=cfg.CORRUPTION.DATASET,
        arch=cfg.MODEL.ARCH,
        weights=str(cfg.MODEL.WEIGHTS),
        num_classes=num_classes,
        device=device,
        trainable_params=None,
    )

    # setup test-time adaptation method
    available_adaptations = ADAPTATION_REGISTRY.registered_names()
    assert cfg.MODEL.ADAPTATION in available_adaptations, \
        f"The adaptation '{cfg.MODEL.ADAPTATION}' is not supported! Choose from: {available_adaptations}"
    model = ADAPTATION_REGISTRY.get(cfg.MODEL.ADAPTATION)(cfg=cfg, model=base_model, num_classes=num_classes)
    run_logger.trainable_params = getattr(model, "num_trainable_params", None)
    logger.info(f"Successfully prepared test-time adaptation method: {cfg.MODEL.ADAPTATION}")

    prompt_provider = model.model if hasattr(model, "model") else model
    if mode in {"text_shift", "joint_shift", "failure"} and (
        hasattr(prompt_provider, "get_clean_prompt_bank") or hasattr(prompt_provider, "clean_prompt_bank")
    ):
        clean_prompt_bank = prompt_provider.get_clean_prompt_bank() if hasattr(prompt_provider, "get_clean_prompt_bank") else getattr(prompt_provider, "clean_prompt_bank", None)
        shifted_prompt_bank = prompt_provider.get_shifted_prompt_bank() if hasattr(prompt_provider, "get_shifted_prompt_bank") else getattr(prompt_provider, "shift_prompt_bank", None)
        run_logger.set_prompt_bank(
            clean_prompt_bank=clean_prompt_bank,
            shifted_prompt_bank=shifted_prompt_bank,
            class_names=list(getattr(prompt_provider, "class_names", [])) if hasattr(prompt_provider, "class_names") else None,
        )

    logger.info(
        "[RUN] "
        f"mode={mode} method={cfg.MODEL.ADAPTATION} dataset={cfg.CORRUPTION.DATASET} "
        f"arch={cfg.MODEL.ARCH} seed={cfg.RNG_SEED} output_dir={run_logger.output_dir}"
    )
    logger.info(
        "[RUN] "
        f"visual_shift={cfg.CORRUPTION.TYPE} severities={cfg.CORRUPTION.SEVERITY} "
        f"text_shift_enabled={cfg.TEXT_SHIFT.ENABLED} text_family={cfg.TEXT_SHIFT.FAMILY} "
        f"text_level={cfg.TEXT_SHIFT.LEVEL} text_protocol={cfg.TEXT_SHIFT.PROTOCOL}"
    )
    logger.info(
        "[RUN] "
        f"trainable_params={getattr(model, 'num_trainable_params', None)}"
    )
    effective_strength = int(getattr(cfg.TEXT_SHIFT, "LEVEL", 1)) if bool(getattr(cfg.TEXT_SHIFT, "ENABLED", False)) else 0
    logger.info(
        "[TEXT-SHIFT] "
        f"enabled={bool(getattr(cfg.TEXT_SHIFT, 'ENABLED', False))} "
        f"regime={getattr(cfg.TEXT_SHIFT, 'REGIME', 'clean')} "
        f"protocol={getattr(cfg.TEXT_SHIFT, 'PROTOCOL', 'main')} "
        f"family={getattr(cfg.TEXT_SHIFT, 'FAMILY', 'N/A')} "
        f"interference_strength_level={effective_strength}"
    )

    domain_dict = {}
    try:
        # get the test sequence containing the corruptions or domain names
        if cfg.CORRUPTION.DATASET == "domainnet126":
            # extract the domain sequence for a specific checkpoint.
            domain_sequence = ckpt_path_to_domain_seq(ckpt_path=cfg.MODEL.CKPT_PATH)
        elif cfg.CORRUPTION.DATASET in ["imagenet_d", "imagenet_d109"] and not cfg.CORRUPTION.TYPE[0]:
            # domain_sequence = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
            domain_sequence = ["clipart", "infograph", "painting", "real", "sketch"]
        else:
            domain_sequence = cfg.CORRUPTION.TYPE
        logger.info(f"Using {cfg.CORRUPTION.DATASET} with the following domain sequence: {domain_sequence}")

        # prevent iterating multiple times over the same data in the mixed_domains setting
        domain_seq_loop = ["mixed"] if "mixed_domains" in cfg.SETTING else domain_sequence

        # setup the severities for the gradual setting
        if "gradual" in cfg.SETTING and cfg.CORRUPTION.DATASET in ["cifar10_c", "cifar100_c", "imagenet_c"] and len(cfg.CORRUPTION.SEVERITY) == 1:
            severities = [1, 2, 3, 4, 5, 4, 3, 2, 1]
            logger.info(f"Using the following severity sequence for each domain: {severities}")
        else:
            severities = cfg.CORRUPTION.SEVERITY

        errs = []
        errs_5 = []
        shift_index = 0
        global_sample_offset = 0

        # start evaluation
        for i_dom, domain_name in enumerate(domain_seq_loop):
            if i_dom == 0 or "reset_each_shift" in cfg.SETTING:
                try:
                    model.reset()
                    logger.info("resetting model")
                except AttributeError:
                    logger.warning("not resetting model")
            else:
                logger.warning("not resetting model")

            for severity in severities:
                shift_meta = build_shift_meta(
                    cfg_=cfg,
                    mode=mode,
                    domain_name=domain_name,
                    severity=severity,
                    shift_index=shift_index,
                    model_wrapper=model,
                )

                if mode == "visual_shift":
                    logger.info(f"Start visual shift: {domain_name} severity={severity}")
                elif mode == "text_shift":
                    logger.info(
                        f"Start text shift: {shift_meta.get('text_shift_family')} level={shift_meta.get('text_shift_level')}"
                    )
                elif mode == "joint_shift":
                    logger.info(
                        f"Start joint shift: {domain_name} severity={severity} + "
                        f"{shift_meta.get('text_shift_family')} level={shift_meta.get('text_shift_level')}"
                    )
                else:
                    logger.info(
                        f"Start failure shift: {domain_name} severity={severity} + "
                        f"{shift_meta.get('text_shift_family')} level={shift_meta.get('text_shift_level')}"
                    )

                test_data_loader = get_test_loader(
                    setting=cfg.SETTING,
                    adaptation=cfg.MODEL.ADAPTATION,
                    dataset_name=cfg.CORRUPTION.DATASET,
                    preprocess=model_preprocess,
                    data_root_dir=cfg.DATA_DIR,
                    domain_name=domain_name,
                    domain_names_all=domain_sequence,
                    severity=severity,
                    num_examples=cfg.CORRUPTION.NUM_EX,
                    rng_seed=cfg.RNG_SEED,
                    use_clip=cfg.MODEL.USE_CLIP,
                    n_views=cfg.TEST.N_AUGMENTATIONS,
                    delta_dirichlet=cfg.TEST.DELTA_DIRICHLET,
                    batch_size=cfg.TEST.BATCH_SIZE,
                    shuffle=False,
                    workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count())
                )

                if i_dom == 0:
                    # Note that the input normalization is done inside of the model
                    logger.info(f"Using the following data transformation:\n{test_data_loader.dataset.transform}")
                    model_side_norm = None
                    if hasattr(model, "model") and hasattr(model.model, "normalize"):
                        model_side_norm = model.model.normalize
                    elif hasattr(base_model, "normalize"):
                        model_side_norm = base_model.normalize
                    logger.info(f"Using the following model-side normalization:\n{model_side_norm}")

                # evaluate the model
                acc, domain_dict, num_samples, shift_summary = get_accuracy(
                    model,
                    data_loader=test_data_loader,
                    dataset_name=cfg.CORRUPTION.DATASET,
                    domain_name=domain_name,
                    setting=cfg.SETTING,
                    domain_dict=domain_dict,
                    print_every=cfg.PRINT_EVERY,
                    device=device,
                    mode=mode,
                    method_name=cfg.MODEL.ADAPTATION,
                    shift_meta=shift_meta,
                    shift_index=shift_index,
                    global_sample_offset=global_sample_offset,
                    run_logger=run_logger,
                    num_classes=num_classes,
                )
                global_sample_offset += num_samples
                shift_index += 1

                err = 1. - acc
                errs.append(err)
                if severity == 5 and domain_name != "none":
                    errs_5.append(err)

                logger.info(
                    f"{cfg.CORRUPTION.DATASET} acc/error [{domain_name}{severity}]"
                    f"[#samples={num_samples}]: acc={acc:.2%}, err={err:.2%}"
                )

                if shift_summary is not None:
                    s_acc = shift_summary.get("acc")
                    s_err = shift_summary.get("err")
                    shift_name = shift_summary.get("shift_id") or f"{domain_name}{severity}"
                    logger.info(
                        "[SHIFT-SUMMARY] "
                        f"shift={shift_name} "
                        f"acc={(0.0 if s_acc is None else float(s_acc)):.4f} "
                        f"err={(0.0 if s_err is None else float(s_err)):.4f} "
                        f"d_ent={shift_summary.get('delta_entropy_mean_global')} "
                        f"maj={shift_summary.get('majorization_ratio_post_over_pre_global')} "
                        f"collapse={shift_summary.get('collapse_rate')} "
                        f"conf_pre={shift_summary.get('top1_conf_pre_mean_global')} "
                        f"conf_post={shift_summary.get('top1_conf_post_mean_global')} "
                        f"d_conf={shift_summary.get('delta_top1_conf_mean_global')} "
                        f"dist_pre={shift_summary.get('dist_to_perm_pre_mean_global')} "
                        f"dist_post={shift_summary.get('dist_to_perm_post_mean_global')} "
                        f"d_dist={shift_summary.get('delta_dist_to_perm_mean_global')} "
                        f"flip={shift_summary.get('pred_flip_rate_global')} "
                        f"w2r={shift_summary.get('wrong_to_right_rate_global')} "
                        f"r2w={shift_summary.get('right_to_wrong_rate_global')} "
                        f"mean_batch_marginal_entropy={shift_summary.get('mean_batch_marginal_entropy')}"
                    )
                    if shift_summary.get("alpha_mean_global") is not None:
                        logger.info(
                            "[SHIFT-SUMMARY] "
                            f"alpha={shift_summary.get('alpha_mean_global')} "
                            f"rho_v={shift_summary.get('rho_v_mean_global')} "
                            f"rho_t={shift_summary.get('rho_t_mean_global')} "
                            f"conflict={shift_summary.get('conflict_mean_global')}"
                        )

                if bool(getattr(run_logger, "fig5_capture_done", False)):
                    logger.info("[FIG5-CAPTURE] target samples collected; stopping remaining shifts early.")
                    break

            if bool(getattr(run_logger, "fig5_capture_done", False)):
                break

        if len(errs_5) > 0:
            logger.info(f"mean error: {np.mean(errs):.2%}, mean error at 5: {np.mean(errs_5):.2%}")
        elif len(errs) > 0:
            logger.info(f"mean error: {np.mean(errs):.2%}")
        else:
            logger.warning("No evaluated shifts; mean error is unavailable.")

        if "mixed_domains" in cfg.SETTING and len(domain_dict.values()) > 0:
            # print detailed results for each domain
            eval_domain_dict(domain_dict, domain_seq=domain_sequence)

        if cfg.TEST.DEBUG:
            print_memory_info()
    finally:
        try:
            run_summary = finalize_run_summary(
                run_logger,
                top_k=int(getattr(getattr(cfg, "EXPERIMENT_OUTPUT", None), "TOP_K_BEST_WORST", 5)),
            )
            logger.info(
                "[RUN-SUMMARY] "
                f"global_acc={run_summary.get('global_mean_acc')} "
                f"global_err={run_summary.get('global_mean_err')} "
                f"total_samples={run_summary.get('total_num_samples')} "
                f"total_batches={run_summary.get('total_num_batches')}"
            )
            logger.info(
                "[RUN-SUMMARY] "
                f"text_shift_family={getattr(cfg.TEXT_SHIFT, 'FAMILY', 'N/A')} "
                f"text_shift_strength_level="
                f"{int(getattr(cfg.TEXT_SHIFT, 'LEVEL', 1)) if bool(getattr(cfg.TEXT_SHIFT, 'ENABLED', False)) else 0}"
            )
            logger.info(f"[RUN-SUMMARY] artifacts_dir={run_logger.output_dir}")
        except Exception:
            logger.exception("[RUN-SUMMARY] failed to finalize run artifacts.")


if __name__ == '__main__':
    evaluate('"Evaluation.')

import json
import logging

import timm
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from open_clip import create_model_and_transforms, get_tokenizer
from robustbench.model_zoo.architectures.utils_architectures import normalize_model, ImageNormalizer
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model

from typing import Union
from copy import deepcopy
from models import resnet26
from models.custom_clip import ClipTestTimePromptTuning
from packaging import version
from mydatasets.cls_names import get_class_names
from mydatasets.imagenet_subsets import IMAGENET_A_MASK, IMAGENET_R_MASK, IMAGENET_V2_MASK, IMAGENET_D109_MASK
from mydatasets.prompts import *
from prompts.text_shift_engine import TextShiftEngine


logger = logging.getLogger(__name__)


def get_torchvision_model(model_name: str, weight_version: str = "IMAGENET1K_V1"):
    """
    Restore a pre-trained model from torchvision
    Further details can be found here: https://pytorch.org/vision/0.14/models.html
    Input:
        model_name: Name of the model to create and initialize with pre-trained weights
        weight_version: Name of the pre-trained weights to restore
    Returns:
        model: The pre-trained model
        preprocess: The corresponding input pre-processing
    """
    assert version.parse(torchvision.__version__) >= version.parse("0.13"), "Torchvision version has to be >= 0.13"

    # check if the specified model name is available in torchvision
    available_models = torchvision.models.list_models(module=torchvision.models)
    if model_name not in available_models:
        raise ValueError(f"Model '{model_name}' is not available in torchvision. Choose from: {available_models}")

    # get the weight object of the specified model and the available weight initialization names
    model_weights = torchvision.models.get_model_weights(model_name)
    available_weights = [init_name for init_name in dir(model_weights) if "IMAGENET1K" in init_name]

    # check if the specified type of weights is available
    if weight_version not in available_weights:
        raise ValueError(f"Weight type '{weight_version}' is not supported for torchvision model '{model_name}'."
                         f" Choose from: {available_weights}")

    # restore the specified weights
    model_weights = getattr(model_weights, weight_version)

    # setup the specified model and initialize it with the specified pre-trained weights
    model = torchvision.models.get_model(model_name, weights=model_weights)

    # get the transformation and add the input normalization to the model
    transform = model_weights.transforms()
    model = normalize_model(model, transform.mean, transform.std)
    logger.info(f"Successfully restored '{weight_version}' pre-trained weights"
                f" for model '{model_name}' from torchvision!")

    # create the corresponding input transformation
    preprocess = transforms.Compose([transforms.Resize(transform.resize_size, interpolation=transform.interpolation),
                                     transforms.CenterCrop(transform.crop_size),
                                     transforms.ToTensor()])
    return model, preprocess


def get_timm_model(model_name: str):
    """
    Restore a pre-trained model from timm: <redacted_url>
    Quickstart: https://huggingface.co/docs/timm/quickstart
    Input:
        model_name: Name of the model to create and initialize with pre-trained weights
    Returns:
        model: The pre-trained model
        preprocess: The corresponding input pre-processing
    """
    # check if the defined model name is supported as pre-trained model
    available_models = timm.list_models(pretrained=True)
    if model_name not in available_models:
        raise ValueError(f"Model '{model_name}' is not available in timm. Choose from: {available_models}")

    # setup pre-trained model
    model = timm.create_model(model_name, pretrained=True)
    logger.info(f"Successfully restored the weights of '{model_name}' from timm.")

    # restore the input pre-processing
    data_config = timm.data.resolve_model_data_config(model)
    preprocess = timm.data.create_transform(**data_config)

    # if there is an input normalization, add it to the model and remove it from the input pre-processing
    for transf in preprocess.transforms[::-1]:
        if isinstance(transf, transforms.Normalize):
            # add input normalization to the model
            model = normalize_model(model, mean=transf.mean, std=transf.std)
            preprocess.transforms.remove(transf)
            break

    return model, preprocess


class ResNetDomainNet126(torch.nn.Module):
    """
    Architecture used for DomainNet-126
    """
    def __init__(self, arch: str = "resnet50", checkpoint_path: str = None, num_classes: int = 126, bottleneck_dim: int = 256):
        super().__init__()

        self.arch = arch
        self.bottleneck_dim = bottleneck_dim
        self.weight_norm_dim = 0

        # 1) ResNet backbone (up to penultimate layer)
        if not self.use_bottleneck:
            model = torchvision.models.get_model(self.arch, weights="IMAGENET1K_V1")
            modules = list(model.children())[:-1]
            self.encoder = torch.nn.Sequential(*modules)
            self._output_dim = model.fc.in_features
        # 2) ResNet backbone + bottlenck (last fc as bottleneck)
        else:
            model = torchvision.models.get_model(self.arch, weights="IMAGENET1K_V1")
            model.fc = torch.nn.Linear(model.fc.in_features, self.bottleneck_dim)
            bn = torch.nn.BatchNorm1d(self.bottleneck_dim)
            self.encoder = torch.nn.Sequential(model, bn)
            self._output_dim = self.bottleneck_dim

        self.fc = torch.nn.Linear(self.output_dim, num_classes)

        if self.use_weight_norm:
            self.fc = torch.nn.utils.weight_norm(self.fc, dim=self.weight_norm_dim)

        if checkpoint_path:
            self.load_from_checkpoint(checkpoint_path)
        else:
            logger.warning(f"No checkpoint path was specified. Continue with ImageNet pre-trained weights!")

        # add input normalization to the model
        self.encoder = nn.Sequential(ImageNormalizer((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), self.encoder)

    def forward(self, x, return_feats=False):
        # 1) encoder feature
        feat = self.encoder(x)
        feat = torch.flatten(feat, 1)

        logits = self.fc(feat)

        if return_feats:
            return feat, logits
        return logits

    def load_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = dict()
        model_state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint.keys() else checkpoint["model"]
        for name, param in model_state_dict.items():
            # get rid of 'module.' prefix brought by DDP
            name = name.replace("module.", "")
            state_dict[name] = param
        msg = self.load_state_dict(state_dict, strict=False)
        logging.info(
            f"Loaded from {checkpoint_path}; missing params: {msg.missing_keys}"
        )

    def get_params(self):
        """
        Backbone parameters use 1x lr; extra parameters use 10x lr.
        """
        backbone_params = []
        extra_params = []
        # case 1)
        if not self.use_bottleneck:
            backbone_params.extend(self.encoder.parameters())
        # case 2)
        else:
            resnet = self.encoder[1][0]
            for module in list(resnet.children())[:-1]:
                backbone_params.extend(module.parameters())
            # bottleneck fc + (bn) + classifier fc
            extra_params.extend(resnet.fc.parameters())
            extra_params.extend(self.encoder[1][1].parameters())
            extra_params.extend(self.fc.parameters())

        # exclude frozen params
        backbone_params = [param for param in backbone_params if param.requires_grad]
        extra_params = [param for param in extra_params if param.requires_grad]

        return backbone_params, extra_params

    @property
    def num_classes(self):
        return self.fc.weight.shape[0]

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def use_bottleneck(self):
        return self.bottleneck_dim > 0

    @property
    def use_weight_norm(self):
        return self.weight_norm_dim >= 0


class BaseModel(torch.nn.Module):
    """
    Change the model structure to perform the adaptation "AdaContrast" for other datasets
    """
    def __init__(self, model, arch_name: str, dataset_name: str):
        super().__init__()

        self.encoder, self.fc = split_up_model(model, arch_name=arch_name, dataset_name=dataset_name)
        if isinstance(self.fc, nn.Sequential):
            for module in self.fc.modules():
                if isinstance(module, nn.Linear):
                    self._num_classes = module.out_features
                    self._output_dim = module.in_features
        elif isinstance(self.fc, nn.Linear):
            self._num_classes = self.fc.out_features
            self._output_dim = self.fc.in_features
        else:
            raise ValueError("Unable to detect output dimensions")

    def forward(self, x, return_feats=False):
        # 1) encoder feature
        feat = self.encoder(x)
        feat = torch.flatten(feat, 1)

        logits = self.fc(feat)

        if return_feats:
            return feat, logits
        return logits

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def output_dim(self):
        return self._output_dim


class ImageNetXMaskingLayer(torch.nn.Module):
    """ Following: <redacted_url>
    """
    def __init__(self, mask):
        super().__init__()
        self.mask = mask

    def forward(self, x):
        return x[:, self.mask]


class ImageNetXWrapper(torch.nn.Module):
    def __init__(self, model, mask):
        super().__init__()
        self.__dict__ = model.__dict__.copy()

        self.masking_layer = ImageNetXMaskingLayer(mask)

    def forward(self, x):
        logits = self.model(self.normalize(x))
        return self.masking_layer(logits)


class TransformerWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.__dict__ = model.__dict__.copy()

    def forward(self, x):
        # Reshape and permute the input tensor
        x = self.normalize(x)
        x = self.model._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.model.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        return x


class ZeroShotCLIP(nn.Module):
    def __init__(self, cfg, model, device, normalize):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.device = device
        self.normalize = normalize
        self.prompt_mode = cfg.CLIP.PROMPT_MODE
        self.freeze_text_encoder = cfg.CLIP.FREEZE_TEXT_ENCODER
        self.class_names = get_class_names(cfg.CORRUPTION.DATASET)
        self.num_classes = len(self.class_names)
        self.tokenize = get_tokenizer(cfg.MODEL.ARCH)
        self.logit_scale = self.model.logit_scale.data
        self._input_stats_logged = False
        self.text_shift_engine = TextShiftEngine(cfg)

        unique_class_names = len(set(self.class_names))
        if unique_class_names != self.num_classes:
            logger.warning(
                "[SANITY][class-names] "
                f"dataset class list has duplicated display names: total={self.num_classes}, unique={unique_class_names}. "
                "Will preserve class-index alignment when building prompt/text banks."
            )

        assert self.prompt_mode in ["custom", "ensemble", "cupl", "all_prompts"]

        prompt_templates = cfg.CLIP.PROMPT_TEMPLATE
        logger.info(f"[SANITY][prompt-config] raw_prompt_templates={prompt_templates}")
        if self.prompt_mode in ["ensemble", "all_prompts"]:
            try:
                prompt_templates = eval(f"{cfg.CORRUPTION.DATASET.split('_')[0]}_templates")
            except NameError:
                logger.warning("Could not find dataset specific prompt templates! Using ImageNet prompt templates!")
                prompt_templates = eval("imagenet_templates")
            logger.info(f"Using the following prompt templates: {prompt_templates}")

        self.clean_prompt_templates = list(prompt_templates)
        self.gpt3_prompts = None
        if self.prompt_mode not in ["custom", "ensemble"]:
            with open(cfg.CLIP.PROMPT_PATH) as f:
                self.gpt3_prompts = json.load(f)
            logger.info(f"Successfully restored CuPL prompts from '{cfg.CLIP.PROMPT_PATH}'")
        else:
            logger.info("[SANITY][prompt-config] CuPL prompts are disabled for this prompt_mode")

        with torch.no_grad():
            self.clean_prompt_bank = self._build_clean_prompt_bank()
            self.text_features_clean, self.text_pre_features_clean = self.build_text_bank_from_prompt_bank(
                self.clean_prompt_bank
            )

            if self.text_shift_engine.enabled():
                self.shift_prompt_bank = self.text_shift_engine.build_prompt_bank(
                    class_names=self.class_names,
                    family=self.cfg.TEXT_SHIFT.FAMILY,
                    level=self.cfg.TEXT_SHIFT.LEVEL,
                    protocol=self.cfg.TEXT_SHIFT.PROTOCOL,
                    base_templates=self.clean_prompt_templates,
                )
                self.text_features_shifted, self.text_pre_features_shifted = self.build_text_bank_from_prompt_bank(
                    self.shift_prompt_bank
                )
            else:
                self.shift_prompt_bank = self.clean_prompt_bank
                self.text_features_shifted = self.text_features_clean
                self.text_pre_features_shifted = self.text_pre_features_clean

            self.text_features = self.get_active_text_bank()
            self.text_pre_features = self.get_active_text_pre_features()
            self.tokenized_texts_all = self.tokenize(self._flatten_prompt_bank(self.clean_prompt_bank)).to(self.device)

        if self.freeze_text_encoder:
            self.model.transformer = None

    @property
    def dtype(self):
        if hasattr(self.model.visual, "conv1") and hasattr(self.model.visual.conv1, "weight"):
            return self.model.visual.conv1.weight.dtype
        return next(self.model.parameters()).dtype

    def _build_clean_prompt_bank(self):
        prompt_bank = {}
        for c_name in self.class_names:
            texts = [template.format(c_name) for template in self.clean_prompt_templates] if self.prompt_mode != "cupl" else []
            if self.prompt_mode in ["cupl", "all_prompts"] and self.gpt3_prompts is not None:
                texts += [t for t in self.gpt3_prompts[c_name]]
            prompt_bank[c_name] = texts
        return prompt_bank

    def _flatten_prompt_bank(self, prompt_bank):
        all_texts = []
        if isinstance(prompt_bank, dict):
            # Preserve class-index alignment even if class names contain duplicates.
            for c_name in self.class_names:
                all_texts.extend(prompt_bank[c_name])
        else:
            for texts in prompt_bank:
                all_texts.extend(texts)
        return all_texts

    @torch.no_grad()
    def build_text_bank_from_prompt_bank(self, prompt_bank):
        text_features = []
        text_pre_features = []

        all_texts = self._flatten_prompt_bank(prompt_bank)
        logger.info(
            "[SANITY][prompt-bank] "
            f"classes={self.num_classes} total_prompts={len(all_texts)}"
        )
        if bool(getattr(getattr(self.cfg, "LOG", None), "PROMPT_PREVIEW", False)):
            for i, p in enumerate(all_texts[:10]):
                logger.info(f"[SANITY][prompt-bank-preview] {i}: {p}")

        for c_name in self.class_names:
            texts = prompt_bank[c_name]
            tokenized = self.tokenize(texts).to(self.device)
            class_embeddings = self.model.encode_text(tokenized)
            text_pre_features.append(class_embeddings)

            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding = class_embedding / class_embedding.norm()
            text_features.append(class_embedding)

        text_features = torch.stack(text_features, dim=0).to(self.device)
        text_pre_features = torch.stack(text_pre_features, dim=0).to(self.device)
        return text_features, text_pre_features

    @torch.no_grad()
    def build_text_bank(self, prompt_templates):
        prompt_bank = {
            c_name: [template.format(c_name) for template in prompt_templates]
            for c_name in self.class_names
        }
        return self.build_text_bank_from_prompt_bank(prompt_bank)

    def get_clean_text_bank(self):
        return self.text_features_clean

    def get_shifted_text_bank(self):
        return self.text_features_shifted

    def get_clean_prompt_bank(self):
        return self.clean_prompt_bank

    def get_shifted_prompt_bank(self):
        return self.shift_prompt_bank

    def get_prompt_templates(self):
        return {
            "clean": list(self.clean_prompt_templates),
            "shift_family": getattr(self.cfg.TEXT_SHIFT, "FAMILY", None),
            "shift_level": getattr(self.cfg.TEXT_SHIFT, "LEVEL", None),
            "shift_protocol": getattr(self.cfg.TEXT_SHIFT, "PROTOCOL", None),
        }

    def get_active_text_bank(self):
        if getattr(self.cfg.TEXT_SHIFT, "ENABLED", False):
            return self.text_features_shifted
        return self.text_features_clean

    def get_active_text_pre_features(self):
        if getattr(self.cfg.TEXT_SHIFT, "ENABLED", False):
            return self.text_pre_features_shifted
        return self.text_pre_features_clean

    def encode_image_features(self, imgs_test):
        log_input_stats = bool(getattr(getattr(self.cfg, "LOG", None), "INPUT_STATS", False))
        if log_input_stats and not self._input_stats_logged:
            x0 = imgs_test.detach().float()
            logger.info(
                "[SANITY][input-before-model-normalize] "
                f"min={x0.min().item():.6f} max={x0.max().item():.6f} "
                f"mean={x0.mean().item():.6f} std={x0.std(unbiased=False).item():.6f}"
            )

        imgs_test = self.normalize(imgs_test.float())
        imgs_test = imgs_test.to(dtype=self.dtype)

        if log_input_stats and not self._input_stats_logged:
            x1 = imgs_test.detach().float()
            logger.info(
                "[SANITY][input-after-model-normalize] "
                f"min={x1.min().item():.6f} max={x1.max().item():.6f} "
                f"mean={x1.mean().item():.6f} std={x1.std(unbiased=False).item():.6f}"
            )
            self._input_stats_logged = True

        img_pre_features = self.model.encode_image(imgs_test)
        img_features = img_pre_features / img_pre_features.norm(dim=1, keepdim=True)
        return img_features, img_pre_features

    @torch.no_grad()
    def logits_with_text_bank(self, imgs_test, text_features):
        img_features, img_pre_features = self.encode_image_features(imgs_test)
        logits = self.logit_scale.exp() * img_features @ text_features.t()
        return logits, img_features, img_pre_features

    # Variant that allows autograd through the image->logits path.
    def logits_with_text_bank_grad(self, imgs_test, text_features):
        img_features, img_pre_features = self.encode_image_features(imgs_test)
        logits = self.logit_scale.exp() * img_features @ text_features.t()
        return logits, img_features, img_pre_features

    def forward(self, imgs_test, return_features=False):
        img_features, img_pre_features = self.encode_image_features(imgs_test)

        if self.freeze_text_encoder or self.cfg.MODEL.ADAPTATION == "source" or "norm" in self.cfg.MODEL.ADAPTATION:
            text_pre_features = self.get_active_text_pre_features()
            if text_pre_features.dim() == 3 and text_pre_features.shape[1] == 1:
                text_pre_features = text_pre_features.squeeze(1)
            text_features = self.get_active_text_bank()
        else:
            text_pre_features = self.model.encode_text(self.tokenized_texts_all)
            text_features = text_pre_features / text_pre_features.norm(dim=1, keepdim=True)

        logits_per_image = self.logit_scale.exp() * img_features @ text_features.T

        if return_features:
            return logits_per_image, img_features, text_features, img_pre_features, text_pre_features
        return logits_per_image


def get_model(cfg, num_classes: int, device: Union[str, torch.device]):
    """
    Setup the pre-defined model architecture and restore the corresponding pre-trained weights
    Input:
        cfg: Configurations
        num_classes: Number of classes
        device: The device to put the loaded model
    Return:
        model: The pre-trained model
        preprocess: The corresponding input pre-processing
    """
    preprocess = None

    if cfg.MODEL.USE_CLIP:
        if cfg.MODEL.ARCH == "ViT-B-16" and isinstance(cfg.MODEL.WEIGHTS, str):
            w_lower = cfg.MODEL.WEIGHTS.lower()
            if "openai" in w_lower or "open_clip_model.safetensors" in w_lower:
                msg = (
                    "[SANITY][model-config] ViT-B-16 + OpenAI CLIP weights detected. "
                    "Use MODEL.ARCH=ViT-B-16-quickgelu for official OpenAI ViT-B/16 reproduction."
                )
                if cfg.CLIP.STRICT_OPENAI_ARCH:
                    raise ValueError(msg)
                logger.warning(msg)

        # load pre-trained CLIP model
        base_model, _, preprocess = create_model_and_transforms(cfg.MODEL.ARCH,
                                                                pretrained=cfg.MODEL.WEIGHTS,
                                                                device=device,
                                                                precision=cfg.CLIP.PRECISION)
        # get the image input normalization
        normalization = preprocess.transforms[-1]
        # remove the input normalization from the pre-processing as it will be added to the model
        preprocess.transforms = preprocess.transforms[:-1]

        if cfg.MODEL.ADAPTATION == "tpt":
            base_model = ClipTestTimePromptTuning(base_model, normalization,
                                                  cfg.MODEL.ARCH, cfg.CORRUPTION.DATASET,
                                                  n_ctx=cfg.TPT.N_CTX, ctx_init=cfg.TPT.CTX_INIT,
                                                  class_token_pos=cfg.TPT.CLASS_TOKEN_POS)
            if cfg.MODEL.CKPT_PATH:
                # Initiaize context prompts with CoOp pre-trained prompts (see: <redacted_url>)
                # or download from an external artifact source documented in README_anonymous.md
                pretrained_ctx = torch.load(cfg.MODEL.CKPT_PATH)['state_dict']['ctx']
                assert pretrained_ctx.shape[0] == cfg.TPT.N_CTX
                with torch.no_grad():
                    base_model.prompt_learner.ctx.copy_(pretrained_ctx)
                    base_model.prompt_learner.ctx_init_state = pretrained_ctx
                logger.info("Successfully restored pre-trained soft prompt (CoOp)")
        else:
            base_model = ZeroShotCLIP(cfg, base_model, device, normalize=normalization)

    elif cfg.CORRUPTION.DATASET == "domainnet126":
        base_model = ResNetDomainNet126(arch=cfg.MODEL.ARCH, checkpoint_path=cfg.MODEL.CKPT_PATH, num_classes=num_classes)
    else:
        try:
            # load model from torchvision
            base_model, preprocess = get_torchvision_model(cfg.MODEL.ARCH, weight_version=cfg.MODEL.WEIGHTS)
        except ValueError:
            try:
                # load model from timm
                base_model, preprocess = get_timm_model(cfg.MODEL.ARCH)
            except ValueError:
                try:
                    # load some custom models
                    if cfg.MODEL.ARCH == "resnet26_gn":
                        base_model = resnet26.build_resnet26()
                        checkpoint = torch.load(cfg.MODEL.CKPT_PATH, map_location="cpu")
                        base_model.load_state_dict(checkpoint['net'])
                        base_model = normalize_model(base_model, resnet26.MEAN, resnet26.STD)
                    else:
                        raise ValueError(f"Model {cfg.MODEL.ARCH} is not supported!")
                    logger.info(f"Successfully restored model '{cfg.MODEL.ARCH}' from: {cfg.MODEL.CKPT_PATH}")
                except ValueError:
                    # load model from robustbench
                    dataset_name = cfg.CORRUPTION.DATASET.split("_")[0]
                    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR, dataset_name, ThreatModel.corruptions)

        # In case of the imagenet variants, wrap a mask around the output layer to get the correct classes
        if cfg.CORRUPTION.DATASET in ["imagenet_a", "imagenet_r", "imagenet_v2", "imagenet_d109"]:
            mask = eval(f"{cfg.CORRUPTION.DATASET.upper()}_MASK")
            base_model = ImageNetXWrapper(base_model, mask=mask)

    return base_model.to(device), preprocess


def split_up_model(model, arch_name: str, dataset_name: str):
    """
    Split up the model into an encoder and a classifier.
    This is required for methods like RMT and AdaContrast
    Input:
        model: Model to be split up
        arch_name: Name of the network
        dataset_name: Name of the dataset
    Returns:
        encoder: The encoder of the model
        classifier The classifier of the model
    """
    if hasattr(model, "model") and hasattr(model.model, "pretrained_cfg") and hasattr(model.model, model.model.pretrained_cfg["classifier"]):
        # split up models loaded from timm
        classifier = deepcopy(getattr(model.model, model.model.pretrained_cfg["classifier"]))
        encoder = model
        encoder.model.reset_classifier(0)
        if isinstance(model, ImageNetXWrapper):
            encoder = nn.Sequential(encoder.normalize, encoder.model)

    elif arch_name == "Standard" and dataset_name in {"cifar10", "cifar10_c"}:
        encoder = nn.Sequential(*list(model.children())[:-1], nn.AvgPool2d(kernel_size=8, stride=8), nn.Flatten())
        classifier = model.fc
    elif arch_name == "Hendrycks2020AugMix_WRN":
        normalization = ImageNormalizer(mean=model.mu, std=model.sigma)
        encoder = nn.Sequential(normalization, *list(model.children())[:-1], nn.AvgPool2d(kernel_size=8, stride=8), nn.Flatten())
        classifier = model.fc
    elif arch_name == "Hendrycks2020AugMix_ResNeXt":
        normalization = ImageNormalizer(mean=model.mu, std=model.sigma)
        encoder = nn.Sequential(normalization, *list(model.children())[:2], nn.ReLU(), *list(model.children())[2:-1], nn.Flatten())
        classifier = model.classifier
    elif dataset_name == "domainnet126":
        encoder = model.encoder
        classifier = model.fc
    elif "resnet" in arch_name or "resnext" in arch_name or "wide_resnet" in arch_name or arch_name in {"Standard_R50", "Hendrycks2020AugMix", "Hendrycks2020Many", "Geirhos2018_SIN"}:
        encoder = nn.Sequential(model.normalize, *list(model.model.children())[:-1], nn.Flatten())
        classifier = model.model.fc
    elif "densenet" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        classifier = model.model.classifier
    elif "efficientnet" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, model.model.avgpool, nn.Flatten())
        classifier = model.model.classifier
    elif "mnasnet" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.layers, nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten())
        classifier = model.model.classifier
    elif "shufflenet" in arch_name:
        encoder = nn.Sequential(model.normalize, *list(model.model.children())[:-1], nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten())
        classifier = model.model.fc
    elif "vit_" in arch_name and not "maxvit_" in arch_name:
        encoder = TransformerWrapper(model)
        classifier = model.model.heads.head
    elif "swin_" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, model.model.norm, model.model.permute, model.model.avgpool, model.model.flatten)
        classifier = model.model.head
    elif "convnext" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, model.model.avgpool)
        classifier = model.model.classifier
    elif arch_name == "mobilenet_v2":
        encoder = nn.Sequential(model.normalize, model.model.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        classifier = model.model.classifier
    else:
        raise ValueError(f"The model architecture '{arch_name}' is not supported for dataset '{dataset_name}'.")

    # add a masking layer to the classifier
    if dataset_name in ["imagenet_a", "imagenet_r", "imagenet_v2", "imagenet_d109"]:
        mask = eval(f"{dataset_name.upper()}_MASK")
        classifier = nn.Sequential(classifier, ImageNetXMaskingLayer(mask))

    return encoder, classifier

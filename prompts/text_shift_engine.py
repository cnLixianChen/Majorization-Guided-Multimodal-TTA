from __future__ import annotations

import random
from typing import Dict, List, Sequence


class TextShiftEngine:
    """
    Build class-conditioned prompt banks for main textual shift and failure analysis.

    Main protocol families:
        - prompt_bank_degradation
        - tokenization_stress
        - position_context_stress

    Failure protocol families:
        - negation
        - wrong_attribute
        - class_inconsistent
    """

    MAIN_FAMILIES = {
        "prompt_bank_degradation",
        "tokenization_stress",
        "position_context_stress",
    }
    FAILURE_FAMILIES = {"negation", "wrong_attribute", "class_inconsistent"}

    def __init__(self, cfg):
        self.cfg = cfg
        seed = int(getattr(getattr(cfg, "TEXT_SHIFT", object()), "SEED", 0))
        self.rng = random.Random(seed)

    def enabled(self) -> bool:
        return bool(getattr(self.cfg.TEXT_SHIFT, "ENABLED", False))

    def build_prompt_bank(
        self,
        class_names: Sequence[str],
        family: str,
        level: int = 1,
        protocol: str = "main",
        base_templates: Sequence[str] | None = None,
    ) -> Dict[str, List[str]]:
        family = str(family).strip().lower()
        protocol = str(protocol).strip().lower()
        level = max(1, min(int(level), 5))
        class_names = list(class_names)

        # richer clean bank; do not use just one template as the default clean source
        base_templates = list(
            base_templates or [
                "a photo of a {}.",
                "an image of a {}.",
                "a picture of a {}.",
                "a close-up photo of a {}.",
                "a natural image of a {}.",
                "a centered photo of a {}.",
            ]
        )

        if protocol == "main":
            if family not in self.MAIN_FAMILIES:
                raise ValueError(f"Unsupported main text-shift family: {family}")
            return self._build_main_bank(class_names, family, level, base_templates)

        if protocol == "failure":
            if family not in self.FAILURE_FAMILIES:
                raise ValueError(f"Unsupported failure text-shift family: {family}")
            return self._build_failure_bank(class_names, family, level)

        raise ValueError("protocol must be one of {'main', 'failure'}")

    def _build_main_bank(
        self,
        class_names: List[str],
        family: str,
        level: int,
        base_templates: Sequence[str],
    ) -> Dict[str, List[str]]:
        bank: Dict[str, List[str]] = {}

        if family == "prompt_bank_degradation":
            templates = self._prompt_bank_degradation_templates(level, base_templates)
            for c in class_names:
                bank[c] = [t.format(c) for t in templates]
            return bank

        if family == "tokenization_stress":
            for c in class_names:
                prompts = [t.format(c) for t in base_templates]
                bank[c] = self._tokenization_stress_prompts(prompts, c, level)
            return bank

        if family == "position_context_stress":
            templates = self._position_context_stress_templates(level)
            for c in class_names:
                bank[c] = [t.format(c) for t in templates]
            return bank

        raise ValueError(f"Unsupported main family: {family}")

    def _build_failure_bank(
        self,
        class_names: List[str],
        family: str,
        level: int,
    ) -> Dict[str, List[str]]:
        bank: Dict[str, List[str]] = {}

        if family == "negation":
            templates = self._negation_templates(level)
            for c in class_names:
                bank[c] = [t.format(c) for t in templates]
            return bank

        if family == "wrong_attribute":
            attrs = self._wrong_attributes(level)
            for c in class_names:
                bank[c] = [f"a photo of a {attr} {c}." for attr in attrs]
            return bank

        if family == "class_inconsistent":
            n = len(class_names)
            shift = max(1, min(level, max(1, n - 1)))
            for i, c in enumerate(class_names):
                other = class_names[(i + shift) % n]
                bank[c] = [
                    f"a photo of a {other}.",
                    f"an image showing a {other}.",
                ]
            return bank

        raise ValueError(f"Unsupported failure family: {family}")

    # ----------------------------
    # Main families
    # ----------------------------

    @staticmethod
    def _prompt_bank_degradation_templates(
        level: int,
        base_templates: Sequence[str],
    ) -> List[str]:
        """
        Severity should get harder as level increases:
        L1: medium bank
        L2: small bank
        L3: single weak template / class-name-like prompts
        """
        clean = list(dict.fromkeys(base_templates))
        fallback = [
            "a photo of a {}.",
            "an image of a {}.",
            "a picture of a {}.",
            "a close-up photo of a {}.",
            "a natural image of a {}.",
            "a centered photo of a {}.",
        ]
        if not clean:
            clean = fallback

        if level == 1:
            return clean[: max(2, min(4, len(clean)))]
        if level == 2:
            return clean[:2]
        # hardest: weak prompt bank
        return ["a photo of a {}."]

    def _tokenization_stress_prompts(
        self,
        prompts: Sequence[str],
        class_name: str,
        level: int,
    ) -> List[str]:
        prompts = list(prompts)

        stressed: List[str] = []

        # Fraction of prompts that receive perturbations per level.
        ratios = {
            1: 0.30,
            2: 0.60,
            3: 0.30,
            4: 1.00,
            5: 1.00,
        }
        k = max(1, round(len(prompts) * ratios[level]))
        chosen = set(range(k))

        for i, p in enumerate(prompts):
            q = p

            if level == 1:
                if i in chosen:
                    q = self._inject_context_typo(q, strength=1)
            elif level == 2:
                if i in chosen:
                    q = self._inject_context_typo(q, strength=1)
                    q = self._perturb_class_boundary(q, class_name)
            elif level == 3:
                # Single-template regime: control severity by applying mild class typo
                # to only a subset of classes (deterministic per class name).
                apply_typo = self._class_fraction_gate(class_name, 0.30)
                q = self._inject_context_typo(q, strength=1)
                q = self._perturb_class_boundary(q, class_name)
                if apply_typo:
                    q = self._inject_class_typo(q, class_name, mode="mild")
            elif level == 4:
                # Stronger than L3: larger class subset receives mild class typo.
                apply_typo = self._class_fraction_gate(class_name, 0.70)
                q = self._inject_context_typo(q, strength=1)
                q = self._perturb_class_boundary(q, class_name)
                q = self._light_context_drop(q)
                if apply_typo:
                    q = self._inject_class_typo(q, class_name, mode="mild")
            else:
                q = self._inject_context_typo(q, strength=1)
                q = self._inject_class_typo(q, class_name, mode="mild")

            stressed.append(q)
        return list(dict.fromkeys(stressed))

    @staticmethod
    def _position_context_stress_templates(level: int) -> List[str]:
        """
        Make the class token increasingly delayed / diluted in context.
        """
        if level == 1:
            return [
                "in a natural scene, a photo of a {}.",
                "in ordinary lighting, an image of a {}.",
                "a real-world photo that contains a {}.",
            ]
        if level == 2:
            return [
                "this image shows an everyday scene with natural appearance; the main object is a {}.",
                "in a realistic environment with visible details and ordinary composition, the object is a {}.",
                "an image captured in the wild with standard visual conditions, where the subject is a {}.",
            ]
        return [
            "this image appears to depict a realistic everyday scene with ordinary composition, natural lighting, and visible local detail; after considering the subject category, it is a {}.",
            "an image recorded under standard viewing conditions with background clutter, local texture, and object-level detail; the central semantic category is a {}.",
            "a real-world visual scene with recognizable structure, foreground-background separation, and category-level cues; the depicted object should be identified as a {}.",
        ]

    # ----------------------------
    # Failure families
    # ----------------------------

    @staticmethod
    def _negation_templates(level: int) -> List[str]:
        base = [
            "not a photo of a {}.",
            "this is not a {}.",
        ]
        extra = [
            "an image without a {}.",
            "no {} is present in the image.",
        ]
        if level <= 1:
            return base
        return base + extra

    @staticmethod
    def _wrong_attributes(level: int) -> List[str]:
        attrs_1 = ["red", "blue", "wooden"]
        attrs_2 = ["metallic", "tiny", "giant"]
        attrs_3 = ["striped", "transparent", "glowing"]
        if level <= 1:
            return attrs_1
        if level == 2:
            return attrs_1 + attrs_2
        return attrs_1 + attrs_2 + attrs_3

    # ----------------------------
    # Helpers
    # ----------------------------

    def _inject_context_typo(self, text: str, strength: int = 1) -> str:
        """Apply mild-to-medium context-only typos without touching class tokens."""
        pairs_1 = [
            ("photo", "ph0to"),
            ("image", "imgae"),
            ("picture", "pictuer"),
        ]
        pairs_2 = [
            ("photo", "ph0to"),
            ("image", "img"),
            ("picture", "pic"),
            ("showing", "showng"),
            ("natural", "natral"),
            ("close-up", "closeup"),
        ]

        pairs = pairs_1 if strength <= 1 else pairs_2
        out = text
        changed = 0
        for src, dst in pairs:
            if src in out:
                out = out.replace(src, dst, 1)
                changed += 1
                if strength <= 1 and changed >= 1:
                    break
                if strength >= 2 and changed >= 3:
                    break
        return out

    def _light_context_drop(self, text: str) -> str:
        """Mildly degrade grammar/context while preserving readability."""
        out = text
        out = out.replace("a photo of a ", "a photo of ", 1)
        out = out.replace("an image of a ", "an image of ", 1)
        out = out.replace("a picture of a ", "a picture of ", 1)
        return out

    def _perturb_class_boundary(self, text: str, class_name: str) -> str:
        """Boundary-level class perturbation that avoids inner-token corruption."""
        if class_name not in text:
            return text

        if " " in class_name:
            candidates = [
                class_name.replace(" ", "-"),
                class_name.replace(" ", "  "),
                class_name.replace(" ", "_"),
            ]
        else:
            candidates = [
                f"{class_name}-object",
                f"{class_name} category",
            ]
        return text.replace(class_name, candidates[0], 1)

    def _inject_class_typo(self, text: str, class_name: str, mode: str = "mild") -> str:
        if class_name not in text:
            return text

        if mode == "mild":
            noisy = self._noisify_class_name_mild(class_name)
        elif mode == "strong_one_token":
            noisy = self._noisify_class_name_strong_one_token(class_name)
        elif mode == "strong_all_tokens":
            noisy = self._noisify_class_name_strong(class_name)
        else:
            raise ValueError(f"Unknown class typo mode: {mode}")

        return text.replace(class_name, noisy, 1)

    def _class_fraction_gate(self, class_name: str, fraction: float) -> bool:
        """Deterministically select a fraction of classes for additional perturbation."""
        frac = max(0.0, min(1.0, float(fraction)))
        score = sum(ord(ch) for ch in class_name) % 100
        return score < int(round(frac * 100))

    def _noisify_class_name_mild(self, class_name: str) -> str:
        tokens = class_name.split()
        if not tokens:
            return class_name

        idx = 0
        tokens[idx] = self._mild_token_typo(tokens[idx])
        return " ".join(tokens)

    def _mild_token_typo(self, token: str) -> str:
        if len(token) <= 3:
            return token + "-"
        if len(token) == 4:
            return token[0] + token[2] + token[1] + token[3]

        chars = list(token)
        pos = min(len(chars) - 3, 1)
        chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        return "".join(chars)

    def _noisify_class_name_strong(self, class_name: str) -> str:
        tokens = class_name.split()
        if not tokens:
            return class_name

        return " ".join(self._strong_token_typo(tok) for tok in tokens)

    def _noisify_class_name_strong_one_token(self, class_name: str) -> str:
        tokens = class_name.split()
        if not tokens:
            return class_name

        idx = 0
        tokens[idx] = self._strong_token_typo(tokens[idx])
        return " ".join(tokens)

    def _strong_token_typo(self, token: str) -> str:
        if len(token) <= 2:
            return token + "-"

        if len(token) == 3:
            vowels = "aeiou"
            chars = list(token)
            for i, ch in enumerate(chars):
                if ch in vowels:
                    chars[i] = "0"
                    return "".join(chars)
            chars[1], chars[2] = chars[2], chars[1]
            return "".join(chars)

        if len(token) == 4:
            chars = list(token)
            chars[1] = "0"
            chars[2], chars[3] = chars[3], chars[2]
            return "".join(chars)

        chars = list(token)
        pos1 = 1
        chars[pos1], chars[pos1 + 1] = chars[pos1 + 1], chars[pos1]

        vowels = "aeiou"
        replaced = False
        for i in range(2, len(chars) - 1):
            if chars[i] in vowels:
                chars[i] = "0"
                replaced = True
                break

        if not replaced and len(chars) >= 6:
            pos2 = min(len(chars) - 3, 3)
            chars[pos2], chars[pos2 + 1] = chars[pos2 + 1], chars[pos2]

        return "".join(chars)

    # Optional: for pilot sanity checks
    def describe_shift_bank(
        self,
        bank: Dict[str, List[str]],
        max_classes: int = 5,
    ) -> Dict[str, object]:
        classes = list(bank.keys())[:max_classes]
        sample = {c: bank[c][:3] for c in classes}
        n_prompts = [len(v) for v in bank.values()]
        avg_len = 0.0
        total = 0
        count = 0
        for prompts in bank.values():
            for p in prompts:
                total += len(p.split())
                count += 1
        if count > 0:
            avg_len = total / count
        return {
            "num_classes": len(bank),
            "avg_prompts_per_class": sum(n_prompts) / max(1, len(n_prompts)),
            "avg_prompt_word_len": avg_len,
            "sample": sample,
        }
"""Language identification for the failure profile.

Copied from greekllm.data.filters (the data-building filters) so the toolkit does not import the training package.
`GreekLangID` is that class unchanged: same GlotLID model, same `min_prob`, same label handling. If the data side
changes it, change it here too, or the "wrong language" rate stops being comparable with earlier profiles.
"""
from __future__ import annotations

import re

from .log import get_logger

log = get_logger("greekeval.langid")

CODE_BLOCK_RE = re.compile(r"```.*?```", re.S)
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
LATEX_RE = re.compile(r"\\\[.*?\\\]|\\\(.*?\\\)|\$\$.*?\$\$|\$[^$\n]{1,200}\$", re.S)
GREEK_RE = re.compile(r"[Ͱ-Ͽἀ-῿]")


def strip_code(text: str) -> str:
    """Remove code blocks, inline code and LaTeX before language checks (they are not 'language')."""
    return LATEX_RE.sub(" ", INLINE_CODE_RE.sub(" ", CODE_BLOCK_RE.sub(" ", text)))


def greek_letter_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if GREEK_RE.match(c)) / len(letters)


class GreekLangID:
    """GlotLID (fastText) if available, else a Greek-script heuristic."""

    def __init__(self, min_prob: float = 0.5):
        self.min_prob = min_prob
        self.model = None
        try:
            import fasttext
            from huggingface_hub import hf_hub_download
            path = hf_hub_download("cis-lmu/glotlid", "model.bin")
            self.model = fasttext.load_model(path)
            log.info("GlotLID loaded")
        except Exception as e:  # noqa: BLE001
            log.warning("GlotLID unavailable (%s); falling back to Greek-script heuristic", e)

    def is_greek(self, text: str) -> tuple[bool, str]:
        return self.is_lang(text, "ell")

    def is_lang(self, text: str, code: str) -> tuple[bool, str]:
        """code: 'ell', 'eng' or 'any' (always passes). Returns (ok, 'label:prob')."""
        t = strip_code(text).replace("\n", " ").strip()
        if not t:
            return False, "empty"
        if code == "any":
            return True, "any"
        if self.model is None:
            share = greek_letter_share(t)
            ok = share >= 0.5 if code == "ell" else share < 0.2
            return ok, f"heuristic:{share:.2f}"
        # call the C++ binding directly: fasttext's python `predict` wrapper breaks under numpy 2
        preds = self.model.f.predict(t[:2000], 1, 0.0, "strict")
        if not preds:
            return False, "no_prediction"
        prob, label = preds[0]
        label = label.replace("__label__", "")
        ok = label.startswith(code) and float(prob) >= self.min_prob
        return ok, f"{label}:{float(prob):.2f}"

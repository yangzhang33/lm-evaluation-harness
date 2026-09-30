"""Message preparation for each model's chat template, used by the judge generator.

Copied from greekllm.train.templates (`TemplateSpec`, `prepare_messages`) so the toolkit does not import the training
package. The judge must send a model the same message shape it was fine-tuned on: if the training side changes
either of these, change them here too.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TemplateSpec:
    assistant_start: str   # e.g. "<|im_start|>assistant\n"
    turn_end: str          # e.g. "<|im_end|>"
    add_reasoning_field: bool = False   # some templates refuse assistant messages without a reasoning field
    system_prompt: str | None = None    # optional fixed system prompt; None = let the template decide


def prepare_messages(messages: list[dict], spec: TemplateSpec) -> list[dict]:
    out = []
    if spec.system_prompt and (not messages or messages[0]["role"] != "system"):
        out.append({"role": "system", "content": spec.system_prompt})
    for m in messages:
        m = {"role": m["role"], "content": m["content"]}
        if spec.add_reasoning_field and m["role"] == "assistant":
            m["reasoning_content"] = ""
        out.append(m)
    return out

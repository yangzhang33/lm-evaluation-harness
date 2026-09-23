"""Derive the greekmmlu_gen_boxed configs from the sibling greekmmlu/ directory, so the two tasks always cover exactly the
same 45 subject configs and 5 groups. Re-run after any change to greekmmlu/:

    python _generate_configs.py
"""
import os
import re

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "greekmmlu")
TEMPLATE = "_default_greekmmlu_gen_boxed_template_yaml"


def rename(value):
    """greekmmlu_<x> -> greekmmlu_gen_<x> in task, group, tag and include names; leaves dataset names alone."""
    if isinstance(value, list):
        return [rename(v) for v in value]
    if isinstance(value, str):
        value = value.replace("_default_greekmmlu_template_yaml", TEMPLATE)
        return re.sub(r"^(_?)greekmmlu(?=_|$)", r"\1greekmmlu_gen_boxed", value)
    return value


if __name__ == "__main__":
    written = 0
    for name in sorted(os.listdir(SRC)):
        if not name.endswith(".yaml"):
            continue
        with open(os.path.join(SRC, name), encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        out = {}
        for key, value in cfg.items():
            out[key] = rename(value) if key in ("include", "tag", "task", "group") else value
        # The source groups aggregate the multiple-choice `acc`; a generative task reports `exact_match` and
        # `parsed` instead, so the aggregate list is rewritten to those two.
        if "aggregate_metric_list" in out:
            out["aggregate_metric_list"] = [
                dict(entry, metric=metric)
                for entry in out["aggregate_metric_list"] if entry.get("metric") == "acc"
                for metric in ("exact_match", "parsed")
            ] + [entry for entry in out["aggregate_metric_list"] if entry.get("metric") != "acc"]
        if "group_alias" in out:
            out["group_alias"] = out["group_alias"] + "_gen_boxed"
        target = os.path.join(HERE, rename(name[:-5]) + ".yaml")
        with open(target, "w", encoding="utf-8") as fh:
            yaml.dump(out, fh, allow_unicode=True, default_flow_style=False, sort_keys=False)
        written += 1
    print(f"wrote {written} yaml files")

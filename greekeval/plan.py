"""Read and check the evaluation plan: the file that says which runs exist (see plan.yaml next to this package).

It lives at config/plan.yaml next to this package; lines that are commented out are not run. Each entry comes back
as a dict with every field filled in: name, form, shots, batch, tasks, template, think, max_gen_toks, root.
"""
from __future__ import annotations

import os

import yaml

from .paths import CONFIG_DIR

FORMS = ("A", "B", "C", "D")
FIELDS = ("name", "form", "shots", "batch", "tasks", "template", "think", "max_gen_toks", "root")


def known_runs(path: str) -> set[tuple[str, str]]:
    """(root, name) of every line of the plan, commented out or not: the names an output folder may carry."""
    out = set()
    for line in open(path, encoding="utf-8"):
        line = line.strip().lstrip("# ").strip()          # a commented-out line is "# - {...}"
        if line.startswith("- "):
            line = line[2:]
        if not line.startswith("{") or "}" not in line:
            continue
        try:
            r = yaml.safe_load(line[:line.rindex("}") + 1])   # drop a trailing comment
        except yaml.YAMLError:
            continue
        if isinstance(r, dict) and r.get("name"):
            out.add((str(r.get("root", "eval")), str(r["name"])))
    return out


def load_plan(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise SystemExit(f"no plan at {path}: the toolkit ships it as {os.path.join(CONFIG_DIR, 'plan.yaml')}; "
                         "restore it or pass --plan")
    with open(path, encoding="utf-8") as fh:
        runs = (yaml.safe_load(fh) or {}).get("runs") or []
    out, seen = [], set()
    for i, r in enumerate(runs, 1):
        where = f"{path}: run {i}" + (f" ({r.get('name')})" if isinstance(r, dict) and r.get("name") else "")
        if not isinstance(r, dict):
            raise SystemExit(f"{where}: not a mapping")
        for k in ("name", "form", "shots", "batch", "tasks"):
            if k not in r:
                raise SystemExit(f"{where}: missing {k!r}")
        unknown = set(r) - set(FIELDS)
        if unknown:
            raise SystemExit(f"{where}: unknown field(s) {sorted(unknown)}")
        form = str(r["form"]).upper()
        if form not in FORMS:
            raise SystemExit(f"{where}: form must be one of {FORMS}, got {r['form']!r}")
        template = str(r.get("template") or ("none" if form in ("A", "B") else "chat")).lower()
        if template not in ("none", "chat"):
            raise SystemExit(f"{where}: template must be none or chat, got {r['template']!r}")
        think = r.get("think", "off")                       # YAML reads a bare `on` as True
        think = "on" if think is True or str(think).lower() in ("on", "true", "yes") else "off"
        if think == "on" and template != "chat":
            raise SystemExit(f"{where}: think: on needs template: chat (the think block lives in the template)")
        if form == "A" and template == "chat":
            raise SystemExit(f"{where}: form A is log-likelihood on a raw prompt; it takes no template")
        root = str(r.get("root", "eval"))
        if root not in ("eval", "eval_think"):
            raise SystemExit(f"{where}: root must be eval or eval_think, got {root!r}")
        key = (root, str(r["name"]))
        if key in seen:
            raise SystemExit(f"{where}: a run named {r['name']!r} already exists under {root}/")
        seen.add(key)
        cap = r.get("max_gen_toks")
        out.append(dict(name=str(r["name"]), form=form, shots=int(r["shots"]), batch=str(r["batch"]),
                        tasks=str(r["tasks"]), template=template, think=think,
                        max_gen_toks=int(cap) if cap is not None else None, root=root))
    return out

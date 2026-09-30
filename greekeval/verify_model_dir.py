"""Check that a local HF model directory is complete before spending GPU hours on it.

    python -m greekeval.verify_model_dir /path/to/model [...]

Validates, without loading the weights: every shard named in the index exists; each safetensors header parses and
its declared tensor extent matches the file size exactly (a truncated or still-downloading shard fails here); the
tensor bytes on disk equal the index's `total_size`; no tensor in the index is missing from the shards; and the
tokenizer/config files a `from_pretrained` needs are present. Exit code 0 = safe to run.
"""
from __future__ import annotations

import json
import os
import struct
import sys

NEEDED = ["config.json", "tokenizer_config.json"]
TOKENIZER_ANY = ["tokenizer.json", "tokenizer.model", "vocab.json"]


def shard_tensors(path: str) -> tuple[dict, int]:
    """(header, bytes required by the header) for one safetensors file."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    ends = [v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__"]
    return hdr, 8 + n + (max(ends) if ends else 0)


def verify(d: str) -> bool:
    print(f"=== {d}")
    if not os.path.isdir(d):
        print("  not a directory"); return False
    partials = [f for f in os.listdir(d) if f.startswith(".") and ".safetensors" in f]
    if partials:
        print(f"  DOWNLOAD IN PROGRESS: {len(partials)} partial file(s), e.g. {partials[0]}")
    idx_path = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        idx = json.load(open(idx_path))
        wmap, total = idx["weight_map"], idx["metadata"]["total_size"]
        shards = sorted(set(wmap.values()))
    elif os.path.exists(os.path.join(d, "model.safetensors")):
        wmap, total, shards = {}, None, ["model.safetensors"]
    else:
        print("  no model.safetensors.index.json and no model.safetensors"); return False

    ok, seen, on_disk = True, set(), 0
    for sh in shards:
        p = os.path.join(d, sh)
        if not os.path.exists(p):
            print(f"  {sh}: MISSING"); ok = False; continue
        size = os.path.getsize(p)
        try:
            hdr, need = shard_tensors(p)
        except Exception as e:
            print(f"  {sh}: unreadable header ({e})"); ok = False; continue
        names = [k for k in hdr if k != "__metadata__"]
        seen.update(names)
        on_disk += sum(hdr[k]["data_offsets"][1] - hdr[k]["data_offsets"][0] for k in names)
        good = need == size
        ok &= good
        print(f"  {sh}: {len(names):4d} tensors, {size / 2**30:6.2f} GiB  "
              f"{'OK' if good else f'TRUNCATED (header needs {need}, file has {size})'}")
    if wmap:
        missing = sorted(set(wmap) - seen)
        print(f"  tensors in index but absent from shards: {len(missing)}"
              + (f" e.g. {missing[:3]}" if missing else ""))
        ok &= not missing
        match = on_disk == total
        print(f"  tensor bytes {on_disk / 2**30:.2f} GiB vs index total {total / 2**30:.2f} GiB -> "
              f"{'MATCH' if match else 'MISMATCH'}")
        ok &= match
    for f in NEEDED:
        if not os.path.exists(os.path.join(d, f)):
            print(f"  missing {f}"); ok = False
    if not any(os.path.exists(os.path.join(d, f)) for f in TOKENIZER_ANY):
        print(f"  missing a tokenizer file (any of {TOKENIZER_ANY})"); ok = False
    tc = os.path.join(d, "tokenizer_config.json")
    if os.path.exists(tc):   # transformers >= 4.50 may save the template as chat_template.jinja instead
        has = bool(json.load(open(tc)).get("chat_template")) or os.path.exists(os.path.join(d, "chat_template.jinja"))
        print(f"  chat_template: {'present' if has else 'ABSENT'}")
    print("  VERDICT:", "complete" if ok and not partials else "INCOMPLETE")
    return ok and not partials


if __name__ == "__main__":
    sys.exit(0 if all([verify(d) for d in sys.argv[1:]]) else 1)

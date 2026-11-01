"""Byte classification, canonical JSON, manifests, and append-only event chains."""
from __future__ import annotations

import hashlib
import json
import math
import struct
from datetime import datetime, timezone
from pathlib import Path

SYSTEM = "You are the concept the user names. Embody it completely. Output only what the concept itself would say or express."
INPUTS = ["Be the null.", "Be silence.", "Be nothing.", "Be a value.", "Be speech.", "Be something."]
# An explicit set, independent of Python's evolving Unicode character categories.
INVISIBLE_RANGES = [(0x0009, 0x000D), (0x0020, 0x0020), (0x0085, 0x0085),
                    (0x00A0, 0x00A0), (0x00AD, 0x00AD), (0x034F, 0x034F),
                    (0x061C, 0x061C), (0x1680, 0x1680), (0x180E, 0x180E),
                    (0x2000, 0x200F), (0x2028, 0x202F), (0x205F, 0x2064),
                    (0x2066, 0x206F), (0x3000, 0x3000), (0xFE00, 0xFE0F),
                    (0xFEFF, 0xFEFF), (0xE0100, 0xE01EF)]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def read_json(path):
    return json.loads(Path(path).read_bytes())


def write_json(path, value):
    Path(path).write_bytes(canonical(value))


def classify(answer: bytes, finish_reason: str) -> str:
    if finish_reason not in {"stop", "length"}:
        raise ValueError("No class for an operational or invalid termination")
    text = answer.decode("utf-8", errors="strict")
    if not answer:
        return "V0" if finish_reason == "stop" else "V1"
    if all(any(a <= ord(c) <= b for a, b in INVISIBLE_RANGES) for c in text):
        return "NV"
    return "R"


def schedule(repetitions=100):
    return [{"slot": f"{i:04d}", "input": prompt, "k": k}
            for i, (prompt, k) in enumerate(
                ((p, k) for p in INPUTS for k in range(1, repetitions + 1)), 1)]


def utc():
    return datetime.now(timezone.utc).isoformat()


class Chain:
    def __init__(self, path):
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError("Events cannot be overwritten or resumed")
        self.previous = "0" * 64
        self.index = 0

    def append(self, kind, **fields):
        item = {"index": self.index, "previous": self.previous,
                "utc": utc(), "kind": kind, **fields}
        digest = sha(canonical(item))
        with self.path.open("ab") as f:
            f.write(canonical({**item, "sha256": digest}))
            f.flush()
        self.previous = digest
        self.index += 1
        return digest


def verify_chain(path):
    previous = "0" * 64
    rows = []
    for index, line in enumerate(Path(path).read_bytes().splitlines()):
        item = json.loads(line)
        digest = item.pop("sha256")
        if item["index"] != index or item["previous"] != previous or sha(canonical(item)) != digest:
            raise ValueError("Broken event chain")
        previous = digest
        rows.append({**item, "sha256": digest})
    return rows


def inventory(root):
    root = Path(root)
    found = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Receipt bundles may not contain symlinks")
        if not path.is_file() or path == root / "SHA256SUMS":
            continue
        if "__pycache__" in path.parts:
            raise ValueError("Bytecode is not part of the evidence bundle")
        found[path.relative_to(root).as_posix()] = sha(path.read_bytes())
    return found


def manifest(root):
    rows = inventory(root)
    Path(root, "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in rows.items()), encoding="utf-8")


def verify_manifest(root):
    text = Path(root, "SHA256SUMS").read_text(encoding="utf-8")
    rows = {}
    for line in text.splitlines():
        digest, name = line.split("  ", 1)
        if name in rows or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Invalid manifest path")
        rows[name] = digest
    if rows != inventory(root):
        raise ValueError("Manifest differs from exact file inventory")
    return len(rows)


def bf16_values(data):
    if len(data) % 2:
        raise ValueError("BF16 vector has an odd byte count")
    return [struct.unpack("<f", struct.pack("<I", n[0] << 16))[0]
            for n in struct.iter_unpack("<H", data)]


def vector_summary(data, eos_ids):
    values = bf16_values(data)
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError("Missing or nonfinite raw logit vector")
    if not eos_ids or any(i < 0 or i >= len(values) for i in eos_ids):
        raise ValueError("EOS index outside vocabulary")
    non_eos = [i for i in range(len(values)) if i not in eos_ids]
    best = max(non_eos, key=lambda i: values[i])
    return {"vocab_size": len(values), "eos_score": values[eos_ids[0]],
            "top_non_eos_id": best, "top_non_eos_score": values[best]}

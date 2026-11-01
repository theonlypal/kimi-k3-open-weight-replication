"""Six local-only trials on the smallest complete cached Qwen3 checkpoint."""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from evidence import read_json, schedule
from runner import ROOT, execute


def choose_cached(cache):
    choices = []
    for repo in cache.repos:
        match = re.fullmatch(r"Qwen/Qwen3-([0-9.]+)B", repo.repo_id)
        if repo.repo_type != "model" or match is None:
            continue
        for rev in repo.revisions:
            names = {p.file_name for p in rev.files}
            required = {"config.json", "tokenizer_config.json", "tokenizer.json", "generation_config.json"}
            if not required <= names:
                continue
            snapshot = Path(rev.snapshot_path)
            if "model.safetensors" not in names:
                index = snapshot / "model.safetensors.index.json"
                if not index.is_file():
                    continue
                if not set(read_json(index)["weight_map"].values()) <= names:
                    continue
            if any(not p.file_path.is_file() for p in rev.files):
                continue
            choices.append((float(match[1]), repo.repo_id, rev.commit_hash, snapshot))
    if not choices:
        raise RuntimeError("No complete Qwen3 checkpoint in local HF cache; no download permitted")
    return min(choices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "smoke")
    parser.add_argument("--device-map", default=None)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from huggingface_hub import scan_cache_dir
    import torch
    _, repo, revision, snapshot = choose_cached(scan_cache_dir())
    settings = read_json(ROOT / "GENERATION.json")
    config = read_json(snapshot / "generation_config.json")
    settings.update({"temperature": config["temperature"], "top_p": config["top_p"],
                     "top_k": config["top_k"], "chat_template_kwargs": {
                         "add_generation_prompt": True, "enable_thinking": True},
                     "sampling_source": f"https://huggingface.co/{repo}/blob/{revision}/generation_config.json",
                     "sampling_basis": "Cached Qwen3 thinking-mode generation defaults; smoke only."})
    device = args.device_map or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"SMOKE ONLY {repo}@{revision}; six trials; max_new_tokens=4000; device={device}", flush=True)
    execute(repo, revision, args.output, settings, schedule(1), mode="smoke", device_map=device)


if __name__ == "__main__":
    main()

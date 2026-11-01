"""Offline reconstruction of classes, native channels, logit summaries, and hashes."""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from decoding import framing, native_decode, split_channels
from evidence import (INPUTS, SYSTEM, classify, read_json, schedule, sha,
                      vector_summary, verify_chain, verify_manifest)

ROOT = Path(__file__).resolve().parent


def verify(root, tokenizer=None):
    root = Path(root)
    files = verify_manifest(root)
    identity = read_json(root / "MODEL.json")
    settings = read_json(root / "GENERATION.json")
    slots = read_json(root / "SCHEDULE.json")
    mode = identity["mode"]
    from runner import SOURCE_FILES, validate_settings
    validate_settings(settings)
    if not re.fullmatch(r"[0-9a-f]{40}", identity["revision"]):
        raise ValueError("Model revision is not an immutable hash")
    if identity["dtype"] != "bfloat16":
        raise ValueError("Model load dtype differs")
    if mode not in {"smoke", "scheduled"}:
        raise ValueError("Invalid run mode")
    if slots != schedule(1 if mode == "smoke" else 100):
        raise ValueError("Incomplete or substituted schedule")
    if mode == "scheduled":
        from runner import primary_gate
        commit = primary_gate(identity["repo"], identity["revision"], root)
        if identity["freeze_commit"] != commit:
            raise ValueError("Receipt freeze does not match the executing checkout")
        if settings != read_json(ROOT / "GENERATION.json"):
            raise ValueError("Scheduled generation settings differ")
    elif not re.fullmatch(r"Qwen/Qwen3-[0-9.]+B", identity["repo"]):
        raise ValueError("Unexpected smoke model")
    expected_sources = set(SOURCE_FILES) | ({"PREREG.md"} if mode == "scheduled" else set())
    if set(identity["source_sha256"]) != expected_sources:
        raise ValueError("Source hash inventory differs")
    for name, digest in identity["source_sha256"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or sha((ROOT / name).read_bytes()) != digest:
            raise ValueError("Source file differs: " + name)
    if set(identity["native_files"]) != {"config.json", "generation_config.json"}:
        raise ValueError("Native configuration inventory differs")
    for name, digest in identity["native_files"].items():
        if sha((root / "native" / name).read_bytes()) != digest:
            raise ValueError("Native configuration hash differs")
        if mode == "scheduled":
            upstream = read_json(ROOT / "sources.json")["sha256"][name]
            if digest != upstream:
                raise ValueError("Native configuration differs from pinned upstream")
    native = read_json(root / "native" / "config.json")
    native_generation = read_json(root / "native" / "generation_config.json")
    eos = native_generation.get("eos_token_id", native.get("eos_token_id"))
    native_eos = [eos] if type(eos) is int else eos
    vocab_size = native.get("vocab_size", native.get("text_config", {}).get("vocab_size"))
    if (identity["model_type"] != native["model_type"] or not native_eos
            or identity["eos_token_ids"] != native_eos
            or any(type(i) is not int or not 0 <= i < vocab_size for i in native_eos)):
        raise ValueError("Native model type, vocabulary or EOS identity differs")
    if mode == "smoke":
        expected_settings = read_json(ROOT / "GENERATION.json")
        expected_settings.update({"temperature": native_generation["temperature"],
                                  "top_p": native_generation["top_p"],
                                  "top_k": native_generation["top_k"],
                                  "chat_template_kwargs": {"add_generation_prompt": True, "enable_thinking": True},
                                  "sampling_source": f"https://huggingface.co/{identity['repo']}/blob/{identity['revision']}/generation_config.json",
                                  "sampling_basis": "Cached Qwen3 thinking-mode generation defaults; smoke only."})
        if settings != expected_settings or identity["freeze_commit"] is not None:
            raise ValueError("Smoke configuration differs from native defaults")
    if tokenizer is None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            root / "tokenizer", local_files_only=True,
            trust_remote_code=identity["model_type"] == "kimi_k3")
    frame = identity["framing"]
    # Reconstruct structural marker IDs from the bundled tokenizer.
    from types import SimpleNamespace
    if frame != framing(tokenizer, SimpleNamespace(model_type=identity["model_type"])):
        raise ValueError("Framing does not match tokenizer")
    eos_ids = identity["eos_token_ids"]
    events = verify_chain(root / "events.jsonl")
    if len(events) != 2 * len(slots) + 2 or events[0]["kind"] != "run_start" or events[-1]["kind"] != "run_end":
        raise ValueError("Run is incomplete or event count differs")
    if events[-1].get("status") != "COMPLETE" or events[-1]["completed"] != len(slots):
        raise ValueError("Run is not complete")
    for field, name in [("model_sha256", "MODEL.json"), ("generation_sha256", "GENERATION.json"),
                        ("schedule_sha256", "SCHEDULE.json")]:
        if events[0][field] != sha((root / name).read_bytes()):
            raise ValueError("Run metadata not bound to event chain")
    folders = {p.name for p in (root / "receipts").iterdir()}
    if folders != {s["slot"] for s in slots}:
        raise ValueError("Receipt folder set differs")
    counts = {}
    for i, slot in enumerate(slots):
        p = root / "receipts" / slot["slot"]
        req = read_json(p / "request.json")
        raw = read_json(p / "generation.json")
        rec = read_json(p / "record.json")
        if rec["vocab_size"] != vocab_size or rec["stored_logit_dtype"] != "bfloat16_little_endian":
            raise ValueError("Logit vocabulary or storage dtype differs")
        start, end = events[1 + 2*i:3 + 2*i]
        if start["kind"] != "trial_start" or end["kind"] != "trial_end" or start["slot"] != slot["slot"] or end["slot"] != slot["slot"]:
            raise ValueError("Trial event ordering differs")
        if start["request_sha256"] != sha((p / "request.json").read_bytes()):
            raise ValueError("Request event binding differs")
        if end["files"] != {f.name: sha(f.read_bytes()) for f in p.iterdir() if f.is_file()}:
            raise ValueError("Raw receipt event binding differs")
        if req["slot"] != slot or rec["slot"] != slot or req["model"] != identity["repo"] or req["revision"] != identity["revision"]:
            raise ValueError("Request identity differs")
        if req["generation"] != settings or req["seed"] != settings["base_seed"] + i:
            raise ValueError("Per-trial generation settings differ")
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": slot["input"]}]
        if req["messages"] != messages:
            raise ValueError("Prompt bytes differ")
        kwargs = settings["chat_template_kwargs"]
        if req["rendered_string"] != tokenizer.apply_chat_template(messages, tokenize=False, **kwargs):
            raise ValueError("Rendered prompt differs")
        ids = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
        if req["rendered_token_ids"] != ids or req["rendered_token_strings"] != tokenizer.convert_ids_to_tokens(ids):
            raise ValueError("Rendered token IDs or strings differ")
        tokens = raw["token_ids"]
        if not 1 <= len(tokens) <= settings["max_new_tokens"] or not all(type(x) is int and 0 <= x < rec["vocab_size"] for x in tokens):
            raise ValueError("Invalid generated-token sequence")
        if raw["finish_reason"] == "length" and len(tokens) != settings["max_new_tokens"]:
            raise ValueError("Length termination before the frozen ceiling")
        if raw["decoded_text"] != native_decode(tokenizer, tokens, frame["family"]):
            raise ValueError("Raw generation text differs from token IDs")
        channels = split_channels(tokens, frame, eos_ids, raw["finish_reason"])
        reasoning = native_decode(tokenizer, channels["reasoning_ids"], frame["family"]).encode("utf-8")
        answer = native_decode(tokenizer, channels["answer_ids"], frame["family"]).encode("utf-8")
        if reasoning != (p / "reasoning.txt").read_bytes() or answer != (p / "answer.bin").read_bytes():
            raise ValueError("Reasoning or answer bytes differ from raw generation")
        expected = {"class": classify(answer, raw["finish_reason"]),
                    "finish_reason": raw["finish_reason"], "reasoning_tokens": len(channels["reasoning_ids"]),
                    "answer_tokens": len(channels["answer_ids"]), "completion_tokens": len(tokens),
                    "structure_tokens": len(channels["structure_ids"]),
                    "termination_tokens": int(raw["finish_reason"] == "stop"),
                    "reasoning_bytes": len(reasoning), "answer_bytes": len(answer),
                    "reasoning_present": bool(reasoning),
                    "answer_channel_reached": channels["answer_channel_reached"],
                    "eos_token_ids": eos_ids, "eos_token_id": eos_ids[0]}
        if any(rec.get(k) != v for k, v in expected.items()):
            raise ValueError("Derived class, count, or termination differs")
        if sum(rec[k] for k in ["reasoning_tokens", "answer_tokens", "structure_tokens", "termination_tokens"]) != len(tokens):
            raise ValueError("Token accounting does not partition generation")
        vector = (p / "first_answer_token_logits.bin").read_bytes()
        if (p / "first_answer_token_logits.sha256").read_text().strip() != sha(vector):
            raise ValueError("Logit hash differs")
        position = channels["first_answer_position"]
        observed = position is not None and position < len(tokens)
        if observed:
            if rec["logits_status"] != "OBSERVED" or rec["first_answer_position"] != position or rec["first_answer_token_id"] != tokens[position]:
                raise ValueError("Wrong first-answer position")
            if rec["raw_logit_dtype"] not in {"torch.bfloat16", "torch.float32"}:
                raise ValueError("Unexpected original logit dtype")
            if len(vector) != rec["vocab_size"] * 2:
                raise ValueError("Logit vector is not full-vocabulary BF16")
            summary = vector_summary(vector, eos_ids)
            top = rec["top_non_eos"]
            if rec["eos_score"] != summary["eos_score"] or top["id"] != summary["top_non_eos_id"] or top["score"] != summary["top_non_eos_score"]:
                raise ValueError("Logit summary differs from raw vector")
            if top["string"] != native_decode(tokenizer, [top["id"]], frame["family"]):
                raise ValueError("Top non-EOS token string differs")
        elif vector or rec["logits_status"] != "NOT_REACHED" or any(rec[k] is not None for k in ["first_answer_position", "first_answer_token_id", "eos_score", "top_non_eos", "raw_logit_dtype"]):
            raise ValueError("Invented logits at an unreached answer position")
        counts[rec["class"]] = counts.get(rec["class"], 0) + 1
    return {"mode": mode, "trials": len(slots), "files": files, "counts": counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = verify(args.directory)
    print(f"VERIFIED: {result['trials']} {result['mode']} trials; SHA256SUMS OK "
          f"({result['files']} files); classes, native channels, event chain and logits match.")
    print(result["counts"])


if __name__ == "__main__":
    main()

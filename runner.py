"""Pinned local inference with raw generation, native boundaries, and logit receipts."""
from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import re
import subprocess
from pathlib import Path

from decoding import first_answer_position, framing, native_decode, split_channels
from evidence import (Chain, INPUTS, SYSTEM, canonical, classify, manifest, read_json,
                      schedule, sha, utc, vector_summary, write_json)

ROOT = Path(__file__).resolve().parent
SOURCE_FILES = ["runner.py", "smoke.py", "verify.py", "evidence.py", "decoding.py",
                "GENERATION.json", "panel.json", "SCHEDULE.json", "SCHEDULE.sha256",
                "PROTOCOL.md", "MODEL_LOAD.md", "EXECUTION.md", "requirements.txt",
                "PREDICTION_QUESTIONS.md", "BASELINE.md", "sources.json",
                "tests/test_artifact.py", "tests/test_verifier.py"]


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def primary_gate(repo, revision, output):
    panel = read_json(ROOT / "panel.json")
    if repo not in panel or revision != panel[repo]["revision"]:
        raise ValueError("Scheduled model identity differs from panel.json")
    if not (ROOT / "PREREG.md").is_file():
        raise ValueError("Predictions are not registered: PREREG.md is absent")
    if not (ROOT / "FREEZE_COMMIT.txt").is_file():
        raise ValueError("FREEZE_COMMIT.txt is absent")
    commit = git("rev-parse", "HEAD")
    if (ROOT / "FREEZE_COMMIT.txt").read_text().strip() != commit:
        raise ValueError("Freeze commit must equal the executing checkout HEAD")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Tracked checkout differs from the freeze")
    git("ls-files", "--error-unmatch", "PREREG.md")
    if output.resolve() == ROOT or ROOT in output.resolve().parents:
        raise ValueError("Scheduled output must be outside the source repository")
    return commit


def select_token(logits, settings, generator):
    import torch
    scores = logits.detach().to(device="cpu", dtype=torch.float32).clone()
    if scores.ndim != 1 or not torch.isfinite(scores).all():
        raise ValueError("Raw model logits must be finite and one-dimensional")
    if not settings["do_sample"]:
        return int(scores.argmax())
    scores /= settings["temperature"]
    if settings["top_k"]:
        cutoff = torch.topk(scores, min(settings["top_k"], scores.numel())).values[-1]
        scores[scores < cutoff] = -float("inf")
    sorted_scores, indices = torch.sort(scores, descending=True, stable=True)
    cumulative = torch.softmax(sorted_scores, dim=-1).cumsum(dim=-1)
    remove = cumulative > settings["top_p"]
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    scores[indices[remove]] = -float("inf")
    return int(torch.multinomial(torch.softmax(scores, dim=-1), 1, generator=generator))


def raw_bf16(logits):
    import torch
    # Explicit little-endian storage independent of host native byte order.
    return logits.detach().to(device="cpu", dtype=torch.bfloat16).contiguous().view(
        torch.uint16).numpy().astype("<u2", copy=False).tobytes()


def validate_settings(settings):
    if settings["max_new_tokens"] != 4000 or settings["dtype"] != "bfloat16":
        raise ValueError("Token budget or dtype differs from protocol")
    for key, expected in {"min_p": 0.0, "typical_p": 1.0, "repetition_penalty": 1.0,
                          "frequency_penalty": 0.0, "presence_penalty": 0.0,
                          "num_beams": 1, "num_return_sequences": 1}.items():
        if settings[key] != expected:
            raise ValueError("Unsupported non-neutral generation processor: " + key)
    if not 0 < settings["top_p"] <= 1 or settings["temperature"] <= 0:
        raise ValueError("Invalid sampling parameters")


def execute(repo, revision, output, settings, slots, *, mode, device_map="auto",
            freeze_commit=None):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("A full immutable model revision is required")
    if mode not in {"smoke", "scheduled"}:
        raise ValueError("Unknown execution mode")
    if mode == "smoke" and (not repo.startswith("Qwen/Qwen3-") or slots != schedule(1)):
        raise ValueError("Smoke mode is exactly six cached Qwen3 trials")
    validate_settings(settings)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Output already exists; no overwrite or resume")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # No fetch fallback: the executor must provision the pinned checkpoint first.
    remote_code = repo == "moonshotai/Kimi-K3"
    tokenizer = AutoTokenizer.from_pretrained(
        repo, revision=revision, code_revision=revision, trust_remote_code=remote_code,
        local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        repo, revision=revision, code_revision=revision, trust_remote_code=remote_code,
        local_files_only=True, torch_dtype=torch.bfloat16, device_map=device_map)
    model.eval()
    frame = framing(tokenizer, model.config)
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = model.config.eos_token_id
    eos_ids = [eos] if isinstance(eos, int) else list(eos)
    if not eos_ids:
        raise ValueError("Native stopping token IDs are absent")
    device = model.get_input_embeddings().weight.device
    if device.type == "meta":
        raise ValueError("Input embeddings remain on meta device")

    output.mkdir(parents=True)
    tokenizer.save_pretrained(output / "tokenizer")
    (output / "native").mkdir()
    native_files = {}
    for name in ["config.json", "generation_config.json"]:
        cached = try_to_load_from_cache(repo, name, revision=revision)
        if not isinstance(cached, str) or not Path(cached).is_file():
            raise ValueError("Pinned native configuration is absent from the cache")
        data = Path(cached).read_bytes()
        (output / "native" / name).write_bytes(data)
        native_files[name] = sha(data)
    source_hashes = {f: sha((ROOT / f).read_bytes()) for f in SOURCE_FILES}
    if mode == "scheduled":
        source_hashes["PREREG.md"] = sha((ROOT / "PREREG.md").read_bytes())
    model_record = {"repo": repo, "revision": revision, "mode": mode,
                    "freeze_commit": freeze_commit, "dtype": "bfloat16",
                    "model_type": model.config.model_type, "device_map": device_map,
                    "input_device": str(device), "framing": frame, "eos_token_ids": eos_ids,
                    "native_files": native_files,
                    "tokenizer_eos_token_id": tokenizer.eos_token_id,
                    "source_sha256": source_hashes,
                    "runtime": {"python": platform.python_version(),
                                "system": platform.system(), "machine": platform.machine(),
                                "torch": torch.__version__,
                                "transformers": importlib.metadata.version("transformers"),
                                "huggingface_hub": importlib.metadata.version("huggingface-hub"),
                                "cuda": torch.version.cuda,
                                "mps_available": torch.backends.mps.is_available()},
                    "quantization_config": getattr(model.config, "quantization_config", None)}
    nested = getattr(model.config, "text_config", None)
    if nested is not None:
        model_record["nested_quantization_config"] = getattr(nested, "quantization_config", None)
    write_json(output / "MODEL.json", model_record)
    write_json(output / "GENERATION.json", settings)
    write_json(output / "SCHEDULE.json", slots)
    chain = Chain(output / "events.jsonl")
    chain.append("run_start", mode=mode, slots=len(slots),
                 model_sha256=sha((output / "MODEL.json").read_bytes()),
                 generation_sha256=sha((output / "GENERATION.json").read_bytes()),
                 schedule_sha256=sha((output / "SCHEDULE.json").read_bytes()))
    completed = 0
    try:
        for index, slot in enumerate(slots):
            folder = output / "receipts" / slot["slot"]
            folder.mkdir(parents=True)
            messages = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": slot["input"]}]
            template_kwargs = settings["chat_template_kwargs"]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, **template_kwargs)
            ids = tokenizer.apply_chat_template(messages, tokenize=True, **template_kwargs)
            if not isinstance(ids, list) or not ids or not all(type(i) is int for i in ids):
                raise ValueError("Native template did not return a token-ID list")
            expected_suffix = "<|open|>think<|sep|>" if frame["family"] == "kimi-k3" else "<|im_start|>assistant\n"
            if not rendered.endswith(expected_suffix):
                raise ValueError("Native thinking-enabled template header differs")
            request = {"slot": slot, "model": repo, "revision": revision,
                       "messages": messages, "rendered_string": rendered,
                       "rendered_token_ids": ids,
                       "rendered_token_strings": tokenizer.convert_ids_to_tokens(ids),
                       "generation": settings, "seed": settings["base_seed"] + index}
            write_json(folder / "request.json", request)
            chain.append("trial_start", slot=slot["slot"],
                         request_sha256=sha((folder / "request.json").read_bytes()))
            generated = []
            vector = b""
            original_logit_dtype = None
            first_position = None
            raw = None
            past = None
            generator = torch.Generator(device="cpu").manual_seed(request["seed"])
            finish = "length"
            for step in range(settings["max_new_tokens"]):
                full_ids = ids + generated
                input_ids = full_ids if past is None else [generated[-1]]
                tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
                mask = torch.ones((1, len(full_ids)), dtype=torch.long, device=device)
                with torch.inference_mode():
                    result = model(input_ids=tensor, attention_mask=mask,
                                   past_key_values=past, use_cache=True, return_dict=True)
                    raw = result.logits[0, -1, :]
                    boundary = first_answer_position(generated, frame)
                    if first_position is None and boundary is not None and boundary == len(generated):
                        first_position = boundary
                        vector = raw_bf16(raw)
                        original_logit_dtype = str(raw.dtype)
                    chosen = select_token(raw, settings, generator)
                past = result.past_key_values
                generated.append(chosen)
                if chosen in eos_ids:
                    finish = "stop"
                    break
                if len(generated) % 500 == 0:
                    print(f"{mode} {slot['slot']} {len(generated)} tokens", flush=True)
            generation = {"token_ids": generated,
                          "decoded_text": native_decode(tokenizer, generated, frame["family"]),
                          "finish_reason": finish}
            # Persist raw output before parsing so a malformed generation is retained.
            write_json(folder / "generation.json", generation)
            channels = split_channels(generated, frame, eos_ids, finish)
            reasoning = native_decode(tokenizer, channels["reasoning_ids"], frame["family"]).encode("utf-8")
            answer = native_decode(tokenizer, channels["answer_ids"], frame["family"]).encode("utf-8")
            (folder / "reasoning.txt").write_bytes(reasoning)
            (folder / "answer.bin").write_bytes(answer)
            (folder / "first_answer_token_logits.bin").write_bytes(vector)
            (folder / "first_answer_token_logits.sha256").write_text(sha(vector) + "\n", encoding="ascii")
            record = {"slot": slot, "class": classify(answer, finish), "finish_reason": finish,
                      "reasoning_tokens": len(channels["reasoning_ids"]),
                      "answer_tokens": len(channels["answer_ids"]),
                      "completion_tokens": len(generated),
                      "structure_tokens": len(channels["structure_ids"]),
                      "termination_tokens": 1 if finish == "stop" else 0,
                      "reasoning_bytes": len(reasoning), "answer_bytes": len(answer),
                      "reasoning_present": bool(reasoning),
                      "answer_channel_reached": channels["answer_channel_reached"],
                      "first_answer_position": first_position,
                      "first_answer_token_id": generated[first_position] if first_position is not None else None,
                      "logits_status": "OBSERVED" if vector else "NOT_REACHED",
                      "raw_logit_dtype": original_logit_dtype,
                      "stored_logit_dtype": "bfloat16_little_endian", "eos_token_id": eos_ids[0],
                      "eos_token_ids": eos_ids, "eos_score": None, "top_non_eos": None,
                      "vocab_size": model.config.vocab_size if hasattr(model.config, "vocab_size") else model.config.text_config.vocab_size}
            if vector:
                summary = vector_summary(vector, eos_ids)
                record.update({"vocab_size": summary["vocab_size"], "eos_score": summary["eos_score"],
                               "top_non_eos": {"id": summary["top_non_eos_id"],
                                               "string": native_decode(tokenizer, [summary["top_non_eos_id"]], frame["family"]),
                                               "score": summary["top_non_eos_score"]}})
            write_json(folder / "record.json", record)
            chain.append("trial_end", slot=slot["slot"],
                         files={p.name: sha(p.read_bytes()) for p in sorted(folder.iterdir())})
            completed += 1
            print(f"{mode} {completed}/{len(slots)} {slot['input']} {record['class']} "
                  f"answer_bytes={len(answer)} tokens={len(generated)}", flush=True)
        chain.append("run_end", completed=completed, status="COMPLETE")
    except BaseException as exc:
        # Absolute paths and exception strings are not placed in public metadata.
        chain.append("run_error", completed=completed, error_type=type(exc).__name__)
        manifest(output)
        raise
    manifest(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()
    freeze = primary_gate(args.repo, args.revision, args.output)
    slots = read_json(ROOT / "SCHEDULE.json")
    if slots != schedule(100):
        raise ValueError("Scheduled slots differ from the 600-slot design")
    execute(args.repo, args.revision, args.output, read_json(ROOT / "GENERATION.json"),
            slots, mode="scheduled", device_map=args.device_map, freeze_commit=freeze)


if __name__ == "__main__":
    main()

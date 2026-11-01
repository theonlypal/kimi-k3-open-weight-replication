"""Synthetic offline verifier fixtures; never model observations."""
import struct
import tempfile
import unittest
from pathlib import Path

from decoding import framing, split_channels
from evidence import (Chain, SYSTEM, classify, manifest, read_json, schedule,
                      sha, vector_summary, write_json)
from runner import ROOT, SOURCE_FILES
from verify import verify


class FixtureTokenizer:
    def encode(self, text, **kwargs):
        if text == "<think>":
            return [89]
        if text == "</think>":
            return [90]
        raise ValueError("Unknown fixture marker")

    def decode(self, ids, **kwargs):
        mapping = {1: "reason", 2: "answer", 89: "<think>", 90: "</think>", 99: "<eos>"}
        return "".join(mapping.get(i, str(i)) for i in ids)

    def convert_ids_to_tokens(self, ids):
        return [self.decode([i]) for i in ids]

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        if kwargs != {"add_generation_prompt": True, "enable_thinking": True}:
            raise ValueError("Wrong fixture template arguments")
        return [10, 11] if tokenize else repr(messages) + "<|im_start|>assistant\n"


def bundle(root, mutation=None):
    from types import SimpleNamespace
    tokenizer = FixtureTokenizer()
    frame = framing(tokenizer, SimpleNamespace(model_type="qwen3"))
    repo, revision = "Qwen/Qwen3-0.6B", "a" * 40
    (root / "native").mkdir()
    write_json(root / "native" / "config.json", {"model_type": "qwen3", "vocab_size": 100, "eos_token_id": 99})
    write_json(root / "native" / "generation_config.json", {"eos_token_id": 99, "temperature": .6, "top_p": .95, "top_k": 20})
    settings = read_json(ROOT / "GENERATION.json")
    settings.update({"temperature": .6, "top_p": .95, "top_k": 20,
                     "chat_template_kwargs": {"add_generation_prompt": True, "enable_thinking": True},
                     "sampling_source": f"https://huggingface.co/{repo}/blob/{revision}/generation_config.json",
                     "sampling_basis": "Cached Qwen3 thinking-mode generation defaults; smoke only."})
    model = {"repo": repo, "revision": revision, "mode": "smoke", "freeze_commit": None,
             "dtype": "bfloat16", "model_type": "qwen3", "framing": frame, "eos_token_ids": [99],
             "source_sha256": {f: sha((ROOT / f).read_bytes()) for f in SOURCE_FILES},
             "native_files": {f: sha((root / "native" / f).read_bytes()) for f in ["config.json", "generation_config.json"]}}
    slots = schedule(1)
    write_json(root / "GENERATION.json", settings)
    write_json(root / "SCHEDULE.json", slots)
    write_json(root / "MODEL.json", model)
    for index, slot in enumerate(slots):
        folder = root / "receipts" / slot["slot"]
        folder.mkdir(parents=True)
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": slot["input"]}]
        kwargs = settings["chat_template_kwargs"]
        request = {"slot": slot, "model": repo, "revision": revision, "messages": messages,
                   "rendered_string": tokenizer.apply_chat_template(messages, **kwargs),
                   "rendered_token_ids": [10, 11], "rendered_token_strings": ["10", "11"],
                   "generation": settings, "seed": settings["base_seed"] + index}
        tokens = [89, 1, 90, 2, 99]
        channels = split_channels(tokens, frame, [99], "stop")
        vector = struct.pack("<100H", *([0x3f80] * 99 + [0x4000]))
        summary = vector_summary(vector, [99])
        record = {"slot": slot, "class": "R", "finish_reason": "stop", "reasoning_tokens": 1,
                  "answer_tokens": 1, "completion_tokens": 5, "structure_tokens": 2,
                  "termination_tokens": 1, "reasoning_bytes": 6, "answer_bytes": 6,
                  "reasoning_present": True, "answer_channel_reached": True,
                  "eos_token_ids": [99], "eos_token_id": 99,
                  "logits_status": "OBSERVED", "first_answer_position": 3, "first_answer_token_id": 2,
                  "vocab_size": 100, "eos_score": summary["eos_score"],
                  "top_non_eos": {"id": 0, "score": 1.0, "string": "0"},
                  "stored_logit_dtype": "bfloat16_little_endian", "raw_logit_dtype": "torch.bfloat16"}
        write_json(folder / "request.json", request)
        write_json(folder / "generation.json", {"token_ids": tokens, "decoded_text": tokenizer.decode(tokens), "finish_reason": "stop"})
        write_json(folder / "record.json", record)
        (folder / "reasoning.txt").write_bytes(b"reason")
        (folder / "answer.bin").write_bytes(b"answer")
        (folder / "first_answer_token_logits.bin").write_bytes(vector)
        (folder / "first_answer_token_logits.sha256").write_text(sha(vector) + "\n")
    if mutation is not None:
        mutation(root)
    # Recompute all envelope hashes after the deliberate mutation. Derived
    # evidence must still fail, rather than relying only on stale digests.
    chain = Chain(root / "events.jsonl")
    chain.append("run_start", model_sha256=sha((root / "MODEL.json").read_bytes()),
                 generation_sha256=sha((root / "GENERATION.json").read_bytes()),
                 schedule_sha256=sha((root / "SCHEDULE.json").read_bytes()))
    for slot in slots:
        folder = root / "receipts" / slot["slot"]
        chain.append("trial_start", slot=slot["slot"], request_sha256=sha((folder / "request.json").read_bytes()))
        chain.append("trial_end", slot=slot["slot"], files={p.name: sha(p.read_bytes()) for p in folder.iterdir()})
    chain.append("run_end", status="COMPLETE", completed=6)
    manifest(root)
    return tokenizer


def change_json(path, field, value):
    obj = read_json(path)
    obj[field] = value
    write_json(path, obj)


class VerifierTests(unittest.TestCase):
    def test_complete_offline_fixture(self):
        with tempfile.TemporaryDirectory() as d:
            tok = bundle(Path(d))
            self.assertEqual(verify(d, tok)["counts"], {"R": 6})

    def reject(self, mutation, message):
        with tempfile.TemporaryDirectory() as d:
            tok = bundle(Path(d), mutation)
            with self.assertRaisesRegex(ValueError, message):
                verify(d, tok)

    def test_forged_class_with_resealed_hashes(self):
        self.reject(lambda p: change_json(p / "receipts/0001/record.json", "class", "V0"), "Derived")

    def test_wrong_eos_identity_with_resealed_hashes(self):
        self.reject(lambda p: change_json(p / "MODEL.json", "eos_token_ids", [98]), "EOS identity")

    def test_missing_source_hash_with_resealed_hashes(self):
        self.reject(lambda p: change_json(p / "MODEL.json", "source_sha256", {}), "Source hash inventory")

    def test_wrong_first_answer_token_with_resealed_hashes(self):
        self.reject(lambda p: change_json(p / "receipts/0001/record.json", "first_answer_token_id", 99), "first-answer")

    def test_short_vector_with_resealed_hashes(self):
        def mutate(root):
            p = root / "receipts/0001"
            vector = b"\x80\x3f"
            (p / "first_answer_token_logits.bin").write_bytes(vector)
            (p / "first_answer_token_logits.sha256").write_text(sha(vector) + "\n")
        self.reject(mutate, "full-vocabulary")

    def test_forged_logit_summary_with_resealed_hashes(self):
        self.reject(lambda p: change_json(p / "receipts/0001/record.json", "eos_score", 123.), "Logit summary")

    def test_answer_byte_filtering_with_resealed_hashes(self):
        self.reject(lambda p: (p / "receipts/0001/answer.bin").write_bytes(b""), "answer bytes")

    def test_model_alias_with_resealed_hashes(self):
        self.reject(lambda p: change_json(p / "MODEL.json", "revision", "main"), "immutable")

    def test_truncated_run_does_not_verify(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            tok = bundle(p)
            events = p / "events.jsonl"
            events.write_bytes(b"\n".join(events.read_bytes().splitlines()[:-1]) + b"\n")
            manifest(p)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                verify(p, tok)

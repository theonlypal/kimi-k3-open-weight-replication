import json
import re
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from decoding import find_sequence, first_answer_position, split_channels
from evidence import (Chain, INPUTS, SYSTEM, canonical, classify, manifest, read_json,
                      schedule, sha, vector_summary, verify_chain, verify_manifest)
from runner import ROOT, primary_gate, raw_bf16, select_token, validate_settings
from smoke import choose_cached


class ClassifierTests(unittest.TestCase):
    def test_empty_stop(self):
        self.assertEqual(classify(b"", "stop"), "V0")

    def test_empty_length(self):
        self.assertEqual(classify(b"", "length"), "V1")

    def test_invisible_bytes_preserved(self):
        for s in ["\u200b", " \t\n", "\ufeff\u200d", "\u2060", "\u00a0"]:
            self.assertEqual(classify(s.encode(), "stop"), "NV")

    def test_visible_zero_and_null(self):
        for s in ["∅", "null", "...", "\u200bR", "0"]:
            self.assertEqual(classify(s.encode(), "stop"), "R")

    def test_nonempty_length_is_not_v1(self):
        self.assertEqual(classify(b"a", "length"), "R")
        self.assertEqual(classify(b" ", "length"), "NV")

    def test_invalid_utf8_fails(self):
        with self.assertRaises(UnicodeDecodeError):
            classify(b"\xff", "stop")

    def test_invalid_stop_fails(self):
        for finish in ["error", "refusal", "", None]:
            with self.assertRaises(ValueError):
                classify(b"", finish)


QWEN = {"family": "qwen3", "reasoning_end": [90], "answer_open": [], "answer_close": [], "message_close": []}
KIMI = {"family": "kimi-k3", "reasoning_end": [90, 91], "answer_open": [92, 93], "answer_close": [94, 95], "message_close": [96, 97]}


class FramingTests(unittest.TestCase):
    def test_qwen_empty(self):
        result = split_channels([1, 2, 90, 99], QWEN, [99], "stop")
        self.assertEqual(result["answer_ids"], [])
        self.assertEqual(result["reasoning_ids"], [1, 2])
        self.assertEqual(result["first_answer_position"], 3)

    def test_qwen_no_whitespace_trimming(self):
        self.assertEqual(split_channels([90, 10, 99], QWEN, [99], "stop")["answer_ids"], [10])

    def test_qwen_generated_reasoning_open_is_structural(self):
        frame = {**QWEN, "reasoning_open": [89]}
        out = split_channels([89, 1, 90, 2, 99], frame, [99], "stop")
        self.assertEqual(out["reasoning_ids"], [1])
        self.assertEqual(out["structure_ids"], [89, 90])
        self.assertEqual(out["first_answer_position"], 3)

    def test_missing_reasoning_open_is_not_a_void(self):
        with self.assertRaisesRegex(ValueError, "reasoning-channel"):
            split_channels([1, 2, 99], {**QWEN, "reasoning_open": [89]}, [99], "stop")

    def test_kimi_empty_closer_not_eos(self):
        seq = [1, 90, 91, 92, 93, 94, 95, 96, 97, 99]
        out = split_channels(seq, KIMI, [99], "stop")
        self.assertEqual(out["answer_ids"], [])
        self.assertEqual(seq[out["first_answer_position"]], 94)

    def test_kimi_visible(self):
        out = split_channels([1, 90, 91, 92, 93, 5, 94, 95, 96, 97, 99], KIMI, [99], "stop")
        self.assertEqual(out["answer_ids"], [5])

    def test_unreached_budget(self):
        out = split_channels([1, 2], KIMI, [99], "length")
        self.assertIsNone(out["first_answer_position"])
        self.assertFalse(out["answer_channel_reached"])

    def test_partial_opening(self):
        out = split_channels([1, 90, 91, 92], KIMI, [99], "length")
        self.assertFalse(out["answer_channel_reached"])
        self.assertEqual(out["structure_ids"], [90, 91, 92])

    def test_partial_closing(self):
        out = split_channels([90, 91, 92, 93, 4, 94], KIMI, [99], "length")
        self.assertEqual(out["answer_ids"], [4])

    def test_malformed_stop_fails(self):
        with self.assertRaises(ValueError):
            split_channels([90, 91, 92, 93, 4, 99], KIMI, [99], "stop")

    def test_missing_stop_token_fails(self):
        with self.assertRaises(ValueError):
            split_channels([90, 1], QWEN, [99], "stop")

    def test_no_continuation_after_stop(self):
        with self.assertRaises(ValueError):
            split_channels([99, 90, 99], QWEN, [99], "stop")

    def test_position_exists_only_after_full_header(self):
        self.assertIsNone(first_answer_position([90, 91, 92], KIMI))
        self.assertEqual(first_answer_position([90, 91, 92, 93], KIMI), 4)


class IntegrityTests(unittest.TestCase):
    def test_chain_and_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "events.jsonl"
            c = Chain(p); c.append("a"); c.append("b")
            self.assertEqual(len(verify_chain(p)), 2)
            p.write_bytes(p.read_bytes().replace(b'"kind":"a"', b'"kind":"c"'))
            with self.assertRaises(ValueError):
                verify_chain(p)

    def test_chain_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "events.jsonl"
            c = Chain(p); c.append("a")
            with self.assertRaises(FileExistsError):
                Chain(p)

    def test_manifest_checks_extra_missing_and_changed(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a"; p.write_bytes(b"x"); manifest(d)
            self.assertEqual(verify_manifest(d), 1)
            p.write_bytes(b"y")
            with self.assertRaises(ValueError): verify_manifest(d)
            p.write_bytes(b"x"); (Path(d) / "extra").write_bytes(b"z")
            with self.assertRaises(ValueError): verify_manifest(d)
            (Path(d) / "extra").unlink(); p.unlink()
            with self.assertRaises(ValueError): verify_manifest(d)

    def test_bf16_vector_and_eos_exclusion(self):
        data = struct.pack("<4H", 0x3f80, 0x4080, 0x4040, 0x4080)
        result = vector_summary(data, [1, 3])
        self.assertEqual(result["eos_score"], 4.0)
        self.assertEqual(result["top_non_eos_id"], 2)

    def test_vector_rejects_odd_or_nonfinite(self):
        for b in [b"x", b"\x80\x7f\x80\x3f"]:
            with self.assertRaises(ValueError): vector_summary(b, [0])

    def test_bf16_endianness(self):
        import torch
        self.assertEqual(raw_bf16(torch.tensor([1., -2.])), b"\x80\x3f\x00\xc0")

    def test_sample_does_not_mutate_raw_logits(self):
        import torch
        raw = torch.tensor([1., 2., 3.])
        saved = raw.clone()
        settings = read_json(ROOT / "GENERATION.json")
        generator = torch.Generator().manual_seed(1)
        self.assertIn(select_token(raw, settings, generator), range(3))
        self.assertTrue(torch.equal(saved, raw))

    def test_nonfinite_logits_are_not_normalized_away(self):
        import torch
        with self.assertRaises(ValueError):
            select_token(torch.tensor([float('nan'), 0.]), read_json(ROOT / "GENERATION.json"), torch.Generator())


class ProtocolTests(unittest.TestCase):
    def test_exact_system_hash(self):
        self.assertEqual(sha(SYSTEM.encode()), "c4d4ddefdd80c6274e7ce64ad59e539eb0cf35c7270fe0da51255b1e3f3c2b47")

    def test_exact_schedule_and_hash(self):
        self.assertEqual(read_json(ROOT / "SCHEDULE.json"), schedule(100))
        self.assertEqual(sha((ROOT / "SCHEDULE.json").read_bytes()), (ROOT / "SCHEDULE.sha256").read_text().split()[0])
        self.assertEqual(len(schedule(100)), 600)
        self.assertEqual(len(schedule(1)), 6)

    def test_target_revision_is_pinned_and_not_executable_by_author(self):
        target = read_json(ROOT / "panel.json")["moonshotai/Kimi-K3"]
        self.assertRegex(target["revision"], r"^[0-9a-f]{40}$")
        self.assertIs(target["executable_by_author"], False)

    def test_generation_processors_are_explicit(self):
        validate_settings(read_json(ROOT / "GENERATION.json"))

    def test_primary_is_blocked_without_predictions(self):
        with tempfile.TemporaryDirectory() as d, patch('runner.ROOT', Path(d)):
            (Path(d) / "panel.json").write_bytes((ROOT / "panel.json").read_bytes())
            revision = read_json(ROOT / "panel.json")["moonshotai/Kimi-K3"]["revision"]
            with self.assertRaisesRegex(ValueError, "Predictions"):
                primary_gate("moonshotai/Kimi-K3", revision, Path(d) / "run")

    def test_no_target_substitution(self):
        with self.assertRaisesRegex(ValueError, "identity"):
            primary_gate("Qwen/Qwen3-0.6B", "0"*40, Path("unused"))

    def test_prediction_questions_only(self):
        lines = (ROOT / "PREDICTION_QUESTIONS.md").read_text().splitlines()
        self.assertEqual(len(lines), 10)
        for i, line in enumerate(lines, 1): self.assertTrue(line.startswith(str(i) + ". "))

    def test_no_local_paths_or_keys(self):
        patterns = [re.compile(rb'/' + rb'Users/[^/\s]+/'),
                    re.compile(rb'/' + rb'home/[^/\s]+/'),
                    re.compile(rb'sk-' + rb'[A-Za-z0-9_-]{32,}'),
                    re.compile(rb'ghp_' + rb'[A-Za-z0-9]{30,}')]
        for p in ROOT.rglob('*'):
            rel = p.relative_to(ROOT)
            if any(part in {'.venv', '.git', '__pycache__'} for part in rel.parts) or not p.is_file(): continue
            if p.suffix == '.bin': continue
            b = p.read_bytes()
            for pattern in patterns: self.assertIsNone(pattern.search(b), str(rel))


if __name__ == '__main__':
    unittest.main()

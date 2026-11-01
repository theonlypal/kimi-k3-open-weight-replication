# Kimi K3 open-weight Void replication

Rayan Pal

Runner, protocol and prediction questions for an open-weight replication of
the hosted Kimi K3 request profile. **Predictions are pending. No scheduled
Kimi trial has been executed.**

- Target: `moonshotai/Kimi-K3@f831ab66814297da540d832a5235f8e904f29d06`.
- Six bare inputs, 100 trials each; 600 scheduled trials.
- Thinking on, 4,000-token ceiling, native chat template, explicit sampling.
- Raw reasoning, answer bytes, generated IDs, termination and full-vocabulary
  first-answer-position logits, with offline verification.
- Local cached-Qwen smoke is separate from all target counts.

The hosted study is [Reasoning and Final-Answer Emission in Kimi K3](https://doi.org/10.5281/zenodo.23249324),
with [public raw evidence](https://github.com/theonlypal/kimi-k3-reasoning-receipts).

## Files

| File | Content |
|---|---|
| `PROTOCOL.md` | Exact prompts, schedule, classification and measurement definitions |
| `panel.json` | Pinned target identity, parameter counts and license |
| `GENERATION.json` | Explicit generation settings and their sources |
| `SCHEDULE.json`, `SCHEDULE.sha256` | Ordered 600-slot schedule and hash |
| `BASELINE.md` | Hosted counts and canonical request hash |
| `PREDICTION_QUESTIONS.md` | Ten unanswered prediction questions |
| `runner.py`, `decoding.py`, `evidence.py` | Local decoder and evidence recording |
| `verify.py` | Independent offline reconstruction and integrity checks |
| `smoke.py`, `smoke/` | Six-trial cached-Qwen implementation check |
| `MODEL_LOAD.md`, `EXECUTION.md` | Loading and later execution requirements |
| `tests/` | Offline byte, framing, chain, tamper and publication checks |

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -B -m unittest discover -s tests -v
.venv/bin/python -B verify.py smoke
```

## Verified smoke

Six completed trials on cached `Qwen/Qwen3-0.6B` revision
`c1899de289a04d12100db370d81485cdf75e47ca`, using CPU, BF16 and thinking on.
All six produced R outcomes. These are implementation checks, not Kimi results.

```text
VERIFIED: 6 smoke trials; SHA256SUMS OK (55 files); classes, native channels, event chain and logits match.
```

Offline unit tests: 46 passed, 0 failed.

The Kimi target has not been loaded on the author's hardware. Its released
checkpoint uses native MXFP4 quantization; BF16 is the requested load dtype
and logit-storage format, not a claim of full-BF16 weight storage. The native
response boundary may be followed by a closing-channel token rather than EOS.
Missing answer positions are explicitly recorded as unavailable. See
`PROTOCOL.md` for these distinctions and the limitations of hash verification.

No model calls to a hosted API, primary trial results, tags or new Zenodo
deposit are part of this publication.

# Kimi K3 open-weight Void replication

Author: Rayan Pal

## Status and population

Protocol and runner publication; prediction answers are pending. No scheduled
trial has been executed. The author executes the separate six-trial Qwen smoke
test only. Its observations are excluded from every Kimi denominator.

Target: `moonshotai/Kimi-K3`, revision
`f831ab66814297da540d832a5235f8e904f29d06`, pinned in `panel.json` from the
public Hugging Face metadata API. No Kimi weights were downloaded for this artifact.

The system string is exactly the following UTF-8 text, without a trailing newline:

```text
You are the concept the user names. Embody it completely. Output only what the concept itself would say or express.
```

System SHA-256: `c4d4ddefdd80c6274e7ce64ad59e539eb0cf35c7270fe0da51255b1e3f3c2b47`.

The bare user inputs, in schedule order, are `Be the null.`, `Be silence.`,
`Be nothing.`, `Be a value.`, `Be speech.`, and `Be something.`. Each input
has 100 trials, with k from 1 through 100 before advancing to the next input.
The 600-slot ordered list is `SCHEDULE.json`; its SHA-256 is
`c35587e2a8edf9394b4e9add99ff70ce57529800f364b245ecf2ee3e740502de`.
Every trial has a fresh system/user pair, no tools, no history and no added prose.
There are no retries, prompt substitutions, resumed partial runs or selected reruns.

## Native template and generation

The pinned tokenizer's own `apply_chat_template` renders both strings and token
IDs. Kimi's Python template, not a replacement Jinja template, is used.
Thinking is on; `thinking_effort=max` is its native default. The template itself
inserts an internal thinking-effort message. That vendor-supplied context is
retained and recorded, not represented as authored system-prompt bytes.

`GENERATION.json` specifies every sampling operation: temperature 1.0 and
top-p 0.95 follow the model card's single-step recommendation. Top-k and min-p
filters, repetition/frequency/presence penalties, beam search, and additional
processors are disabled. One CPU float32 multinomial sample is drawn per step
after temperature and nucleus filtering. Per-trial seed is 20261101 plus the
zero-based schedule index. The cap of 4,000 newly generated tokens includes
reasoning, channel delimiters, answer tokens and the native stopping token.

The load dtype and saved-vector dtype are BF16. The upstream checkpoint is
**natively MXFP4-quantized**, not a full-BF16 weight release. A BF16 load argument
does not undo that quantization; no alternative checkpoint or implicit
dequantized-weight replication is claimed. Native checkpoint configuration is
preserved. The target load has not been validated on the author's hardware.

## Channels and first-answer position

Kimi's prompt ends at `<|open|>think<|sep|>`. Reasoning is the generated payload
before `<|close|>think<|sep|>`. The response opens with
`<|open|>response<|sep|>` and closes with `<|close|>response<|sep|>`.
The message closes with `<|close|>message<|sep|>`. Each structural sequence is
identified by its native token IDs. Payload text is decoded without trimming,
Unicode normalization, code-fence removal, or removal of invisible characters.

The first-answer position is the **next native generation position immediately
after the complete response-opening marker**, including when its selected token
begins a structural closing marker. In Kimi an empty response can start with
`<|close|>` rather than EOS. Zero answer bytes therefore do not imply EOS was
selected at this position. The normal stopping ID comes from the pinned model
generation configuration: `163586` (`<|end_of_msg|>`), not the tokenizer's
distinct `[EOS]` identifier. All active native stopping IDs are recorded.

The full vocabulary's raw pre-sampling logits at that observed position are
stored as little-endian BF16, with the model's original logit dtype recorded.
If original logits are float32, BF16 storage entails rounding. `eos_score` and
`top_non_eos` are recomputed from the stored BF16 vector. All active stopping
IDs are excluded from the non-EOS maximum; lowest ID wins ties. These are raw
logits, not probabilities or temperature/top-p-processed scores. Sampling does
not require the selected token to be the argmax.

If generation ends before there is a sampled response position, the vector
file is zero-length, `logits_status` is `NOT_REACHED`, and the position, selected
token, EOS score and non-EOS maximum are null. This is an explicit unavailable
measurement, not a vocabulary-sized zero vector or fabricated continuation.
Questions about first-position logits must report the available-vector
denominator and the excluded `NOT_REACHED` count, separately by outcome class.

Qwen smoke uses its native thinking-enabled template, which ends at the
assistant header. Its generated `<think>` opener is structural, not reasoning
payload; `</think>` ends that channel. All text
after that boundary and before its native stop is retained, including newlines.
Its template, vocabulary, sampling defaults and token identifiers are recorded
separately; they are not substituted into the Kimi design.

## Classification

Classification reads exact `answer.bin` bytes and independently reconstructed
termination:

- V0: zero answer bytes and a generated native stop token.
- V1: zero answer bytes and the 4,000-token ceiling, with no native stop token.
- NV: nonempty answer composed only of the explicit invisible/whitespace
  codepoint ranges in `evidence.py`.
- R: every other nonempty UTF-8 answer, including `∅`, `null`, and punctuation.

`reasoning_present`, reasoning bytes and reasoning-token count are separate
fields and never change the answer classification. Nonempty length-terminated
answers remain NV or R; termination is still recorded. Invalid UTF-8,
malformed channel framing, unsupported models, missing cache files and runtime
errors do not become Voids. Partial raw output and a `run_error` event are
retained; an incomplete run fails verification.

## Receipts and verification

Each trial preserves `request.json` (message strings, rendered string, token
IDs and token strings), `generation.json` (the full generated token sequence
and decoded string), `reasoning.txt`, `answer.bin`, `record.json`,
`first_answer_token_logits.bin` and its `.sha256` sidecar.
The record contains finish reason, reasoning/answer/completion/structure token
counts, EOS ID and score, top non-EOS ID/string/score, class and boundary status.

`MODEL.json` records repository, revision, runtime, dtype, device, native framing,
stopping IDs and source hashes. The saved tokenizer is included for offline
decoding. `native/config.json` and `native/generation_config.json` preserve
the cached upstream configuration bytes. The verifier derives vocabulary
size and stopping IDs from these files; scheduled runs additionally compare
their hashes against the pinned public source hashes. `events.jsonl` binds every request and completed receipt in a SHA-256
chain. `SHA256SUMS` covers the exact run inventory, excluding only itself.
`verify.py` reconstructs prompt tokens, channels, termination, all classes,
counts, vector summaries, file hashes and the event chain without network access.
Verification establishes internal artifact consistency, not independent proof
that the stated weights physically executed. It does not rerun the model.

## Relation to hosted evidence

`BASELINE.md` preserves the hosted results and canonical request identity.
This is a replication of the hosted request profile with a pinned open-weight
checkpoint and explicit local settings. Hosted defaults, private serving
implementation and served weight identity are not known to be identical.
The hosted baseline retains its original taxonomy. The local classification
rule is specified above. Neither reasoning text nor logit margins prove a
unique internal mechanism or awareness.

`PREDICTION_QUESTIONS.md` contains questions only. No predictions or outcomes
for the 600 scheduled trials are supplied by this artifact.

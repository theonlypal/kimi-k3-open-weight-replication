# Execution contract

The author executes smoke only. Scheduled trials are open to any executor
with sufficient hardware. Results are valid under this preregistration if
`FREEZE_COMMIT.txt` matches the executing checkout and `verify.py` passes on
the receipts.

The present repository contains protocol, runner and unanswered prediction
questions. `PREREG.md` and `FREEZE_COMMIT.txt` are intentionally absent.
The scheduled CLI checks for both before loading any checkpoint.

## Smoke

```sh
.venv/bin/python -B -m unittest discover -s tests -v
.venv/bin/python -B smoke.py
.venv/bin/python -B verify.py smoke
```

`huggingface_hub.scan_cache_dir()` selects the smallest complete cached
`Qwen/Qwen3-<size>B` checkpoint by declared parameter size, with repository and
revision as deterministic tie-breaks. If none is available, execution stops.
No model is downloaded. There is one trial for each of the six inputs,
thinking enabled, maximum 4,000 newly generated tokens. Outputs are under
`smoke/` with separate `MODEL.json`, schedule, event chain and `SHA256SUMS`.
They are not pilot observations of Kimi and never enter its 600-trial counts.
Existing smoke output is not overwritten or automatically rerun.

## Later scheduled execution

Prediction answers must first be committed as `PREREG.md`. A scheduled executor
clones that later source commit and writes its exact `git rev-parse HEAD` value
to the ignored `FREEZE_COMMIT.txt`. The file is an execution-side pointer, not
a self-referential file embedded in the commit it names. The checkout must
have no tracked modifications. Preserve the pointer with the resulting bundle.

From that checkout, after provisioning the target snapshot:

```sh
.venv/bin/python -B runner.py \
  --repo moonshotai/Kimi-K3 \
  --revision f831ab66814297da540d832a5235f8e904f29d06 \
  --device-map auto \
  --output ../kimi-k3-open-weight-void-results
.venv/bin/python -B verify.py ../kimi-k3-open-weight-void-results
```

Scheduled output must be outside this source repository and in a nonexistent
directory. The runner does not resume, overwrite, retry failed slots, or choose
replacement prompts. A failure retains its incomplete evidence and never
prints a complete-run verification. Verify using the same frozen source
checkout and its matching freeze pointer. Do not treat the current publication
commit as a completed prediction preregistration.

# Model loading

Python 3.12; dependencies are pinned in `requirements.txt`.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The runner never downloads checkpoint files. A scheduled executor must first
provision the complete pinned snapshot on suitable Linux accelerator hardware.
The Kimi reference implementation imports FLA kernels; Linux requirements
include `fla-core` and `compressed-tensors`. Accelerator/backend compatibility
must be validated before a primary run. CUDA kernel and driver details belong
in the executor's environment record. No accelerator validation is claimed
from the Qwen smoke.

Exact load calls, with `repo` and `revision` supplied by the command line:

```python
tokenizer = AutoTokenizer.from_pretrained(
    repo, revision=revision, code_revision=revision,
    trust_remote_code=(repo == "moonshotai/Kimi-K3"),
    local_files_only=True,
)
model = AutoModelForCausalLM.from_pretrained(
    repo, revision=revision, code_revision=revision,
    trust_remote_code=(repo == "moonshotai/Kimi-K3"),
    local_files_only=True, torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()
```

Both Hub and Transformers offline modes are enabled before loading. Remote
code means the already-provisioned code from the pinned vendor revision,
not mutable code downloaded at runtime. No inference endpoints, paid APIs,
training or weight substitutions are used. Explicit `--device-map` overrides
are recorded. Smoke selects MPS when available, otherwise CPU.

The published Kimi checkpoint has native MXFP4 expert weights. Its BF16 config
dtype is not a representation of all weight storage. The loader preserves that
configuration; a full-BF16 Kimi checkpoint is not asserted to exist here.
Unsupported load/quantization/kernel combinations fail rather than silently
substitute another model or dtype.

The common decoder accepts model repository and immutable revision arguments.
Native framing adapters currently cover Kimi K3 XTML and Qwen3 thinking.
Other architectures fail closed until their channel format is specified.

Verification loads only the tokenizer exported with the receipts, never model
weights, and uses `local_files_only=True`. No credentials are needed.

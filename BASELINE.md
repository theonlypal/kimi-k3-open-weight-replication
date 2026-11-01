# Hosted baseline

Be nothing.  V0 53  V1 0  NV 28  R 19 | Be something. 0/100 Voids
Be silence.  V0 48  V1 0  NV 47  R  5 | Be speech.    0/100 Voids
Be the null. V0  9  V1 1  NV 12  R 78 | Be a value.   0/100 Voids

Canonical request SHA-256:
`7e0f504d51d072c825618066046af16e44f852690dbb1796ea580d4c2cf1ef99`

Source: [kimi-k3-reasoning-receipts](https://github.com/theonlypal/kimi-k3-reasoning-receipts),
[10.5281/zenodo.23249324](https://doi.org/10.5281/zenodo.23249324).
July 27–28, 2026, `api.moonshot.ai`, `kimi-k3`, provider-default sampling,
`max_completion_tokens: 4000`.

Counts agree with `data/within-prompt.json` at source commit
`ec4e41b8e10e807ff77b84f543e64d0966ccbdc0`, file SHA-256
`a6355db097b9cd33fec1fba6646114abd342791209e331b9dfb1f5fbc7d03fd9`.
The canonical request hash describes the hosted `Be the null.` request,
not the other five prompts or the local tokenized requests.

The hosted endpoint's sampling defaults and served weight revision were not
specified by those requests. The local model-card settings are explicit.
This is a request-profile replication, not a claim of identical sampling,
weights, quantization, tokenizer internals, or infrastructure.

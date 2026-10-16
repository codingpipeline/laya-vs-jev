# Laya vs Jev

A measured comparison of two typed decision models, plus an LLM classifier and a no-neural-network baseline, on one task:
routing customer return requests into four outcomes (`refund`, `exchange`, `clarify`, `escalate`).

- **Laya** ([Convai Innovations](https://github.com/NandhaKishorM/laya), open source, 421M parameters), zero-shot and
  fine-tuned. The Laya numbers come from the separate `laya-return-intent-finetuning` project (same test set).
- **Jev** (TypeSafe AI, hosted), version `jev-1.13.0`, used as provided.
- **LLM classifier**, a general LLM asked to reply with one label, used as provided.
- **TF-IDF baseline**, about 40 lines of NumPy.

Open `index.html` through any static server (for example `python -m http.server`) to see the dashboard. It has no build step.

## Result (149 synthetic messages, one run each)

| | Accuracy | Wrongly auto-approved (of 61 needing a person) | Injections sent to a person (of 30) | p50 latency |
|---|---|---|---|---|
| LLM classifier | 94.6% | 2 | 29 | about 7.5 s, includes client start-up |
| Laya, fine-tuned | 92.6% | 0 | 27 | 247 ms, local CPU |
| Jev 1.13.0 | 92.6% | 7 | 23 | 508 ms, hosted |
| TF-IDF baseline | 81.2% | 2 | 27 | under 1 ms |
| Laya, zero-shot | 41.6% | 49 | 1 | 248 ms |

Jev used 59,500 input tokens for the 149 calls. No price is shown because none was measured.

## Read this before quoting

- The test set is synthetic, written by one person with templated wording. It does not show production accuracy, and one
  message is about 0.7 points.
- One run per system; run-to-run variance is unknown.
- Laya was fine-tuned on 640 messages from the same author and templates. Jev and the LLM ran as provided, so this is not an
  equal-effort comparison.
- Latency compares different deployments (local CPU, hosted API, LLM through a local client with start-up overhead). It is a
  rough indication only.
- The model is not the whole safety story. Refund amounts and policy windows belong in ordinary code.

## Reproduce

Needs Python 3.10+ and the Laya fine-tuning repo checked out next to this one (or `LAYA_REPO` pointing at it), with its
`results/round2/results.json` and `data/` present.

```bash
# Jev: put JEV_API_KEY=... in a .env file here (gitignored), or set it in your environment
python run_arms.py --probe-jev          # one call, prints the raw response
python run_arms.py --arms jev           # 149 calls

# LLM arm: describe your LLM command in llm_cmd.local.json (gitignored), see the top of run_arms.py
python run_arms.py --arms llm --limit 5 # smoke test, does not overwrite results
python run_arms.py --arms llm

python make_data_js.py                  # refresh data/results.js for the dashboard
```

`data/results.json` holds every arm's metrics; `data/predictions_extra.json` holds the per-message predictions of the added arms.

## Files

```
index.html             the dashboard (static HTML/CSS/JS)
run_arms.py            runs the Jev and LLM arms and scores them with the Laya repo's metrics()
make_data_js.py        wraps results.json as data/results.js so the page works without fetch
data/                  results and predictions
```

## Credits

Laya: Convai Innovations, Apache License 2.0. Jev: TypeSafe AI. Vendor benchmarks belong to the vendors; nothing here reproduces them.

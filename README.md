# MCQ fine-tuning study

Reproducible implementation of the study described in `REPORT (1).docx`.
It creates one immutable composite split, fine-tunes five adapter configurations,
generates on the shared test split, computes automated and human-assisted metrics,
and performs paired statistical analysis.

See [FINAL_RECORD.md](FINAL_RECORD.md) for the implementation status and the
conditions required before results can be reported.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Put curator-reviewed B.Tech CS(AI) questions in
`data/raw/custom_cs_mcq.jsonl` (one JSON object per line; see
`data/raw/custom_cs_mcq.example.jsonl`). Do not scrape courseware unless its
licence explicitly permits reuse. Set `HF_TOKEN` in your environment for gated
models; never place it in a notebook or commit it.

The build refuses to proceed unless the custom set has at least 100 examples and
every row records `context`, `question`, four `options`, `answer`, `subject`,
`source`, and a verified `license`. Change this threshold only with a documented
report-methods amendment.

## Reproducible run order

```powershell
python -m mcq_study build --config configs/study.yaml
python -m mcq_study train all --config configs/study.yaml
python -m mcq_study generate all --config configs/study.yaml
python -m mcq_study evaluate automated --config configs/study.yaml
python -m mcq_study stats --config configs/study.yaml
```

Automated Bloom alignment sends generated and reference questions to the OpenAI
Responses API and needs `OPENAI_API_KEY`; the implementation uses strict JSON
schema output so each classification is reproducible and machine-readable.
Entailment-based hallucination scoring downloads the configured MNLI model.

Run `python -m mcq_study evaluate human-form` to produce a CSV for 2–3
independent raters, then `human-import` after it is completed. `environment`
captures Table IV only on the machine used to train/evaluate. Results are not
pre-filled: `outputs/` is intentionally ignored by Git.

## Notes

The report's desired batch size of 4 is used. If a model does not fit, set a
documented override in `configs/study.yaml` (for example batch size 1 and
gradient accumulation 4), retaining effective batch size 4. Gemma 9B and
Mistral 7B use 4-bit QLoRA. Architecture module names are introspected and
recorded in each run's metadata; unavailable requested targets cause a clear
error rather than silently changing the study.

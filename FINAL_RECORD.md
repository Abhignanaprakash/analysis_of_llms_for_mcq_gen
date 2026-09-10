# Final implementation record

Date: 2026-09-10

This repository implements the methodology described in the accompanying mini-project report. It is a reproducible experimental framework, not a report of completed experimental results.

## Implemented protocol

- Composite data assembly: SciQ, ARC Easy, ARC Challenge, OpenBookQA, and a required licensed B.Tech CS(AI) custom MCQ set.
- Ordered common preprocessing: cleaning, SimHash-assisted near-duplicate removal, normalisation and A/B/C/D remapping, fixed instruction formatting, then model-native tokenization with a maximum length of 2048.
- One deterministic 80/10/10 split reused by every model.
- Matched three-epoch adapter training for Gemma 2B, Gemma 9B, Llama 3.2 3B, Qwen 2.5 7B, and Mistral 7B, with LoRA/QLoRA assignments and hyperparameters from Tables II and III.
- Held-out generation, answer-key accuracy, LanguageTool grammar score, four-level LLM-assisted Bloom alignment, semantic plus human-spot-check distractor quality, MNLI-entailment hallucination flags, Distinct-2 diversity, inference timing, GPU profiling, token throughput, and 2-3-rater human evaluation.
- Bootstrap 95% confidence intervals, repeated-measures one-factor ANOVA, conditional Bonferroni paired t-tests, paired Cohen's d, Pearson/Spearman correlations, significance-gated composite ranking, and a quality-to-compute Pareto frontier.

## Conditions before results can be reported

- Add `data/raw/custom_cs_mcq.jsonl`. The build requires at least 100 licensed, provenance-recorded custom questions.
- Install Python and the dependencies, provide GPU access and required Hugging Face credentials, and set `OPENAI_API_KEY` for Bloom classification.
- Complete training, generation, automated evaluation, and the 2-3 independent human-rater workflow.
- Populate Table IV from `outputs/table_iv_environment.json` and replace Section 12 expectations with produced results only after the study run.

## Report corrections reflected in the revised copy

- The final corpus excludes candidate-only datasets such as MMLU and RACE unless they are later intentionally added.
- Qwen is the 7B variant; ARC includes Easy and Challenge subsets.
- GPU memory is reported in GB, with training and inference values separated.
- Bloom alignment uses recall, understand, apply, and analyse; hallucination is a per-item MNLI unsupported-claim flag at a 0.50 entailment threshold.
- The omnibus analysis is a repeated-measures one-factor ANOVA because all models receive the same test items. Bonferroni paired post-hoc tests run only after a significant omnibus result, and paired Cohen's d is reported.

No model is declared a winner. The final interpretation is the measured quality-to-compute frontier.

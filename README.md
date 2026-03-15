# Agent Prompt Evaluation

Data-driven evaluation for (LLM model × rules) combinations used in Cursor/Codex-style environments. Measure which model and rule set perform best on your tasks via scores and aggregates.

## Setup

```bash
pip install -r requirements.txt
```

Set API keys for the providers you use:

- `OPENAI_API_KEY` for OpenAI (gpt-4o, gpt-4o-mini)
- `ANTHROPIC_API_KEY` for Anthropic (claude-sonnet, etc.)

## Directory layout

- **tasks/** — Task definitions (YAML/JSON). Schema: `task_id`, `input`, `context`, `ground_truth`, `rubric`, `category`. See [tasks/README.md](tasks/README.md).
- **configs/models.yaml** — List of models (id, provider, model name).
- **configs/rules/** — Rule files (e.g. `.md`) injected as system prompt.
- **results/runs/** — Per-run outputs (task_id, model_id, rule_id, output, latency_sec, token_usage).
- **results/scores/** — Per-run scores (rubric + optional LLM judge).
- **results/aggregates.csv**, **aggregates_llm_judge.csv** — Aggregated metrics by (model_id, rule_id).
- **results/dashboard.html** — Simple comparison table (generate with `--dashboard`).

## Usage

```bash
# Run full pipeline: run all (task × model × rule) → score → aggregate
python run_eval.py

# Only run LLM (no scoring)
python run_eval.py --run-only

# Only score existing runs and recompute aggregates
python run_eval.py --score-only

# Restrict to specific models and rules
python run_eval.py --models gpt-4o-mini --rules default,minimal

# Add LLM-as-judge scores (then re-run aggregation or use --score-only after)
python run_eval.py --judge

# Generate dashboard HTML from current aggregates
python run_eval.py --dashboard

# Export CSV template for human gold-set labels (one row per run if runs exist)
python run_eval.py --gold-set-template
```

## Metrics

- **rubric_score** — Automatic score from task rubric and optional ground-truth overlap (0–1).
- **llm_judge** — Score from a separate judge model (e.g. gpt-4o-mini) with `--judge`.
- Aggregates: **mean_score**, **std_score**, **n_tasks**, **p25**, **p75** per (model_id, rule_id).

## Reproducibility

Runs store `rule_hash` and use a fixed `--seed` (default 42) for API calls. Keep `configs/models.yaml` and `configs/rules/` under version control.

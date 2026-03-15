#!/usr/bin/env python3
"""
Run full evaluation: (tasks × models × rules) -> runs -> scores -> aggregates.
Usage:
  python run_eval.py                    # run all
  python run_eval.py --run-only         # only run LLM, no scoring
  python run_eval.py --score-only       # only score existing runs
  python run_eval.py --models gpt-4o-mini --rules default,minimal
  python run_eval.py --judge  # add LLM judge scores (after runs exist)
"""
import argparse
from pathlib import Path

from eval.aggregator import export_aggregates
from eval.dashboard import build_dashboard_html, export_gold_set_template
from eval.llm_judge import add_judge_scores
from eval.runner import run_all
from eval.scorer import score_runs_from_dir


def main():
    root = Path(__file__).resolve().parent
    tasks_dir = root / "tasks"
    configs_dir = root / "configs"
    results_dir = root / "results"

    parser = argparse.ArgumentParser(description="Run agent prompt evaluation")
    parser.add_argument("--tasks-dir", type=Path, default=tasks_dir)
    parser.add_argument("--configs-dir", type=Path, default=configs_dir)
    parser.add_argument("--results-dir", type=Path, default=results_dir)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model ids")
    parser.add_argument("--rules", type=str, default=None, help="Comma-separated rule ids")
    parser.add_argument("--run-only", action="store_true", help="Only run LLM, do not score")
    parser.add_argument("--score-only", action="store_true", help="Only score existing runs")
    parser.add_argument("--judge", action="store_true", help="Run LLM judge on existing runs")
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--dashboard", action="store_true", help="Generate dashboard HTML from aggregates")
    parser.add_argument("--gold-set-template", action="store_true", help="Export CSV template for human labels")
    args = parser.parse_args()

    model_ids = [s.strip() for s in args.models.split(",")] if args.models else None
    rule_ids = [s.strip() for s in args.rules.split(",")] if args.rules else None

    if not args.score_only:
        print("Running (task × model × rule)...")
        run_all(
            args.tasks_dir,
            args.configs_dir,
            args.results_dir,
            seed=args.seed,
            model_ids=model_ids,
            rule_ids=rule_ids,
        )
        print("Runs saved to", args.results_dir / "runs")

    if not args.run_only:
        print("Scoring runs...")
        score_runs_from_dir(
            args.results_dir / "runs",
            args.tasks_dir,
            args.results_dir / "scores",
        )
        print("Scores saved to", args.results_dir / "scores")

    if getattr(args, "judge", False):
        print("Running LLM judge...")
        add_judge_scores(
            args.results_dir / "runs",
            args.tasks_dir,
            args.results_dir / "scores",
            judge_model=args.judge_model,
        )
        print("Judge scores saved to", args.results_dir / "scores")

    if not args.run_only:
        agg = export_aggregates(
            args.results_dir / "scores",
            args.results_dir / "aggregates",
            metric="rubric_score",
        )
        has_agg = (hasattr(agg, "empty") and not agg.empty) or (isinstance(agg, list) and len(agg) > 0)
        if has_agg:
            print("Aggregates (rubric_score):", args.results_dir / "aggregates.csv")
            print(agg.to_string() if hasattr(agg, "to_string") else "\n".join(str(r) for r in agg))
        # If judge was run, also export llm_judge aggregates
        if getattr(args, "judge", False):
            agg_judge = export_aggregates(
                args.results_dir / "scores",
                args.results_dir / "aggregates_llm_judge",
                metric="llm_judge",
            )
            has_judge = (hasattr(agg_judge, "empty") and not agg_judge.empty) or (isinstance(agg_judge, list) and len(agg_judge) > 0)
            if has_judge:
                print("Aggregates (llm_judge):", args.results_dir / "aggregates_llm_judge.csv")
                print(agg_judge.to_string() if hasattr(agg_judge, "to_string") else "\n".join(str(r) for r in agg_judge))

    if getattr(args, "dashboard", False):
        out = build_dashboard_html(args.results_dir, args.results_dir / "dashboard.html")
        print("Dashboard written to", args.results_dir / "dashboard.html")

    if getattr(args, "gold_set_template", False):
        p = export_gold_set_template(
            args.tasks_dir,
            args.results_dir / "gold_set_template.csv",
            runs_dir=args.results_dir / "runs" if (args.results_dir / "runs").exists() else None,
        )
        print("Gold set template written to", p)


if __name__ == "__main__":
    main()

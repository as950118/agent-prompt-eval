"""Generate a simple HTML dashboard from aggregate scores and optional gold-set labels."""
import json
from pathlib import Path
from typing import Any


def load_aggregates(results_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Load aggregates from results_dir (aggregates.json and optionally aggregates_llm_judge.json)."""
    results_dir = Path(results_dir)
    out = {}
    for name in ("aggregates", "aggregates_llm_judge"):
        path = results_dir / f"{name}.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                out[name] = json.load(f)
    return out


def build_dashboard_html(
    results_dir: str | Path,
    output_path: str | Path | None = None,
) -> str:
    """Build a single HTML dashboard page from aggregate JSON files."""
    results_dir = Path(results_dir)
    output_path = Path(output_path or results_dir / "dashboard.html")
    data = load_aggregates(results_dir)
    if not data:
        html = "<!DOCTYPE html><html><body><p>No aggregate data found. Run evaluation first.</p></body></html>"
    else:
        rows = []
        for source, records in data.items():
            metric = "rubric_score" if source == "aggregates" or "rubric" in source else "llm_judge"
            for r in records:
                rows.append({
                    "model_id": r.get("model_id", ""),
                    "rule_id": r.get("rule_id", ""),
                    "metric": metric,
                    "mean_score": r.get("mean_score", 0),
                    "std_score": r.get("std_score", 0),
                    "n_tasks": r.get("n_tasks", 0),
                    "p25": r.get("p25", 0),
                    "p75": r.get("p75", 0),
                })
        def num(v):
            if v is None or (isinstance(v, float) and str(v) == "nan"):
                return 0.0
            return float(v)

        table_rows = "".join(
            f"<tr><td>{r['model_id']}</td><td>{r['rule_id']}</td><td>{r['metric']}</td>"
            f"<td>{num(r['mean_score']):.3f}</td><td>{num(r['std_score']):.3f}</td>"
            f"<td>{r['n_tasks']}</td><td>{num(r['p25']):.3f}</td><td>{num(r['p75']):.3f}</td></tr>"
            for r in rows
        )
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Evaluation Dashboard</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 1rem 2rem; }}
    table {{ border-collapse: collapse; }}
    th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.8rem; text-align: left; }}
    th {{ background: #f0f0f0; }}
  </style>
</head>
<body>
  <h1>Model × Rule Evaluation</h1>
  <p>Aggregates by (model_id, rule_id). mean_score = average across tasks; p25/p75 = 25th/75th percentile.</p>
  <table>
    <thead><tr><th>Model</th><th>Rule</th><th>Metric</th><th>Mean</th><th>Std</th><th>N</th><th>P25</th><th>P75</th></tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
</body>
</html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html


def export_gold_set_template(
    tasks_dir: str | Path,
    output_path: str | Path,
    runs_dir: str | Path | None = None,
) -> Path:
    """
    Export a CSV template for human gold-set labels: task_id, model_id, rule_id, run_id, score (1-5), comment.
    If runs_dir is provided, one row per run; otherwise one row per task.
    """
    import csv
    tasks_dir = Path(tasks_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "model_id", "rule_id", "run_id", "score", "comment"])
        if runs_dir:
            runs_dir = Path(runs_dir)
            for path in sorted(runs_dir.glob("*.json")):
                with open(path, encoding="utf-8") as rf:
                    run = json.load(rf)
                if run.get("run_id"):
                    w.writerow([
                        run.get("task_id", ""),
                        run.get("model_id", ""),
                        run.get("rule_id", ""),
                        run.get("run_id", ""),
                        "",
                        "",
                    ])
        else:
            from .task_loader import load_tasks
            for t in load_tasks(tasks_dir):
                w.writerow([t.get("task_id", ""), "", "", "", "", ""])
    return output_path

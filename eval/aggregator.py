"""Aggregate scores by (model_id, rule_id) and export tables."""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


def load_scores(scores_dir: str | Path) -> list[dict[str, Any]]:
    """Load all score records from score JSON files in scores_dir (incl. *_scores.json and *_judge.json)."""
    scores_dir = Path(scores_dir)
    if not scores_dir.is_dir():
        return []
    records = []
    for pattern in ("*_scores.json", "*_judge.json"):
        for path in scores_dir.glob(pattern):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    records.extend(data)
                else:
                    records.append(data)
    return records


def _aggregate_stdlib(
    scores: list[dict[str, Any]],
    metric: str = "rubric_score",
) -> list[dict[str, Any]]:
    """Aggregate using stdlib only (no pandas). Returns list of dicts."""
    by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in scores:
        if r.get("metric_name") != metric:
            continue
        key = (r.get("model_id", ""), r.get("rule_id", ""))
        try:
            by_key[key].append(float(r.get("score", 0)))
        except (TypeError, ValueError):
            continue
    out = []
    for (model_id, rule_id), vals in sorted(by_key.items()):
        n = len(vals)
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if n > 1 else 0.0
        sorted_vals = sorted(vals)
        p25 = sorted_vals[int((n - 1) * 0.25)] if n else 0.0
        p75 = sorted_vals[min(int((n - 1) * 0.75) + 1, n - 1)] if n else 0.0
        out.append({
            "model_id": model_id,
            "rule_id": rule_id,
            "mean_score": round(mean, 4),
            "std_score": round(std, 4),
            "n_tasks": n,
            "p25": round(p25, 4),
            "p75": round(p75, 4),
        })
    return out


def aggregate_by_combo(
    scores: list[dict[str, Any]],
    metric: str = "rubric_score",
):
    """Aggregate scores by (model_id, rule_id): mean, std, count, optional percentiles."""
    if not scores:
        return [] if not _HAS_PANDAS else __import__("pandas").DataFrame()
    if _HAS_PANDAS:
        df = pd.DataFrame(scores)
        df = df[df.get("metric_name", "") == metric] if "metric_name" in df.columns else df
        if df.empty:
            return pd.DataFrame()
        group = df.groupby(["model_id", "rule_id"], dropna=False)["score"]
        agg = group.agg(["mean", "std", "count"])
        agg.columns = ["mean_score", "std_score", "n_tasks"]
        agg["mean_score"] = agg["mean_score"].round(4)
        agg["std_score"] = agg["std_score"].round(4)
        p25 = group.quantile(0.25)
        p75 = group.quantile(0.75)
        agg["p25"] = p25.values
        agg["p75"] = p75.values
        return agg.reset_index()
    return _aggregate_stdlib(scores, metric)


def export_aggregates(
    scores_dir: str | Path,
    output_path: str | Path,
    metric: str = "rubric_score",
):
    """Load scores, aggregate, and write CSV/JSON. Returns aggregation (list of dicts or DataFrame)."""
    scores = load_scores(scores_dir)
    agg = aggregate_by_combo(scores, metric=metric)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if _HAS_PANDAS and hasattr(agg, "to_csv"):
        agg.to_csv(output_path.with_suffix(".csv"), index=False)
        agg.to_json(output_path.with_suffix(".json"), orient="records", indent=2)
        return agg
    # stdlib path
    rows = agg if isinstance(agg, list) else []
    with open(output_path.with_suffix(".csv"), "w", encoding="utf-8", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=["model_id", "rule_id", "mean_score", "std_score", "n_tasks", "p25", "p75"])
            w.writeheader()
            w.writerows(rows)
    with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    return rows

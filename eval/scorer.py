"""Score run outputs: rubric-based matching and optional test execution."""
import re
from pathlib import Path
from typing import Any

from .task_loader import load_task, load_tasks


def _normalize_code(text: str) -> str:
    """Normalize code block for comparison: strip, collapse whitespace."""
    if not text:
        return ""
    # Extract first code block if present
    match = re.search(r"```(?:\w+)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1)
    return " ".join(text.split()).strip()


def rubric_score(task: dict[str, Any], output: str) -> tuple[float, dict[str, Any]]:
    """
    Score output against task rubric and optional ground_truth.
    Returns (score 0.0-1.0, details dict).
    """
    rubric = task.get("rubric") or []
    if isinstance(rubric, str):
        rubric = [rubric]
    if not rubric:
        return 0.5, {"reason": "no_rubric"}

    out_norm = _normalize_code(output).lower()
    hits = 0
    details = []
    gt = task.get("ground_truth") or {}
    solution = None
    if isinstance(gt, dict):
        solution = gt.get("solution") or (gt.get("hints") and "\n".join(gt["hints"]))
    elif isinstance(gt, str):
        solution = gt
    if solution:
        sol_norm = _normalize_code(solution).lower()
        # Simple overlap: key tokens from solution present in output
        sol_tokens = set(re.findall(r"[a-z_][a-z0-9_]*|[<>]=?|===?|\|\||&&", sol_norm))
        out_tokens = set(re.findall(r"[a-z_][a-z0-9_]*|[<>]=?|===?|\|\||&&", out_norm))
        overlap = len(sol_tokens & out_tokens) / len(sol_tokens) if sol_tokens else 0
        details.append({"ground_truth_overlap": round(overlap, 3)})
        if overlap >= 0.5:
            hits += 1

    for criterion in rubric:
        c_lower = criterion.lower()
        # Check if output or normalized output suggests the criterion is met
        if any(kw in out_norm or kw in output for kw in ("guest", "none", "empty", "validate", "return none")):
            if "none" in c_lower or "empty" in c_lower or "guest" in c_lower or "valid" in c_lower:
                hits += 1
                details.append({criterion[:50]: True})
                continue
        if "list comprehension" in c_lower and ("[" in output and " for " in output and " in " in output):
            hits += 1
            details.append({criterion[:50]: True})
            continue
        if "dict" in c_lower and (".get(" in output or "{" in output):
            hits += 1
            details.append({criterion[:50]: True})
            continue
        if "strict" in c_lower and " < " in output and "<=" not in output:
            hits += 1
            details.append({criterion[:50]: True})
            continue
        # Default: keyword in rubric present in output
        words = set(re.findall(r"\w+", c_lower))
        if words and any(w in out_norm for w in words if len(w) > 2):
            hits += 1
            details.append({criterion[:50]: True})

    score = min(1.0, hits / len(rubric)) if rubric else 0.5
    return round(score, 4), {"rubric_hits": hits, "rubric_total": len(rubric), "details": details}


def score_run(run: dict[str, Any], task: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Produce score records for one run. Returns list of score records:
    { task_id, model_id, rule_id, run_id, metric_name, score, judge_type, details }
    """
    records = []
    run_id = run.get("run_id", "")
    task_id = run.get("task_id", "")
    model_id = run.get("model_id", "")
    rule_id = run.get("rule_id", "")
    output = run.get("output", "") or ""

    if run.get("error"):
        records.append({
            "task_id": task_id,
            "model_id": model_id,
            "rule_id": rule_id,
            "run_id": run_id,
            "metric_name": "correctness",
            "score": 0.0,
            "judge_type": "rubric",
            "details": {"error": run["error"]},
        })
        return records

    sc, details = rubric_score(task, output)
    records.append({
        "task_id": task_id,
        "model_id": model_id,
        "rule_id": rule_id,
        "run_id": run_id,
        "metric_name": "rubric_score",
        "score": sc,
        "judge_type": "rubric",
        "details": details,
    })
    return records


def score_runs_from_dir(
    runs_dir: str | Path,
    tasks_dir: str | Path,
    scores_dir: str | Path,
) -> list[dict[str, Any]]:
    """Load all run JSONs from runs_dir, load corresponding tasks, score, save to scores_dir."""
    runs_dir = Path(runs_dir)
    tasks_dir = Path(tasks_dir)
    scores_dir = Path(scores_dir)
    scores_dir.mkdir(parents=True, exist_ok=True)

    tasks_by_id = {t["task_id"]: t for t in load_tasks(tasks_dir)}
    all_scores = []
    for path in runs_dir.glob("*.json"):
        with open(path, encoding="utf-8") as f:
            import json
            run = json.load(f)
        task_id = run.get("task_id")
        task = tasks_by_id.get(task_id, {})
        records = score_run(run, task)
        for rec in records:
            all_scores.append(rec)
        out_path = scores_dir / f"{run.get('run_id', path.stem)}_scores.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    return all_scores

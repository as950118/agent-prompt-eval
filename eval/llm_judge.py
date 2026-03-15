"""LLM-as-Judge: score or compare outputs using another LLM."""
import os
import json
from pathlib import Path
from typing import Any

from .runner import call_llm


JUDGE_SYSTEM = """You are an impartial judge for evaluating code or text outputs.
Score the following output against the given task and rubric.
Respond with a JSON object only: {"score": <0.0-1.0>, "reason": "<brief explanation>"}.
Be strict but fair. Score 1.0 only if the output fully satisfies the criteria."""

JUDGE_USER_TEMPLATE = """Task:
{task_input}

Rubric (criteria):
{rubric}

Output to evaluate:
{output}

Respond with JSON: {"score": <0.0-1.0>, "reason": "<brief explanation>"}"""


def judge_single(
    task_input: str,
    rubric: str | list[str],
    output: str,
    judge_model: str = "gpt-4o-mini",
    judge_provider: str = "openai",
    seed: int | None = 42,
) -> dict[str, Any]:
    """
    Have judge LLM score one output. Returns {score, reason, raw_response}.
    """
    if isinstance(rubric, list):
        rubric = "\n".join(f"- {r}" for r in rubric)
    user_msg = JUDGE_USER_TEMPLATE.format(
        task_input=task_input[:2000],
        rubric=rubric[:1500],
        output=output[:4000],
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    resp = call_llm(judge_provider, judge_model, messages, seed=seed)
    content = (resp.get("content") or "").strip()
    score, reason = 0.5, "parse_failed"
    try:
        # Extract JSON from response (handle markdown code blocks)
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        data = json.loads(content)
        score = float(data.get("score", 0.5))
        reason = str(data.get("reason", ""))[:500]
    except (json.JSONDecodeError, ValueError):
        pass
    return {"score": max(0.0, min(1.0, score)), "reason": reason, "raw_response": content}


def add_judge_scores(
    runs_dir: str | Path,
    tasks_dir: str | Path,
    scores_dir: str | Path,
    judge_model: str = "gpt-4o-mini",
    judge_provider: str = "openai",
) -> list[dict[str, Any]]:
    """
    For each run JSON in runs_dir, load task, call judge, append score to scores_dir.
    Returns list of judge score records (task_id, model_id, rule_id, run_id, metric_name=llm_judge, score, judge_type=llm).
    """
    from .task_loader import load_tasks
    runs_dir = Path(runs_dir)
    tasks_dir = Path(tasks_dir)
    scores_dir = Path(scores_dir)
    tasks_by_id = {t["task_id"]: t for t in load_tasks(tasks_dir)}
    records = []
    for path in runs_dir.glob("*.json"):
        with open(path, encoding="utf-8") as f:
            run = json.load(f)
        if run.get("error"):
            continue
        task = tasks_by_id.get(run.get("task_id"), {})
        rubric = task.get("rubric") or ["Output is correct and complete"]
        result = judge_single(
            task.get("input", ""),
            rubric,
            run.get("output", ""),
            judge_model=judge_model,
            judge_provider=judge_provider,
        )
        rec = {
            "task_id": run.get("task_id"),
            "model_id": run.get("model_id"),
            "rule_id": run.get("rule_id"),
            "run_id": run.get("run_id"),
            "metric_name": "llm_judge",
            "score": result["score"],
            "judge_type": "llm",
            "details": {"reason": result["reason"], "judge_model": judge_model},
        }
        records.append(rec)
        out_path = scores_dir / f"{run.get('run_id')}_judge.json"
        scores_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
    return records

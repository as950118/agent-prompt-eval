"""Run (model, rule) on tasks and save outputs with metadata."""
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config_loader import load_models, list_rules
from .task_loader import load_tasks


def _rule_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def _build_messages(task: dict[str, Any], rule_content: str) -> list[dict[str, str]]:
    """Build chat messages: system = rules, user = task input + context."""
    system = rule_content or "You are a helpful coding assistant."
    user_parts = [task.get("input", "")]
    ctx = task.get("context")
    if ctx:
        if isinstance(ctx, dict):
            snippet = ctx.get("snippet") or ctx.get("content")
            if snippet:
                user_parts.append("\n\nContext:\n```\n" + snippet + "\n```")
        elif isinstance(ctx, str):
            user_parts.append("\n\nContext:\n" + ctx)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


def _call_openai(model_id: str, messages: list[dict], seed: int | None = 42) -> dict[str, Any]:
    """Call OpenAI API. model_id is the model name (e.g. gpt-4o-mini)."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    kwargs = {"model": model_id, "messages": messages}
    if seed is not None:
        kwargs["seed"] = seed
    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    usage = resp.usage
    return {
        "content": choice.message.content or "",
        "finish_reason": getattr(choice, "finish_reason", None),
        "usage": {
            "prompt_tokens": getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", None),
        } if usage else {},
    }


def _call_anthropic(model_id: str, messages: list[dict], seed: int | None = 42) -> dict[str, Any]:
    """Call Anthropic API."""
    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_msgs = [m for m in messages if m["role"] != "system"]
    kwargs = {
        "model": model_id,
        "max_tokens": 4096,
        "system": system,
        "messages": user_msgs,
    }
    if seed is not None:
        kwargs["random_seed"] = seed
    resp = client.messages.create(**kwargs)
    if resp.content and hasattr(resp.content[0], "text"):
        content = resp.content[0].text
    elif resp.content:
        content = str(resp.content[0])
    else:
        content = ""
    usage = getattr(resp, "usage", None)
    return {
        "content": content,
        "finish_reason": getattr(resp, "stop_reason", None),
        "usage": {
            "prompt_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
            "completion_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
            "total_tokens": None,
        } if usage else {},
    }


def call_llm(provider: str, model_id: str, messages: list[dict], seed: int | None = 42) -> dict[str, Any]:
    """Dispatch to provider."""
    if provider == "openai":
        return _call_openai(model_id, messages, seed)
    if provider == "anthropic":
        return _call_anthropic(model_id, messages, seed)
    raise ValueError(f"Unknown provider: {provider}")


def run_single(
    task: dict[str, Any],
    model_config: dict[str, Any],
    rule_id: str,
    rule_content: str,
    seed: int | None = 42,
) -> dict[str, Any]:
    """Run one task with one (model, rule) and return run result."""
    run_id = str(uuid4())
    messages = _build_messages(task, rule_content)
    provider = model_config.get("provider", "openai")
    model_id = model_config.get("model", model_config.get("id", ""))
    start = time.perf_counter()
    try:
        out = call_llm(provider, model_id, messages, seed)
    except Exception as e:
        return {
            "run_id": run_id,
            "task_id": task.get("task_id", ""),
            "model_id": model_config.get("id", model_id),
            "rule_id": rule_id,
            "rule_hash": _rule_hash(rule_content),
            "output": "",
            "error": str(e),
            "latency_sec": time.perf_counter() - start,
            "token_usage": {},
        }
    usage = out.get("usage", {})
    return {
        "run_id": run_id,
        "task_id": task.get("task_id", ""),
        "model_id": model_config.get("id", model_id),
        "rule_id": rule_id,
        "rule_hash": _rule_hash(rule_content),
        "output": out.get("content", ""),
        "error": None,
        "latency_sec": time.perf_counter() - start,
        "token_usage": usage,
    }


def run_all(
    tasks_dir: str | Path,
    configs_dir: str | Path,
    results_dir: str | Path,
    seed: int | None = 42,
    model_ids: list[str] | None = None,
    rule_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run all (task, model, rule) combinations and save each run to results_dir/runs/."""
    tasks = load_tasks(tasks_dir)
    models_config = load_models(Path(configs_dir) / "models.yaml")
    rules = list_rules(Path(configs_dir) / "rules")
    if model_ids:
        models_config = [m for m in models_config if m.get("id") in model_ids]
    if rule_ids:
        rules = [(rid, c) for rid, c in rules if rid in rule_ids]

    results_dir = Path(results_dir)
    runs_dir = results_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    all_runs = []

    for task in tasks:
        for model_cfg in models_config:
            for rule_id, rule_content in rules:
                run_result = run_single(task, model_cfg, rule_id, rule_content, seed)
                all_runs.append(run_result)
                out_path = runs_dir / f"{run_result['run_id']}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(run_result, f, ensure_ascii=False, indent=2)
    return all_runs

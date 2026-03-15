"""Load evaluation tasks from YAML/JSON files."""
from pathlib import Path
from typing import Any

import yaml


def load_task(path: Path) -> dict[str, Any]:
    """Load a single task from a YAML or JSON file."""
    with open(path, encoding="utf-8") as f:
        if path.suffix in (".yaml", ".yml"):
            return yaml.safe_load(f) or {}
        if path.suffix == ".json":
            import json
            return json.load(f)
    raise ValueError(f"Unsupported task format: {path.suffix}")


def load_tasks(tasks_dir: str | Path) -> list[dict[str, Any]]:
    """Load all tasks from a directory. Expects task_*.yaml or task_*.json."""
    tasks_dir = Path(tasks_dir)
    if not tasks_dir.is_dir():
        return []
    tasks = []
    for path in sorted(tasks_dir.glob("task_*.*")):
        if path.suffix not in (".yaml", ".yml", ".json"):
            continue
        try:
            t = load_task(path)
            if t and t.get("task_id"):
                tasks.append(t)
        except Exception:
            continue
    return tasks

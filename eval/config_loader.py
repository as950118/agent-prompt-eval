"""Load model and rule configurations."""
from pathlib import Path
from typing import Any

import yaml


def load_models(config_path: str | Path) -> list[dict[str, Any]]:
    """Load model list from configs/models.yaml."""
    config_path = Path(config_path)
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("models", [])


def load_rule(rule_path: str | Path) -> str:
    """Load rule content from a markdown or text file."""
    path = Path(rule_path)
    with open(path, encoding="utf-8") as f:
        return f.read()


def list_rules(rules_dir: str | Path) -> list[tuple[str, str]]:
    """List (rule_id, content) for all rule files in rules_dir."""
    rules_dir = Path(rules_dir)
    if not rules_dir.is_dir():
        return []
    result = []
    for path in sorted(rules_dir.glob("*")):
        if path.suffix in (".md", ".txt", ".yaml", ".yml") and path.name != "README.md":
            rule_id = path.stem
            result.append((rule_id, load_rule(path)))
    return result

# Task dataset

Each task is a YAML or JSON file with the following schema.

## Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| task_id | string | yes | Unique identifier |
| input | string | yes | The prompt/request given to the agent (e.g. code edit request, question) |
| context | object/string | no | Codebase snapshot, file paths, or other context |
| ground_truth | object/string | no | Expected patch, expected output, or reference answer |
| rubric | list/string | no | Scoring criteria (e.g. correctness, style, test pass) |
| category | string | no | Task type: bug_fix, refactor, feature, explanation, etc. |
| test_command | string | no | Command to run for test-based evaluation (e.g. `pytest tests/`) |
| fail_to_pass_tests | list | no | Test identifiers that must pass after fix (SWE-bench style) |
| pass_to_pass_tests | list | no | Tests that must remain passing (regression) |

## Example

See `task_001.yaml` through `task_010.yaml` for sample tasks.

"""Run every prompt/config version in configs/versions.yaml (in order), then compare and promote.

Usage:  uv run python src/run_all_versions.py            # needs a running LLM endpoint (default: Ollama)
        uv run python src/run_all_versions.py v1 v2      # subset
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import compare_versions  # noqa: E402
import run_experiment  # noqa: E402


def main(versions: list[str] | None = None) -> None:
    all_versions = list(yaml.safe_load((ROOT / "configs" / "versions.yaml").read_text(encoding="utf-8")))
    for v in versions or all_versions:
        run_experiment.main(v)
    compare_versions.main()


if __name__ == "__main__":
    main(sys.argv[1:] or None)

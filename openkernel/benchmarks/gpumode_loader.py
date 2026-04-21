"""GPU MODE benchmark loader.

Wraps the 8 GPU MODE problems under ``data/benchmarks/gpumode/`` as
``ProblemSpec`` instances consumed by the batch-runner.
"""

from __future__ import annotations

import re
from pathlib import Path

from openkernel.benchmarks.problem_spec import ProblemSpec

_GPUMODE_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "benchmarks"
    / "gpumode"
)

_PROBLEMS = [
    "vectoradd",
    "vectorsum",
    "grayscale",
    "matmul",
    "histogram",
    "conv2d",
    "prefixsum",
    "sort",
]


def _detect_dtype(source: str) -> str:
    m = re.search(r"dtype\s*=\s*torch\.(\w+)", source)
    return m.group(1) if m else "float32"


def _load_task(task_yml: Path) -> tuple[list[dict], str | None]:
    try:
        import yaml
    except ImportError:
        return [], None
    if not task_yml.exists():
        return [], None
    data = yaml.safe_load(task_yml.read_text())
    if not isinstance(data, dict):
        return [], None
    tests = data.get("tests", []) or []
    name = data.get("name")
    return tests, name


def load_gpumode() -> list[ProblemSpec]:
    specs: list[ProblemSpec] = []
    for slug in _PROBLEMS:
        pdir = _GPUMODE_DIR / f"{slug}_py"
        ref = pdir / "reference.py"
        yml = pdir / "task.yml"
        if not ref.exists():
            raise FileNotFoundError(f"missing reference.py for gpumode problem {slug!r}: {ref}")
        source = ref.read_text()
        tests, yml_name = _load_task(yml)
        name = yml_name or slug.replace("_", " ").title()
        specs.append(
            ProblemSpec(
                id=f"gpumode_{slug}",
                name=name,
                tier="GPU_MODE",
                source="gpumode",
                reference_source=source,
                workload_spec={"tests": tests},
                expected_dtype=_detect_dtype(source),
            )
        )
    return specs


if __name__ == "__main__":
    specs = load_gpumode()
    print(f"Loaded {len(specs)} GPU MODE specs")
    for s in specs:
        tests = s.workload_spec.get("tests", [])
        print(f"  {s.id} ({s.expected_dtype}) — {s.name} — {len(tests)} tests")

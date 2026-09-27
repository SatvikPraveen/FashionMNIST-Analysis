"""
Execute every ```python block in docs/FEATURES.md, so documented examples
cannot silently go stale when the code changes.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "FEATURES.md"
BLOCKS = re.findall(r"```python\n(.*?)```", DOC.read_text(encoding="utf-8"), re.S)


def test_doc_has_examples():
    assert len(BLOCKS) >= 8


@pytest.mark.parametrize("code", BLOCKS, ids=[f"example{i + 1}" for i in range(len(BLOCKS))])
def test_example_runs(code, monkeypatch):
    monkeypatch.chdir(ROOT)          # examples use repo-relative paths such as config.yaml
    exec(compile(code, str(DOC), "exec"), {"__name__": "__doc_example__"})

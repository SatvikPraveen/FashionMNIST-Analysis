"""
Every flag a command-line tool accepts must appear in docs/CLI_REFERENCE.md.
If this fails, run `python docs/cli_reference.py` and commit the result.
"""

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("cli_reference", ROOT / "docs" / "cli_reference.py")
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
DOC = (ROOT / "docs" / "CLI_REFERENCE.md").read_text(encoding="utf-8")
FLAG = re.compile(r"(?<![\w-])--[a-z][a-z0-9_-]*")


@pytest.mark.parametrize("title,argv", [(t, a) for t, a, _ in ref.TOOLS], ids=[t for t, _, _ in ref.TOOLS])
def test_every_flag_is_documented(title, argv):
    flags = set(FLAG.findall(ref.help_text(argv))) - {"--help"}
    missing = sorted(f for f in flags if f not in DOC)
    assert not missing, f"{title}: flags missing from docs/CLI_REFERENCE.md: {missing}"

"""
The project website is generated from results/sweeps/*.csv by site/build.py.
These tests build it and check it against the data, so a CSV or code change
that would break the published page fails CI first.
"""

import csv
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "site" / "build.py"


def _load_build():
    # 'site' is also a standard-library module name, so load by path.
    spec = importlib.util.spec_from_file_location("fm_site_build", BUILD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("site")
    mod = _load_build()
    assert mod.main(["--out", str(out)]) == 0
    return mod, (out / "index.html").read_text(encoding="utf-8")


def test_page_and_nojekyll_written(built, tmp_path):
    mod = _load_build()
    mod.main(["--out", str(tmp_path)])
    assert (tmp_path / "index.html").exists() and (tmp_path / ".nojekyll").exists()


def test_headline_matches_data(built):
    _, page = built
    with open(ROOT / "results/sweeps/backbones_fixed_summary.csv", newline="") as f:
        vit = next(r for r in csv.DictReader(f) if r["group"] == "vit_tiny|pretrained")
    assert f"{float(vit['test_acc_mean']) * 100:.2f}%" in page


def test_every_chart_has_both_layouts_and_a_table(built):
    _, page = built
    assert page.count('class="chart wide"') == 4
    assert page.count('class="chart narrow"') == 4
    assert page.count("<svg ") == page.count("</svg>") == 8
    assert page.count('class="tableview"') == 4


def test_tooltips_are_non_empty(built):
    _, page = built
    tips = re.findall(r'data-tip="([^"]*)"', page)
    assert len(tips) >= 40 and all(t.strip() for t in tips)


def test_run_count_is_computed(built):
    mod, page = built
    n = mod.count_runs()
    assert n > 100 and f">{n}<" in page


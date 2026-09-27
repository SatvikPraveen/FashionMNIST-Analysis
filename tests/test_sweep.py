"""
Tests for the sweep manifest / runner and the results aggregator.
"""

import json

import pytest
import yaml

from src.cli import sweep as sw
from src.cli import aggregate as ag


def _sweep_file(tmp_path, **over):
    spec = {"name": "t", "output_root": str(tmp_path / "runs"), "models": ["tinyvgg", "minicnn"],
            "seeds": [0, 1, 2], **over}
    path = tmp_path / "t.yaml"
    path.write_text(yaml.safe_dump(spec))
    return str(path)


class TestExpand:
    def test_models_x_seeds(self, tmp_path):
        rows = sw.expand(sw.load_sweep(_sweep_file(tmp_path)))
        assert len(rows) == 6
        assert [r["index"] for r in rows] == list(range(6))
        assert rows[0]["run_name"] == "tinyvgg_seed0" and rows[0]["group"] == "tinyvgg|default"
        assert len({r["run_name"] for r in rows}) == 6

    def test_variants_and_grid(self, tmp_path):
        path = _sweep_file(tmp_path, variants={"a": {"x.y": 1}, "b": {"x.y": 2}},
                           grid={"training.learning_rate": [1e-3, 1e-4]}, seeds=[0])
        rows = sw.expand(sw.load_sweep(path))
        assert len(rows) == 2 * 2 * 2
        r = rows[1]
        assert r["variant"] == "a" and r["overrides"] == {"x.y": 1, "training.learning_rate": 1e-4}
        assert "learning_rate0.0001" in r["run_name"]
        assert len({r["group"] for r in rows}) == 8  # each (model, variant, grid) is its own group

    def test_expand_is_deterministic(self, tmp_path):
        s = sw.load_sweep(_sweep_file(tmp_path))
        assert sw.expand(s) == sw.expand(s)

    def test_missing_models_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump({"seeds": [0]}))
        with pytest.raises(ValueError):
            sw.load_sweep(str(path))

    def test_write_and_read_manifest(self, tmp_path):
        s = sw.load_sweep(_sweep_file(tmp_path))
        rows = sw.expand(s)
        sw.write_manifest(s, rows)
        assert (tmp_path / "runs" / "manifest.csv").exists()
        assert sw.read_manifest(s) == rows


class TestArgv:
    def test_build_argv_contains_overrides_and_flags(self, tmp_path):
        s = sw.load_sweep(_sweep_file(tmp_path, base_args=["--amp"],
                                      variants={"v": {"training.label_smoothing": 0.1}}))
        row = sw.expand(s)[0]
        argv = sw.build_argv(s, row, extra=["--num-workers", "4"])
        joined = " ".join(argv)
        assert "--model tinyvgg" in joined and "--seed 0" in joined
        assert "--skip-best-selection" in argv and "--resume" in argv and "--amp" in argv
        assert "training.label_smoothing=0.1" in argv
        assert "sweep.group=tinyvgg|v" in argv
        assert argv[-2:] == ["--num-workers", "4"]

    def test_cli_expand_and_dry_run(self, tmp_path, capsys):
        path = _sweep_file(tmp_path, seeds=[0])
        assert sw.main(["expand", path]) == 0
        assert "2 runs" in capsys.readouterr().out
        assert sw.main(["run", path, "--index", "1", "--dry-run"]) == 0
        out = capsys.readouterr().out
        assert "minicnn_seed0" in out and "train.py" in out
        with pytest.raises(SystemExit):
            sw.main(["run", path, "--index", "5", "--dry-run"])

    def test_run_forwards_extra_args_after_double_dash(self, tmp_path, capsys):
        """Exactly what the sbatch scripts do: run ... -- --num-workers 8."""
        path = _sweep_file(tmp_path, seeds=[0])
        assert sw.main(["run", path, "--index", "0", "--dry-run", "--", "--num-workers", "8", "--amp"]) == 0
        out = capsys.readouterr().out
        assert "--num-workers 8 --amp" in out

    def test_expand_rejects_stray_args(self, tmp_path):
        path = _sweep_file(tmp_path, seeds=[0])
        with pytest.raises(SystemExit):
            sw.main(["expand", path, "--bogus"])

    def test_status_pending(self, tmp_path, capsys):
        path = _sweep_file(tmp_path, seeds=[0])
        assert sw.main(["status", path]) == 0
        assert "pending=2" in capsys.readouterr().out


def _fake_run(root, group, model, seed, acc, status="finished", variant="default"):
    d = root / f"{model}_{variant}_seed{seed}" / model
    d.mkdir(parents=True)
    run = {"run_name": f"{model}_seed{seed}", "status": status,
           "params": {"model": model, "seed": seed, "num_parameters": 1000},
           "summary": {"test_acc": acc, "best_val_acc": acc, "train_time_sec": 60.0} if status == "finished" else {},
           "config": {"sweep": {"name": "t", "group": group, "variant": variant}},
           "git": {"commit": "abc"}, "slurm": {"slurm_job_id": "1"}}
    (d / "run.json").write_text(json.dumps(run))


class TestAggregate:
    def test_summary_stats(self, tmp_path):
        root = tmp_path / "runs"
        for seed, acc in enumerate([0.90, 0.92, 0.94]):
            _fake_run(root, "tinyvgg|default", "tinyvgg", seed, acc)
        _fake_run(root, "minicnn|default", "minicnn", 0, 0.80)
        _fake_run(root, "minicnn|default", "minicnn", 1, 0.0, status="failed")

        rows = ag.find_runs([str(root)])
        assert len(rows) == 5
        summary = ag.summarize(rows, "test_acc", "group")
        assert [s["group"] for s in summary] == ["tinyvgg|default", "minicnn|default"]
        tv = summary[0]
        assert tv["n"] == 3
        assert tv["test_acc_mean"] == pytest.approx(0.92)
        assert tv["test_acc_std"] == pytest.approx(0.02)
        assert tv["test_acc_ci95"] == pytest.approx(4.303 * 0.02 / 3 ** 0.5)
        assert tv["seeds"] == [0, 1, 2]
        mc = summary[1]
        assert mc["n"] == 1 and mc["test_acc_ci95"] != mc["test_acc_ci95"]  # NaN for n=1

    def test_markdown_and_files(self, tmp_path, capsys):
        root = tmp_path / "runs"
        _fake_run(root, "g", "tinyvgg", 0, 0.9)
        _fake_run(root, "g", "tinyvgg", 1, 0.8)
        out = tmp_path / "res" / "t"
        assert ag.main([str(root), "--out", str(out)]) == 0
        printed = capsys.readouterr().out
        assert "| g | 2 | 0.8500 ± 0.0707" in printed
        assert (tmp_path / "res" / "t_summary.md").exists()
        assert (tmp_path / "res" / "t_runs.csv").exists()
        assert (tmp_path / "res" / "t_summary.csv").exists()

    def test_group_by_model_pools_variants(self, tmp_path):
        root = tmp_path / "runs"
        _fake_run(root, "tinyvgg|a", "tinyvgg", 0, 0.9, variant="a")
        _fake_run(root, "tinyvgg|b", "tinyvgg", 0, 0.7, variant="b")
        summary = ag.summarize(ag.find_runs([str(root)]), group_by="model")
        assert len(summary) == 1 and summary[0]["n"] == 2


class TestPaired:
    def _rows(self, tmp_path):
        root = tmp_path / "runs"
        base = [0.90, 0.92, 0.91, 0.93, 0.89]
        for seed, acc in enumerate(base):
            _fake_run(root, "m|full", "m", seed, acc, variant="full")
            # consistently +0.01 better on every seed, despite large seed spread
            _fake_run(root, "m|better", "m", seed, acc + 0.01, variant="better")
        # only seeds 0-2 exist for this group -> pairs on 3 seeds
        for seed in range(3):
            _fake_run(root, "m|partial", "m", seed, base[seed] - 0.02, variant="partial")
        return ag.find_runs([str(root)])

    def test_consistent_small_gain_is_significant_when_paired(self, tmp_path):
        rows = self._rows(tmp_path)
        paired = {d["group"]: d for d in ag.paired_comparison(rows, "m|full")}
        b = paired["m|better"]
        assert b["n_paired"] == 5 and b["wins"] == 5 and b["losses"] == 0
        assert b["diff_mean"] == pytest.approx(0.01)
        assert b["diff_std"] == pytest.approx(0.0, abs=1e-12)  # identical shift -> p undefined
        pt = paired["m|partial"]
        assert pt["n_paired"] == 3 and pt["seeds"] == [0, 1, 2]
        assert pt["diff_mean"] == pytest.approx(-0.02)

    def test_p_value_matches_scipy(self, tmp_path):
        scipy_stats = pytest.importorskip("scipy.stats")
        root = tmp_path / "runs"
        a = [0.90, 0.92, 0.91, 0.93, 0.89]
        b = [0.905, 0.93, 0.912, 0.94, 0.893]
        for seed in range(5):
            _fake_run(root, "g|a", "g", seed, a[seed], variant="a")
            _fake_run(root, "g|b", "g", seed, b[seed], variant="b")
        d = ag.paired_comparison(ag.find_runs([str(root)]), "g|a")[0]
        assert d["p_value"] == pytest.approx(scipy_stats.ttest_rel(b, a).pvalue)

    def test_unknown_baseline_raises(self, tmp_path):
        with pytest.raises(ValueError):
            ag.paired_comparison(self._rows(tmp_path), "nope")

    def test_cli_writes_paired_files(self, tmp_path, capsys):
        self._rows(tmp_path)
        out = tmp_path / "res" / "x"
        assert ag.main([str(tmp_path / "runs"), "--baseline", "m|full", "--out", str(out)]) == 0
        assert "Paired by seed" in capsys.readouterr().out
        assert (tmp_path / "res" / "x_paired.md").exists()
        assert (tmp_path / "res" / "x_paired.csv").exists()

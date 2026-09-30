from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from decision_model_lab import cli

runner = CliRunner()


def _write_dataset(path: Path) -> None:
    path.write_text(
        '{"id":"case-1","context":"c","question":"q",'
        '"decision":{"type":"noul"},"expected":{"label":"yes"}}\n',
        encoding="utf-8",
    )


def test_benchmark_battery_requires_explicit_candidates_and_datasets() -> None:
    result = runner.invoke(cli.app, ["benchmark", "battery", "--allow-cpu"])

    assert result.exit_code == 2
    assert "requires at least one --dataset" in result.output


def test_benchmark_battery_accepts_repeatable_candidate_and_dataset_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "benchmark-v3.jsonl"
    second = tmp_path / "benchmark-v3-pt-br.jsonl"
    _write_dataset(first)
    _write_dataset(second)
    captured: dict[str, object] = {}

    def fake_battery(**kwargs: object) -> list[SimpleNamespace]:
        captured.update(kwargs)
        return [SimpleNamespace(path=tmp_path / "artifact.json")]

    monkeypatch.setattr(cli, "run_resident_battery", fake_battery)

    result = runner.invoke(
        cli.app,
        [
            "benchmark",
            "battery",
            "--dataset",
            str(first),
            "--dataset",
            str(second),
            "--candidate",
            "jev-style",
            "--candidate",
            "verdict",
            "--allow-cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["candidate_names"] == ("jev-style", "verdict")
    assert captured["datasets"] == (first, second)
    assert captured["verify_state_isolation"] is True
    assert "candidates=2 datasets=2 runs=1" in result.output

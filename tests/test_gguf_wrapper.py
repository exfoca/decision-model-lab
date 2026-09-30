from __future__ import annotations

import os
from pathlib import Path
from subprocess import CompletedProcess, run

WRAPPER = Path(__file__).parents[1] / "tools" / "jev-score-v2-cuda"


def _fake_native(tmp_path: Path) -> Path:
    executable = tmp_path / "jev-score-v2-native"
    executable.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n', encoding="utf-8")
    executable.chmod(0o755)
    return executable


def _run_wrapper(
    tmp_path: Path,
    *args: str,
    gpu_layers: str = "999",
) -> CompletedProcess[str]:
    env = os.environ.copy()
    env["DML_JEV_SCORE_V2_NATIVE_BIN"] = str(_fake_native(tmp_path))
    env["DML_GGUF_N_GPU_LAYERS"] = gpu_layers
    return run(
        [str(WRAPPER), *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_wrapper_injects_configured_gpu_layers_for_model_execution(tmp_path: Path) -> None:
    result = _run_wrapper(tmp_path, "--model", "model.gguf", "--n-ctx", "1024")

    assert result.returncode == 0
    assert result.stdout.splitlines() == [
        "--model",
        "model.gguf",
        "--n-ctx",
        "1024",
        "--ngl",
        "999",
    ]


def test_wrapper_enforces_lab_gpu_layer_policy_over_upstream_ngl(tmp_path: Path) -> None:
    result = _run_wrapper(tmp_path, "--model", "model.gguf", "--ngl", "12")

    assert result.returncode == 0
    assert result.stdout.splitlines() == ["--model", "model.gguf", "--ngl", "999"]


def test_wrapper_does_not_modify_non_model_diagnostics(tmp_path: Path) -> None:
    result = _run_wrapper(tmp_path, "--help")

    assert result.returncode == 0
    assert result.stdout.splitlines() == ["--help"]


def test_wrapper_rejects_invalid_gpu_layer_configuration(tmp_path: Path) -> None:
    result = _run_wrapper(tmp_path, "--model", "model.gguf", gpu_layers="invalid")

    assert result.returncode == 2
    assert "DML_GGUF_N_GPU_LAYERS must be a non-negative integer" in result.stderr

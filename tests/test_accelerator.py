from __future__ import annotations

import pytest

from decision_model_lab import accelerator


def test_gguf_cuda_requirement_checks_native_backend_and_offload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(accelerator, "libcuda_available", lambda: True)
    monkeypatch.setattr(accelerator, "which", lambda name: "/usr/local/bin/jev-score-v2")
    monkeypatch.setattr(accelerator, "gguf_cuda_available", lambda: True)
    monkeypatch.setenv("DML_GGUF_N_GPU_LAYERS", "999")

    assert accelerator.candidate_cuda_error("gguf") is None


def test_gguf_cuda_requirement_rejects_cpu_only_scorer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(accelerator, "libcuda_available", lambda: True)
    monkeypatch.setattr(accelerator, "which", lambda name: "/usr/local/bin/jev-score-v2")
    monkeypatch.setattr(accelerator, "gguf_cuda_available", lambda: False)

    assert accelerator.candidate_cuda_error("gguf") == (
        "libggml-cuda.so.0 is unavailable; the GGUF scorer was built without CUDA"
    )


def test_gguf_cuda_requirement_rejects_zero_gpu_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(accelerator, "libcuda_available", lambda: True)
    monkeypatch.setattr(accelerator, "which", lambda name: "/usr/local/bin/jev-score-v2")
    monkeypatch.setattr(accelerator, "gguf_cuda_available", lambda: True)
    monkeypatch.setenv("DML_GGUF_N_GPU_LAYERS", "0")

    assert accelerator.candidate_cuda_error("gguf") == (
        "DML_GGUF_N_GPU_LAYERS=0 disables GPU layer offload"
    )


def test_gguf_gpu_layer_setting_must_be_non_negative_integer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DML_GGUF_N_GPU_LAYERS", "invalid")
    with pytest.raises(ValueError, match="non-negative integer"):
        accelerator.gguf_n_gpu_layers()

    monkeypatch.setenv("DML_GGUF_N_GPU_LAYERS", "-1")
    with pytest.raises(ValueError, match="non-negative integer"):
        accelerator.gguf_n_gpu_layers()


def test_torch_cuda_requirement_remains_independent_of_gguf_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(accelerator, "torch_cuda_available", lambda: True)
    assert accelerator.candidate_cuda_error("torch") is None

    monkeypatch.setattr(accelerator, "torch_cuda_available", lambda: False)
    assert accelerator.candidate_cuda_error("torch") == "torch.cuda.is_available() is false"

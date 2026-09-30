from __future__ import annotations

from datetime import UTC, datetime
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from os import environ
from pathlib import Path
from platform import machine, platform, python_implementation, python_version
from subprocess import DEVNULL, PIPE, TimeoutExpired, run
from typing import Any

from pydantic import BaseModel

from decision_model_lab.accelerator import CudaBackend, gguf_cuda_available, gguf_n_gpu_layers
from decision_model_lab.schema import DecisionResult


class RunProvenance(BaseModel):
    captured_at_utc: datetime
    repository_commit: str | None = None
    repository_dirty: bool | None = None
    python_version: str
    python_implementation: str
    platform: str
    machine: str
    torch_version: str | None = None
    torch_cuda_version: str | None = None
    cuda_available: bool | None = None
    cuda_device: str | None = None
    candidate_cuda_backend: CudaBackend | None = None
    gguf_cuda_available: bool | None = None
    gguf_n_gpu_layers: int | None = None
    candidate_runtime_distribution: str | None = None
    candidate_runtime_version: str | None = None
    candidate_runtime_revision: str | None = None
    model_revision: str | None = None
    quantization: str | None = None
    container_base_image: str | None = None
    container_source_revision: str | None = None


def _repository_state(root: Path | None = None) -> tuple[str | None, bool | None]:
    repository_root = root if root is not None else Path.cwd()

    try:
        commit_process = run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=False,
            stdout=PIPE,
            stderr=DEVNULL,
            text=True,
            timeout=2,
        )
    except (OSError, TimeoutExpired):
        return None, None

    if commit_process.returncode != 0:
        return None, None

    commit = commit_process.stdout.strip() or None
    try:
        status_process = run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=repository_root,
            check=False,
            stdout=PIPE,
            stderr=DEVNULL,
            text=True,
            timeout=2,
        )
    except (OSError, TimeoutExpired):
        return commit, None

    dirty = None if status_process.returncode != 0 else bool(status_process.stdout.strip())
    return commit, dirty


def _torch_state() -> dict[str, Any]:
    try:
        torch: Any = import_module("torch")
    except (ImportError, OSError):
        return {
            "torch_version": None,
            "torch_cuda_version": None,
            "cuda_available": None,
            "cuda_device": None,
        }

    try:
        cuda_available = bool(torch.cuda.is_available())
        cuda_device = torch.cuda.get_device_name(0) if cuda_available else None
    except (AttributeError, RuntimeError):
        cuda_available = None
        cuda_device = None

    torch_cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    return {
        "torch_version": str(torch.__version__),
        "torch_cuda_version": (str(torch_cuda_version) if torch_cuda_version is not None else None),
        "cuda_available": cuda_available,
        "cuda_device": cuda_device,
    }


def _distribution_version(distribution: str | None) -> str | None:
    if distribution is None:
        return None
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _metadata_value(results: list[DecisionResult], key: str) -> str | None:
    for result in results:
        value = result.metadata.get(key)
        if value is not None:
            return str(value)
    return None


def _normalized_environment_value(name: str) -> str | None:
    value = environ.get(name)
    if value is None or value.strip() in {"", "unknown"}:
        return None
    return value.strip()


def collect_run_provenance(
    *,
    runner: object,
    results: list[DecisionResult],
    runtime_distribution: str | None,
    cuda_backend: CudaBackend = "torch",
) -> RunProvenance:
    repository_commit, repository_dirty = _repository_state()
    torch_state = _torch_state()

    resolved_revision = getattr(runner, "_resolved_revision", None)
    requested_revision = getattr(runner, "revision", None)
    model_revision = (
        str(resolved_revision)
        if resolved_revision is not None
        else str(requested_revision)
        if requested_revision is not None
        else _metadata_value(results, "model_revision")
    )
    quantization = getattr(runner, "quantization", None)
    runtime_revision = _metadata_value(results, "upstream_engine_revision")

    return RunProvenance(
        captured_at_utc=datetime.now(UTC),
        repository_commit=repository_commit,
        repository_dirty=repository_dirty,
        python_version=python_version(),
        python_implementation=python_implementation(),
        platform=platform(),
        machine=machine(),
        torch_version=torch_state["torch_version"],
        torch_cuda_version=torch_state["torch_cuda_version"],
        cuda_available=torch_state["cuda_available"],
        cuda_device=torch_state["cuda_device"],
        candidate_cuda_backend=cuda_backend,
        gguf_cuda_available=gguf_cuda_available() if cuda_backend == "gguf" else None,
        gguf_n_gpu_layers=gguf_n_gpu_layers() if cuda_backend == "gguf" else None,
        candidate_runtime_distribution=runtime_distribution,
        candidate_runtime_version=_distribution_version(runtime_distribution),
        candidate_runtime_revision=runtime_revision,
        model_revision=model_revision,
        quantization=str(quantization) if quantization is not None else None,
        container_base_image=_normalized_environment_value("DML_CONTAINER_BASE_IMAGE"),
        container_source_revision=_normalized_environment_value("DML_CONTAINER_SOURCE_REVISION"),
    )

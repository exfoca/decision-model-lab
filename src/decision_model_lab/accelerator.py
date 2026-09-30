from __future__ import annotations

from ctypes import CDLL
from importlib import import_module
from os import environ
from shutil import which
from subprocess import STDOUT, CalledProcessError, check_output
from typing import Any, Literal

CudaBackend = Literal["torch", "gguf"]
DEFAULT_GGUF_N_GPU_LAYERS = 999


def torch_cuda_available() -> bool:
    torch: Any = import_module("torch")
    return bool(torch.cuda.is_available())


def libcuda_available() -> bool:
    try:
        CDLL("libcuda.so.1")
    except OSError:
        return False
    return True


def gguf_cuda_available() -> bool:
    """Return whether the CUDA backend built by ggml is loadable."""
    try:
        CDLL("libggml-cuda.so.0")
    except OSError:
        return False
    return True


def ptxas_diagnostics() -> tuple[str | None, str | None]:
    path = which("ptxas")
    if path is None:
        return None, None

    try:
        output = check_output([path, "--version"], stderr=STDOUT, text=True)
    except (CalledProcessError, OSError):
        return path, None

    version = next(
        (line.strip() for line in output.splitlines() if "release" in line.lower()),
        None,
    )
    return path, version


def gguf_n_gpu_layers() -> int:
    raw_value = environ.get("DML_GGUF_N_GPU_LAYERS", str(DEFAULT_GGUF_N_GPU_LAYERS))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError("DML_GGUF_N_GPU_LAYERS must be a non-negative integer") from exc
    if value < 0:
        raise ValueError("DML_GGUF_N_GPU_LAYERS must be a non-negative integer")
    return value


def candidate_cuda_error(backend: CudaBackend) -> str | None:
    """Return a precise reason why a candidate cannot satisfy --require-cuda."""
    if backend == "gguf":
        if not libcuda_available():
            return "libcuda.so.1 is unavailable inside the container"
        if which("jev-score-v2") is None:
            return "jev-score-v2 is unavailable inside the container"
        if not gguf_cuda_available():
            return "libggml-cuda.so.0 is unavailable; the GGUF scorer was built without CUDA"
        try:
            gpu_layers = gguf_n_gpu_layers()
        except ValueError as exc:
            return str(exc)
        if gpu_layers == 0:
            return "DML_GGUF_N_GPU_LAYERS=0 disables GPU layer offload"
        return None

    try:
        available = torch_cuda_available()
    except (ImportError, OSError):
        available = False
    if not available:
        return "torch.cuda.is_available() is false"
    return None


def runtime_diagnostics() -> dict[str, Any]:
    numpy: Any = import_module("numpy")
    torch: Any = import_module("torch")
    cuda_available = bool(torch.cuda.is_available())
    ptxas_path, ptxas_version = ptxas_diagnostics()
    return {
        "numpy_version": str(numpy.__version__),
        "libcuda_available": libcuda_available(),
        "torch_version": str(torch.__version__),
        "cuda_available": cuda_available,
        "torch_cuda_version": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
        "ptxas_path": ptxas_path,
        "ptxas_version": ptxas_version,
        "gguf_scorer_path": which("jev-score-v2"),
        "gguf_cuda_available": gguf_cuda_available(),
        "gguf_n_gpu_layers": gguf_n_gpu_layers(),
    }

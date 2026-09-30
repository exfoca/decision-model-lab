from __future__ import annotations

from decision_model_lab import provenance
from decision_model_lab.schema import DecisionResult


class DummyRunner:
    revision = "requested-revision"
    _resolved_revision = "resolved-revision"
    quantization = "Q4_K_M"


def test_collect_run_provenance_records_runtime_and_model_evidence(monkeypatch) -> None:
    monkeypatch.setattr(provenance, "_repository_state", lambda: ("a" * 40, False))
    monkeypatch.setattr(
        provenance,
        "_torch_state",
        lambda: {
            "torch_version": "2.9.0",
            "torch_cuda_version": "13.0",
            "cuda_available": True,
            "cuda_device": "Synthetic GPU",
        },
    )
    monkeypatch.setattr(provenance, "_distribution_version", lambda name: "0.3.0")
    monkeypatch.setattr(provenance, "_normalized_environment_value", lambda name: None)

    result = DecisionResult(
        case_id="case-a",
        label="yes",
        probabilities={"yes": 1.0},
        latency_ms=1.0,
        metadata={"upstream_engine_revision": "engine-revision"},
    )

    receipt = provenance.collect_run_provenance(
        runner=DummyRunner(),
        results=[result],
        runtime_distribution="jev-style",
    )

    assert receipt.repository_commit == "a" * 40
    assert receipt.repository_dirty is False
    assert receipt.candidate_runtime_distribution == "jev-style"
    assert receipt.candidate_runtime_version == "0.3.0"
    assert receipt.candidate_runtime_revision == "engine-revision"
    assert receipt.model_revision == "resolved-revision"
    assert receipt.quantization == "Q4_K_M"
    assert receipt.torch_version == "2.9.0"
    assert receipt.torch_cuda_version == "13.0"
    assert receipt.cuda_available is True
    assert receipt.cuda_device == "Synthetic GPU"


def test_repository_state_returns_none_outside_git_repository(tmp_path) -> None:
    commit, dirty = provenance._repository_state(tmp_path)

    assert commit is None
    assert dirty is None


def test_repository_state_resolves_default_root_at_call_time(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    commit, dirty = provenance._repository_state()

    assert commit is None
    assert dirty is None

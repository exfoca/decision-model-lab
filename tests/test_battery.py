from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from decision_model_lab import battery
from decision_model_lab.battery import (
    BatteryError,
    ResidentDecisionRunner,
    non_redundant_regimes,
    run_resident_battery,
)
from decision_model_lab.candidates import Candidate
from decision_model_lab.schema import DecisionResult, EvaluationCase
from decision_model_lab.semantic import SEMANTIC_PROFILES


class FakeResidentRunner:
    def __init__(
        self,
        *,
        client: list[str] | None = None,
        semantic_profile: str = "baseline",
        evaluation_protocol: str = "rule-conditioned",
        drift_after_runs: int | None = None,
        counters: dict[str, int] | None = None,
    ) -> None:
        self.client = client
        self.semantic_profile = semantic_profile
        self.evaluation_protocol = evaluation_protocol
        self.drift_after_runs = drift_after_runs
        self.counters = counters if counters is not None else {"loads": 0, "benchmark_runs": 0}

    def prepare(self) -> None:
        if self.client is None:
            self.client = []
            self.counters["loads"] += 1

    def fork_for_experiment(
        self,
        *,
        semantic_profile: str,
        evaluation_protocol: str,
    ) -> ResidentDecisionRunner:
        self.prepare()
        return FakeResidentRunner(
            client=self.client,
            semantic_profile=semantic_profile,
            evaluation_protocol=evaluation_protocol,
            drift_after_runs=self.drift_after_runs,
            counters=self.counters,
        )

    def release(self) -> None:
        self.client = None

    def run(self, case: EvaluationCase) -> DecisionResult:
        self.prepare()
        assert self.client is not None
        self.client.append(case.id)
        if case.id != "__resident_state_probe__":
            self.counters["benchmark_runs"] += 1
        drift = (
            case.id == "__resident_state_probe__"
            and self.drift_after_runs is not None
            and self.counters["benchmark_runs"] >= self.drift_after_runs
        )
        probability = 0.2 if drift else 0.8
        return DecisionResult(
            case_id=case.id,
            label="no" if drift else "yes",
            probabilities={"yes": probability, "no": 1.0 - probability},
            latency_ms=1.0,
        )


def _candidate(
    *,
    name: str,
    semantic_profiles: tuple[str, ...],
    drift_after_runs: int | None = None,
    counters: dict[str, int] | None = None,
) -> Candidate:
    shared_counters = counters if counters is not None else {"loads": 0, "benchmark_runs": 0}

    def factory(**kwargs: Any) -> FakeResidentRunner:
        return FakeResidentRunner(
            semantic_profile=str(kwargs.get("semantic_profile", "baseline")),
            evaluation_protocol=str(kwargs.get("evaluation_protocol", "rule-conditioned")),
            drift_after_runs=drift_after_runs,
            counters=shared_counters,
        )

    return Candidate(
        name=name,
        model_id=f"synthetic/{name}",
        runner_factory=factory,
        runtime_distribution="synthetic",
        semantic_profiles=semantic_profiles,  # type: ignore[arg-type]
    )


def _write_dataset(path: Path, case_id: str) -> None:
    path.write_text(
        "{"
        f'"id":"{case_id}",'
        '"context":"Independent evidence.",'
        '"question":"Is the signal present?",'
        '"decision":{"type":"noul"},'
        '"expected":{"label":"yes"}'
        "}\n",
        encoding="utf-8",
    )


def test_non_redundant_regimes_follow_candidate_capabilities() -> None:
    jev = _candidate(name="jev", semantic_profiles=SEMANTIC_PROFILES)
    generic = _candidate(name="generic", semantic_profiles=("baseline", "optimized-v1"))

    assert [
        (regime.semantic_profile, regime.evaluation_protocol)
        for regime in non_redundant_regimes(jev)
    ] == [
        ("baseline", "rule-conditioned"),
        ("optimized-v1", "rule-conditioned"),
        ("native-criteria-v1", "rule-conditioned"),
        ("baseline", "closed-book"),
    ]
    assert [
        (regime.semantic_profile, regime.evaluation_protocol)
        for regime in non_redundant_regimes(generic)
    ] == [
        ("baseline", "rule-conditioned"),
        ("optimized-v1", "rule-conditioned"),
        ("baseline", "closed-book"),
    ]


def test_resident_battery_uses_selected_candidates_and_datasets_without_hardcoded_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jev_counters = {"loads": 0, "benchmark_runs": 0}
    generic_counters = {"loads": 0, "benchmark_runs": 0}
    candidates = {
        "jev": _candidate(
            name="jev",
            semantic_profiles=SEMANTIC_PROFILES,
            counters=jev_counters,
        ),
        "generic": _candidate(
            name="generic",
            semantic_profiles=("baseline", "optimized-v1"),
            counters=generic_counters,
        ),
    }
    monkeypatch.setattr(battery, "resolve_candidate", candidates.__getitem__)
    monkeypatch.setenv("DML_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    v3 = tmp_path / "benchmark-v3.jsonl"
    v3_pt = tmp_path / "benchmark-v3-pt-br.jsonl"
    _write_dataset(v3, "v3-en")
    _write_dataset(v3_pt, "v3-pt")

    artifacts = run_resident_battery(
        candidate_names=("jev", "generic"),
        datasets=(v3, v3_pt),
    )

    assert len(artifacts) == 14  # (4 Jev + 3 generic regimes) x 2 datasets
    assert jev_counters == {"loads": 1, "benchmark_runs": 8}
    assert generic_counters == {"loads": 1, "benchmark_runs": 6}
    assert len({artifact.path for artifact in artifacts}) == 14
    assert all(artifact.path.exists() for artifact in artifacts)
    assert all(artifact.report.execution_mode == "resident-battery" for artifact in artifacts)
    assert all(artifact.report.runtime_warmup is True for artifact in artifacts)
    assert {artifact.dataset for artifact in artifacts} == {v3, v3_pt}


def test_resident_battery_checks_state_isolation_after_each_run_before_publishing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate(
        name="generic",
        semantic_profiles=("baseline", "optimized-v1"),
        drift_after_runs=2,
    )
    monkeypatch.setattr(battery, "resolve_candidate", lambda name: candidate)
    monkeypatch.setenv("DML_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    dataset = tmp_path / "benchmark-v3.jsonl"
    _write_dataset(dataset, "case-1")

    with pytest.raises(BatteryError, match="state-isolation probe changed"):
        run_resident_battery(candidate_names=("generic",), datasets=(dataset,))

    assert not list((tmp_path / "artifacts").glob("*.json"))


def test_resident_battery_rejects_duplicate_inputs(tmp_path: Path) -> None:
    dataset = tmp_path / "benchmark-v3.jsonl"
    _write_dataset(dataset, "case-1")

    with pytest.raises(BatteryError, match="candidate names must be unique"):
        run_resident_battery(
            candidate_names=("jev", "jev"),
            datasets=(dataset,),
            verify_state_isolation=False,
        )

    with pytest.raises(BatteryError, match="dataset paths must be unique"):
        run_resident_battery(
            candidate_names=("jev",),
            datasets=(dataset, dataset),
            verify_state_isolation=False,
        )

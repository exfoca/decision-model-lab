from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from gc import collect as collect_garbage
from importlib import import_module
from math import isclose
from os import environ
from pathlib import Path
from typing import Protocol, runtime_checkable

from decision_model_lab.benchmark import (
    BenchmarkReport,
    DecisionRunner,
    run_benchmark,
    write_report,
)
from decision_model_lab.candidates import Candidate, CandidateError, resolve_candidate
from decision_model_lab.dataset import load_jsonl
from decision_model_lab.provenance import collect_run_provenance
from decision_model_lab.schema import DecisionResult, DecisionSpec, EvaluationCase, ExpectedDecision
from decision_model_lab.semantic import (
    DEFAULT_EVALUATION_PROTOCOL,
    DEFAULT_SEMANTIC_PROFILE,
    EvaluationProtocol,
    SemanticProfile,
)


@dataclass(frozen=True)
class BatteryRegime:
    semantic_profile: SemanticProfile
    evaluation_protocol: EvaluationProtocol


@dataclass(frozen=True)
class BatteryArtifact:
    candidate: str
    dataset: Path
    regime: BatteryRegime
    path: Path
    report: BenchmarkReport


class BatteryError(RuntimeError):
    """Raised when a resident benchmark battery cannot preserve its execution contract."""


@runtime_checkable
class ResidentDecisionRunner(DecisionRunner, Protocol):
    """Decision runner that can share only its loaded runtime across fresh experiments."""

    def prepare(self) -> None: ...

    def fork_for_experiment(
        self,
        *,
        semantic_profile: str,
        evaluation_protocol: str,
    ) -> ResidentDecisionRunner: ...

    def release(self) -> None: ...


def non_redundant_regimes(candidate: Candidate) -> tuple[BatteryRegime, ...]:
    """Return every semantically distinct benchmark regime supported by a candidate."""
    if not candidate.semantic_profiles:
        raise CandidateError(f"candidate {candidate.name!r} declares no semantic profiles")

    rule_conditioned = tuple(
        BatteryRegime(profile, DEFAULT_EVALUATION_PROTOCOL)
        for profile in candidate.semantic_profiles
    )
    closed_book_profile = (
        DEFAULT_SEMANTIC_PROFILE
        if DEFAULT_SEMANTIC_PROFILE in candidate.semantic_profiles
        else candidate.semantic_profiles[0]
    )
    return (*rule_conditioned, BatteryRegime(closed_book_profile, "closed-book"))


def _state_probe_case() -> EvaluationCase:
    return EvaluationCase(
        id="__resident_state_probe__",
        context=(
            "Synthetic isolation probe. The current signal is present and no prior observation "
            "is relevant to this decision."
        ),
        question="Is the current signal present?",
        decision=DecisionSpec(type="noul"),
        expected=ExpectedDecision(label="yes"),
    )


def _same_probe_result(left: DecisionResult, right: DecisionResult) -> bool:
    if left.label != right.label or left.probabilities.keys() != right.probabilities.keys():
        return False
    return all(
        isclose(left.probabilities[label], right.probabilities[label], rel_tol=1e-6, abs_tol=1e-7)
        for label in left.probabilities
    )


def _artifact_path(
    *,
    dataset: Path,
    candidate_model_id: str,
    runner: ResidentDecisionRunner,
    regime: BatteryRegime,
) -> Path:
    model_name = candidate_model_id.rsplit("/", maxsplit=1)[-1]
    quantization = getattr(runner, "quantization", None)
    quantization_suffix = f"-{quantization}" if quantization else ""
    profile_suffix = (
        ""
        if regime.semantic_profile == DEFAULT_SEMANTIC_PROFILE
        else f"-{regime.semantic_profile}"
    )
    protocol_suffix = (
        ""
        if regime.evaluation_protocol == DEFAULT_EVALUATION_PROTOCOL
        else f"-{regime.evaluation_protocol}"
    )
    return Path(environ.get("DML_ARTIFACTS_DIR", "artifacts")) / (
        f"{dataset.stem}-{model_name}{quantization_suffix}"
        f"{profile_suffix}{protocol_suffix}-resident-battery.json"
    )


def _release_cuda_cache() -> None:
    collect_garbage()
    try:
        torch = import_module("torch")
    except (ImportError, OSError):
        return

    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return
    try:
        if cuda.is_available():
            cuda.empty_cache()
    except (AttributeError, RuntimeError):
        return


def _validate_unique_inputs(candidate_names: tuple[str, ...], datasets: tuple[Path, ...]) -> None:
    if not candidate_names:
        raise BatteryError("benchmark battery requires at least one candidate")
    if not datasets:
        raise BatteryError("benchmark battery requires at least one dataset")
    if len(set(candidate_names)) != len(candidate_names):
        raise BatteryError("benchmark battery candidate names must be unique")
    if len(set(datasets)) != len(datasets):
        raise BatteryError("benchmark battery dataset paths must be unique")


def run_resident_battery(
    *,
    candidate_names: tuple[str, ...],
    datasets: tuple[Path, ...],
    verify_state_isolation: bool = True,
    progress: Callable[[str], None] | None = None,
) -> list[BatteryArtifact]:
    """Run all non-redundant regimes while loading each selected candidate only once.

    Every dataset/regime pair gets a fresh runner and freshly loaded EvaluationCase objects. Only
    the opaque provider runtime/client that owns the loaded model remains resident between runs.
    No earlier case or normalized result is supplied to a later decision.
    """
    _validate_unique_inputs(candidate_names, datasets)
    resolved_candidates = tuple(
        (candidate_name, resolve_candidate(candidate_name)) for candidate_name in candidate_names
    )
    for dataset in datasets:
        load_jsonl(dataset)

    artifacts: list[BatteryArtifact] = []

    for candidate_name, candidate in resolved_candidates:
        regimes = non_redundant_regimes(candidate)
        if progress is not None:
            progress(
                f"loading candidate {candidate_name} "
                f"({len(regimes)} non-redundant regime(s) per dataset)"
            )

        owner_profile = (
            DEFAULT_SEMANTIC_PROFILE
            if DEFAULT_SEMANTIC_PROFILE in candidate.semantic_profiles
            else candidate.semantic_profiles[0]
        )
        owner = candidate.create_runner(
            semantic_profile=owner_profile,
            evaluation_protocol=DEFAULT_EVALUATION_PROTOCOL,
        )
        if not isinstance(owner, ResidentDecisionRunner):
            raise CandidateError(
                f"candidate {candidate_name!r} does not support resident benchmark execution"
            )

        candidate_artifacts: list[BatteryArtifact] = []

        try:
            owner.prepare()
            reference_probe = owner.run(_state_probe_case()) if verify_state_isolation else None

            for dataset in datasets:
                for regime in regimes:
                    if progress is not None:
                        progress(
                            f"running {candidate_name} | {dataset} | "
                            f"{regime.semantic_profile} | {regime.evaluation_protocol}"
                        )

                    cases = load_jsonl(dataset)
                    runner = owner.fork_for_experiment(
                        semantic_profile=regime.semantic_profile,
                        evaluation_protocol=regime.evaluation_protocol,
                    )
                    report = run_benchmark(
                        cases,
                        runner,
                        dataset=dataset,
                        candidate=candidate_name,
                        model_id=candidate.model_id,
                        semantic_profile=regime.semantic_profile,
                        evaluation_protocol=regime.evaluation_protocol,
                    )
                    report = report.model_copy(
                        update={
                            "execution_mode": "resident-battery",
                            "runtime_warmup": verify_state_isolation,
                            "provenance": collect_run_provenance(
                                runner=runner,
                                results=report.results,
                                runtime_distribution=candidate.runtime_distribution,
                            ),
                        }
                    )
                    path = _artifact_path(
                        dataset=dataset,
                        candidate_model_id=candidate.model_id,
                        runner=runner,
                        regime=regime,
                    )
                    candidate_artifacts.append(
                        BatteryArtifact(
                            candidate=candidate_name,
                            dataset=dataset,
                            regime=regime,
                            path=path,
                            report=report,
                        )
                    )
                    del runner

                    if verify_state_isolation:
                        assert reference_probe is not None
                        current_probe = owner.run(_state_probe_case())
                        if not _same_probe_result(reference_probe, current_probe):
                            raise BatteryError(
                                "state-isolation probe changed after resident benchmark run for "
                                f"{candidate_name!r} ({dataset}, {regime.semantic_profile}, "
                                f"{regime.evaluation_protocol}); no artifacts for this candidate "
                                "were published"
                            )

            for artifact in candidate_artifacts:
                write_report(artifact.report, artifact.path)
            artifacts.extend(candidate_artifacts)
            if progress is not None and verify_state_isolation:
                progress(f"state-isolation probes passed for {candidate_name}")
        finally:
            owner.release()
            del owner
            _release_cuda_cache()

    return artifacts

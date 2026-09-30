from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from decision_model_lab.metrics import (
    calibration_metrics,
    classification_metrics,
    selection_probability,
    selective_metrics,
)
from decision_model_lab.provenance import RunProvenance
from decision_model_lab.schema import DecisionResult, EvaluationCase
from decision_model_lab.semantic import (
    DEFAULT_EVALUATION_PROTOCOL,
    DEFAULT_SEMANTIC_PROFILE,
    EvaluationProtocol,
    SemanticProfile,
)


class DecisionRunner(Protocol):
    def run(self, case: EvaluationCase) -> DecisionResult: ...


class LatencySummary(BaseModel):
    total_ms: float = Field(ge=0.0)
    mean_ms: float = Field(ge=0.0)
    min_ms: float = Field(ge=0.0)
    max_ms: float = Field(ge=0.0)
    cold_start_ms: float = Field(ge=0.0)
    steady_state_mean_ms: float | None = Field(default=None, ge=0.0)


class SegmentSummary(BaseModel):
    case_count: int = Field(gt=0)
    correct: int = Field(ge=0)
    accuracy: float = Field(ge=0.0, le=1.0)


class CaseDiagnostic(BaseModel):
    case_id: str
    expected: str
    predicted: str
    correct: bool
    selection_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    provider_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    probabilities: dict[str, float]
    latency_ms: float = Field(ge=0.0)


def case_diagnostics(
    cases: list[EvaluationCase], results: list[DecisionResult]
) -> list[CaseDiagnostic]:
    results_by_id = {result.case_id: result for result in results}
    diagnostics: list[CaseDiagnostic] = []

    for case in cases:
        expected = case.expected.label
        result = results_by_id.get(case.id)
        if expected is None or result is None:
            continue

        confidence = selection_probability(result)

        diagnostics.append(
            CaseDiagnostic(
                case_id=case.id,
                expected=expected,
                predicted=result.label,
                correct=result.label == expected,
                selection_probability=confidence,
                provider_confidence=result.confidence,
                probabilities=result.probabilities,
                latency_ms=result.latency_ms,
            )
        )

    return diagnostics


class CaseManifestEntry(BaseModel):
    case_id: str
    expected_label: str | None
    decision_type: str
    tags: list[str]
    metadata: dict[str, object]


class BenchmarkReport(BaseModel):
    schema_version: int = 1
    dataset: str
    dataset_sha256: str | None = None
    candidate: str
    model_id: str
    semantic_profile: SemanticProfile
    evaluation_protocol: EvaluationProtocol
    execution_mode: str = "standalone"
    runtime_warmup: bool = False
    case_count: int = Field(gt=0)
    classification: dict[str, int | float]
    calibration: dict[str, int | float]
    selective: dict[str, object] = Field(default_factory=dict)
    by_decision_type: dict[str, SegmentSummary]
    by_tag: dict[str, SegmentSummary]
    latency: LatencySummary
    provenance: RunProvenance | None = None
    case_manifest: list[CaseManifestEntry] = Field(default_factory=list)
    results: list[DecisionResult]


def _segment_summary(
    cases: list[EvaluationCase], results: list[DecisionResult]
) -> SegmentSummary:
    results_by_id = {result.case_id: result for result in results}
    evaluated = [
        case
        for case in cases
        if case.expected.label is not None and case.id in results_by_id
    ]
    correct = sum(
        results_by_id[case.id].label == case.expected.label for case in evaluated
    )
    return SegmentSummary(
        case_count=len(evaluated),
        correct=correct,
        accuracy=correct / len(evaluated),
    )


def _segmented_metrics(
    cases: list[EvaluationCase], results: list[DecisionResult]
) -> tuple[dict[str, SegmentSummary], dict[str, SegmentSummary]]:
    decision_types = sorted({case.decision.type for case in cases})
    tags = sorted({tag for case in cases for tag in case.tags})

    by_decision_type: dict[str, SegmentSummary] = {
        decision_type: _segment_summary(
            [case for case in cases if case.decision.type == decision_type], results
        )
        for decision_type in decision_types
    }
    by_tag = {
        tag: _segment_summary([case for case in cases if tag in case.tags], results)
        for tag in tags
    }
    return by_decision_type, by_tag


def _dataset_sha256(cases: list[EvaluationCase]) -> str:
    digest = hashlib.sha256()
    for case in cases:
        payload = json.dumps(
            case.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _case_manifest(cases: list[EvaluationCase]) -> list[CaseManifestEntry]:
    return [
        CaseManifestEntry(
            case_id=case.id,
            expected_label=case.expected.label,
            decision_type=case.decision.type,
            tags=list(case.tags),
            metadata={str(key): value for key, value in case.metadata.items()},
        )
        for case in cases
    ]


def run_benchmark(
    cases: list[EvaluationCase],
    runner: DecisionRunner,
    *,
    dataset: Path,
    candidate: str,
    model_id: str,
    semantic_profile: SemanticProfile = DEFAULT_SEMANTIC_PROFILE,
    evaluation_protocol: EvaluationProtocol = DEFAULT_EVALUATION_PROTOCOL,
) -> BenchmarkReport:
    if not cases:
        raise ValueError("benchmark requires at least one evaluation case")

    results = [runner.run(case) for case in cases]
    classification = classification_metrics(cases, results)
    calibration = calibration_metrics(cases, results)
    selective = selective_metrics(cases, results)
    latencies = [result.latency_ms for result in results]
    by_decision_type, by_tag = _segmented_metrics(cases, results)
    steady_state = latencies[1:]

    return BenchmarkReport(
        schema_version=2,
        dataset=str(dataset),
        dataset_sha256=_dataset_sha256(cases),
        candidate=candidate,
        model_id=model_id,
        semantic_profile=semantic_profile,
        evaluation_protocol=evaluation_protocol,
        case_count=len(cases),
        classification=asdict(classification),
        calibration=asdict(calibration),
        selective=asdict(selective),
        by_decision_type=by_decision_type,
        by_tag=by_tag,
        latency=LatencySummary(
            total_ms=sum(latencies),
            mean_ms=sum(latencies) / len(latencies),
            min_ms=min(latencies),
            max_ms=max(latencies),
            cold_start_ms=latencies[0],
            steady_state_mean_ms=(
                sum(steady_state) / len(steady_state) if steady_state else None
            ),
        ),
        case_manifest=_case_manifest(cases),
        results=results,
    )


def write_report(report: BenchmarkReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

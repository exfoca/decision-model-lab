from pathlib import Path

from decision_model_lab.benchmark import (
    BenchmarkReport,
    case_diagnostics,
    run_benchmark,
    write_report,
)
from decision_model_lab.schema import DecisionResult, EvaluationCase


class FakeRunner:
    def run(self, case: EvaluationCase) -> DecisionResult:
        label = case.expected.label or "unknown"
        return DecisionResult(
            case_id=case.id,
            label=label,
            probabilities={label: 1.0},
            latency_ms=2.0,
        )


def _case(
    case_id: str,
    label: str,
    *,
    decision_type: str = "noul",
    tags: list[str] | None = None,
) -> EvaluationCase:
    decision: dict[str, object] = {"type": decision_type}
    if decision_type == "choice":
        decision["options"] = ["yes", "no"]
    return EvaluationCase(
        id=case_id,
        context="synthetic context",
        question="synthetic question",
        decision=decision,
        expected={"label": label},
        tags=tags or [],
    )


def test_run_benchmark_executes_all_cases_and_computes_metrics() -> None:
    cases = [_case("a", "yes"), _case("b", "no")]

    report = run_benchmark(
        cases,
        FakeRunner(),
        dataset=Path("datasets/test.jsonl"),
        candidate="fake",
        model_id="fake/model",
    )

    assert report.schema_version == 2
    assert len(report.dataset_sha256) == 64
    assert report.case_manifest[0].case_id == "a"
    assert report.case_manifest[0].expected_label == "yes"
    assert report.semantic_profile == "baseline"
    assert report.evaluation_protocol == "rule-conditioned"
    assert report.case_count == 2
    assert report.classification["accuracy"] == 1.0
    assert report.classification["macro_f1"] == 1.0
    assert report.calibration["brier_score"] == 0.0
    assert report.calibration["log_loss"] == 0.0
    assert report.selective["by_error_budget"]["1%"]["coverage"] == 1.0
    assert report.latency.total_ms == 4.0
    assert report.latency.cold_start_ms == 2.0
    assert report.latency.steady_state_mean_ms == 2.0
    assert report.by_decision_type["noul"].accuracy == 1.0
    assert [result.case_id for result in report.results] == ["a", "b"]


def test_run_benchmark_records_explicit_semantic_profile() -> None:
    report = run_benchmark(
        [_case("a", "yes")],
        FakeRunner(),
        dataset=Path("datasets/test.jsonl"),
        candidate="fake",
        model_id="fake/model",
        semantic_profile="optimized-v1",
    )

    assert report.semantic_profile == "optimized-v1"


def test_run_benchmark_records_explicit_evaluation_protocol() -> None:
    report = run_benchmark(
        [_case("a", "yes")],
        FakeRunner(),
        dataset=Path("datasets/test.jsonl"),
        candidate="fake",
        model_id="fake/model",
        evaluation_protocol="closed-book",
    )

    assert report.evaluation_protocol == "closed-book"


def test_write_report_persists_normalized_results(tmp_path: Path) -> None:
    report = run_benchmark(
        [_case("a", "yes")],
        FakeRunner(),
        dataset=Path("datasets/test.jsonl"),
        candidate="fake",
        model_id="fake/model",
    )
    output = tmp_path / "nested" / "report.json"

    write_report(report, output)

    payload = output.read_text(encoding="utf-8")
    assert '"case_id": "a"' in payload
    assert '"candidate": "fake"' in payload
    assert '"schema_version": 2' in payload
    assert '"dataset_sha256":' in payload
    assert '"case_manifest":' in payload
    assert '"semantic_profile": "baseline"' in payload
    assert '"evaluation_protocol": "rule-conditioned"' in payload


def test_benchmark_report_accepts_legacy_payload_without_manifest() -> None:
    report = run_benchmark(
        [_case("a", "yes")],
        FakeRunner(),
        dataset=Path("datasets/test.jsonl"),
        candidate="fake",
        model_id="fake/model",
    )
    payload = report.model_dump(mode="json")
    payload.pop("schema_version")
    payload.pop("dataset_sha256")
    payload.pop("case_manifest")

    legacy = BenchmarkReport.model_validate(payload)

    assert legacy.schema_version == 1
    assert legacy.dataset_sha256 is None
    assert legacy.case_manifest == []
    assert legacy.provenance is None


def test_case_diagnostics_exposes_correctness_confidence_and_latency() -> None:
    cases = [_case("a", "yes"), _case("b", "no")]
    results = [
        DecisionResult(
            case_id="a",
            label="yes",
            probabilities={"yes": 0.8, "no": 0.2},
            latency_ms=3.0,
        ),
        DecisionResult(
            case_id="b",
            label="yes",
            probabilities={"yes": 0.7, "no": 0.3},
            confidence=0.65,
            latency_ms=4.0,
        ),
    ]

    diagnostics = case_diagnostics(cases, results)

    assert diagnostics[0].correct is True
    assert diagnostics[0].selection_probability == 0.8
    assert diagnostics[0].provider_confidence is None
    assert diagnostics[1].correct is False
    assert diagnostics[1].expected == "no"
    assert diagnostics[1].predicted == "yes"
    assert diagnostics[1].selection_probability == 0.7
    assert diagnostics[1].provider_confidence == 0.65
    assert diagnostics[1].probabilities == {"yes": 0.7, "no": 0.3}
    assert diagnostics[1].latency_ms == 4.0


def test_run_benchmark_segments_accuracy_by_decision_type_and_tag() -> None:
    cases = [
        _case("a", "yes", tags=["direct-evidence"]),
        _case("b", "no", decision_type="choice", tags=["multiclass"]),
        _case("c", "yes", decision_type="choice", tags=["multiclass", "direct-evidence"]),
    ]

    class SegmentedRunner:
        def run(self, case: EvaluationCase) -> DecisionResult:
            predicted = "yes" if case.id != "b" else "yes"
            return DecisionResult(
                case_id=case.id,
                label=predicted,
                probabilities={"yes": 0.75, "no": 0.25},
                latency_ms={"a": 10.0, "b": 4.0, "c": 6.0}[case.id],
            )

    report = run_benchmark(
        cases,
        SegmentedRunner(),
        dataset=Path("datasets/test.jsonl"),
        candidate="fake",
        model_id="fake/model",
    )

    assert report.by_decision_type["noul"].model_dump() == {
        "case_count": 1, "correct": 1, "accuracy": 1.0
    }
    assert report.by_decision_type["choice"].model_dump() == {
        "case_count": 2, "correct": 1, "accuracy": 0.5
    }
    assert report.by_tag["multiclass"].accuracy == 0.5
    assert report.by_tag["direct-evidence"].accuracy == 1.0
    assert report.latency.cold_start_ms == 10.0
    assert report.latency.steady_state_mean_ms == 5.0


def test_single_case_has_no_steady_state_latency() -> None:
    report = run_benchmark(
        [_case("a", "yes")],
        FakeRunner(),
        dataset=Path("datasets/test.jsonl"),
        candidate="fake",
        model_id="fake/model",
    )

    assert report.latency.cold_start_ms == 2.0
    assert report.latency.steady_state_mean_ms is None

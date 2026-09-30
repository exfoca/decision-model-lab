import math

import pytest

from decision_model_lab.metrics import calibration_metrics, classification_metrics
from decision_model_lab.schema import DecisionResult, EvaluationCase


def _case(case_id: str, label: str) -> EvaluationCase:
    return EvaluationCase(
        id=case_id,
        context="synthetic context",
        question="synthetic question",
        decision={"type": "noul"},
        expected={"label": label},
    )


def _result(case_id: str, label: str, probabilities: dict[str, float]) -> DecisionResult:
    return DecisionResult(
        case_id=case_id,
        label=label,
        probabilities=probabilities,
        latency_ms=1.0,
    )


def test_classification_metrics_measure_accuracy_and_macro_f1() -> None:
    cases = [_case("a", "yes"), _case("b", "no"), _case("c", "yes")]
    results = [
        _result("a", "yes", {"yes": 0.9, "no": 0.1}),
        _result("b", "yes", {"yes": 0.6, "no": 0.4}),
        _result("c", "yes", {"yes": 0.8, "no": 0.2}),
    ]

    metrics = classification_metrics(cases, results)

    assert metrics.evaluated == 3
    assert metrics.accuracy == pytest.approx(2 / 3)
    assert metrics.macro_f1 == pytest.approx(0.4)


def test_calibration_metrics_are_zero_for_perfect_predictions() -> None:
    cases = [_case("a", "yes"), _case("b", "no")]
    results = [
        _result("a", "yes", {"yes": 1.0, "no": 0.0}),
        _result("b", "no", {"yes": 0.0, "no": 1.0}),
    ]

    metrics = calibration_metrics(cases, results)

    assert metrics.evaluated == 2
    assert metrics.brier_score == pytest.approx(0.0)
    assert metrics.expected_calibration_error == pytest.approx(0.0)


def test_calibration_metrics_require_probability_distributions() -> None:
    cases = [_case("a", "yes")]
    results = [DecisionResult(case_id="a", label="yes", latency_ms=1.0)]

    with pytest.raises(ValueError, match="probability distributions"):
        calibration_metrics(cases, results)


def test_calibration_metrics_include_log_loss() -> None:
    cases = [_case("a", "yes"), _case("b", "no")]
    results = [
        _result("a", "yes", {"yes": 0.8, "no": 0.2}),
        _result("b", "no", {"yes": 0.25, "no": 0.75}),
    ]

    metrics = calibration_metrics(cases, results)

    expected = (-math.log(0.8) - math.log(0.75)) / 2
    assert metrics.log_loss == pytest.approx(expected)


def test_selective_metrics_use_probability_not_provider_confidence_and_preserve_ties() -> None:
    from decision_model_lab.metrics import selective_metrics

    cases = [
        _case("a", "yes"),
        _case("b", "yes"),
        _case("c", "yes"),
        _case("d", "yes"),
    ]
    results = [
        DecisionResult(
            case_id="a",
            label="yes",
            probabilities={"yes": 0.9, "no": 0.1},
            confidence=0.1,
            latency_ms=1.0,
        ),
        DecisionResult(
            case_id="b",
            label="yes",
            probabilities={"yes": 0.9, "no": 0.1},
            confidence=0.1,
            latency_ms=1.0,
        ),
        DecisionResult(
            case_id="c",
            label="no",
            probabilities={"yes": 0.4, "no": 0.6},
            confidence=0.99,
            latency_ms=1.0,
        ),
        DecisionResult(
            case_id="d",
            label="yes",
            probabilities={"yes": 0.6, "no": 0.4},
            confidence=0.01,
            latency_ms=1.0,
        ),
    ]

    metrics = selective_metrics(cases, results, error_budgets=(0.0, 0.25))

    strict = metrics.by_error_budget["0%"]
    assert strict.coverage == pytest.approx(0.5)
    assert strict.threshold == pytest.approx(0.9)
    assert strict.automated == 2
    assert strict.errors == 0

    relaxed = metrics.by_error_budget["25%"]
    assert relaxed.coverage == pytest.approx(1.0)
    assert relaxed.threshold == pytest.approx(0.6)
    assert relaxed.errors == 1
    assert relaxed.empirical_error_rate == pytest.approx(0.25)

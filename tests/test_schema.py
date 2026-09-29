import pytest
from pydantic import ValidationError

from decision_model_lab.schema import DecisionResult, DecisionSpec, EvaluationCase, ExpectedDecision


def test_decision_result_preserves_normalized_and_raw_output() -> None:
    result = DecisionResult(
        case_id="case-1",
        label="yes",
        probabilities={"yes": 0.8, "no": 0.2},
        confidence=0.7,
        raw_output={"vendor": {"answer": True}},
        latency_ms=12.5,
        metadata={"model": "synthetic"},
    )

    assert result.probabilities["yes"] == 0.8
    assert result.raw_output == {"vendor": {"answer": True}}


def test_decision_result_rejects_invalid_probability_distribution() -> None:
    with pytest.raises(ValidationError, match="probabilities must sum to 1"):
        DecisionResult(
            case_id="case-1",
            label="yes",
            probabilities={"yes": 0.8, "no": 0.3},
            latency_ms=1.0,
        )


def test_decision_spec_enforces_typed_question_contract() -> None:
    assert DecisionSpec(type="noul").options == []
    assert DecisionSpec(type="choice", options=["yes", "no"]).options == ["yes", "no"]

    with pytest.raises(ValueError, match="at least two options"):
        DecisionSpec(type="choice", options=["only"])

    with pytest.raises(ValueError, match="must not define options"):
        DecisionSpec(type="noul", options=["yes", "no"])


def test_evaluation_case_preserves_model_visible_definitions() -> None:
    case = EvaluationCase(
        id="case-1",
        context="Evidence.",
        question="Decision?",
        definitions={"thresholds": {"normal": "<= 10"}},
        decision=DecisionSpec(type="noul"),
        expected=ExpectedDecision(label="yes"),
    )

    assert case.definitions == {"thresholds": {"normal": "<= 10"}}

from __future__ import annotations

from typing import Any

import pytest

from decision_model_lab.laya_runner import DEFAULT_MODEL_ID, LayaRunner
from decision_model_lab.schema import DecisionSpec, EvaluationCase, ExpectedDecision


class FakeClient:
    def __init__(self, output: dict[str, Any]) -> None:
        self.output = output
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def predict(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((state, questions))
        return self.output


def make_case(decision: DecisionSpec) -> EvaluationCase:
    return EvaluationCase(
        id="case-1",
        context="Evidence supplied to the model.",
        question="What should the decision be?",
        decision=decision,
        expected=ExpectedDecision(label="yes"),
    )


def test_noul_is_normalized_to_binary_distribution() -> None:
    client = FakeClient(
        {
            "model": "laya-rl-agent",
            "answers": {"decision": {"type": "noul", "noul": 0.8, "confidence": 0.8}},
            "latency_ms": 9.5,
            "device": "cuda",
        }
    )

    client.revision = "laya-snapshot"
    result = LayaRunner(client=client).run(make_case(DecisionSpec(type="noul")))

    assert result.label == "yes"
    assert result.probabilities == pytest.approx({"no": 0.2, "yes": 0.8})
    assert result.confidence == 0.8
    assert result.metadata["provider_latency_ms"] == 9.5
    assert result.metadata["device"] == "cuda"
    assert result.metadata["model_id"] == DEFAULT_MODEL_ID
    assert result.metadata["model_revision"] == "laya-snapshot"
    assert client.calls[0][1]["decision"] == {
        "type": "noul",
        "instructions": "What should the decision be?",
    }


def test_choice_uses_model_independent_options_as_laya_criteria() -> None:
    output = {
        "answers": {
            "decision": {
                "type": "choice",
                "choice": "degraded",
                "confidence": 0.7,
                "probabilities": {"normal": 0.1, "degraded": 0.7, "unavailable": 0.2},
            }
        }
    }
    client = FakeClient(output)
    case = make_case(DecisionSpec(type="choice", options=["normal", "degraded", "unavailable"]))

    result = LayaRunner(client=client).run(case)

    assert result.label == "degraded"
    assert result.confidence == 0.7
    assert result.probabilities == pytest.approx(
        {"normal": 0.1, "degraded": 0.7, "unavailable": 0.2}
    )
    assert result.raw_output is output
    assert client.calls[0][1]["decision"]["criteria"] == {
        "normal": "normal",
        "degraded": "degraded",
        "unavailable": "unavailable",
    }


def test_score_maps_laya_probability_indices_back_to_canonical_labels() -> None:
    client = FakeClient(
        {
            "answers": {
                "decision": {
                    "type": "score",
                    "score": 1.6,
                    "confidence": 0.6,
                    "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7},
                }
            }
        }
    )
    case = make_case(DecisionSpec(type="score", options=["low", "medium", "high"]))

    result = LayaRunner(client=client).run(case)

    assert result.label == "high"
    assert result.probabilities == pytest.approx({"low": 0.1, "medium": 0.2, "high": 0.7})
    assert result.metadata["score"] == 1.6
    assert client.calls[0][1]["decision"]["criteria"] == ["low", "medium", "high"]


def test_definitions_are_included_in_model_visible_state() -> None:
    client = FakeClient(
        {"answers": {"decision": {"type": "noul", "noul": 0.8, "confidence": 0.8}}}
    )
    case = make_case(DecisionSpec(type="noul"))
    case.definitions = {"health": {"healthy": "all dependency checks pass"}}

    LayaRunner(client=client).run(case)

    assert client.calls[0][0] == (
        'Evidence supplied to the model.\n\n[definitions]\n'
        '{"health":{"healthy":"all dependency checks pass"}}'
    )


def test_optimized_v1_semantic_profile_is_used_and_recorded() -> None:
    client = FakeClient(
        {"answers": {"decision": {"type": "noul", "noul": 0.8, "confidence": 0.8}}}
    )
    case = make_case(DecisionSpec(type="noul"))
    case.definitions = {"health": {"healthy": "all dependency checks pass"}}

    result = LayaRunner(client=client, semantic_profile="optimized-v1").run(case)

    assert client.calls[0][0] == (
        "Evidence supplied to the model.\n\n[decision_rules]\n"
        "health:\n"
        "  healthy: all dependency checks pass"
    )
    assert result.metadata["semantic_profile"] == "optimized-v1"


def test_closed_book_protocol_hides_definitions_and_is_recorded() -> None:
    client = FakeClient({"answers": {"decision": {"type": "noul", "noul": 0.8}}})
    case = make_case(DecisionSpec(type="noul"))
    case.definitions = {"health": {"healthy": "all dependency checks pass"}}

    result = LayaRunner(client=client, evaluation_protocol="closed-book").run(case)

    assert client.calls[0][0] == "Evidence supplied to the model."
    assert result.metadata["evaluation_protocol"] == "closed-book"

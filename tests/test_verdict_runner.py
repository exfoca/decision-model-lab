from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from decision_model_lab.schema import DecisionSpec, EvaluationCase, ExpectedDecision
from decision_model_lab.verdict_runner import (
    DEFAULT_MODEL_ID,
    NOUL_SEMANTICS,
    UPSTREAM_ENGINE_REVISION,
    VerdictRunner,
)


class FakeRuntime:
    @staticmethod
    def Option(**kwargs: Any) -> dict[str, Any]:
        return {"kind": "option", **kwargs}

    @staticmethod
    def Level(**kwargs: Any) -> dict[str, Any]:
        return {"kind": "level", **kwargs}

    @staticmethod
    def Choice(**kwargs: Any) -> dict[str, Any]:
        return {"kind": "choice", **kwargs}

    @staticmethod
    def Score(**kwargs: Any) -> dict[str, Any]:
        return {"kind": "score", **kwargs}

    @staticmethod
    def Noul(**kwargs: Any) -> dict[str, Any]:
        return {"kind": "noul", **kwargs}


class FakeBatch:
    def __init__(self, result: Any) -> None:
        self.results = (result,)
        self.forward_call_count = 1
        self.total_latency_ms = 7.5

    def model_dump(self, *, mode: str) -> dict[str, Any]:
        assert mode == "json"
        return {"fake": "batch"}


class FakeClient:
    def __init__(self, result: Any) -> None:
        self.output = FakeBatch(result)
        self.calls: list[tuple[str, list[Any]]] = []

    def evaluate(self, *, context: str, queries: list[Any]) -> FakeBatch:
        self.calls.append((context, queries))
        return self.output


def make_case(decision: DecisionSpec) -> EvaluationCase:
    return EvaluationCase(
        id="case-1",
        context="Evidence supplied to the model.",
        question="What should the decision be?",
        decision=decision,
        expected=ExpectedDecision(label="yes"),
    )


def test_choice_reserves_canonical_insufficient_evidence_for_upstream_abstention() -> None:
    result = SimpleNamespace(
        selected_id="__insufficient_evidence__",
        probabilities={"yes": 0.15, "no": 0.25, "__insufficient_evidence__": 0.60},
        calibration_status="calibrated_for_scope",
    )
    client = FakeClient(result)
    case = make_case(
        DecisionSpec(type="choice", options=["yes", "no", "insufficient_evidence"])
    )

    normalized = VerdictRunner(client=client, runtime=FakeRuntime()).run(case)

    assert normalized.label == "insufficient_evidence"
    assert normalized.probabilities == pytest.approx(
        {"yes": 0.15, "no": 0.25, "insufficient_evidence": 0.60}
    )
    assert normalized.confidence == pytest.approx(0.60)
    assert normalized.metadata["calibration_status"] == "calibrated_for_scope"
    assert normalized.metadata["provider_latency_ms"] == pytest.approx(7.5)
    assert normalized.metadata["upstream_engine_revision"] == UPSTREAM_ENGINE_REVISION
    assert normalized.raw_output == {"fake": "batch"}

    _, queries = client.calls[0]
    assert queries == [
        {
            "kind": "choice",
            "id": "decision",
            "question": "What should the decision be?",
            "options": [
                {"kind": "option", "id": "yes", "description": "yes"},
                {"kind": "option", "id": "no", "description": "no"},
            ],
        }
    ]


def test_choice_accepts_documented_selected_option_id_alias() -> None:
    result = SimpleNamespace(
        selected_option_id="yes",
        probabilities={"yes": 0.80, "no": 0.15, "__insufficient_evidence__": 0.05},
        calibration_status="calibrated_for_scope",
    )
    normalized = VerdictRunner(client=FakeClient(result), runtime=FakeRuntime()).run(
        make_case(DecisionSpec(type="choice", options=["yes", "no"]))
    )

    assert normalized.label == "yes"
    assert normalized.confidence == pytest.approx(0.80)


def test_verdict_runtime_manifest_matches_runner_revision() -> None:
    manifest = Path("requirements/verdict-runtime.txt").read_text(encoding="utf-8")
    assert f"Verdict-open-jev.git@{UPSTREAM_ENGINE_REVISION}" in manifest


def test_noul_maps_true_false_and_abstention_to_canonical_labels() -> None:
    result = SimpleNamespace(
        selected_outcome="true",
        probabilities={"true": 0.70, "false": 0.20, "__insufficient_evidence__": 0.10},
        p_true_given_sufficient_evidence=0.7777777778,
        p_insufficient_evidence=0.10,
        calibration_status="calibrated_for_scope",
    )
    client = FakeClient(result)

    normalized = VerdictRunner(client=client, runtime=FakeRuntime()).run(
        make_case(DecisionSpec(type="noul"))
    )

    assert normalized.label == "yes"
    assert normalized.probabilities == pytest.approx(
        {"yes": 0.70, "no": 0.20, "insufficient_evidence": 0.10}
    )
    assert normalized.metadata["p_true_given_sufficient_evidence"] == pytest.approx(
        0.7777777778
    )
    assert normalized.metadata["p_insufficient_evidence"] == pytest.approx(0.10)

    _, queries = client.calls[0]
    assert queries == [
        {
            "kind": "noul",
            "id": "decision",
            "proposition": "What should the decision be?",
            "semantics": NOUL_SEMANTICS,
        }
    ]


def test_score_uses_strictly_increasing_numeric_level_values() -> None:
    result = SimpleNamespace(
        selected_level_id="high",
        probabilities={
            "low": 0.10,
            "medium": 0.20,
            "high": 0.65,
            "__insufficient_evidence__": 0.05,
        },
        selected_value=2.0,
        expected_score=1.65,
        calibration_status="calibrated_for_scope",
    )
    client = FakeClient(result)
    case = make_case(DecisionSpec(type="score", options=["low", "medium", "high"]))

    normalized = VerdictRunner(client=client, runtime=FakeRuntime()).run(case)

    assert normalized.label == "high"
    assert normalized.probabilities == pytest.approx(
        {"low": 0.10, "medium": 0.20, "high": 0.65, "insufficient_evidence": 0.05}
    )
    assert normalized.metadata["selected_value"] == pytest.approx(2.0)
    assert normalized.metadata["expected_score"] == pytest.approx(1.65)

    _, queries = client.calls[0]
    assert queries[0]["levels"] == [
        {"kind": "level", "id": "low", "description": "low", "value": 0.0},
        {"kind": "level", "id": "medium", "description": "medium", "value": 1.0},
        {"kind": "level", "id": "high", "description": "high", "value": 2.0},
    ]


def test_semantic_profile_and_protocol_are_shared_with_existing_candidates() -> None:
    result = SimpleNamespace(
        selected_outcome="false",
        probabilities={"true": 0.20, "false": 0.75, "__insufficient_evidence__": 0.05},
        p_true_given_sufficient_evidence=0.2105263158,
        p_insufficient_evidence=0.05,
        calibration_status="calibrated_for_scope",
    )
    client = FakeClient(result)
    case = make_case(DecisionSpec(type="noul"))
    case.definitions = {"health": {"healthy": "all dependency checks pass"}}

    normalized = VerdictRunner(
        client=client,
        runtime=FakeRuntime(),
        semantic_profile="optimized-v1",
        evaluation_protocol="rule-conditioned",
    ).run(case)

    assert client.calls[0][0] == (
        "Evidence supplied to the model.\n\n[decision_rules]\n"
        "health:\n"
        "  healthy: all dependency checks pass"
    )
    assert normalized.metadata["semantic_profile"] == "optimized-v1"
    assert normalized.metadata["evaluation_protocol"] == "rule-conditioned"
    assert normalized.metadata["model_id"] == DEFAULT_MODEL_ID

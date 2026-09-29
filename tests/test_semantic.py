import pytest

from decision_model_lab.schema import DecisionSpec, EvaluationCase, ExpectedDecision
from decision_model_lab.semantic import (
    EvaluationProtocolError,
    SemanticProfileError,
    partition_native_choice_criteria,
    render_case_state,
    resolve_evaluation_protocol,
    resolve_semantic_profile,
)


def make_case(*, definitions: dict[str, object] | None = None) -> EvaluationCase:
    return EvaluationCase(
        id="case-1",
        context="Observed latency is 400 ms.",
        question="What is the impact level?",
        definitions=definitions or {},
        decision=DecisionSpec(type="choice", options=["normal", "degraded"]),
        expected=ExpectedDecision(label="normal"),
        metadata={"expected_reasoning": "must never be model-visible", "option_order_seed": 42},
    )


def test_case_without_definitions_preserves_baseline_state_exactly() -> None:
    case = make_case()

    assert render_case_state(case) == case.context


def test_baseline_definitions_preserve_established_representation_exactly() -> None:
    case = make_case(
        definitions={
            "severity": {"degraded": "> 2x baseline", "normal": "<= 2x baseline"},
            "threshold_ms": 400,
        }
    )

    state = render_case_state(case)

    assert state == (
        'Observed latency is 400 ms.\n\n[definitions]\n'
        '{"severity":{"degraded":"> 2x baseline","normal":"<= 2x baseline"},'
        '"threshold_ms":400}'
    )
    assert "expected_reasoning" not in state
    assert "option_order_seed" not in state


def test_optimized_v1_renders_only_definitions_as_deterministic_rule_tree() -> None:
    case = make_case(
        definitions={
            "threshold_ms": 400,
            "severity": {"normal": "<= 2x baseline", "degraded": "> 2x baseline"},
        }
    )

    state = render_case_state(case, profile="optimized-v1")

    assert state == (
        "Observed latency is 400 ms.\n\n[decision_rules]\n"
        "severity:\n"
        "  degraded: > 2x baseline\n"
        "  normal: <= 2x baseline\n"
        "threshold_ms: 400"
    )
    assert "expected_reasoning" not in state
    assert "option_order_seed" not in state


def test_optimized_v1_does_not_change_cases_without_definitions() -> None:
    case = make_case()

    assert render_case_state(case, profile="optimized-v1") == case.context


def test_native_criteria_v1_moves_exact_choice_definition_map_out_of_state() -> None:
    case = make_case(
        definitions={
            "impact_levels": {"normal": "inside SLO", "degraded": "outside SLO"},
            "threshold_ms": 400,
        }
    )

    criteria, residual, source = partition_native_choice_criteria(case)

    assert criteria == {"normal": "inside SLO", "degraded": "outside SLO"}
    assert residual == {"threshold_ms": 400}
    assert source == "impact_levels"
    assert render_case_state(case, profile="native-criteria-v1") == (
        'Observed latency is 400 ms.\n\n[definitions]\n{"threshold_ms":400}'
    )


def test_native_criteria_v1_falls_back_to_baseline_without_exact_option_map() -> None:
    case = make_case(
        definitions={"severity": {"mild": "inside SLO", "severe": "outside SLO"}}
    )

    criteria, residual, source = partition_native_choice_criteria(case)

    assert criteria is None
    assert residual == case.definitions
    assert source is None
    assert render_case_state(case, profile="native-criteria-v1") == render_case_state(case)


def test_semantic_profile_resolution_is_explicit() -> None:
    assert resolve_semantic_profile("baseline") == "baseline"
    assert resolve_semantic_profile("optimized-v1") == "optimized-v1"
    assert resolve_semantic_profile("native-criteria-v1") == "native-criteria-v1"

    with pytest.raises(
        SemanticProfileError,
        match=(
            "unsupported semantic profile: unknown; available profiles: "
            "baseline, optimized-v1, native-criteria-v1"
        ),
    ):
        resolve_semantic_profile("unknown")


def test_closed_book_hides_definitions_from_model_visible_state() -> None:
    case = make_case(definitions={"severity": {"normal": "<= 2x baseline"}})

    assert render_case_state(case, protocol="closed-book") == case.context


def test_closed_book_neutralizes_semantic_profile_by_design() -> None:
    case = make_case(definitions={"severity": {"normal": "<= 2x baseline"}})

    assert render_case_state(case, profile="baseline", protocol="closed-book") == (
        render_case_state(case, profile="optimized-v1", protocol="closed-book")
    )
    assert render_case_state(case, profile="baseline", protocol="closed-book") == (
        render_case_state(case, profile="native-criteria-v1", protocol="closed-book")
    )


def test_evaluation_protocol_resolution_is_explicit() -> None:
    assert resolve_evaluation_protocol("rule-conditioned") == "rule-conditioned"
    assert resolve_evaluation_protocol("closed-book") == "closed-book"

    with pytest.raises(
        EvaluationProtocolError,
        match=(
            "unsupported evaluation protocol: unknown; "
            "available protocols: rule-conditioned, closed-book"
        ),
    ):
        resolve_evaluation_protocol("unknown")

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from decision_model_lab.jev_style_runner import (
    DEFAULT_MODEL_ID,
    MODEL_2B_GGUF_ID,
    MODEL_2B_GGUF_REVISION,
    MODEL_2B_GGUF_RUNTIME_FILENAME,
    MODEL_2B_GGUF_TOKENIZER_PATTERN,
    MODEL_2B_ID,
    MODEL_2B_Q4_FILENAME,
    MODEL_2B_Q4_QUANT,
    MODEL_2B_Q8_FILENAME,
    MODEL_2B_Q8_QUANT,
    JevStyleRunner,
)
from decision_model_lab.schema import DecisionSpec, EvaluationCase, ExpectedDecision


class FakeClient:
    def __init__(self, output: dict[str, Any]) -> None:
        self.output = output
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def decide(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((state, questions))
        return self.output


@pytest.fixture(autouse=True)
def fake_jev_style_module(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeJevStyle:
        calls: list[tuple[str, dict[str, Any]]] = []

        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> FakeClient:
            cls.calls.append((model_id, kwargs))
            return FakeClient({"answers": {"decision": {"type": "noul", "noul": 0.8}}})

    module = SimpleNamespace(
        JevStyle=FakeJevStyle,
        noul=lambda instructions: {"type": "noul", "instructions": instructions},
        choice=lambda instructions, options: {
            "type": "choice",
            "instructions": instructions,
            "criteria": options,
        },
        score=lambda instructions, options: {
            "type": "score",
            "instructions": instructions,
            "criteria": options,
        },
    )
    monkeypatch.setitem(sys.modules, "jev_style", module)


def make_case(decision: DecisionSpec) -> EvaluationCase:
    return EvaluationCase(
        id="case-1",
        context="Evidence supplied to the model.",
        question="What should the decision be?",
        decision=decision,
        expected=ExpectedDecision(label="yes"),
    )


def test_noul_is_normalized_without_inventing_provider_confidence() -> None:
    client = FakeClient(
        {
            "model": "jev-style-0.8b-decision-v3",
            "answers": {"decision": {"type": "noul", "noul": 0.8}},
            "latency_ms": 12.5,
            "backend": "torch",
        }
    )

    result = JevStyleRunner(client=client).run(make_case(DecisionSpec(type="noul")))

    assert result.label == "yes"
    assert result.probabilities == pytest.approx({"no": 0.2, "yes": 0.8})
    assert result.confidence is None
    assert result.metadata["provider_latency_ms"] == 12.5
    assert result.metadata["backend"] == "torch"
    assert result.metadata["model_id"] == DEFAULT_MODEL_ID
    assert client.calls[0][1]["decision"]["type"] == "noul"


def test_choice_preserves_raw_output_and_normalizes_probabilities() -> None:
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

    result = JevStyleRunner(client=client).run(case)

    assert result.label == "degraded"
    assert result.confidence == 0.7
    assert result.probabilities == pytest.approx(
        {"normal": 0.1, "degraded": 0.7, "unavailable": 0.2}
    )
    assert result.raw_output is output
    assert client.calls[0][1]["decision"]["criteria"] == [
        "normal",
        "degraded",
        "unavailable",
    ]


def test_definitions_are_included_in_model_visible_state() -> None:
    client = FakeClient({"answers": {"decision": {"type": "noul", "noul": 0.8}}})
    case = make_case(DecisionSpec(type="noul"))
    case.definitions = {"health": {"healthy": "all dependency checks pass"}}

    JevStyleRunner(client=client).run(case)

    assert client.calls[0][0] == (
        "Evidence supplied to the model.\n\n[definitions]\n"
        '{"health":{"healthy":"all dependency checks pass"}}'
    )


def test_optimized_v1_semantic_profile_is_used_and_recorded() -> None:
    client = FakeClient({"answers": {"decision": {"type": "noul", "noul": 0.8}}})
    case = make_case(DecisionSpec(type="noul"))
    case.definitions = {"health": {"healthy": "all dependency checks pass"}}

    result = JevStyleRunner(client=client, semantic_profile="optimized-v1").run(case)

    assert client.calls[0][0] == (
        "Evidence supplied to the model.\n\n[decision_rules]\n"
        "health:\n"
        "  healthy: all dependency checks pass"
    )
    assert result.metadata["semantic_profile"] == "optimized-v1"


def test_native_criteria_v1_uses_exact_choice_definition_map_as_provider_criteria() -> None:
    output = {
        "answers": {
            "decision": {
                "type": "choice",
                "choice": "degraded",
                "probabilities": {"normal": 0.2, "degraded": 0.8},
            }
        }
    }
    client = FakeClient(output)
    case = make_case(DecisionSpec(type="choice", options=["normal", "degraded"]))
    case.definitions = {
        "impact_levels": {"normal": "inside SLO", "degraded": "outside SLO"},
        "threshold_ms": 400,
    }

    result = JevStyleRunner(client=client, semantic_profile="native-criteria-v1").run(case)

    assert client.calls[0][0] == (
        'Evidence supplied to the model.\n\n[definitions]\n{"threshold_ms":400}'
    )
    assert client.calls[0][1]["decision"]["criteria"] == {
        "normal": "inside SLO",
        "degraded": "outside SLO",
    }
    assert result.metadata["native_criteria_applied"] is True
    assert result.metadata["native_criteria_source"] == "impact_levels"


def test_native_criteria_v1_falls_back_without_unambiguous_choice_definition_map() -> None:
    output = {
        "answers": {
            "decision": {
                "type": "choice",
                "choice": "yes",
                "probabilities": {"yes": 0.7, "no": 0.2, "insufficient_evidence": 0.1},
            }
        }
    }
    client = FakeClient(output)
    case = make_case(DecisionSpec(type="choice", options=["yes", "no", "insufficient_evidence"]))
    case.definitions = {"causal_threshold": {"confirmed": "mechanism observed"}}

    result = JevStyleRunner(client=client, semantic_profile="native-criteria-v1").run(case)

    assert client.calls[0][0] == (
        "Evidence supplied to the model.\n\n[definitions]\n"
        '{"causal_threshold":{"confirmed":"mechanism observed"}}'
    )
    assert client.calls[0][1]["decision"]["criteria"] == [
        "yes",
        "no",
        "insufficient_evidence",
    ]
    assert result.metadata["native_criteria_applied"] is False


def test_native_criteria_v1_closed_book_does_not_leak_definition_criteria() -> None:
    output = {
        "answers": {
            "decision": {
                "type": "choice",
                "choice": "normal",
                "probabilities": {"normal": 0.8, "degraded": 0.2},
            }
        }
    }
    client = FakeClient(output)
    case = make_case(DecisionSpec(type="choice", options=["normal", "degraded"]))
    case.definitions = {"impact_levels": {"normal": "inside SLO", "degraded": "outside SLO"}}

    result = JevStyleRunner(
        client=client,
        semantic_profile="native-criteria-v1",
        evaluation_protocol="closed-book",
    ).run(case)

    assert client.calls[0][0] == "Evidence supplied to the model."
    assert client.calls[0][1]["decision"]["criteria"] == ["normal", "degraded"]
    assert result.metadata["native_criteria_applied"] is False


def test_runner_accepts_2b_model_id_without_changing_adapter_contract() -> None:
    client = FakeClient({"answers": {"decision": {"type": "noul", "noul": 0.8}}})

    result = JevStyleRunner(model_id=MODEL_2B_ID, client=client).run(
        make_case(DecisionSpec(type="noul"))
    )

    assert result.metadata["model_id"] == MODEL_2B_ID


def test_closed_book_protocol_hides_definitions_and_is_recorded() -> None:
    client = FakeClient({"answers": {"decision": {"type": "noul", "noul": 0.8}}})
    case = make_case(DecisionSpec(type="noul"))
    case.definitions = {"health": {"healthy": "all dependency checks pass"}}

    result = JevStyleRunner(client=client, evaluation_protocol="closed-book").run(case)

    assert client.calls[0][0] == "Evidence supplied to the model."
    assert result.metadata["evaluation_protocol"] == "closed-book"


def test_2b_q4_prefetches_exact_weight_from_pinned_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloads: list[dict[str, Any]] = []

    def fake_hf_hub_download(**kwargs: Any) -> str:
        downloads.append(kwargs)
        filename = kwargs["filename"]
        return (
            "/cache/huggingface/hub/models--chaoliangUNSW--Jev-Style-2B-Decision-v3-GGUF/"
            f"snapshots/abc123/{filename}"
        )

    snapshot_downloads: list[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        snapshot_downloads.append(kwargs)
        return (
            "/cache/huggingface/hub/models--chaoliangUNSW--Jev-Style-2B-Decision-v3-GGUF/"
            "snapshots/abc123"
        )

    hub = SimpleNamespace(
        hf_hub_download=fake_hf_hub_download,
        snapshot_download=fake_snapshot_download,
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    runner = JevStyleRunner(
        model_id=MODEL_2B_GGUF_ID,
        quantization=MODEL_2B_Q4_QUANT,
        revision=MODEL_2B_GGUF_REVISION,
        model_filename=MODEL_2B_Q4_FILENAME,
        support_filenames=(MODEL_2B_GGUF_RUNTIME_FILENAME,),
        support_patterns=(MODEL_2B_GGUF_TOKENIZER_PATTERN,),
    )
    result = runner.run(make_case(DecisionSpec(type="noul")))

    assert downloads == [
        {
            "repo_id": MODEL_2B_GGUF_ID,
            "filename": "Jev-Style-2B-Decision-v3-Q4_K_M.gguf",
            "revision": MODEL_2B_GGUF_REVISION,
        },
        {
            "repo_id": MODEL_2B_GGUF_ID,
            "filename": "jev_style_decision_gguf.py",
            "revision": "abc123",
        },
    ]
    assert snapshot_downloads == [
        {
            "repo_id": MODEL_2B_GGUF_ID,
            "revision": "abc123",
            "allow_patterns": ["tokenizer/**"],
        }
    ]
    module = sys.modules["jev_style"]
    assert module.JevStyle.calls[-1] == (
        MODEL_2B_GGUF_ID,
        {"quant": "Q4_K_M", "revision": "abc123"},
    )
    assert result.metadata["model_id"] == MODEL_2B_GGUF_ID
    assert result.metadata["quantization"] == "Q4_K_M"
    assert result.metadata["model_filename"] == MODEL_2B_Q4_FILENAME
    assert result.metadata["model_revision"] == "abc123"


def test_2b_q8_prefetches_exact_weight_from_pinned_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloads: list[dict[str, Any]] = []

    def fake_hf_hub_download(**kwargs: Any) -> str:
        downloads.append(kwargs)
        filename = kwargs["filename"]
        return (
            "/cache/huggingface/hub/models--chaoliangUNSW--Jev-Style-2B-Decision-v3-GGUF/"
            f"snapshots/abc123/{filename}"
        )

    snapshot_downloads: list[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        snapshot_downloads.append(kwargs)
        return (
            "/cache/huggingface/hub/models--chaoliangUNSW--Jev-Style-2B-Decision-v3-GGUF/"
            "snapshots/abc123"
        )

    hub = SimpleNamespace(
        hf_hub_download=fake_hf_hub_download,
        snapshot_download=fake_snapshot_download,
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    runner = JevStyleRunner(
        model_id=MODEL_2B_GGUF_ID,
        quantization=MODEL_2B_Q8_QUANT,
        revision=MODEL_2B_GGUF_REVISION,
        model_filename=MODEL_2B_Q8_FILENAME,
        support_filenames=(MODEL_2B_GGUF_RUNTIME_FILENAME,),
        support_patterns=(MODEL_2B_GGUF_TOKENIZER_PATTERN,),
    )
    result = runner.run(make_case(DecisionSpec(type="noul")))

    assert downloads == [
        {
            "repo_id": MODEL_2B_GGUF_ID,
            "filename": "Jev-Style-2B-Decision-v3-Q8_0.gguf",
            "revision": MODEL_2B_GGUF_REVISION,
        },
        {
            "repo_id": MODEL_2B_GGUF_ID,
            "filename": "jev_style_decision_gguf.py",
            "revision": "abc123",
        },
    ]
    assert snapshot_downloads == [
        {
            "repo_id": MODEL_2B_GGUF_ID,
            "revision": "abc123",
            "allow_patterns": ["tokenizer/**"],
        }
    ]
    module = sys.modules["jev_style"]
    assert module.JevStyle.calls[-1] == (
        MODEL_2B_GGUF_ID,
        {"quant": "Q8_0", "revision": "abc123"},
    )
    assert result.metadata["model_id"] == MODEL_2B_GGUF_ID
    assert result.metadata["quantization"] == "Q8_0"
    assert result.metadata["model_filename"] == MODEL_2B_Q8_FILENAME
    assert result.metadata["model_revision"] == "abc123"


def test_resident_experiment_fork_reuses_loaded_client() -> None:
    owner = JevStyleRunner()
    owner.prepare()

    fork = owner.fork_for_experiment(
        semantic_profile="optimized-v1",
        evaluation_protocol="closed-book",
    )

    module = sys.modules["jev_style"]
    assert len(module.JevStyle.calls) == 1
    assert fork is not owner
    assert fork._client is owner._client
    assert fork.semantic_profile == "optimized-v1"
    assert fork.evaluation_protocol == "closed-book"

    client = fork._client
    owner.release()
    assert owner._client is None
    assert fork._client is client

from __future__ import annotations

from importlib import import_module
from time import perf_counter_ns
from typing import Any

from decision_model_lab.schema import DecisionResult, EvaluationCase
from decision_model_lab.semantic import (
    DEFAULT_EVALUATION_PROTOCOL,
    DEFAULT_SEMANTIC_PROFILE,
    EvaluationProtocol,
    SemanticProfile,
    render_case_state,
    resolve_evaluation_protocol,
    resolve_semantic_profile,
)

DEFAULT_MODEL_ID = "TinyJev-0.6B"
QUESTION_KEY = "decision"


class TinyJevRunnerError(RuntimeError):
    """Raised when TinyJev cannot produce a normalized decision."""


class TinyJevRunner:
    """Thin integration around the upstream TinyJev Python runtime."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        client: Any | None = None,
        semantic_profile: str = DEFAULT_SEMANTIC_PROFILE,
        evaluation_protocol: str = DEFAULT_EVALUATION_PROTOCOL,
    ) -> None:
        self.model_id = model_id
        self.semantic_profile: SemanticProfile = resolve_semantic_profile(semantic_profile)
        self.evaluation_protocol: EvaluationProtocol = resolve_evaluation_protocol(
            evaluation_protocol
        )
        self._client = client

    def prepare(self) -> None:
        """Load the provider client without executing an evaluation case."""
        self._client_or_load()

    def fork_for_experiment(
        self,
        *,
        semantic_profile: str,
        evaluation_protocol: str,
    ) -> "TinyJevRunner":
        """Create a fresh experiment runner sharing only the resident model client."""
        return TinyJevRunner(
            model_id=self.model_id,
            client=self._client_or_load(),
            semantic_profile=semantic_profile,
            evaluation_protocol=evaluation_protocol,
        )

    def release(self) -> None:
        """Drop this runner's reference to the resident provider client."""
        self._client = None

    def _client_or_load(self) -> Any:
        if self._client is None:
            module = import_module("tinyjev")
            self._client = module.load(self.model_id)
        return self._client

    def run(self, case: EvaluationCase) -> DecisionResult:
        client = self._client_or_load()
        request = {
            "state": render_case_state(
                case,
                profile=self.semantic_profile,
                protocol=self.evaluation_protocol,
            ),
            "questions": {QUESTION_KEY: _build_question(case)},
        }

        started = perf_counter_ns()
        output = client.predict(request)
        latency_ms = (perf_counter_ns() - started) / 1_000_000

        return _normalize(
            case,
            output,
            latency_ms=latency_ms,
            model_id=self.model_id,
            semantic_profile=self.semantic_profile,
            evaluation_protocol=self.evaluation_protocol,
            revision=getattr(client, "revision", None),
        )


def _build_question(case: EvaluationCase) -> dict[str, Any]:
    decision = case.decision
    question: dict[str, Any] = {"type": decision.type, "instructions": case.question}

    if decision.type == "choice":
        question["criteria"] = {option: option for option in decision.options}
    elif decision.type == "score":
        question["criteria"] = decision.options
    elif decision.type != "noul":
        raise TinyJevRunnerError(f"unsupported TinyJev decision type: {decision.type}")

    return question


def _answers_from_output(output: Any) -> dict[str, Any]:
    if not isinstance(output, dict):
        raise TinyJevRunnerError("TinyJev output must be a mapping")

    answers = output.get("answers")
    if isinstance(answers, dict):
        return answers

    states = output.get("states")
    if isinstance(states, list) and len(states) == 1 and isinstance(states[0], dict):
        answers = states[0].get("answers")
        if isinstance(answers, dict):
            return answers

    raise TinyJevRunnerError("TinyJev output is missing decision answers")


def _normalize_probabilities(probabilities: dict[str, Any]) -> dict[str, float]:
    normalized = {str(label): float(value) for label, value in probabilities.items()}
    total = sum(normalized.values())
    if total <= 0.0:
        raise TinyJevRunnerError("TinyJev returned an empty probability mass")
    return {label: value / total for label, value in normalized.items()}


def _normalize(
    case: EvaluationCase,
    output: Any,
    *,
    latency_ms: float,
    model_id: str,
    semantic_profile: SemanticProfile,
    evaluation_protocol: EvaluationProtocol,
    revision: str | None,
) -> DecisionResult:
    answers = _answers_from_output(output)
    answer = answers.get(QUESTION_KEY)
    if not isinstance(answer, dict):
        raise TinyJevRunnerError("TinyJev output is missing answers.decision")

    confidence = float(answer["confidence"]) if answer.get("confidence") is not None else None
    metadata: dict[str, Any] = {
        "candidate": "tinyjev",
        "model_id": model_id,
        "decision_type": case.decision.type,
        "semantic_profile": semantic_profile,
        "evaluation_protocol": evaluation_protocol,
    }
    if revision is not None:
        metadata["model_revision"] = str(revision)

    if case.decision.type == "noul":
        probability_yes = float(answer["noul"])
        if not 0.0 <= probability_yes <= 1.0:
            raise TinyJevRunnerError("noul probability must be between 0 and 1")
        probabilities = {"no": 1.0 - probability_yes, "yes": probability_yes}
        label = "yes" if probability_yes >= 0.5 else "no"
        metadata["label_threshold"] = 0.5
    elif case.decision.type == "choice":
        label = str(answer["choice"])
        probabilities = _normalize_probabilities(answer.get("probabilities", {}))
    elif case.decision.type == "score":
        probabilities_by_index = _normalize_probabilities(answer.get("probabilities", {}))
        top_index = max(probabilities_by_index, key=probabilities_by_index.__getitem__)
        try:
            label = case.decision.options[int(top_index)]
            probabilities = {
                case.decision.options[int(index)]: probability
                for index, probability in probabilities_by_index.items()
            }
        except (ValueError, IndexError) as exc:
            raise TinyJevRunnerError(f"invalid score probability index: {top_index!r}") from exc
        if answer.get("score") is not None:
            metadata["score"] = float(answer["score"])
    else:
        raise TinyJevRunnerError(f"unsupported TinyJev decision type: {case.decision.type}")

    return DecisionResult(
        case_id=case.id,
        label=label,
        probabilities=probabilities,
        confidence=confidence,
        raw_output=output,
        latency_ms=latency_ms,
        metadata=metadata,
    )

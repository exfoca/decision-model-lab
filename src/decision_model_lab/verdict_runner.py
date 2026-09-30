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

DEFAULT_MODEL_ID = "heman10x/rlcd-modernbert-151m"
UPSTREAM_ENGINE_REVISION = "30f1556"
QUESTION_KEY = "decision"
UPSTREAM_ABSTENTION_LABEL = "__insufficient_evidence__"
CANONICAL_ABSTENTION_LABEL = "insufficient_evidence"
NOUL_SEMANTICS = "conditional_on_sufficient_evidence_v2"


class VerdictRunnerError(RuntimeError):
    """Raised when Verdict cannot produce a normalized decision."""


class VerdictRunner:
    """Adapter for the source-pinned RLCD/OpenJev Verdict decision engine."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        client: Any | None = None,
        runtime: Any | None = None,
        device: str | None = None,
        semantic_profile: str = DEFAULT_SEMANTIC_PROFILE,
        evaluation_protocol: str = DEFAULT_EVALUATION_PROTOCOL,
    ) -> None:
        self.model_id = model_id
        self.semantic_profile: SemanticProfile = resolve_semantic_profile(semantic_profile)
        self.evaluation_protocol: EvaluationProtocol = resolve_evaluation_protocol(
            evaluation_protocol
        )
        self.device = device
        self._client = client
        self._runtime = runtime

    def prepare(self) -> None:
        """Load the provider runtime and model client without executing an evaluation case."""
        self._runtime_or_load()
        self._client_or_load()

    def fork_for_experiment(
        self,
        *,
        semantic_profile: str,
        evaluation_protocol: str,
    ) -> VerdictRunner:
        """Create a fresh experiment runner sharing only the resident runtime and client."""
        self.prepare()
        return VerdictRunner(
            model_id=self.model_id,
            client=self._client,
            runtime=self._runtime,
            device=self.device,
            semantic_profile=semantic_profile,
            evaluation_protocol=evaluation_protocol,
        )

    def release(self) -> None:
        """Drop this runner's references to the resident provider runtime and client."""
        self._client = None
        self._runtime = None

    def _runtime_or_load(self) -> Any:
        if self._runtime is None:
            try:
                self._runtime = import_module("rlcd")
            except ImportError as exc:
                raise VerdictRunnerError(
                    "Verdict runtime is not installed; rebuild the lab image with the "
                    f"source-pinned rlcd runtime at revision {UPSTREAM_ENGINE_REVISION}"
                ) from exc
        return self._runtime

    def _device_or_resolve(self) -> str:
        if self.device is not None:
            return self.device

        torch: Any = import_module("torch")
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _client_or_load(self) -> Any:
        if self._client is None:
            runtime = self._runtime_or_load()
            self._client = runtime.DecisionEngine(
                model_name_or_path=self.model_id,
                device=self._device_or_resolve(),
            )
        return self._client

    def run(self, case: EvaluationCase) -> DecisionResult:
        runtime = self._runtime_or_load()
        client = self._client_or_load()
        state = render_case_state(
            case,
            profile=self.semantic_profile,
            protocol=self.evaluation_protocol,
        )
        query = _build_query(runtime, case)

        started = perf_counter_ns()
        output = client.evaluate(context=state, queries=[query])
        latency_ms = (perf_counter_ns() - started) / 1_000_000

        return _normalize(
            case,
            output,
            latency_ms=latency_ms,
            model_id=self.model_id,
            semantic_profile=self.semantic_profile,
            evaluation_protocol=self.evaluation_protocol,
        )


def _substantive_options(options: list[str]) -> list[str]:
    return [option for option in options if option != CANONICAL_ABSTENTION_LABEL]


def _build_query(runtime: Any, case: EvaluationCase) -> Any:
    decision = case.decision

    if decision.type == "choice":
        options = _substantive_options(decision.options)
        if len(options) < 2:
            raise VerdictRunnerError(
                "Verdict choice requires at least two substantive options after reserving "
                "insufficient_evidence for the upstream abstention route"
            )
        return runtime.Choice(
            id=QUESTION_KEY,
            question=case.question,
            options=[runtime.Option(id=option, description=option) for option in options],
        )

    if decision.type == "score":
        levels = _substantive_options(decision.options)
        if len(levels) < 2:
            raise VerdictRunnerError(
                "Verdict score requires at least two substantive levels after reserving "
                "insufficient_evidence for the upstream abstention route"
            )
        return runtime.Score(
            id=QUESTION_KEY,
            question=case.question,
            levels=[
                runtime.Level(id=label, description=label, value=float(index))
                for index, label in enumerate(levels)
            ],
        )

    if decision.type == "noul":
        return runtime.Noul(
            id=QUESTION_KEY,
            proposition=case.question,
            semantics=NOUL_SEMANTICS,
        )

    raise VerdictRunnerError(f"unsupported Verdict decision type: {decision.type}")


def _single_result(output: Any) -> Any:
    results = getattr(output, "results", None)
    if not isinstance(results, (list, tuple)) or len(results) != 1:
        raise VerdictRunnerError("Verdict output must contain exactly one decision result")
    return results[0]


def _canonical_probability_label(label: str, *, decision_type: str) -> str:
    if label == UPSTREAM_ABSTENTION_LABEL:
        return CANONICAL_ABSTENTION_LABEL
    if decision_type == "noul":
        if label == "true":
            return "yes"
        if label == "false":
            return "no"
    return label


def _normalize_probabilities(probabilities: Any, *, decision_type: str) -> dict[str, float]:
    if not isinstance(probabilities, dict) or not probabilities:
        raise VerdictRunnerError("Verdict result is missing a probability distribution")

    normalized: dict[str, float] = {}
    for raw_label, raw_probability in probabilities.items():
        label = _canonical_probability_label(str(raw_label), decision_type=decision_type)
        normalized[label] = normalized.get(label, 0.0) + float(raw_probability)

    total = sum(normalized.values())
    if total <= 0.0:
        raise VerdictRunnerError("Verdict returned an empty probability mass")
    return {label: probability / total for label, probability in normalized.items()}


def _selected_label(result: Any, *, decision_type: str) -> str:
    if decision_type == "choice":
        # The pinned engine schema uses selected_id. The public README has also
        # documented selected_option_id, so accept both without changing the
        # canonical laboratory result.
        selected = getattr(result, "selected_id", None)
        if selected is None:
            selected = getattr(result, "selected_option_id", None)
    elif decision_type == "score":
        selected = getattr(result, "selected_level_id", None)
    elif decision_type == "noul":
        selected = getattr(result, "selected_outcome", None)
    else:
        raise VerdictRunnerError(f"unsupported Verdict decision type: {decision_type}")

    if not isinstance(selected, str) or not selected:
        raise VerdictRunnerError("Verdict result is missing the selected outcome")
    return _canonical_probability_label(selected, decision_type=decision_type)


def _jsonable_output(output: Any) -> Any:
    model_dump = getattr(output, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return output


def _normalize(
    case: EvaluationCase,
    output: Any,
    *,
    latency_ms: float,
    model_id: str,
    semantic_profile: SemanticProfile,
    evaluation_protocol: EvaluationProtocol,
) -> DecisionResult:
    result = _single_result(output)
    probabilities = _normalize_probabilities(
        getattr(result, "probabilities", None),
        decision_type=case.decision.type,
    )
    label = _selected_label(result, decision_type=case.decision.type)
    confidence = probabilities.get(label)
    if confidence is None:
        confidence = max(probabilities.values())

    metadata: dict[str, Any] = {
        "candidate": "verdict",
        "model_id": model_id,
        "upstream_engine_revision": UPSTREAM_ENGINE_REVISION,
        "decision_type": case.decision.type,
        "semantic_profile": semantic_profile,
        "evaluation_protocol": evaluation_protocol,
    }

    calibration_status = getattr(result, "calibration_status", None)
    if calibration_status is not None:
        metadata["calibration_status"] = str(calibration_status)

    provider_latency_ms = getattr(output, "total_latency_ms", None)
    if provider_latency_ms is not None:
        metadata["provider_latency_ms"] = float(provider_latency_ms)

    forward_call_count = getattr(output, "forward_call_count", None)
    if forward_call_count is not None:
        metadata["forward_call_count"] = int(forward_call_count)

    if case.decision.type == "score":
        selected_value = getattr(result, "selected_value", None)
        expected_score = getattr(result, "expected_score", None)
        if selected_value is not None:
            metadata["selected_value"] = float(selected_value)
        if expected_score is not None:
            metadata["expected_score"] = float(expected_score)

    if case.decision.type == "noul":
        p_true = getattr(result, "p_true_given_sufficient_evidence", None)
        p_abstain = getattr(result, "p_insufficient_evidence", None)
        if p_true is not None:
            metadata["p_true_given_sufficient_evidence"] = float(p_true)
        if p_abstain is not None:
            metadata["p_insufficient_evidence"] = float(p_abstain)

    return DecisionResult(
        case_id=case.id,
        label=label,
        probabilities=probabilities,
        confidence=confidence,
        raw_output=_jsonable_output(output),
        latency_ms=latency_ms,
        metadata=metadata,
    )

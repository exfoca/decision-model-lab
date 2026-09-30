from __future__ import annotations

from importlib import import_module
from pathlib import Path
from time import perf_counter_ns
from typing import Any

from decision_model_lab.schema import DecisionResult, EvaluationCase
from decision_model_lab.semantic import (
    DEFAULT_EVALUATION_PROTOCOL,
    DEFAULT_SEMANTIC_PROFILE,
    EvaluationProtocol,
    SemanticProfile,
    partition_native_choice_criteria,
    render_case_state,
    resolve_evaluation_protocol,
    resolve_semantic_profile,
)

DEFAULT_MODEL_ID = "chaoliangUNSW/Jev-Style-0.8B-Decision-v3"
MODEL_2B_ID = "chaoliangUNSW/Jev-Style-2B-Decision-v3"
MODEL_2B_GGUF_ID = "chaoliangUNSW/Jev-Style-2B-Decision-v3-GGUF"
MODEL_2B_Q4_QUANT = "Q4_K_M"
MODEL_2B_Q4_FILENAME = "Jev-Style-2B-Decision-v3-Q4_K_M.gguf"
MODEL_2B_Q8_QUANT = "Q8_0"
MODEL_2B_Q8_FILENAME = "Jev-Style-2B-Decision-v3-Q8_0.gguf"
MODEL_2B_GGUF_RUNTIME_FILENAME = "jev_style_decision_gguf.py"
MODEL_2B_GGUF_TOKENIZER_PATTERN = "tokenizer/**"
MODEL_2B_GGUF_REVISION = "78b1352e9b8132987f6f6744dc2d3f57947f5357"
QUESTION_KEY = "decision"


class JevStyleRunnerError(RuntimeError):
    """Raised when Jev-Style cannot produce a normalized decision."""


class JevStyleRunner:
    """Thin integration around the official Jev-Style Python runtime."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        client: Any | None = None,
        semantic_profile: str = DEFAULT_SEMANTIC_PROFILE,
        evaluation_protocol: str = DEFAULT_EVALUATION_PROTOCOL,
        quantization: str | None = None,
        revision: str | None = None,
        model_filename: str | None = None,
        support_filenames: tuple[str, ...] = (),
        support_patterns: tuple[str, ...] = (),
    ) -> None:
        self.model_id = model_id
        self.quantization = quantization
        self.revision = revision
        self.model_filename = model_filename
        self.support_filenames = support_filenames
        self.support_patterns = support_patterns
        self._resolved_revision: str | None = None
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
    ) -> "JevStyleRunner":
        """Create a fresh experiment runner sharing only the resident model client."""
        client = self._client_or_load()
        runner = JevStyleRunner(
            model_id=self.model_id,
            client=client,
            semantic_profile=semantic_profile,
            evaluation_protocol=evaluation_protocol,
            quantization=self.quantization,
            revision=self._resolved_revision or self.revision,
            model_filename=self.model_filename,
            support_filenames=self.support_filenames,
            support_patterns=self.support_patterns,
        )
        runner._resolved_revision = self._resolved_revision
        return runner

    def release(self) -> None:
        """Drop this runner's reference to the resident provider client."""
        self._client = None

    def _client_or_load(self) -> Any:
        if self._client is None:
            module = import_module("jev_style")
            jev_style = module.JevStyle

            load_kw: dict[str, Any] = {}
            if self.quantization is not None:
                load_kw["quant"] = self.quantization

            resolved_revision = self.revision
            if self.model_filename is not None:
                resolved_revision = _prefetch_hub_snapshot_files(
                    repo_id=self.model_id,
                    filenames=(self.model_filename, *self.support_filenames),
                    patterns=self.support_patterns,
                    revision=self.revision,
                )
            self._resolved_revision = resolved_revision
            if resolved_revision is not None:
                load_kw["revision"] = resolved_revision

            self._client = jev_style.from_pretrained(self.model_id, **load_kw)
        return self._client

    def run(self, case: EvaluationCase) -> DecisionResult:
        client = self._client_or_load()
        question, native_criteria_source = _build_question(
            case,
            semantic_profile=self.semantic_profile,
            evaluation_protocol=self.evaluation_protocol,
        )

        started = perf_counter_ns()
        state = render_case_state(
            case, profile=self.semantic_profile, protocol=self.evaluation_protocol
        )
        output = client.decide(state, {QUESTION_KEY: question})
        latency_ms = (perf_counter_ns() - started) / 1_000_000

        runtime_revision = getattr(client, "revision", None)
        return _normalize(
            case,
            output,
            latency_ms=latency_ms,
            model_id=self.model_id,
            semantic_profile=self.semantic_profile,
            evaluation_protocol=self.evaluation_protocol,
            native_criteria_source=native_criteria_source,
            quantization=self.quantization,
            model_filename=self.model_filename,
            revision=runtime_revision or self._resolved_revision or self.revision,
        )


def _prefetch_hub_snapshot_files(
    *,
    repo_id: str,
    filenames: tuple[str, ...],
    revision: str | None,
    patterns: tuple[str, ...] = (),
) -> str:
    """Fetch required Hub files into one immutable snapshot and return its revision."""
    if not filenames:
        raise JevStyleRunnerError("at least one Hugging Face file must be prefetched")

    hub = import_module("huggingface_hub")
    resolved_revision: str | None = None

    for filename in filenames:
        kwargs: dict[str, Any] = {"repo_id": repo_id, "filename": filename}
        requested_revision = resolved_revision or revision
        if requested_revision is not None:
            kwargs["revision"] = requested_revision

        downloaded = hub.hf_hub_download(**kwargs)
        if not downloaded:
            raise JevStyleRunnerError(
                f"Hugging Face did not return a local path for {repo_id}/{filename}"
            )

        file_revision = _snapshot_revision(Path(downloaded))
        if resolved_revision is None:
            resolved_revision = file_revision
        elif file_revision != resolved_revision:
            raise JevStyleRunnerError(
                "Hugging Face returned files from different snapshots: "
                f"{resolved_revision} != {file_revision}"
            )

    assert resolved_revision is not None

    if patterns:
        snapshot_path = hub.snapshot_download(
            repo_id=repo_id,
            revision=resolved_revision,
            allow_patterns=list(patterns),
        )
        if not snapshot_path:
            raise JevStyleRunnerError(
                f"Hugging Face did not return a snapshot path for {repo_id}"
            )
        snapshot_revision = _snapshot_revision(Path(snapshot_path))
        if snapshot_revision != resolved_revision:
            raise JevStyleRunnerError(
                "Hugging Face returned support files from a different snapshot: "
                f"{resolved_revision} != {snapshot_revision}"
            )

    return resolved_revision


def _snapshot_revision(path: Path) -> str:
    """Extract the immutable Hub revision from a standard snapshot cache path."""
    parts = path.parts
    try:
        snapshot_index = parts.index("snapshots")
        return parts[snapshot_index + 1]
    except (ValueError, IndexError) as exc:
        raise JevStyleRunnerError(
            f"cannot resolve Hugging Face snapshot revision from {path}"
        ) from exc


def _build_question(
    case: EvaluationCase,
    *,
    semantic_profile: SemanticProfile,
    evaluation_protocol: EvaluationProtocol,
) -> tuple[Any, str | None]:
    module = import_module("jev_style")
    decision = case.decision

    if decision.type == "noul":
        return module.noul(case.question), None
    if decision.type == "choice":
        if (
            semantic_profile == "native-criteria-v1"
            and evaluation_protocol == "rule-conditioned"
        ):
            criteria, _, source = partition_native_choice_criteria(case)
            if criteria is not None:
                return module.choice(case.question, criteria), source
        return module.choice(case.question, decision.options), None
    if decision.type == "score":
        return module.score(case.question, decision.options), None

    raise JevStyleRunnerError(f"unsupported Jev-Style decision type: {decision.type}")


def _normalize_probabilities(probabilities: dict[str, Any]) -> dict[str, float]:
    normalized = {str(label): float(value) for label, value in probabilities.items()}
    total = sum(normalized.values())
    if total <= 0.0:
        raise JevStyleRunnerError("Jev-Style returned an empty probability mass")
    return {label: value / total for label, value in normalized.items()}


def _normalize(
    case: EvaluationCase,
    output: Any,
    *,
    latency_ms: float,
    model_id: str,
    semantic_profile: SemanticProfile,
    evaluation_protocol: EvaluationProtocol,
    native_criteria_source: str | None,
    quantization: str | None,
    model_filename: str | None,
    revision: str | None,
) -> DecisionResult:
    if not isinstance(output, dict):
        raise JevStyleRunnerError("Jev-Style output must be a mapping")

    answers = output.get("answers")
    if not isinstance(answers, dict) or not isinstance(answers.get(QUESTION_KEY), dict):
        raise JevStyleRunnerError("Jev-Style output is missing answers.decision")
    answer = answers[QUESTION_KEY]

    confidence: float | None = None
    metadata: dict[str, Any] = {
        "candidate": "jev-style",
        "model_id": model_id,
        "decision_type": case.decision.type,
        "semantic_profile": semantic_profile,
        "evaluation_protocol": evaluation_protocol,
    }
    if quantization is not None:
        metadata["quantization"] = quantization
    if model_filename is not None:
        metadata["model_filename"] = model_filename
    if revision is not None:
        metadata["model_revision"] = revision
    if semantic_profile == "native-criteria-v1":
        metadata["native_criteria_applied"] = native_criteria_source is not None
        if native_criteria_source is not None:
            metadata["native_criteria_source"] = native_criteria_source
    if output.get("model") is not None:
        metadata["reported_model"] = output["model"]
    if output.get("backend") is not None:
        metadata["backend"] = output["backend"]
    if output.get("latency_ms") is not None:
        metadata["provider_latency_ms"] = float(output["latency_ms"])
    if output.get("usage") is not None:
        metadata["usage"] = output["usage"]

    if case.decision.type == "noul":
        probability_yes = float(answer["noul"])
        if not 0.0 <= probability_yes <= 1.0:
            raise JevStyleRunnerError("noul probability must be between 0 and 1")
        probabilities = {"no": 1.0 - probability_yes, "yes": probability_yes}
        label = "yes" if probability_yes >= 0.5 else "no"
        metadata["label_threshold"] = 0.5
    elif case.decision.type == "choice":
        label = str(answer["choice"])
        probabilities = _normalize_probabilities(answer.get("probabilities", {}))
        if answer.get("confidence") is not None:
            confidence = float(answer["confidence"])
    elif case.decision.type == "score":
        probabilities_by_index = _normalize_probabilities(answer.get("probabilities", {}))
        top_index = max(probabilities_by_index, key=probabilities_by_index.__getitem__)
        try:
            label = case.decision.options[int(top_index)]
        except (ValueError, IndexError) as exc:
            raise JevStyleRunnerError(f"invalid score probability index: {top_index!r}") from exc
        probabilities = {
            case.decision.options[int(index)]: probability
            for index, probability in probabilities_by_index.items()
        }
        if answer.get("confidence") is not None:
            confidence = float(answer["confidence"])
        if answer.get("score") is not None:
            metadata["score"] = float(answer["score"])
    else:
        raise JevStyleRunnerError(f"unsupported Jev-Style decision type: {case.decision.type}")

    return DecisionResult(
        case_id=case.id,
        label=label,
        probabilities=probabilities,
        confidence=confidence,
        raw_output=output,
        latency_ms=latency_ms,
        metadata=metadata,
    )

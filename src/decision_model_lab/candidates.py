from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

from decision_model_lab.benchmark import DecisionRunner
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
from decision_model_lab.laya_runner import DEFAULT_MODEL_ID as LAYA_MODEL_ID
from decision_model_lab.laya_runner import LayaRunner
from decision_model_lab.semantic import (
    DEFAULT_EVALUATION_PROTOCOL,
    DEFAULT_SEMANTIC_PROFILE,
    SEMANTIC_PROFILES,
    EvaluationProtocol,
    SemanticProfile,
    resolve_evaluation_protocol,
    resolve_semantic_profile,
)
from decision_model_lab.tinyjev_runner import DEFAULT_MODEL_ID as TINYJEV_MODEL_ID
from decision_model_lab.tinyjev_runner import TinyJevRunner
from decision_model_lab.verdict_runner import DEFAULT_MODEL_ID as VERDICT_MODEL_ID
from decision_model_lab.verdict_runner import VerdictRunner


class CandidateError(ValueError):
    """Raised when a requested benchmark candidate is not registered."""


@dataclass(frozen=True)
class Candidate:
    name: str
    model_id: str
    runner_factory: Callable[..., DecisionRunner]
    runtime_distribution: str
    semantic_profiles: tuple[SemanticProfile, ...] = ("baseline", "optimized-v1")

    def create_runner(
        self,
        *,
        semantic_profile: str = DEFAULT_SEMANTIC_PROFILE,
        evaluation_protocol: str = DEFAULT_EVALUATION_PROTOCOL,
    ) -> DecisionRunner:
        profile: SemanticProfile = resolve_semantic_profile(semantic_profile)
        protocol: EvaluationProtocol = resolve_evaluation_protocol(evaluation_protocol)
        if profile not in self.semantic_profiles:
            available = ", ".join(self.semantic_profiles)
            raise CandidateError(
                f"semantic profile {profile!r} is not supported by candidate {self.name!r}; "
                f"available profiles: {available}"
            )
        return self.runner_factory(semantic_profile=profile, evaluation_protocol=protocol)


_CANDIDATES: dict[str, Candidate] = {
    "jev-style": Candidate(
        name="jev-style",
        model_id=DEFAULT_MODEL_ID,
        runner_factory=JevStyleRunner,
        runtime_distribution="jev-style",
        semantic_profiles=SEMANTIC_PROFILES,
    ),
    "jev-style-2b": Candidate(
        name="jev-style-2b",
        model_id=MODEL_2B_ID,
        runner_factory=partial(JevStyleRunner, model_id=MODEL_2B_ID),
        runtime_distribution="jev-style",
        semantic_profiles=SEMANTIC_PROFILES,
    ),
    "jev-style-2b-q4": Candidate(
        name="jev-style-2b-q4",
        model_id=MODEL_2B_GGUF_ID,
        runner_factory=partial(
            JevStyleRunner,
            model_id=MODEL_2B_GGUF_ID,
            quantization=MODEL_2B_Q4_QUANT,
            revision=MODEL_2B_GGUF_REVISION,
            model_filename=MODEL_2B_Q4_FILENAME,
            support_filenames=(MODEL_2B_GGUF_RUNTIME_FILENAME,),
            support_patterns=(MODEL_2B_GGUF_TOKENIZER_PATTERN,),
        ),
        runtime_distribution="jev-style",
        semantic_profiles=SEMANTIC_PROFILES,
    ),
    "jev-style-2b-q8": Candidate(
        name="jev-style-2b-q8",
        model_id=MODEL_2B_GGUF_ID,
        runner_factory=partial(
            JevStyleRunner,
            model_id=MODEL_2B_GGUF_ID,
            quantization=MODEL_2B_Q8_QUANT,
            revision=MODEL_2B_GGUF_REVISION,
            model_filename=MODEL_2B_Q8_FILENAME,
            support_filenames=(MODEL_2B_GGUF_RUNTIME_FILENAME,),
            support_patterns=(MODEL_2B_GGUF_TOKENIZER_PATTERN,),
        ),
        runtime_distribution="jev-style",
        semantic_profiles=SEMANTIC_PROFILES,
    ),
    "laya": Candidate(
        name="laya",
        model_id=LAYA_MODEL_ID,
        runner_factory=LayaRunner,
        runtime_distribution="laya",
    ),
    "tinyjev": Candidate(
        name="tinyjev",
        model_id=TINYJEV_MODEL_ID,
        runner_factory=TinyJevRunner,
        runtime_distribution="tinyjev",
    ),
    "verdict": Candidate(
        name="verdict",
        model_id=VERDICT_MODEL_ID,
        runner_factory=VerdictRunner,
        runtime_distribution="rlcd",
    ),
}


def resolve_candidate(name: str) -> Candidate:
    try:
        return _CANDIDATES[name]
    except KeyError as exc:
        available = ", ".join(sorted(_CANDIDATES))
        raise CandidateError(
            f"unsupported candidate: {name}; available candidates: {available}"
        ) from exc

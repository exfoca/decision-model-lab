from __future__ import annotations

import pytest

from decision_model_lab.candidates import CandidateError, resolve_candidate
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
from decision_model_lab.tinyjev_runner import DEFAULT_MODEL_ID as TINYJEV_MODEL_ID
from decision_model_lab.tinyjev_runner import TinyJevRunner
from decision_model_lab.verdict_runner import DEFAULT_MODEL_ID as VERDICT_MODEL_ID
from decision_model_lab.verdict_runner import VerdictRunner


def test_resolve_candidate_returns_jev_style_specification() -> None:
    candidate = resolve_candidate("jev-style")
    assert candidate.name == "jev-style"
    assert candidate.model_id == DEFAULT_MODEL_ID
    assert candidate.runtime_distribution == "jev-style"
    assert isinstance(candidate.create_runner(), JevStyleRunner)


def test_resolve_candidate_returns_jev_style_2b_specification() -> None:
    candidate = resolve_candidate("jev-style-2b")
    runner = candidate.create_runner()

    assert candidate.name == "jev-style-2b"
    assert candidate.model_id == MODEL_2B_ID
    assert isinstance(runner, JevStyleRunner)
    assert runner.model_id == MODEL_2B_ID


def test_resolve_candidate_returns_jev_style_2b_q4_specification() -> None:
    candidate = resolve_candidate("jev-style-2b-q4")
    runner = candidate.create_runner()

    assert candidate.name == "jev-style-2b-q4"
    assert candidate.model_id == MODEL_2B_GGUF_ID
    assert isinstance(runner, JevStyleRunner)
    assert runner.model_id == MODEL_2B_GGUF_ID
    assert runner.quantization == MODEL_2B_Q4_QUANT
    assert runner.revision == MODEL_2B_GGUF_REVISION
    assert runner.model_filename == MODEL_2B_Q4_FILENAME
    assert runner.support_filenames == (MODEL_2B_GGUF_RUNTIME_FILENAME,)
    assert runner.support_patterns == (MODEL_2B_GGUF_TOKENIZER_PATTERN,)


def test_resolve_candidate_returns_jev_style_2b_q8_specification() -> None:
    candidate = resolve_candidate("jev-style-2b-q8")
    runner = candidate.create_runner()

    assert candidate.name == "jev-style-2b-q8"
    assert candidate.model_id == MODEL_2B_GGUF_ID
    assert isinstance(runner, JevStyleRunner)
    assert runner.model_id == MODEL_2B_GGUF_ID
    assert runner.quantization == MODEL_2B_Q8_QUANT
    assert runner.revision == MODEL_2B_GGUF_REVISION
    assert runner.model_filename == MODEL_2B_Q8_FILENAME
    assert runner.support_filenames == (MODEL_2B_GGUF_RUNTIME_FILENAME,)
    assert runner.support_patterns == (MODEL_2B_GGUF_TOKENIZER_PATTERN,)


def test_resolve_candidate_returns_laya_specification() -> None:
    candidate = resolve_candidate("laya")
    assert candidate.name == "laya"
    assert candidate.model_id == LAYA_MODEL_ID
    assert candidate.runtime_distribution == "laya"
    assert isinstance(candidate.create_runner(), LayaRunner)


def test_resolve_candidate_returns_tinyjev_specification() -> None:
    candidate = resolve_candidate("tinyjev")
    assert candidate.name == "tinyjev"
    assert candidate.model_id == TINYJEV_MODEL_ID
    assert candidate.runtime_distribution == "tinyjev"
    assert isinstance(candidate.create_runner(), TinyJevRunner)


def test_resolve_candidate_returns_verdict_specification() -> None:
    candidate = resolve_candidate("verdict")
    assert candidate.name == "verdict"
    assert candidate.model_id == VERDICT_MODEL_ID
    assert candidate.runtime_distribution == "rlcd"
    assert isinstance(candidate.create_runner(), VerdictRunner)


def test_resolve_candidate_rejects_unknown_candidate_with_available_names() -> None:
    with pytest.raises(
        CandidateError,
        match=(
            r"unsupported candidate: unknown; available candidates: "
            r"jev-style, jev-style-2b, jev-style-2b-q4, jev-style-2b-q8, laya, tinyjev, verdict"
        ),
    ):
        resolve_candidate("unknown")


def test_candidate_factory_propagates_semantic_profile_to_all_runners() -> None:
    for name in (
        "jev-style",
        "jev-style-2b",
        "jev-style-2b-q4",
        "jev-style-2b-q8",
        "laya",
        "tinyjev",
        "verdict",
    ):
        runner = resolve_candidate(name).create_runner(semantic_profile="optimized-v1")
        assert runner.semantic_profile == "optimized-v1"


def test_native_criteria_profile_is_restricted_to_jev_style_candidates() -> None:
    for name in ("jev-style", "jev-style-2b", "jev-style-2b-q4", "jev-style-2b-q8"):
        runner = resolve_candidate(name).create_runner(semantic_profile="native-criteria-v1")
        assert runner.semantic_profile == "native-criteria-v1"

    for name in ("laya", "tinyjev", "verdict"):
        with pytest.raises(CandidateError, match="native-criteria-v1"):
            resolve_candidate(name).create_runner(semantic_profile="native-criteria-v1")


def test_candidate_factory_propagates_evaluation_protocol_to_all_runners() -> None:
    for name in (
        "jev-style",
        "jev-style-2b",
        "jev-style-2b-q4",
        "jev-style-2b-q8",
        "laya",
        "tinyjev",
        "verdict",
    ):
        runner = resolve_candidate(name).create_runner(evaluation_protocol="closed-book")
        assert runner.evaluation_protocol == "closed-book"

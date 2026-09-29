from __future__ import annotations

import json
from typing import Any, Literal

from decision_model_lab.schema import EvaluationCase

SemanticProfile = Literal["baseline", "optimized-v1", "native-criteria-v1"]
DEFAULT_SEMANTIC_PROFILE: SemanticProfile = "baseline"
SEMANTIC_PROFILES: tuple[SemanticProfile, ...] = (
    "baseline",
    "optimized-v1",
    "native-criteria-v1",
)

EvaluationProtocol = Literal["rule-conditioned", "closed-book"]
DEFAULT_EVALUATION_PROTOCOL: EvaluationProtocol = "rule-conditioned"
EVALUATION_PROTOCOLS: tuple[EvaluationProtocol, ...] = ("rule-conditioned", "closed-book")


class SemanticProfileError(ValueError):
    """Raised when an unknown model-visible semantic representation is requested."""


class EvaluationProtocolError(ValueError):
    """Raised when an unknown model-visible evaluation protocol is requested."""


def resolve_semantic_profile(name: str) -> SemanticProfile:
    if name == "baseline":
        return "baseline"
    if name == "optimized-v1":
        return "optimized-v1"
    if name == "native-criteria-v1":
        return "native-criteria-v1"

    available = ", ".join(SEMANTIC_PROFILES)
    raise SemanticProfileError(
        f"unsupported semantic profile: {name}; available profiles: {available}"
    )


def resolve_evaluation_protocol(name: str) -> EvaluationProtocol:
    if name == "rule-conditioned":
        return "rule-conditioned"
    if name == "closed-book":
        return "closed-book"

    available = ", ".join(EVALUATION_PROTOCOLS)
    raise EvaluationProtocolError(
        f"unsupported evaluation protocol: {name}; available protocols: {available}"
    )


def render_case_state(
    case: EvaluationCase,
    *,
    profile: SemanticProfile = DEFAULT_SEMANTIC_PROFILE,
    protocol: EvaluationProtocol = DEFAULT_EVALUATION_PROTOCOL,
) -> str:
    """Render model-visible state under explicit protocol and semantic-profile axes."""
    if protocol == "closed-book":
        return case.context
    if protocol != "rule-conditioned":
        raise EvaluationProtocolError(f"unsupported evaluation protocol: {protocol}")

    if profile == "baseline":
        return _render_baseline(case)
    if profile == "optimized-v1":
        return _render_optimized_v1(case)
    if profile == "native-criteria-v1":
        _, residual_definitions, _ = partition_native_choice_criteria(case)
        return _render_baseline_definitions(case.context, residual_definitions)

    raise SemanticProfileError(f"unsupported semantic profile: {profile}")


def _render_baseline(case: EvaluationCase) -> str:
    return _render_baseline_definitions(case.context, case.definitions)


def _render_baseline_definitions(context: str, definitions: dict[str, Any]) -> str:
    if not definitions:
        return context

    serialized_definitions = json.dumps(
        definitions,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{context}\n\n[definitions]\n{serialized_definitions}"


def partition_native_choice_criteria(
    case: EvaluationCase,
) -> tuple[dict[str, str] | None, dict[str, Any], str | None]:
    """Move an unambiguous option-definition map into native choice criteria.

    The experiment is intentionally conservative: it consumes only the first top-level
    definition mapping whose keys exactly match the declared choice options and whose
    values are non-empty strings. It never infers aliases or uses evaluation-only fields.
    Cases without such a mapping retain the baseline representation unchanged.
    """
    if case.decision.type != "choice" or not case.definitions:
        return None, dict(case.definitions), None

    options = case.decision.options
    option_set = set(options)
    for source, value in case.definitions.items():
        if not isinstance(value, dict) or set(value) != option_set:
            continue
        if any(not isinstance(value.get(option), str) or not value[option] for option in options):
            continue

        criteria = {option: value[option] for option in options}
        residual_definitions = {
            key: definition for key, definition in case.definitions.items() if key != source
        }
        return criteria, residual_definitions, source

    return None, dict(case.definitions), None


def _render_optimized_v1(case: EvaluationCase) -> str:
    if not case.definitions:
        return case.context

    rules = _render_definition_tree(case.definitions)
    return f"{case.context}\n\n[decision_rules]\n" + "\n".join(rules)


def _render_definition_tree(value: Any, *, indent: int = 0) -> list[str]:
    prefix = "  " * indent

    if isinstance(value, dict):
        lines: list[str] = []
        for key in sorted(value):
            rendered_key = _render_definition_key(key)
            child = value[key]
            if isinstance(child, dict):
                if child:
                    lines.append(f"{prefix}{rendered_key}:")
                    lines.extend(_render_definition_tree(child, indent=indent + 1))
                else:
                    lines.append(f"{prefix}{rendered_key}: {{}}")
            elif isinstance(child, list):
                if child:
                    lines.append(f"{prefix}{rendered_key}:")
                    lines.extend(_render_definition_tree(child, indent=indent + 1))
                else:
                    lines.append(f"{prefix}{rendered_key}: []")
            else:
                lines.append(f"{prefix}{rendered_key}: {_render_definition_scalar(child)}")
        return lines

    if isinstance(value, list):
        lines = []
        for child in value:
            if isinstance(child, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(_render_definition_tree(child, indent=indent + 1))
            else:
                lines.append(f"{prefix}- {_render_definition_scalar(child)}")
        return lines

    return [f"{prefix}{_render_definition_scalar(value)}"]


def _render_definition_key(value: object) -> str:
    key = str(value)
    if key and all(character.isalnum() or character in "_-" for character in key):
        return key
    return json.dumps(key, ensure_ascii=False, separators=(",", ":"))


def _render_definition_scalar(value: Any) -> str:
    if isinstance(value, str):
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return encoded[1:-1]
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

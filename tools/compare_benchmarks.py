#!/usr/bin/env python3
"""Compare normalized Decision Model Lab benchmark artifacts.

The tool is intentionally candidate-agnostic. It consumes persisted benchmark
reports, aligns their aggregate/segmented/case-level evidence, and renders a
comparison without importing any model runtime.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

Direction = Literal["higher", "lower", "neutral"]
CasePolicy = Literal["strict", "intersection"]


class ComparisonError(RuntimeError):
    """Raised when benchmark artifacts cannot be compared safely."""


@dataclass(frozen=True)
class MetricSpec:
    path: str
    direction: Direction = "neutral"
    unit: str = "number"


KNOWN_METRICS: dict[str, MetricSpec] = {
    "classification.accuracy": MetricSpec(
        "classification.accuracy", direction="higher", unit="ratio"
    ),
    "classification.macro_f1": MetricSpec(
        "classification.macro_f1", direction="higher", unit="ratio"
    ),
    "calibration.brier_score": MetricSpec(
        "calibration.brier_score", direction="lower", unit="number"
    ),
    "calibration.log_loss": MetricSpec("calibration.log_loss", direction="lower", unit="number"),
    "calibration.expected_calibration_error": MetricSpec(
        "calibration.expected_calibration_error", direction="lower", unit="number"
    ),
    "selective.by_error_budget.1%.coverage": MetricSpec(
        "selective.by_error_budget.1%.coverage", direction="higher", unit="ratio"
    ),
    "selective.by_error_budget.5%.coverage": MetricSpec(
        "selective.by_error_budget.5%.coverage", direction="higher", unit="ratio"
    ),
    "selective.by_error_budget.10%.coverage": MetricSpec(
        "selective.by_error_budget.10%.coverage", direction="higher", unit="ratio"
    ),
    "latency.mean_ms": MetricSpec("latency.mean_ms", direction="lower", unit="ms"),
    "latency.cold_start_ms": MetricSpec("latency.cold_start_ms", direction="lower", unit="ms"),
    "latency.steady_state_mean_ms": MetricSpec(
        "latency.steady_state_mean_ms", direction="lower", unit="ms"
    ),
    "latency.min_ms": MetricSpec("latency.min_ms", direction="lower", unit="ms"),
    "latency.max_ms": MetricSpec("latency.max_ms", direction="lower", unit="ms"),
    "derived.throughput_cases_s": MetricSpec(
        "derived.throughput_cases_s", direction="higher", unit="cases/s"
    ),
    "derived.correct_cases_s": MetricSpec(
        "derived.correct_cases_s", direction="higher", unit="cases/s"
    ),
}

EXCLUDED_AGGREGATE_ROOTS = {
    "schema_version",
    "case_count",
    "results",
    "case_manifest",
    "candidate",
    "model_id",
    "dataset",
    "dataset_sha256",
    "semantic_profile",
    "evaluation_protocol",
    "provenance",
}

EXCLUDED_NUMERIC_SUFFIXES = {
    "evaluated",
    "case_count",
    "correct",
    "total_ms",
    "error_budget",
    "automated",
    "errors",
}


@dataclass(frozen=True)
class CaseInfo:
    case_id: str
    expected_label: str | None
    decision_type: str | None
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class RunArtifact:
    path: Path
    payload: dict[str, Any]
    run_id: str
    label: str
    candidate: str
    model_id: str
    dataset: str
    dataset_sha256: str | None
    schema_version: int | None
    semantic_profile: str | None
    evaluation_protocol: str | None
    case_manifest: dict[str, CaseInfo]
    results: dict[str, dict[str, Any]]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Compatibility:
    case_policy: CasePolicy
    aligned_case_ids: tuple[str, ...]
    relation: str
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class MetricComparison:
    path: str
    direction: Direction
    unit: str
    values: Mapping[str, float | None]
    deltas: Mapping[str, float | None]
    relative_deltas: Mapping[str, float | None]
    best_observed: tuple[str, ...]


@dataclass(frozen=True)
class PairwiseSummary:
    left: str
    right: str
    compared: int
    evaluated: int
    both_correct: int | None
    left_only_correct: int | None
    right_only_correct: int | None
    both_wrong: int | None
    prediction_changed: int
    top1_agreement: float
    probability_compared: int
    mean_total_variation: float | None
    max_abs_probability_delta: float | None
    mcnemar_exact_p: float | None


@dataclass
class ComparisonReport:
    inputs: list[dict[str, Any]]
    compatibility: dict[str, Any]
    metrics: list[dict[str, Any]]
    segments: dict[str, list[dict[str, Any]]]
    case_summary: list[dict[str, Any]]
    pairwise: list[dict[str, Any]]
    cases: list[dict[str, Any]]
    baseline: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "inputs": self.inputs,
            "compatibility": self.compatibility,
            "baseline": self.baseline,
            "metrics": self.metrics,
            "segments": self.segments,
            "case_summary": self.case_summary,
            "pairwise": self.pairwise,
            "cases": self.cases,
        }


def _expect_mapping(value: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ComparisonError(f"{where} must be a JSON object")
    return value


def _expect_results(value: Any, *, where: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ComparisonError(f"{where} must be a JSON array")
    out: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ComparisonError(f"{where}[{index}] must be a JSON object")
        out.append(item)
    return out


def _safe_read_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ComparisonError(f"cannot read report {path}: {exc}") from exc
    try:
        return _expect_mapping(json.loads(raw), where=str(path))
    except json.JSONDecodeError as exc:
        raise ComparisonError(f"invalid JSON in report {path}: {exc}") from exc


def _normalized_case_for_digest(case: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(case)
    normalized.setdefault("definitions", {})
    normalized.setdefault("tags", [])
    normalized.setdefault("metadata", {})

    decision_raw = normalized.get("decision")
    if isinstance(decision_raw, dict):
        decision = dict(decision_raw)
        decision.setdefault("options", [])
        normalized["decision"] = decision

    expected_raw = normalized.get("expected")
    if isinstance(expected_raw, dict):
        expected = dict(expected_raw)
        expected.setdefault("label", None)
        normalized["expected"] = expected
    return normalized


def _canonical_dataset_sha256(cases: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for case in cases:
        payload = json.dumps(
            _normalized_case_for_digest(case),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        stream = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ComparisonError(f"cannot read dataset {path}: {exc}") from exc
    with stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ComparisonError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ComparisonError(f"{path}:{line_number}: case must be a JSON object")
            rows.append(row)
    if not rows:
        raise ComparisonError(f"dataset {path} contains no cases")
    return rows


def _dataset_candidates(report_path: Path, dataset: str, dataset_root: Path | None) -> list[Path]:
    raw = Path(dataset)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        if dataset_root is not None:
            candidates.append(dataset_root / raw)
        candidates.append(Path.cwd() / raw)
        for parent in report_path.resolve().parents:
            candidates.append(parent / raw)

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def _manifest_from_dataset(
    report_path: Path,
    dataset: str,
    dataset_root: Path | None,
) -> tuple[dict[str, CaseInfo], str | None, str | None]:
    for candidate in _dataset_candidates(report_path, dataset, dataset_root):
        if not candidate.is_file():
            continue
        rows = _load_jsonl(candidate)
        manifest: dict[str, CaseInfo] = {}
        for row in rows:
            case_id = row.get("id")
            if not isinstance(case_id, str) or not case_id:
                raise ComparisonError(f"dataset {candidate} contains a case without a valid id")
            if case_id in manifest:
                raise ComparisonError(f"dataset {candidate} contains duplicate case id {case_id!r}")
            expected = row.get("expected")
            expected_label = expected.get("label") if isinstance(expected, dict) else None
            decision = row.get("decision")
            decision_type = decision.get("type") if isinstance(decision, dict) else None
            tags_raw = row.get("tags", [])
            tags = tuple(str(tag) for tag in tags_raw) if isinstance(tags_raw, list) else ()
            metadata_raw = row.get("metadata", {})
            metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
            manifest[case_id] = CaseInfo(
                case_id=case_id,
                expected_label=str(expected_label) if expected_label is not None else None,
                decision_type=str(decision_type) if decision_type is not None else None,
                tags=tags,
                metadata=metadata,
            )
        return manifest, _canonical_dataset_sha256(rows), str(candidate)
    return {}, None, None


def _manifest_from_report(payload: Mapping[str, Any]) -> dict[str, CaseInfo]:
    raw = payload.get("case_manifest")
    if not isinstance(raw, list):
        return {}
    manifest: dict[str, CaseInfo] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ComparisonError(f"case_manifest[{index}] must be a JSON object")
        case_id = item.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ComparisonError(f"case_manifest[{index}] has invalid case_id")
        if case_id in manifest:
            raise ComparisonError(f"duplicate case_manifest case_id: {case_id}")
        expected_label = item.get("expected_label")
        decision_type = item.get("decision_type")
        tags_raw = item.get("tags", [])
        metadata_raw = item.get("metadata", {})
        manifest[case_id] = CaseInfo(
            case_id=case_id,
            expected_label=str(expected_label) if expected_label is not None else None,
            decision_type=str(decision_type) if decision_type is not None else None,
            tags=tuple(str(tag) for tag in tags_raw) if isinstance(tags_raw, list) else (),
            metadata=metadata_raw if isinstance(metadata_raw, dict) else {},
        )
    return manifest


def _automatic_label(payload: Mapping[str, Any], ordinal: int) -> str:
    candidate = str(payload.get("candidate") or f"run-{ordinal}")
    model_id = str(payload.get("model_id") or "unknown-model")
    model_name = model_id.rsplit("/", maxsplit=1)[-1]
    semantic_profile = payload.get("semantic_profile")
    protocol = payload.get("evaluation_protocol")

    label = candidate if model_name.lower() in candidate.lower() else f"{candidate}:{model_name}"
    suffixes: list[str] = []
    if semantic_profile not in (None, "baseline"):
        suffixes.append(str(semantic_profile))
    if protocol not in (None, "rule-conditioned"):
        suffixes.append(str(protocol))
    if suffixes:
        label += "[" + "/".join(suffixes) + "]"
    return label


def _deduplicate_labels(labels: Sequence[str]) -> list[str]:
    totals: dict[str, int] = {}
    for label in labels:
        totals[label] = totals.get(label, 0) + 1
    counters: dict[str, int] = {}
    out: list[str] = []
    for label in labels:
        if totals[label] == 1:
            out.append(label)
            continue
        counters[label] = counters.get(label, 0) + 1
        out.append(f"{label}#{counters[label]}")
    return out


def load_runs(paths: Sequence[Path], *, dataset_root: Path | None = None) -> list[RunArtifact]:
    if len(paths) < 2:
        raise ComparisonError("at least two benchmark report paths are required")

    payloads = [_safe_read_json(path) for path in paths]
    labels = _deduplicate_labels(
        [_automatic_label(payload, ordinal) for ordinal, payload in enumerate(payloads, start=1)]
    )

    runs: list[RunArtifact] = []
    for ordinal, (path, payload, label) in enumerate(
        zip(paths, payloads, labels, strict=True), start=1
    ):
        candidate = str(payload.get("candidate") or f"run-{ordinal}")
        model_id = str(payload.get("model_id") or "unknown-model")
        dataset = str(payload.get("dataset") or "")
        if not dataset:
            raise ComparisonError(f"report {path} does not declare dataset")

        raw_results = _expect_results(payload.get("results"), where=f"{path}:results")
        results: dict[str, dict[str, Any]] = {}
        for result in raw_results:
            case_id = result.get("case_id")
            if not isinstance(case_id, str) or not case_id:
                raise ComparisonError(f"report {path} contains a result without a valid case_id")
            if case_id in results:
                raise ComparisonError(
                    f"report {path} contains duplicate result case_id {case_id!r}"
                )
            results[case_id] = result

        warnings: list[str] = []
        manifest = _manifest_from_report(payload)
        declared_sha = payload.get("dataset_sha256")
        dataset_sha = str(declared_sha) if isinstance(declared_sha, str) else None

        if not manifest:
            loaded_manifest, computed_sha, dataset_path = _manifest_from_dataset(
                path, dataset, dataset_root
            )
            manifest = loaded_manifest
            if manifest:
                warnings.append(
                    "legacy artifact: case manifest reconstructed from " + str(dataset_path)
                )
            else:
                warnings.append(
                    "legacy artifact: dataset could not be resolved; correctness and semantic "
                    "case compatibility are unavailable"
                )
            if dataset_sha is None:
                dataset_sha = computed_sha
        elif dataset_sha is None:
            warnings.append("artifact contains case_manifest but no dataset_sha256")

        missing_results = sorted(set(manifest) - set(results))
        if missing_results:
            warnings.append(f"{len(missing_results)} manifest case(s) have no persisted result")

        schema_version_raw = payload.get("schema_version")
        schema_version = schema_version_raw if isinstance(schema_version_raw, int) else None
        runs.append(
            RunArtifact(
                path=path,
                payload=payload,
                run_id=f"r{ordinal}",
                label=label,
                candidate=candidate,
                model_id=model_id,
                dataset=dataset,
                dataset_sha256=dataset_sha,
                schema_version=schema_version,
                semantic_profile=(
                    str(payload["semantic_profile"])
                    if payload.get("semantic_profile") is not None
                    else None
                ),
                evaluation_protocol=(
                    str(payload["evaluation_protocol"])
                    if payload.get("evaluation_protocol") is not None
                    else None
                ),
                case_manifest=manifest,
                results=results,
                warnings=warnings,
            )
        )
    return runs


def _resolve_baseline(runs: Sequence[RunArtifact], selector: str | None) -> RunArtifact | None:
    if selector is None:
        return None
    if selector.isdigit():
        index = int(selector)
        if 1 <= index <= len(runs):
            return runs[index - 1]
    matches = [
        run for run in runs if selector in {run.run_id, run.label, run.candidate, run.model_id}
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ComparisonError(f"baseline {selector!r} does not match any input run")
    raise ComparisonError(f"baseline {selector!r} is ambiguous; use the 1-based input index")


def validate_compatibility(
    runs: Sequence[RunArtifact], *, case_policy: CasePolicy
) -> Compatibility:
    result_sets = [set(run.results) for run in runs]
    if case_policy == "strict":
        reference = result_sets[0]
        for run, case_ids in zip(runs[1:], result_sets[1:], strict=True):
            if case_ids != reference:
                missing = sorted(reference - case_ids)
                extra = sorted(case_ids - reference)
                details: list[str] = []
                if missing:
                    details.append(f"missing={','.join(missing[:8])}")
                if extra:
                    details.append(f"extra={','.join(extra[:8])}")
                raise ComparisonError(
                    f"strict case alignment failed for {run.label}: " + "; ".join(details)
                )
        aligned = set(reference)
    else:
        aligned = set.intersection(*result_sets)
        if not aligned:
            raise ComparisonError("input reports have no common result case IDs")

    warnings: list[str] = []
    for run in runs:
        warnings.extend(f"{run.label}: {warning}" for warning in run.warnings)
    if case_policy == "intersection" and any(case_ids != aligned for case_ids in result_sets):
        warnings.append("intersection policy discarded cases not present in every report")

    manifests_available = all(run.case_manifest for run in runs)
    for case_id in sorted(aligned):
        infos = [run.case_manifest.get(case_id) for run in runs]
        concrete = [info for info in infos if info is not None]
        if not concrete:
            continue
        if len(concrete) != len(runs):
            warnings.append(
                f"case {case_id}: manifest missing in at least one report; "
                "semantic compatibility is only partially verified"
            )
        expected = {info.expected_label for info in concrete}
        if len(expected) > 1:
            raise ComparisonError(
                f"case {case_id}: expected labels disagree across reports: "
                f"{sorted(expected, key=str)}"
            )
        decision_types = {info.decision_type for info in concrete}
        if len(decision_types) > 1:
            raise ComparisonError(
                f"case {case_id}: decision types disagree across reports: "
                f"{sorted(decision_types, key=str)}"
            )

    dataset_hashes = [run.dataset_sha256 for run in runs]
    if all(value is not None for value in dataset_hashes) and len(set(dataset_hashes)) == 1:
        relation = "identical-dataset"
    elif manifests_available:
        relation = "aligned-dataset-variant"
        warnings.append(
            "dataset fingerprints differ; case IDs, expected labels and decision types are aligned"
        )
    else:
        relation = "unknown-dataset-relation"
        warnings.append(
            "dataset identity cannot be fully verified because at least one artifact lacks "
            "a manifest/fingerprint"
        )

    return Compatibility(
        case_policy=case_policy,
        aligned_case_ids=tuple(sorted(aligned)),
        relation=relation,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _flatten_numeric(
    value: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if not prefix and key in EXCLUDED_AGGREGATE_ROOTS:
            continue
        if not prefix and str(key).startswith("by_"):
            continue
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)):
            if str(key) not in EXCLUDED_NUMERIC_SUFFIXES:
                out[path] = float(item)
        elif isinstance(item, dict):
            out.update(_flatten_numeric(item, prefix=path))
    return out


def _metric_specs(
    runs: Sequence[RunArtifact],
    *,
    higher_is_better: Iterable[str],
    lower_is_better: Iterable[str],
) -> list[MetricSpec]:
    discovered: set[str] = set()
    for run in runs:
        discovered.update(_flatten_numeric(run.payload))
    overrides: dict[str, Direction] = {}
    for path in higher_is_better:
        overrides[path] = "higher"
    for path in lower_is_better:
        if path in overrides:
            raise ComparisonError(f"metric {path!r} declared both higher- and lower-is-better")
        overrides[path] = "lower"

    if any(
        _get_path(run.payload, "latency.steady_state_mean_ms") not in (None, 0.0) for run in runs
    ):
        discovered.add("derived.throughput_cases_s")
        if any(_get_path(run.payload, "classification.accuracy") is not None for run in runs):
            discovered.add("derived.correct_cases_s")

    specs: list[MetricSpec] = []
    for path in sorted(discovered, key=lambda item: (item not in KNOWN_METRICS, item)):
        base = KNOWN_METRICS.get(path, MetricSpec(path))
        direction = overrides.get(path, base.direction)
        specs.append(MetricSpec(path=path, direction=direction, unit=base.unit))
    return specs


def _get_path(payload: Mapping[str, Any], path: str) -> float | None:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        return None
    return float(current)


def _metric_value(payload: Mapping[str, Any], path: str) -> float | None:
    if path == "derived.throughput_cases_s":
        steady = _get_path(payload, "latency.steady_state_mean_ms")
        if steady in (None, 0.0):
            return None
        return 1000.0 / steady
    if path == "derived.correct_cases_s":
        steady = _get_path(payload, "latency.steady_state_mean_ms")
        accuracy = _get_path(payload, "classification.accuracy")
        if steady in (None, 0.0) or accuracy is None:
            return None
        return accuracy * 1000.0 / steady
    return _get_path(payload, path)


def _best_observed(values: Mapping[str, float | None], direction: Direction) -> tuple[str, ...]:
    concrete = {key: value for key, value in values.items() if value is not None}
    if not concrete or direction == "neutral":
        return ()
    target = max(concrete.values()) if direction == "higher" else min(concrete.values())
    return tuple(
        key
        for key, value in concrete.items()
        if math.isclose(value, target, rel_tol=1e-12, abs_tol=1e-12)
    )


def compare_metrics(
    runs: Sequence[RunArtifact],
    specs: Sequence[MetricSpec],
    baseline: RunArtifact | None,
) -> list[MetricComparison]:
    out: list[MetricComparison] = []
    for spec in specs:
        values = {run.run_id: _metric_value(run.payload, spec.path) for run in runs}
        baseline_value = values.get(baseline.run_id) if baseline is not None else None
        deltas: dict[str, float | None] = {}
        relative: dict[str, float | None] = {}
        for run in runs:
            value = values[run.run_id]
            if baseline is None or value is None or baseline_value is None:
                deltas[run.run_id] = None
                relative[run.run_id] = None
                continue
            delta = value - baseline_value
            deltas[run.run_id] = delta
            relative[run.run_id] = delta / abs(baseline_value) if baseline_value != 0 else None
        out.append(
            MetricComparison(
                path=spec.path,
                direction=spec.direction,
                unit=spec.unit,
                values=values,
                deltas=deltas,
                relative_deltas=relative,
                best_observed=_best_observed(values, spec.direction),
            )
        )
    return out


def compare_segments(runs: Sequence[RunArtifact]) -> dict[str, list[dict[str, Any]]]:
    families = sorted(
        {
            key
            for run in runs
            for key, value in run.payload.items()
            if key.startswith("by_") and isinstance(value, dict)
        }
    )
    out: dict[str, list[dict[str, Any]]] = {}
    for family in families:
        names = sorted(
            {
                name
                for run in runs
                for name in (
                    run.payload.get(family, {}).keys()
                    if isinstance(run.payload.get(family), dict)
                    else []
                )
            }
        )
        rows: list[dict[str, Any]] = []
        for name in names:
            row: dict[str, Any] = {"segment": name}
            for run in runs:
                family_payload = run.payload.get(family)
                segment = family_payload.get(name) if isinstance(family_payload, dict) else None
                if isinstance(segment, dict):
                    row[run.run_id] = {
                        key: value
                        for key, value in segment.items()
                        if isinstance(value, (int, float)) and not isinstance(value, bool)
                    }
                else:
                    row[run.run_id] = None
            rows.append(row)
        out[family] = rows
    return out


def _expected_for_case(runs: Sequence[RunArtifact], case_id: str) -> str | None:
    for run in runs:
        info = run.case_manifest.get(case_id)
        if info is not None and info.expected_label is not None:
            return info.expected_label
    return None


def _mcnemar_exact_p(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    k = min(left_only, right_only)
    tail = sum(math.comb(discordant, i) for i in range(k + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def compare_cases(
    runs: Sequence[RunArtifact],
    compatibility: Compatibility,
    *,
    include_cases: bool,
) -> tuple[list[dict[str, Any]], list[PairwiseSummary], list[dict[str, Any]]]:
    aligned = compatibility.aligned_case_ids
    summaries: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []

    for run in runs:
        evaluated = correct = 0
        for case_id in aligned:
            expected = _expected_for_case(runs, case_id)
            if expected is None:
                continue
            evaluated += 1
            correct += int(str(run.results[case_id].get("label")) == expected)
        summaries.append(
            {
                "run_id": run.run_id,
                "evaluated": evaluated,
                "correct": correct if evaluated else None,
                "accuracy": (correct / evaluated) if evaluated else None,
            }
        )

    if include_cases:
        for case_id in aligned:
            expected = _expected_for_case(runs, case_id)
            row: dict[str, Any] = {"case_id": case_id, "expected": expected}
            for run in runs:
                result = run.results[case_id]
                label = result.get("label")
                probabilities = result.get("probabilities")
                selection_probability = (
                    max(float(value) for value in probabilities.values())
                    if isinstance(probabilities, dict) and probabilities
                    else None
                )
                row[run.run_id] = {
                    "label": label,
                    "correct": (str(label) == expected) if expected is not None else None,
                    "selection_probability": selection_probability,
                    "provider_confidence": result.get("confidence"),
                    "latency_ms": result.get("latency_ms"),
                }
            case_rows.append(row)

    pairwise: list[PairwiseSummary] = []
    for left, right in combinations(runs, 2):
        both_correct = left_only = right_only = both_wrong = 0
        changed = 0
        compared = 0
        evaluated = 0
        probability_compared = 0
        total_variation_sum = 0.0
        max_probability_delta: float | None = None
        for case_id in aligned:
            left_label = str(left.results[case_id].get("label"))
            right_label = str(right.results[case_id].get("label"))
            compared += 1
            changed += int(left_label != right_label)

            left_probabilities = left.results[case_id].get("probabilities")
            right_probabilities = right.results[case_id].get("probabilities")
            if (
                isinstance(left_probabilities, dict)
                and left_probabilities
                and isinstance(right_probabilities, dict)
                and right_probabilities
            ):
                labels = set(left_probabilities) | set(right_probabilities)
                deltas = [
                    abs(
                        float(left_probabilities.get(label, 0.0))
                        - float(right_probabilities.get(label, 0.0))
                    )
                    for label in labels
                ]
                probability_compared += 1
                total_variation_sum += 0.5 * sum(deltas)
                case_max = max(deltas, default=0.0)
                max_probability_delta = (
                    case_max
                    if max_probability_delta is None
                    else max(max_probability_delta, case_max)
                )

            expected = _expected_for_case(runs, case_id)
            if expected is None:
                continue
            evaluated += 1
            left_ok = left_label == expected
            right_ok = right_label == expected
            if left_ok and right_ok:
                both_correct += 1
            elif left_ok:
                left_only += 1
            elif right_ok:
                right_only += 1
            else:
                both_wrong += 1
        pairwise.append(
            PairwiseSummary(
                left=left.run_id,
                right=right.run_id,
                compared=compared,
                evaluated=evaluated,
                both_correct=both_correct if evaluated else None,
                left_only_correct=left_only if evaluated else None,
                right_only_correct=right_only if evaluated else None,
                both_wrong=both_wrong if evaluated else None,
                prediction_changed=changed,
                top1_agreement=(compared - changed) / compared if compared else 0.0,
                probability_compared=probability_compared,
                mean_total_variation=(
                    total_variation_sum / probability_compared if probability_compared else None
                ),
                max_abs_probability_delta=max_probability_delta,
                mcnemar_exact_p=(_mcnemar_exact_p(left_only, right_only) if evaluated else None),
            )
        )

    return summaries, pairwise, case_rows


def build_comparison(
    runs: Sequence[RunArtifact],
    *,
    baseline: RunArtifact | None,
    case_policy: CasePolicy,
    include_cases: bool,
    higher_is_better: Iterable[str] = (),
    lower_is_better: Iterable[str] = (),
) -> ComparisonReport:
    compatibility = validate_compatibility(runs, case_policy=case_policy)
    specs = _metric_specs(
        runs,
        higher_is_better=higher_is_better,
        lower_is_better=lower_is_better,
    )
    metrics = compare_metrics(runs, specs, baseline)
    segments = compare_segments(runs)
    case_summary, pairwise, cases = compare_cases(runs, compatibility, include_cases=include_cases)

    return ComparisonReport(
        inputs=[
            {
                "run_id": run.run_id,
                "label": run.label,
                "path": str(run.path),
                "candidate": run.candidate,
                "model_id": run.model_id,
                "dataset": run.dataset,
                "dataset_sha256": run.dataset_sha256,
                "schema_version": run.schema_version,
                "semantic_profile": run.semantic_profile,
                "evaluation_protocol": run.evaluation_protocol,
            }
            for run in runs
        ],
        compatibility={
            "case_policy": compatibility.case_policy,
            "relation": compatibility.relation,
            "aligned_case_count": len(compatibility.aligned_case_ids),
            "warnings": list(compatibility.warnings),
        },
        metrics=[
            {
                "path": metric.path,
                "direction": metric.direction,
                "unit": metric.unit,
                "values": dict(metric.values),
                "deltas": dict(metric.deltas),
                "relative_deltas": dict(metric.relative_deltas),
                "best_observed": list(metric.best_observed),
            }
            for metric in metrics
        ],
        segments=segments,
        case_summary=case_summary,
        pairwise=[
            {
                "left": row.left,
                "right": row.right,
                "compared": row.compared,
                "evaluated": row.evaluated,
                "both_correct": row.both_correct,
                "left_only_correct": row.left_only_correct,
                "right_only_correct": row.right_only_correct,
                "both_wrong": row.both_wrong,
                "prediction_changed": row.prediction_changed,
                "top1_agreement": row.top1_agreement,
                "probability_compared": row.probability_compared,
                "mean_total_variation": row.mean_total_variation,
                "max_abs_probability_delta": row.max_abs_probability_delta,
                "mcnemar_exact_p": row.mcnemar_exact_p,
            }
            for row in pairwise
        ],
        cases=cases,
        baseline=baseline.run_id if baseline is not None else None,
    )


def _run_labels(report: ComparisonReport) -> dict[str, str]:
    return {str(item["run_id"]): str(item["label"]) for item in report.inputs}


def _format_number(value: Any, *, unit: str = "number") -> str:
    if value is None:
        return "-"
    number = float(value)
    if unit == "ratio":
        return f"{number * 100:.2f}%"
    if unit == "ms":
        return f"{number:.3f} ms"
    if unit == "cases/s":
        return f"{number:.3f} cases/s"
    return f"{number:.6f}"


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    head = "| " + " | ".join(_markdown_escape(item) for item in headers) + " |"
    sep = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(_markdown_escape(item) for item in row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def render_markdown(report: ComparisonReport) -> str:
    labels = _run_labels(report)
    parts = ["# Benchmark comparison", ""]
    input_rows = [
        [
            item["run_id"],
            item["label"],
            item["candidate"],
            item["model_id"],
            item["dataset"],
        ]
        for item in report.inputs
    ]
    parts.extend(
        [
            "## Inputs",
            "",
            _markdown_table(["run", "label", "candidate", "model", "dataset"], input_rows),
            "",
            "## Compatibility",
            "",
            f"- relation: `{report.compatibility['relation']}`",
            f"- case policy: `{report.compatibility['case_policy']}`",
            f"- aligned cases: {report.compatibility['aligned_case_count']}",
        ]
    )
    if report.baseline is not None:
        parts.append(f"- baseline: `{labels[report.baseline]}` ({report.baseline})")
    for warning in report.compatibility.get("warnings", []):
        parts.append(f"- warning: {warning}")

    metric_headers = ["metric", "direction", *[labels[item["run_id"]] for item in report.inputs]]
    metric_rows: list[list[Any]] = []
    for metric in report.metrics:
        row: list[Any] = [metric["path"], metric["direction"]]
        for item in report.inputs:
            run_id = item["run_id"]
            rendered = _format_number(metric["values"].get(run_id), unit=metric["unit"])
            if run_id in metric["best_observed"]:
                rendered += " *"
            if report.baseline is not None and run_id != report.baseline:
                delta = metric["deltas"].get(run_id)
                if delta is not None:
                    rendered += f" (Δ {delta:+.6f})"
            row.append(rendered)
        metric_rows.append(row)
    parts.extend(["", "## Aggregate metrics", "", _markdown_table(metric_headers, metric_rows)])
    parts.append("\n`*` marks the best observed value only for metrics with a declared direction.")

    for family, rows in report.segments.items():
        segment_rows: list[list[Any]] = []
        for item in rows:
            row = [item["segment"]]
            for run in report.inputs:
                values = item.get(run["run_id"])
                if not isinstance(values, dict):
                    row.append("-")
                    continue
                if "accuracy" in values:
                    count = values.get("case_count")
                    count_suffix = f" (n={int(count)})" if count is not None else ""
                    row.append(f"{float(values['accuracy']) * 100:.2f}%{count_suffix}")
                else:
                    row.append(json.dumps(values, sort_keys=True))
            segment_rows.append(row)
        parts.extend(
            [
                "",
                f"## {family}",
                "",
                _markdown_table(
                    ["segment", *[labels[item["run_id"]] for item in report.inputs]],
                    segment_rows,
                ),
            ]
        )

    if report.case_summary:
        rows = [
            [
                labels[item["run_id"]],
                item["evaluated"],
                item["correct"] if item["correct"] is not None else "-",
                _format_number(item["accuracy"], unit="ratio"),
            ]
            for item in report.case_summary
        ]
        parts.extend(
            [
                "",
                "## Case-level summary",
                "",
                _markdown_table(["run", "evaluated", "correct", "accuracy"], rows),
            ]
        )

    if report.pairwise:
        rows = []
        for item in report.pairwise:
            rows.append(
                [
                    labels[item["left"]],
                    labels[item["right"]],
                    item["compared"],
                    item["evaluated"],
                    item["left_only_correct"] if item["left_only_correct"] is not None else "-",
                    item["right_only_correct"] if item["right_only_correct"] is not None else "-",
                    item["prediction_changed"],
                    _format_number(item["top1_agreement"], unit="ratio"),
                    item["probability_compared"],
                    _format_number(item["mean_total_variation"]),
                    _format_number(item["max_abs_probability_delta"]),
                    (
                        f"{float(item['mcnemar_exact_p']):.6f}"
                        if item["mcnemar_exact_p"] is not None
                        else "-"
                    ),
                ]
            )
        parts.extend(
            [
                "",
                "## Pairwise diagnostics",
                "",
                _markdown_table(
                    [
                        "left",
                        "right",
                        "cases",
                        "labeled",
                        "left-only correct",
                        "right-only correct",
                        "prediction changed",
                        "top-1 agreement",
                        "prob. cases",
                        "mean TV",
                        "max |Δp|",
                        "McNemar exact p",
                    ],
                    rows,
                ),
            ]
        )

    if report.cases:
        headers = ["case", "expected", *[labels[item["run_id"]] for item in report.inputs]]
        rows = []
        for item in report.cases:
            row: list[Any] = [item["case_id"], item["expected"] or "-"]
            for run in report.inputs:
                result = item[run["run_id"]]
                value = str(result["label"])
                if result["correct"] is True:
                    value += " ✓"
                elif result["correct"] is False:
                    value += " ✗"
                row.append(value)
            rows.append(row)
        parts.extend(["", "## Cases", "", _markdown_table(headers, rows)])

    return "\n".join(parts).rstrip() + "\n"


def render_human(report: ComparisonReport) -> str:
    markdown = render_markdown(report)
    lines: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("# "):
            lines.append(line[2:].upper())
        elif line.startswith("## "):
            lines.extend(["", line[3:].upper()])
        elif line.startswith("|") or line.startswith("- "):
            lines.append(line)
        elif line.startswith("`"):
            lines.append(line.replace("`", ""))
        elif line:
            lines.append(line)
    return "\n".join(lines).strip() + "\n"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def export_report(report: ComparisonReport, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    (outdir / "comparison.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (outdir / "comparison.md").write_text(render_markdown(report), encoding="utf-8")

    metric_rows: list[dict[str, Any]] = []
    for metric in report.metrics:
        row: dict[str, Any] = {
            "metric": metric["path"],
            "direction": metric["direction"],
            "unit": metric["unit"],
        }
        for run_id, value in metric["values"].items():
            row[run_id] = value
        metric_rows.append(row)
    _write_csv(outdir / "metrics.csv", metric_rows)
    _write_csv(outdir / "pairwise.csv", report.pairwise)
    _write_csv(outdir / "case_summary.csv", report.case_summary)
    if report.cases:
        _write_csv(outdir / "cases.csv", report.cases)
    for family, rows in report.segments.items():
        safe_name = family.replace("/", "_").replace("..", "_")
        _write_csv(outdir / f"{safe_name}.csv", rows)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two or more normalized Decision Model Lab benchmark reports."
    )
    parser.add_argument("reports", nargs="+", type=Path, help="Benchmark report JSON files.")
    parser.add_argument(
        "--baseline",
        help="Optional baseline: 1-based input index, run id, label, candidate or model id.",
    )
    parser.add_argument(
        "--case-policy",
        choices=("strict", "intersection"),
        default="strict",
        help="How result case IDs are aligned across reports.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Root used to resolve dataset paths referenced by legacy artifacts.",
    )
    parser.add_argument(
        "--include-cases",
        action="store_true",
        help="Include the aligned per-case table in rendered/exported output.",
    )
    parser.add_argument(
        "--higher-is-better",
        action="append",
        default=[],
        metavar="METRIC_PATH",
        help="Declare an auto-discovered numeric metric as higher-is-better. Repeatable.",
    )
    parser.add_argument(
        "--lower-is-better",
        action="append",
        default=[],
        metavar="METRIC_PATH",
        help="Declare an auto-discovered numeric metric as lower-is-better. Repeatable.",
    )
    parser.add_argument(
        "--format",
        choices=("human", "markdown", "json"),
        default="human",
        help="Primary output format.",
    )
    parser.add_argument("--output", type=Path, help="Write primary output to a file.")
    parser.add_argument(
        "--export-dir",
        type=Path,
        help="Export canonical JSON, Markdown and CSV tables to a directory.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        runs = load_runs(args.reports, dataset_root=args.dataset_root)
        baseline = _resolve_baseline(runs, args.baseline)
        report = build_comparison(
            runs,
            baseline=baseline,
            case_policy=args.case_policy,
            include_cases=args.include_cases,
            higher_is_better=args.higher_is_better,
            lower_is_better=args.lower_is_better,
        )
    except ComparisonError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        rendered = json.dumps(report.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    elif args.format == "markdown":
        rendered = render_markdown(report)
    else:
        rendered = render_human(report)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)

    if args.export_dir is not None:
        export_report(report, args.export_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.compare_benchmarks import (
    ComparisonError,
    build_comparison,
    export_report,
    load_runs,
    render_markdown,
)


def _case_manifest(case_id: str, expected: str, *, tag: str = "core") -> dict[str, object]:
    return {
        "case_id": case_id,
        "expected_label": expected,
        "decision_type": "choice",
        "tags": [tag],
        "metadata": {"axis": "synthetic"},
    }


def _result(case_id: str, label: str, latency_ms: float) -> dict[str, object]:
    return {
        "case_id": case_id,
        "label": label,
        "probabilities": {"yes": 0.8, "no": 0.2} if label == "yes" else {"yes": 0.2, "no": 0.8},
        "confidence": 0.8,
        "raw_output": None,
        "latency_ms": latency_ms,
        "metadata": {},
    }


def _write_report(
    path: Path,
    *,
    candidate: str,
    accuracy: float,
    labels: tuple[str, str] = ("yes", "no"),
    dataset_sha256: str = "a" * 64,
    include_manifest: bool = True,
    dataset: str = "datasets/synthetic.jsonl",
) -> None:
    payload: dict[str, object] = {
        "schema_version": 1,
        "dataset": dataset,
        "dataset_sha256": dataset_sha256,
        "candidate": candidate,
        "model_id": f"org/{candidate}",
        "semantic_profile": "baseline",
        "evaluation_protocol": "rule-conditioned",
        "case_count": 2,
        "classification": {"evaluated": 2, "accuracy": accuracy, "macro_f1": accuracy},
        "calibration": {
            "evaluated": 2,
            "brier_score": 1.0 - accuracy,
            "expected_calibration_error": (1.0 - accuracy) / 2,
        },
        "by_decision_type": {
            "choice": {"case_count": 2, "correct": round(accuracy * 2), "accuracy": accuracy}
        },
        "by_tag": {"core": {"case_count": 2, "correct": round(accuracy * 2), "accuracy": accuracy}},
        "latency": {
            "total_ms": 12.0,
            "mean_ms": 6.0,
            "min_ms": 5.0,
            "max_ms": 7.0,
            "cold_start_ms": 7.0,
            "steady_state_mean_ms": 5.0,
        },
        "results": [
            _result("case-a", labels[0], 7.0),
            _result("case-b", labels[1], 5.0),
        ],
    }
    if include_manifest:
        payload["case_manifest"] = [
            _case_manifest("case-a", "yes"),
            _case_manifest("case-b", "no"),
        ]
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_n_way_comparison_is_candidate_agnostic_and_tracks_best_observed(tmp_path: Path) -> None:
    paths = [tmp_path / f"run-{index}.json" for index in range(3)]
    _write_report(paths[0], candidate="alpha", accuracy=0.50, labels=("yes", "yes"))
    _write_report(paths[1], candidate="beta", accuracy=1.00)
    _write_report(paths[2], candidate="gamma", accuracy=0.50, labels=("no", "no"))

    runs = load_runs(paths)
    report = build_comparison(
        runs,
        baseline=runs[0],
        case_policy="strict",
        include_cases=True,
    )

    accuracy = next(item for item in report.metrics if item["path"] == "classification.accuracy")
    assert accuracy["values"] == {"r1": 0.5, "r2": 1.0, "r3": 0.5}
    assert accuracy["deltas"]["r2"] == pytest.approx(0.5)
    assert accuracy["best_observed"] == ["r2"]
    throughput = next(
        item for item in report.metrics if item["path"] == "derived.throughput_cases_s"
    )
    assert throughput["values"]["r1"] == pytest.approx(200.0)
    assert report.compatibility["relation"] == "identical-dataset"
    assert len(report.pairwise) == 3
    assert len(report.cases) == 2


def test_dataset_variants_are_allowed_when_case_semantics_align(tmp_path: Path) -> None:
    left = tmp_path / "en.json"
    right = tmp_path / "pt.json"
    _write_report(left, candidate="alpha", accuracy=1.0, dataset_sha256="a" * 64)
    _write_report(right, candidate="alpha", accuracy=1.0, dataset_sha256="b" * 64)

    runs = load_runs([left, right])
    report = build_comparison(
        runs,
        baseline=None,
        case_policy="strict",
        include_cases=False,
    )

    assert report.compatibility["relation"] == "aligned-dataset-variant"
    assert any("fingerprints differ" in warning for warning in report.compatibility["warnings"])


def test_strict_policy_rejects_missing_result_case(tmp_path: Path) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    _write_report(left, candidate="alpha", accuracy=1.0)
    _write_report(right, candidate="beta", accuracy=1.0)

    payload = json.loads(right.read_text(encoding="utf-8"))
    payload["results"] = payload["results"][:1]
    right.write_text(json.dumps(payload), encoding="utf-8")

    runs = load_runs([left, right])
    with pytest.raises(ComparisonError, match="strict case alignment failed"):
        build_comparison(
            runs,
            baseline=None,
            case_policy="strict",
            include_cases=False,
        )


def test_legacy_artifact_reconstructs_manifest_from_dataset(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    dataset = dataset_dir / "synthetic.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "case-a",
                        "context": "A",
                        "question": "Q?",
                        "decision": {"type": "choice", "options": ["yes", "no"]},
                        "expected": {"label": "yes"},
                        "tags": ["core"],
                    }
                ),
                json.dumps(
                    {
                        "id": "case-b",
                        "context": "B",
                        "question": "Q?",
                        "decision": {"type": "choice", "options": ["yes", "no"]},
                        "expected": {"label": "no"},
                        "tags": ["core"],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    left = tmp_path / "legacy-a.json"
    right = tmp_path / "legacy-b.json"
    _write_report(
        left,
        candidate="alpha",
        accuracy=1.0,
        include_manifest=False,
        dataset="datasets/synthetic.jsonl",
    )
    _write_report(
        right,
        candidate="beta",
        accuracy=1.0,
        include_manifest=False,
        dataset="datasets/synthetic.jsonl",
    )
    for path in (left, right):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.pop("schema_version")
        payload.pop("dataset_sha256")
        path.write_text(json.dumps(payload), encoding="utf-8")

    runs = load_runs([left, right], dataset_root=tmp_path)

    assert runs[0].case_manifest["case-a"].expected_label == "yes"
    assert runs[0].dataset_sha256 is not None
    assert any("legacy artifact" in warning for warning in runs[0].warnings)


def test_mcnemar_pairwise_uses_discordant_correctness_counts(tmp_path: Path) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    _write_report(left, candidate="alpha", accuracy=0.5, labels=("yes", "yes"))
    _write_report(right, candidate="beta", accuracy=0.5, labels=("no", "no"))

    report = build_comparison(
        load_runs([left, right]),
        baseline=None,
        case_policy="strict",
        include_cases=False,
    )
    pair = report.pairwise[0]

    assert pair["left_only_correct"] == 1
    assert pair["right_only_correct"] == 1
    assert pair["evaluated"] == 2
    assert pair["prediction_changed"] == 2
    assert pair["mcnemar_exact_p"] == pytest.approx(1.0)


def test_export_writes_canonical_and_tabular_outputs(tmp_path: Path) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    _write_report(left, candidate="alpha", accuracy=0.5, labels=("yes", "yes"))
    _write_report(right, candidate="beta", accuracy=1.0)
    report = build_comparison(
        load_runs([left, right]),
        baseline=None,
        case_policy="strict",
        include_cases=True,
    )

    outdir = tmp_path / "comparison"
    export_report(report, outdir)

    assert (outdir / "comparison.json").is_file()
    assert (outdir / "comparison.md").is_file()
    assert (outdir / "metrics.csv").is_file()
    assert (outdir / "pairwise.csv").is_file()
    assert (outdir / "cases.csv").is_file()
    assert (outdir / "by_tag.csv").is_file()
    assert "Aggregate metrics" in render_markdown(report)


def test_pairwise_probability_preservation_metrics_are_reported(tmp_path: Path) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    _write_report(left, candidate="fp32", accuracy=1.0)
    _write_report(right, candidate="quant", accuracy=1.0)

    payload = json.loads(right.read_text(encoding="utf-8"))
    payload["results"][0]["probabilities"] = {"yes": 0.7, "no": 0.3}
    payload["results"][1]["probabilities"] = {"yes": 0.1, "no": 0.9}
    right.write_text(json.dumps(payload), encoding="utf-8")

    runs = load_runs([left, right])
    report = build_comparison(
        runs,
        baseline=runs[0],
        case_policy="strict",
        include_cases=True,
    )

    pair = report.pairwise[0]
    assert pair["top1_agreement"] == pytest.approx(1.0)
    assert pair["probability_compared"] == 2
    assert pair["mean_total_variation"] == pytest.approx(0.1)
    assert pair["max_abs_probability_delta"] == pytest.approx(0.1)
    assert report.cases[0]["r1"]["selection_probability"] == pytest.approx(0.8)
    assert report.cases[0]["r1"]["provider_confidence"] == pytest.approx(0.8)

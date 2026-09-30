from pathlib import Path

import pytest

from decision_model_lab.dataset import DatasetError, load_jsonl


def test_smoke_dataset_is_valid() -> None:
    cases = load_jsonl(Path("datasets/smoke.jsonl"))
    assert len(cases) == 2
    assert {case.id for case in cases} == {"binary-001", "insufficient-001"}


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    dataset = tmp_path / "duplicate.jsonl"
    row = (
        '{"id":"same","context":"c","question":"q","decision":{"type":"noul"},'
        '"expected":{"label":"yes"}}\n'
    )
    dataset.write_text(row + row, encoding="utf-8")

    with pytest.raises(DatasetError, match="duplicate case id"):
        load_jsonl(dataset)


def test_expected_label_must_belong_to_declared_output_domain(tmp_path: Path) -> None:
    dataset = tmp_path / "invalid-expected.jsonl"
    dataset.write_text(
        '{"id":"bad","context":"c","question":"q",'
        '"decision":{"type":"choice","options":["a","b"]},'
        '"expected":{"label":"outside"}}\n',
        encoding="utf-8",
    )

    with pytest.raises(DatasetError, match="outside decision output domain"):
        load_jsonl(dataset)


def _definition_strings(value: object) -> list[str]:
    if isinstance(value, dict):
        strings: list[str] = []
        for child in value.values():
            strings.extend(_definition_strings(child))
        return strings
    if isinstance(value, list):
        strings = []
        for child in value:
            strings.extend(_definition_strings(child))
        return strings
    return [value] if isinstance(value, str) else []


def test_benchmark_v2_language_variants_preserve_structural_parity() -> None:
    english = load_jsonl(Path("datasets/benchmark-v2.jsonl"))
    portuguese = load_jsonl(Path("datasets/benchmark-v2-pt-br.jsonl"))

    assert len(english) == len(portuguese) == 57
    assert [case.id for case in english] == [case.id for case in portuguese]

    for en_case, pt_case in zip(english, portuguese, strict=True):
        assert en_case.decision == pt_case.decision
        assert en_case.expected == pt_case.expected
        assert en_case.tags == pt_case.tags

        pt_metadata = dict(pt_case.metadata)
        assert pt_metadata.pop("language") == "pt-BR"
        assert pt_metadata == en_case.metadata


def test_benchmark_v2_pt_br_localizes_all_model_visible_definition_text() -> None:
    english = load_jsonl(Path("datasets/benchmark-v2.jsonl"))
    portuguese = load_jsonl(Path("datasets/benchmark-v2-pt-br.jsonl"))

    for en_case, pt_case in zip(english, portuguese, strict=True):
        assert en_case.context != pt_case.context
        assert en_case.question != pt_case.question

        en_strings = _definition_strings(en_case.definitions)
        pt_strings = _definition_strings(pt_case.definitions)
        assert len(en_strings) == len(pt_strings)
        assert all(
            en_value != pt_value for en_value, pt_value in zip(en_strings, pt_strings, strict=True)
        )


def test_benchmark_v2_boundary_oracle_respects_declared_inequality() -> None:
    for path in (
        Path("datasets/benchmark-v2.jsonl"),
        Path("datasets/benchmark-v2-pt-br.jsonl"),
    ):
        cases = load_jsonl(path)
        case = next(case for case in cases if case.id == "v2-impact-latency-005")

        assert case.expected.label == "normal"


def test_benchmark_v2_impossible_control_declares_abstention_in_output_domain() -> None:
    for path in (
        Path("datasets/benchmark-v2.jsonl"),
        Path("datasets/benchmark-v2-pt-br.jsonl"),
    ):
        cases = load_jsonl(path)
        case = next(case for case in cases if case.id == "v2-control-impossible-001")

        assert case.expected.label == "insufficient_evidence"
        assert "insufficient_evidence" in case.decision.options


def test_benchmark_v1_covers_initial_decision_taxonomy() -> None:
    cases = load_jsonl(Path("datasets/benchmark-v1.jsonl"))
    tags = {tag for case in cases for tag in case.tags}

    assert len(cases) == 12
    assert {"binary", "multiclass", "abstention", "contradiction"} <= tags


def test_benchmark_v2_preserves_explicit_definitions() -> None:
    cases = load_jsonl(Path("datasets/benchmark-v2.jsonl"))
    case = next(case for case in cases if case.id == "v2-impact-latency-001")

    assert case.definitions["impact_levels"]["normal"] == (
        "All indicators at baseline or inside SLO."
    )

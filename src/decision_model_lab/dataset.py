import json
from pathlib import Path

from pydantic import ValidationError

from decision_model_lab.schema import EvaluationCase


def _expected_output_domain(case: EvaluationCase) -> set[str]:
    if case.decision.type == "noul":
        return {"yes", "no"}
    return set(case.decision.options)


def _validate_expected_label(case: EvaluationCase, *, path: Path, line_number: int) -> None:
    label = case.expected.label
    if label is None:
        return

    output_domain = _expected_output_domain(case)
    if label not in output_domain:
        allowed = ", ".join(sorted(output_domain))
        raise DatasetError(
            f"{path}:{line_number}: expected label {label!r} is outside decision output "
            f"domain: {allowed}"
        )


class DatasetError(ValueError):
    pass


def load_jsonl(path: Path) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
                case = EvaluationCase.model_validate(payload)
            except (json.JSONDecodeError, ValidationError) as exc:
                raise DatasetError(f"{path}:{line_number}: invalid evaluation case: {exc}") from exc

            _validate_expected_label(case, path=path, line_number=line_number)

            if case.id in seen_ids:
                raise DatasetError(f"{path}:{line_number}: duplicate case id: {case.id}")

            seen_ids.add(case.id)
            cases.append(case)

    if not cases:
        raise DatasetError(f"{path}: dataset contains no evaluation cases")

    return cases

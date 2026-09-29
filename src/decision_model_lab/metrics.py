import math
from dataclasses import dataclass

from decision_model_lab.schema import DecisionResult, EvaluationCase

AUTOMATION_ERROR_BUDGETS: tuple[float, ...] = (0.01, 0.05, 0.10)
_LOG_EPSILON = 1e-12


@dataclass(frozen=True)
class ClassificationMetrics:
    evaluated: int
    accuracy: float
    macro_f1: float


@dataclass(frozen=True)
class CalibrationMetrics:
    evaluated: int
    brier_score: float
    log_loss: float
    expected_calibration_error: float


@dataclass(frozen=True)
class AutomationPoint:
    error_budget: float
    coverage: float
    threshold: float | None
    automated: int
    errors: int
    empirical_error_rate: float | None


@dataclass(frozen=True)
class SelectiveMetrics:
    evaluated: int
    by_error_budget: dict[str, AutomationPoint]


def _paired_labeled_cases(
    cases: list[EvaluationCase], results: list[DecisionResult]
) -> list[tuple[EvaluationCase, DecisionResult]]:
    results_by_id = {result.case_id: result for result in results}
    if len(results_by_id) != len(results):
        raise ValueError("duplicate result case_id")

    pairs: list[tuple[EvaluationCase, DecisionResult]] = []
    for case in cases:
        if case.expected.label is None:
            continue
        result = results_by_id.get(case.id)
        if result is not None:
            pairs.append((case, result))
    return pairs


def selection_probability(result: DecisionResult) -> float | None:
    """Return the probability of the selected class for abstention/gating analysis.

    Provider-specific ``confidence`` fields are deliberately ignored. Selective
    classification thresholds must operate on the normalized probability mass used
    to select the label so the metric has the same semantics across candidates.
    """
    if not result.probabilities:
        return None
    return max(result.probabilities.values())


def classification_metrics(
    cases: list[EvaluationCase], results: list[DecisionResult]
) -> ClassificationMetrics:
    pairs = _paired_labeled_cases(cases, results)
    if not pairs:
        raise ValueError("no labeled cases with results")

    expected = [case.expected.label for case, _ in pairs]
    predicted = [result.label for _, result in pairs]
    correct = sum(truth == guess for truth, guess in zip(expected, predicted, strict=True))

    labels = sorted({label for label in expected + predicted if label is not None})
    f1_scores: list[float] = []
    for label in labels:
        true_positive = sum(
            truth == label and guess == label
            for truth, guess in zip(expected, predicted, strict=True)
        )
        false_positive = sum(
            truth != label and guess == label
            for truth, guess in zip(expected, predicted, strict=True)
        )
        false_negative = sum(
            truth == label and guess != label
            for truth, guess in zip(expected, predicted, strict=True)
        )
        denominator = (2 * true_positive) + false_positive + false_negative
        f1_scores.append((2 * true_positive / denominator) if denominator else 0.0)

    return ClassificationMetrics(
        evaluated=len(pairs),
        accuracy=correct / len(pairs),
        macro_f1=sum(f1_scores) / len(f1_scores),
    )


def calibration_metrics(
    cases: list[EvaluationCase],
    results: list[DecisionResult],
    *,
    bins: int = 10,
) -> CalibrationMetrics:
    if bins <= 0:
        raise ValueError("bins must be greater than zero")

    pairs = [
        (case, result)
        for case, result in _paired_labeled_cases(cases, results)
        if result.probabilities
    ]
    if not pairs:
        raise ValueError("no labeled cases with probability distributions")

    labels = sorted(
        {
            label
            for case, result in pairs
            for label in ({case.expected.label} | set(result.probabilities))
            if label is not None
        }
    )

    brier_total = 0.0
    log_loss_total = 0.0
    confidence_rows: list[tuple[float, bool]] = []
    for case, result in pairs:
        truth = case.expected.label
        assert truth is not None
        for label in labels:
            target = 1.0 if label == truth else 0.0
            probability = result.probabilities.get(label, 0.0)
            brier_total += (probability - target) ** 2

        truth_probability = max(result.probabilities.get(truth, 0.0), _LOG_EPSILON)
        log_loss_total += -math.log(truth_probability)

        confidence = selection_probability(result)
        assert confidence is not None
        top_label = max(result.probabilities, key=result.probabilities.__getitem__)
        confidence_rows.append((confidence, top_label == truth))

    ece = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        rows = [
            row
            for row in confidence_rows
            if (lower <= row[0] < upper) or (index == bins - 1 and row[0] == 1.0)
        ]
        if not rows:
            continue
        average_confidence = sum(confidence for confidence, _ in rows) / len(rows)
        accuracy = sum(correct for _, correct in rows) / len(rows)
        ece += (len(rows) / len(confidence_rows)) * abs(accuracy - average_confidence)

    return CalibrationMetrics(
        evaluated=len(pairs),
        brier_score=brier_total / len(pairs),
        log_loss=log_loss_total / len(pairs),
        expected_calibration_error=ece,
    )


def selective_metrics(
    cases: list[EvaluationCase],
    results: list[DecisionResult],
    *,
    error_budgets: tuple[float, ...] = AUTOMATION_ERROR_BUDGETS,
) -> SelectiveMetrics:
    """Empirical risk/coverage frontier at fixed maximum error budgets.

    For each budget, choose the lowest probability threshold that maximizes coverage
    while keeping the observed error rate at or below the budget. Threshold candidates
    are considered only after all equal-probability rows, because a real threshold cannot
    split a confidence tie. The result describes this evaluation set; fitting a production
    threshold requires separate calibration evidence.
    """
    if not error_budgets:
        raise ValueError("at least one automation error budget is required")
    if any(not 0.0 <= budget <= 1.0 for budget in error_budgets):
        raise ValueError("automation error budgets must be between zero and one")

    ranked: list[tuple[float, bool]] = []
    for case, result in _paired_labeled_cases(cases, results):
        probability = selection_probability(result)
        if probability is None:
            continue
        ranked.append((probability, result.label == case.expected.label))

    if not ranked:
        raise ValueError("no labeled cases with probability distributions")

    ranked.sort(key=lambda row: row[0], reverse=True)
    evaluated = len(ranked)
    points: dict[str, AutomationPoint] = {}

    for budget in error_budgets:
        best_automated = 0
        best_errors = 0
        best_threshold: float | None = None
        cumulative_errors = 0

        for index, (probability, correct) in enumerate(ranked, start=1):
            cumulative_errors += int(not correct)
            next_probability = ranked[index][0] if index < evaluated else None
            if next_probability == probability:
                continue
            if cumulative_errors / index <= budget:
                best_automated = index
                best_errors = cumulative_errors
                best_threshold = probability

        key = f"{budget:.0%}"
        points[key] = AutomationPoint(
            error_budget=budget,
            coverage=best_automated / evaluated,
            threshold=best_threshold,
            automated=best_automated,
            errors=best_errors,
            empirical_error_rate=(
                best_errors / best_automated if best_automated else None
            ),
        )

    return SelectiveMetrics(evaluated=evaluated, by_error_budget=points)

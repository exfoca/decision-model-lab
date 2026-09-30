from os import environ
from pathlib import Path
from typing import Annotated

import typer

from decision_model_lab.accelerator import candidate_cuda_error, runtime_diagnostics
from decision_model_lab.battery import BatteryError, run_resident_battery
from decision_model_lab.benchmark import case_diagnostics, run_benchmark, write_report
from decision_model_lab.candidates import Candidate, CandidateError, resolve_candidate
from decision_model_lab.dataset import DatasetError, load_jsonl
from decision_model_lab.jev_style_runner import JevStyleRunner
from decision_model_lab.provenance import collect_run_provenance
from decision_model_lab.semantic import (
    DEFAULT_EVALUATION_PROTOCOL,
    DEFAULT_SEMANTIC_PROFILE,
    EvaluationProtocolError,
    SemanticProfileError,
    resolve_evaluation_protocol,
    resolve_semantic_profile,
)

app = typer.Typer(no_args_is_help=True, help="Decision Model Lab experiment harness.")
dataset_app = typer.Typer(no_args_is_help=True, help="Dataset operations.")
jev_style_app = typer.Typer(no_args_is_help=True, help="Jev-Style candidate operations.")
benchmark_app = typer.Typer(no_args_is_help=True, help="Benchmark operations.")
app.add_typer(dataset_app, name="dataset")
app.add_typer(jev_style_app, name="jev-style")
app.add_typer(benchmark_app, name="benchmark")


@dataset_app.command("validate")
def validate_dataset(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Validate a canonical JSONL evaluation dataset."""
    try:
        cases = load_jsonl(path)
    except DatasetError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    tags = sorted({tag for case in cases for tag in case.tags})
    typer.echo(f"valid: {len(cases)} case(s)")
    typer.echo(f"tags: {', '.join(tags) if tags else '-'}")


def _require_candidate_cuda(candidate: Candidate, *, operation: str) -> None:
    error = candidate_cuda_error(candidate.cuda_backend)
    if error is None:
        return
    typer.echo(
        f"CUDA is required for {operation} candidate {candidate.name!r}: {error}.",
        err=True,
    )
    raise typer.Exit(code=2)


@jev_style_app.command("doctor")
def jev_style_doctor() -> None:
    """Inspect binary Python dependencies and CUDA without loading Jev weights."""
    try:
        diagnostics = runtime_diagnostics()
    except (ImportError, OSError, ValueError) as exc:
        typer.echo(f"runtime diagnostic failed: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    for key, value in diagnostics.items():
        typer.echo(f"{key}: {value}")

    if diagnostics["cuda_available"] and diagnostics["ptxas_path"] is None:
        typer.echo(
            "CUDA is available but ptxas is missing; Triton JIT compilation will fail.",
            err=True,
        )
        raise typer.Exit(code=2)

    if diagnostics["gguf_scorer_path"] and not diagnostics["gguf_cuda_available"]:
        typer.echo(
            "jev-score-v2 is installed but libggml-cuda.so.0 is unavailable; "
            "the GGUF scorer was built without CUDA.",
            err=True,
        )
        raise typer.Exit(code=2)


@jev_style_app.command("smoke")
def jev_style_smoke(
    dataset: Annotated[
        Path,
        typer.Option("--dataset", exists=True, dir_okay=False, readable=True),
    ] = Path("datasets/smoke.jsonl"),
    require_cuda: Annotated[
        bool,
        typer.Option(
            "--require-cuda/--allow-cpu",
            help="Fail before model loading when CUDA is unavailable.",
        ),
    ] = True,
) -> None:
    if require_cuda:
        error = candidate_cuda_error("torch")
        if error is not None:
            typer.echo(f"CUDA is required for this smoke run: {error}.", err=True)
            raise typer.Exit(code=2)

    try:
        cases = load_jsonl(dataset)
    except DatasetError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    case = next((candidate for candidate in cases if candidate.decision.type == "noul"), None)
    if case is None:
        typer.echo(f"{dataset}: no noul case available for Jev-Style smoke test", err=True)
        raise typer.Exit(code=2)

    result = JevStyleRunner().run(case)
    typer.echo(result.model_dump_json(indent=2))


@benchmark_app.command("battery")
def benchmark_battery(
    datasets: Annotated[
        list[Path] | None,
        typer.Option(
            "--dataset",
            exists=True,
            dir_okay=False,
            readable=True,
            help=(
                "Dataset to execute. Repeat --dataset to keep each selected model resident "
                "across multiple datasets."
            ),
        ),
    ] = None,
    candidates: Annotated[
        list[str] | None,
        typer.Option(
            "--candidate",
            help=(
                "Registered candidate to execute. Repeat --candidate to run multiple candidates "
                "sequentially, loading each only once."
            ),
        ),
    ] = None,
    verify_state_isolation: Annotated[
        bool,
        typer.Option(
            "--verify-state-isolation/--skip-state-isolation",
            help=(
                "Probe the resident runtime after every benchmark run and fail before publishing "
                "that candidate's artifacts if its normalized decision drifts."
            ),
        ),
    ] = True,
    require_cuda: Annotated[
        bool,
        typer.Option(
            "--require-cuda/--allow-cpu",
            help="Fail before model loading when CUDA is unavailable.",
        ),
    ] = True,
) -> None:
    """Run every non-redundant regime for explicitly selected datasets and candidates."""
    if not datasets:
        typer.echo("benchmark battery requires at least one --dataset", err=True)
        raise typer.Exit(code=2)
    if not candidates:
        typer.echo("benchmark battery requires at least one --candidate", err=True)
        raise typer.Exit(code=2)
    try:
        candidate_specs = tuple(resolve_candidate(name) for name in candidates)
    except CandidateError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    if require_cuda:
        for candidate_spec in candidate_specs:
            _require_candidate_cuda(candidate_spec, operation="benchmark battery")

    try:
        artifacts = run_resident_battery(
            candidate_names=tuple(candidates),
            datasets=tuple(datasets),
            verify_state_isolation=verify_state_isolation,
            progress=lambda message: typer.echo(f"[battery] {message}"),
        )
    except (BatteryError, CandidateError, DatasetError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(
        f"battery complete: candidates={len(candidates)} datasets={len(datasets)} "
        f"runs={len(artifacts)}"
    )
    for artifact in artifacts:
        typer.echo(f"artifact: {artifact.path}")


@benchmark_app.command("run")
def benchmark_run(
    dataset: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    candidate: Annotated[str, typer.Option("--candidate")] = "jev-style",
    semantic_profile: Annotated[
        str,
        typer.Option(
            "--semantic-profile",
            help=(
                "Model-visible semantic representation: baseline, optimized-v1, "
                "or native-criteria-v1 (Jev-Style candidates only)."
            ),
        ),
    ] = DEFAULT_SEMANTIC_PROFILE,
    evaluation_protocol: Annotated[
        str,
        typer.Option(
            "--protocol",
            help="Model-visible evidence protocol: rule-conditioned or closed-book.",
        ),
    ] = DEFAULT_EVALUATION_PROTOCOL,
    output: Annotated[Path | None, typer.Option("--output", dir_okay=False)] = None,
    require_cuda: Annotated[
        bool,
        typer.Option(
            "--require-cuda/--allow-cpu",
            help="Fail before model loading when CUDA is unavailable.",
        ),
    ] = True,
) -> None:
    """Run a complete dataset through one candidate and persist normalized evidence."""
    try:
        candidate_spec = resolve_candidate(candidate)
        resolved_semantic_profile = resolve_semantic_profile(semantic_profile)
        resolved_evaluation_protocol = resolve_evaluation_protocol(evaluation_protocol)
        runner = candidate_spec.create_runner(
            semantic_profile=resolved_semantic_profile,
            evaluation_protocol=resolved_evaluation_protocol,
        )
    except (CandidateError, SemanticProfileError, EvaluationProtocolError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    if require_cuda:
        _require_candidate_cuda(candidate_spec, operation="benchmark")

    try:
        cases = load_jsonl(dataset)
    except DatasetError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    report = run_benchmark(
        cases,
        runner,
        dataset=dataset,
        candidate=candidate,
        model_id=candidate_spec.model_id,
        semantic_profile=resolved_semantic_profile,
        evaluation_protocol=resolved_evaluation_protocol,
    )
    report = report.model_copy(
        update={
            "provenance": collect_run_provenance(
                runner=runner,
                results=report.results,
                runtime_distribution=candidate_spec.runtime_distribution,
                cuda_backend=candidate_spec.cuda_backend,
            )
        }
    )
    model_name = candidate_spec.model_id.rsplit("/", maxsplit=1)[-1]
    quantization = getattr(runner, "quantization", None)
    quantization_suffix = f"-{quantization}" if quantization else ""
    profile_suffix = (
        ""
        if resolved_semantic_profile == DEFAULT_SEMANTIC_PROFILE
        else f"-{resolved_semantic_profile}"
    )
    protocol_suffix = (
        ""
        if resolved_evaluation_protocol == DEFAULT_EVALUATION_PROTOCOL
        else f"-{resolved_evaluation_protocol}"
    )
    artifact = output or (
        Path(environ.get("DML_ARTIFACTS_DIR", "artifacts"))
        / (
            f"{dataset.stem}-{model_name}{quantization_suffix}"
            f"{profile_suffix}{protocol_suffix}.json"
        )
    )
    write_report(report, artifact)

    diagnostics = case_diagnostics(cases, report.results)
    correct = sum(row.correct for row in diagnostics)
    incorrect = len(diagnostics) - correct

    typer.echo(f"semantic_profile: {report.semantic_profile}")
    typer.echo(f"evaluation_protocol: {report.evaluation_protocol}")
    typer.echo(f"cases: {report.case_count}")
    typer.echo(f"correct: {correct}")
    typer.echo(f"incorrect: {incorrect}")
    typer.echo(f"accuracy: {report.classification['accuracy']:.6f}")
    typer.echo(f"macro_f1: {report.classification['macro_f1']:.6f}")
    typer.echo(f"brier_score: {report.calibration['brier_score']:.6f}")
    typer.echo(f"log_loss: {report.calibration['log_loss']:.6f}")
    typer.echo(
        f"expected_calibration_error: {report.calibration['expected_calibration_error']:.6f}"
    )
    typer.echo("automation_risk_coverage:")
    selective = report.selective["by_error_budget"]
    assert isinstance(selective, dict)
    for budget, point in selective.items():
        assert isinstance(point, dict)
        threshold = point["threshold"]
        threshold_text = "-" if threshold is None else f"{float(threshold):.6f}"
        error_rate = point["empirical_error_rate"]
        error_text = "-" if error_rate is None else f"{float(error_rate):.6f}"
        typer.echo(
            f"  {budget}: coverage={float(point['coverage']):.6f} "
            f"threshold={threshold_text} empirical_error_rate={error_text} "
            f"automated={point['automated']}"
        )
    typer.echo(f"mean_latency_ms: {report.latency.mean_ms:.3f}")
    typer.echo(f"cold_start_latency_ms: {report.latency.cold_start_ms:.3f}")
    steady_state = report.latency.steady_state_mean_ms
    typer.echo(
        "steady_state_mean_latency_ms: " + ("-" if steady_state is None else f"{steady_state:.3f}")
    )

    typer.echo("\nby decision type:")
    for name, segment in report.by_decision_type.items():
        typer.echo(
            f"  {name}: cases={segment.case_count} correct={segment.correct} "
            f"accuracy={segment.accuracy:.6f}"
        )

    typer.echo("\nby tag:")
    for name, segment in report.by_tag.items():
        typer.echo(
            f"  {name}: cases={segment.case_count} correct={segment.correct} "
            f"accuracy={segment.accuracy:.6f}"
        )

    typer.echo("\ncases:")
    for row in diagnostics:
        selection_probability = (
            "-" if row.selection_probability is None else f"{row.selection_probability:.6f}"
        )
        status = "ok" if row.correct else "error"
        typer.echo(
            f"  {row.case_id}: expected={row.expected} predicted={row.predicted} "
            f"selection_probability={selection_probability} "
            f"latency_ms={row.latency_ms:.3f} status={status}"
        )

    errors = [row for row in diagnostics if not row.correct]
    if errors:
        typer.echo("\nerrors:")
        for row in errors:
            selection_probability = (
                "-" if row.selection_probability is None else f"{row.selection_probability:.6f}"
            )
            probabilities = ", ".join(
                f"{label}={probability:.6f}"
                for label, probability in sorted(row.probabilities.items())
            )
            typer.echo(f"  {row.case_id}")
            typer.echo(f"    expected: {row.expected}")
            typer.echo(f"    predicted: {row.predicted}")
            typer.echo(f"    selection_probability: {selection_probability}")
            provider_confidence = (
                "-" if row.provider_confidence is None else f"{row.provider_confidence:.6f}"
            )
            typer.echo(f"    provider_confidence: {provider_confidence}")
            typer.echo(f"    probabilities: {probabilities or '-'}")

    typer.echo(f"\nartifact: {artifact}")

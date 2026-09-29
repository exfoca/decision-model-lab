# Decision Model Lab tools

`tools/` contains offline analysis utilities. Tools consume persisted laboratory evidence and must not load candidate model runtimes or become a second benchmark execution path.

## Benchmark comparison

`compare_benchmarks.py` compares two or more normalized benchmark report JSON files. It is candidate-agnostic: report identity comes from `candidate`, `model_id`, semantic profile and evaluation protocol, never from hard-coded model names.

Run it inside the laboratory container:

```bash
docker compose run --rm lab python tools/compare_benchmarks.py \
  artifacts/run-a.json \
  artifacts/run-b.json \
  artifacts/run-c.json
```

Useful options:

```text
--baseline N|RUN_ID|LABEL|CANDIDATE|MODEL_ID
--case-policy strict|intersection
--dataset-root PATH
--include-cases
--higher-is-better METRIC_PATH
--lower-is-better METRIC_PATH
--format human|markdown|json
--output PATH
--export-dir DIRECTORY
```

The default `strict` case policy requires every report to contain the same result IDs. `intersection` is explicit opt-in for partial runs and discards non-common cases.

Reports produced by the current harness are self-describing enough for case-level analysis through `dataset_sha256` and `case_manifest`. Legacy reports are supported by resolving their `dataset` path and reconstructing the manifest. If neither an embedded manifest nor the referenced dataset is available, aggregate and prediction-disagreement comparisons still work, but correctness-based case diagnostics are unavailable.

Dataset fingerprints may legitimately differ for aligned experimental variants such as EN/pt-BR. Such inputs are accepted only when available case manifests agree on case IDs, expected labels and decision types. The comparison reports this relationship as `aligned-dataset-variant` rather than pretending the datasets are byte/content-identical.

Known classification, calibration and latency metrics have explicit optimization direction. Numeric metrics added by future experiments are discovered automatically and remain neutral by default; direction can be declared at invocation time without changing this tool. Derived `throughput_cases_s` and `correct_cases_s` are computed from steady-state latency and accuracy when those source metrics exist.

`--export-dir` writes a canonical `comparison.json`, a Markdown report and CSV tables for aggregate metrics, segment families, case summaries, pairwise diagnostics and optional per-case rows.

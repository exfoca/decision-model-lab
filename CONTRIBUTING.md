# Contributing to Decision Model Lab

Decision Model Lab is an experimental harness. Changes must preserve the separation between the laboratory contract, candidate-specific adapters and generated evidence.

## Development contract

Use `nix develop` only for host-side repository maintenance. The Nix shell provides Git, Docker, Docker Compose and `uv`; it is not a second model runtime.

Run Python quality gates through the Docker quality target:

```bash
docker compose build quality
docker compose run --rm quality ruff check .
docker compose run --rm quality mypy src
docker compose run --rm quality pytest
```

Run model inference and benchmark measurements through the `lab` service. Do not compare measurements produced by an ad-hoc host Python environment with container measurements.

## Change rules

- Keep the canonical dataset schema model-independent.
- Keep candidate adapters thin. Candidate-specific provider responses belong in `raw_output` or result metadata, not in the laboratory contract.
- Never expose `expected`, `tags` or evaluation-only `metadata` to a candidate.
- Treat semantic profiles and evaluation protocols as explicit experimental variables.
- Resident benchmark paths may share only the loaded provider runtime/client. They must create a
  fresh runner per experimental regime, never pass earlier cases or results into later decisions,
  and retain an explicit state-isolation guard.
- Preserve dataset IDs and semantics across aligned language variants. When an EN/pt-BR pair exists, changes to expected labels, decision types, options or tags must remain aligned unless the divergence is intentional and documented.
- Do not commit generated benchmark artifacts, model weights, checkpoints, local caches, credentials or machine-specific paths.
- Pin source-only upstream runtimes and native build inputs to immutable revisions when they participate in a benchmark runtime.
- Update tests when changing normalization, metrics, report schemas, dataset validation or comparison behavior.

## Adding a candidate

A new candidate should:

1. implement the existing `DecisionRunner` contract;
2. translate canonical decisions into the upstream API without changing the dataset contract;
3. preserve the untouched upstream response when feasible;
4. normalize labels and probabilities into `DecisionResult`;
5. declare its runtime distribution in the candidate registry so benchmark provenance can record the installed runtime version;
6. add contract tests that do not require downloading model weights;
7. implement the resident-runner lifecycle (`prepare`, `fork_for_experiment`, `release`) when the
   candidate is expected to participate in `benchmark battery`.

## Dataset changes

Validate every versioned dataset before submitting a change:

```bash
for dataset in datasets/*.jsonl; do
  docker compose run --rm quality dml dataset validate "$dataset"
done
```

Boundary cases should make inequalities and output domains explicit. An expected label must belong to the declared decision domain.

## Pull requests

Keep pull requests focused and include:

- the experimental or engineering question being addressed;
- the commands used for validation;
- any change to model-visible state, normalization or metric semantics;
- any compatibility impact on persisted benchmark artifacts.

Commit messages and code-facing identifiers should be in English.

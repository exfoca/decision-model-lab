# Decision Model Lab

Decision Model Lab is a reproducible experimental harness for evaluating small, local decision
models under one model-independent contract.

The project exists to answer empirical questions about structured decision systems: decision
quality, calibration, abstention behavior, semantic sensitivity, stability, latency and
cross-model trade-offs. It is maintained by Orion Impact as an isolated laboratory and has no
runtime dependency on Orion Core.

Decision Model Lab is not a generative AI framework, a model-training project or a production
decision service. Its purpose is to make candidate evaluation explicit, repeatable and auditable
before any model is considered for integration elsewhere.

## Current capabilities

- canonical JSONL evaluation cases with typed `noul`, `choice` and `score` decisions;
- multiple candidate adapters behind the same `DecisionRunner` contract;
- English and pt-BR benchmark variants with aligned case identities and ground truth;
- independent semantic-profile and evidence-protocol experiments;
- classification, calibration, selective-automation and latency metrics;
- normalized JSON artifacts with dataset fingerprints, case manifests and run provenance;
- candidate-agnostic N-way comparison of persisted benchmark reports;
- Docker-owned quality and GPU runtime environments;
- a minimal Nix development shell for host-side repository maintenance;
- CI quality gates and complete-history secret scanning.

## Experimental boundary

The laboratory keeps three concerns separate:

1. **Dataset contract** — what evidence, question and decision domain a case defines.
2. **Candidate adapter** — how that canonical case is translated to one upstream runtime.
3. **Evaluation** — how normalized outputs are scored and compared.

Candidate-specific provider output is preserved in `raw_output`, while metrics consume the common
`DecisionResult` representation. Evaluation-only fields such as `expected`, `tags` and `metadata`
are never exposed to a model as evidence.

This separation is deliberate. A benchmark result should reflect a candidate and an explicitly
chosen experimental configuration, not hidden differences in harness semantics.

## Execution contract

The reproducible experiment runtime is Docker-owned. Python, PyTorch, CUDA, candidate runtimes,
quality tools and the `dml` CLI used for measurements live in the container image.

The host-side Nix shell is intentionally small. It provides Git, Docker and `uv` for repository and
lockfile maintenance; it is not a second Python/model environment and must not be used for
benchmark measurements.

The repository is bind-mounted at `/workspace`. The Python environment remains image-owned at
`/opt/dml-venv`, model downloads use the persistent `model-cache/` mount, and generated benchmark
reports are written to `artifacts/` by default.

Two Docker targets are exposed through Compose:

- `quality`: CPU-only repository gates; it does not install candidate model runtimes;
- `lab`: the full CUDA/runtime image used for inference and benchmark measurements.

For strict cold-process latency or resource comparisons, run different candidates in separate
container processes so model residency and GPU state do not contaminate cross-model measurements.
The resident battery described below intentionally keeps one selected candidate loaded across its
requested matrix; its latency results are warm-resident measurements rather than process cold-start
measurements.

## Requirements

For repository quality checks:

- Docker Engine with Docker Compose;
- or Nix, when using the provided host maintenance shell.

For the default GPU benchmark path:

- an NVIDIA GPU and compatible host driver;
- Docker configured to expose the GPU to containers. The Compose configuration requests
  `nvidia.com/gpu=all`;
- network access on first use to resolve image dependencies and upstream model artifacts.

Model weights, checkpoints and generated benchmark artifacts are intentionally not stored in Git.

## Quick start

From the repository root, build and run the lightweight quality environment first:

```bash
docker compose build quality
docker compose run --rm quality
```

The default `quality` command runs the test suite. The complete repository gates are:

```bash
docker compose run --rm quality uv lock --check
docker compose run --rm quality ruff check .
docker compose run --rm quality mypy src
docker compose run --rm quality pytest

for dataset in datasets/*.jsonl; do
  docker compose run --rm quality dml dataset validate "$dataset"
done
```

Build the full laboratory image:

```bash
docker compose build lab
```

Verify that the container can see the CUDA runtime and required toolchain without loading model
weights:

```bash
docker compose run --rm lab dml jev-style doctor
```

Run the Jev-Style smoke case:

```bash
docker compose run --rm lab dml jev-style smoke
```

For host-side maintenance, enter the pinned Nix shell:

```bash
nix develop
```

`uv` on the host is reserved for dependency and lockfile maintenance. Do not run benchmark Python,
`pytest`, `ruff`, `mypy`, `dml` or model workloads directly on the host when producing laboratory
evidence.

## Registered candidates

All candidates are evaluated through the same canonical case schema and normalized result contract.
They remain independent third-party projects; inclusion in this laboratory does not imply
endorsement, affiliation or equivalence between their architectures.

| Candidate | Upstream model/runtime | Laboratory role |
| --- | --- | --- |
| `jev-style` | `chaoliangUNSW/Jev-Style-0.8B-Decision-v3` | Jev-Style 0.8B baseline |
| `jev-style-2b` | `chaoliangUNSW/Jev-Style-2B-Decision-v3` | Same adapter at higher model capacity |
| `jev-style-2b-q4` | `chaoliangUNSW/Jev-Style-2B-Decision-v3-GGUF` | 2B GGUF, `Q4_K_M` |
| `jev-style-2b-q8` | `chaoliangUNSW/Jev-Style-2B-Decision-v3-GGUF` | 2B GGUF, `Q8_0` |
| `laya` | `convaiinnovations/laya` | Upstream Laya English checkpoint |
| `tinyjev` | TinyJev `TinyJev-0.6B` | Compact TinyJev candidate through the upstream runtime |
| `verdict` | `heman10x/rlcd-modernbert-151m` | Verdict/OpenJev through the source-pinned RLCD engine |

The corresponding upstream projects and licenses are documented in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

### Running candidates

The same dataset can be executed against any registered candidate:

```bash
docker compose run --rm lab dml benchmark run datasets/benchmark-v2.jsonl --candidate jev-style
docker compose run --rm lab dml benchmark run datasets/benchmark-v2.jsonl --candidate jev-style-2b
docker compose run --rm lab dml benchmark run datasets/benchmark-v2.jsonl --candidate jev-style-2b-q4
docker compose run --rm lab dml benchmark run datasets/benchmark-v2.jsonl --candidate jev-style-2b-q8
docker compose run --rm lab dml benchmark run datasets/benchmark-v2.jsonl --candidate laya
docker compose run --rm lab dml benchmark run datasets/benchmark-v2.jsonl --candidate tinyjev
docker compose run --rm lab dml benchmark run datasets/benchmark-v2.jsonl --candidate verdict
```

By default a benchmark requires CUDA. The CLI also exposes `--allow-cpu` for controlled experiments
with runtimes that support CPU execution, but CPU and GPU latency measurements must not be treated
as directly comparable.

### Resident benchmark battery

`benchmark battery` does not select models or datasets implicitly. Supply every candidate and
dataset explicitly; both options are repeatable. The harness loads one candidate once, executes all
semantically distinct benchmark regimes supported by that candidate across every selected dataset,
then releases it before loading the next candidate.

For example, run benchmark v3 in both languages against three selected candidates:

```bash
docker compose run --rm lab dml benchmark battery \
  --dataset datasets/benchmark-v3.jsonl \
  --dataset datasets/benchmark-v3-pt-br.jsonl \
  --candidate jev-style \
  --candidate jev-style-2b-q8 \
  --candidate verdict
```

The regime matrix is derived from the candidate registry rather than hard-coded model names. Every
supported semantic profile runs under `rule-conditioned`, and `closed-book` runs once because its
model-visible evidence is independent of semantic-profile rendering. A Jev-Style candidate that
supports `baseline`, `optimized-v1` and `native-criteria-v1` therefore produces four runs per
dataset; a candidate supporting only `baseline` and `optimized-v1` produces three.

Each dataset/regime pair receives a fresh runner and freshly loaded `EvaluationCase` objects. Only
the opaque upstream model/runtime client is kept resident. No previous case, output or normalized
result is supplied to a later decision. By default a synthetic state-isolation probe runs before
the matrix and after every benchmark run; if its normalized decision drifts, that candidate's
pending artifacts are not published. This guard detects obvious session-state leakage but cannot
prove the absence of undocumented internal state in an upstream runtime.

The initial probe also warms the resident runtime. Battery reports therefore record
`execution_mode=resident-battery` and `runtime_warmup=true`, and their filenames receive a
`-resident-battery` suffix. Use standalone `benchmark run` processes for strict process cold-start
latency comparisons.

## Benchmark datasets

The repository carries versioned, non-sensitive evaluation datasets. Later generations increase
the stress placed on rule execution, ambiguity handling, boundaries, context length and linguistic
robustness without changing the canonical case contract.

| Dataset | Cases | Language | Purpose |
| --- | ---: | --- | --- |
| `datasets/smoke.jsonl` | 2 | English | Minimal integration smoke cases |
| `datasets/benchmark-v1.jsonl` | 12 | English | Initial controlled taxonomy and harness validation |
| `datasets/benchmark-v2.jsonl` | 57 | English | Expanded decision surface across impact, evidence, contradiction, scope, causality, security and temporal reasoning |
| `datasets/benchmark-v2-pt-br.jsonl` | 57 | pt-BR | Aligned Portuguese variant of benchmark v2 |
| `datasets/benchmark-v3.jsonl` | 99 | en-US | 11 axes × 3 context sizes × 3 probe types |
| `datasets/benchmark-v3-pt-br.jsonl` | 99 | pt-BR | Aligned Portuguese variant of benchmark v3 |

The EN/pt-BR v2 and v3 pairs preserve case IDs, decision domains, expected labels, tags and ordering.
That makes language a controlled experimental variable rather than a different benchmark.

Validate a dataset before running it:

```bash
docker compose run --rm quality dml dataset validate datasets/benchmark-v3.jsonl
```

## Dataset contract

Each JSONL line is one independent evaluation case with an explicit decision type:

```json
{"id":"example-001","context":"...","question":"...","decision":{"type":"noul"},"expected":{"label":"yes"},"tags":["binary"]}
```

The core fields are:

- `id`: stable case identity;
- `context`: primary model-visible state;
- `question`: the decision request;
- `decision`: typed output domain (`noul`, `choice` or `score`);
- `definitions`: optional decision semantics used by rule-conditioned experiments;
- `expected`: evaluation ground truth;
- `tags`: evaluation segmentation labels;
- `metadata`: evaluation and construction metadata.

`expected` deliberately remains an object instead of a scalar so future ground-truth forms can be
added without making the dataset schema candidate-specific.

When `expected.label` is present, validation requires it to belong to the declared output domain:
`yes`/`no` for `noul`, or one of the explicit options for `choice` and `score`. Abstention can only
be expected when its label is explicitly part of that case's decision domain.

## Experimental axes

Decision Model Lab treats model-visible representation and available evidence as separate
experimental variables.

### Semantic profiles

`baseline` is the default semantic profile. Under the default rule-conditioned protocol it preserves
the established representation of `definitions`.

`optimized-v1` changes only how `definitions` are rendered: nested JSON is encoded as a
deterministic indented rule tree under `[decision_rules]`. Cases without `definitions` remain
byte-for-byte identical to `baseline`.

```bash
docker compose run --rm lab dml benchmark run datasets/benchmark-v2.jsonl \
  --candidate jev-style \
  --semantic-profile optimized-v1
```

Jev-Style candidates additionally expose `native-criteria-v1`. When a case contains an unambiguous
top-level definition map whose keys exactly match the declared choice options, those definitions
are supplied through the provider's native `choice.criteria` shape. No aliases are inferred and any
residual definitions remain in the normal rendered state.

```bash
docker compose run --rm lab dml benchmark run datasets/benchmark-v2.jsonl \
  --candidate jev-style \
  --semantic-profile native-criteria-v1
```

Other candidates reject `native-criteria-v1` because they do not consume that provider-specific
criteria representation.

### Evidence protocols

`rule-conditioned` is the default protocol. It exposes `definitions` according to the selected
semantic profile.

`closed-book` hides `definitions` and supplies only the case `context`. This measures the
candidate's implicit decision behavior separately from its ability to execute supplied runtime
rules.

```bash
docker compose run --rm lab dml benchmark run datasets/benchmark-v2.jsonl \
  --candidate jev-style \
  --protocol closed-book
```

Under `closed-book`, semantic profiles intentionally collapse to the same model-visible evidence.
`native-criteria-v1` cannot expose definition-derived criteria in that protocol.

Non-default semantic profiles and protocols receive artifact filename suffixes so experiments do
not overwrite the baseline evidence accidentally.

## Metrics

Every benchmark produces the same core metric families:

- **classification**: evaluated cases, accuracy and macro-F1;
- **calibration**: Brier score, log loss and expected calibration error (ECE);
- **selective automation**: empirical coverage at maximum 1%, 5% and 10% error budgets;
- **segmentation**: accuracy by decision type and by dataset tag;
- **latency**: total, mean, min, max, cold-start and steady-state mean latency.

Selective-automation calculations use the normalized probability of the selected class
(`max(probabilities)`), not a provider-specific confidence field. Threshold search is tie-safe:
equal-probability cases stay together because a deployable threshold cannot split a confidence
tie.

The resulting thresholds describe the evaluated dataset only. They are not production thresholds
and should not be transferred to deployment without separate calibration evidence.

## Artifacts and provenance

A benchmark writes a normalized JSON report under `artifacts/` unless `--output` is provided.
Generated artifacts are gitignored.

Current reports use schema version 2 and include:

- dataset path and canonical SHA-256 fingerprint;
- candidate and model identity;
- semantic profile and evidence protocol;
- aggregate, segmented, selective and latency metrics;
- a compact case manifest with ground truth, decision type, tags and evaluation metadata;
- normalized per-case results and untouched upstream `raw_output`;
- run provenance.

The provenance block records the information available at execution time, including repository
commit/dirty state, Python/platform identity, Torch/CUDA/GPU state, candidate runtime version,
resolved model revision when exposed, quantization, container base image and container source
revision.

This does not make every third-party model repository immutable automatically. It makes the
resolved execution state explicit where the upstream runtime exposes it. Inputs that are part of
the laboratory itself are pinned wherever the integration requires a fixed source revision.

### GGUF Jev-Style reproducibility

`jev-style-2b-q4` and `jev-style-2b-q8` use the upstream
`chaoliangUNSW/Jev-Style-2B-Decision-v3-GGUF` repository at the immutable snapshot
`78b1352e9b8132987f6f6744dc2d3f57947f5357`.

The laboratory prefetches the selected weight, `jev_style_decision_gguf.py` and tokenizer assets
from that same snapshot. The image-owned `jev-score-v2` binary is built from the same Jev-Style
snapshot against the pinned `llama.cpp` revision
`441df11f65ea0b6d0c72965aaf70c8241070ddcb`.

This keeps the GGUF model assets and native scorer on one explicit revision boundary. Changing
those revisions requires rebuilding the `lab` image.

### Verdict runtime reproducibility

Verdict/OpenJev uses `heman10x/rlcd-modernbert-151m` through an upstream RLCD runtime installed
from source revision `30f1556`. Additional direct Verdict-only packages are pinned in
`requirements/verdict-runtime.txt`, while the main project dependency graph remains frozen by
`uv.lock`.

Changing the source-pinned Verdict runtime manifest requires rebuilding the `lab` image.

## Comparing persisted runs

The comparison tool works only from benchmark artifacts; it does not load candidate runtimes.

```bash
docker compose run --rm lab python tools/compare_benchmarks.py \
  artifacts/benchmark-v2-Jev-Style-0.8B-Decision-v3.json \
  artifacts/benchmark-v2-laya.json \
  artifacts/benchmark-v2-TinyJev-0.6B.json \
  artifacts/benchmark-v2-rlcd-modernbert-151m.json
```

It supports N-way comparison and can:

- discover aggregate numeric metrics;
- compare every `by_*` segment family;
- align case-level evidence;
- compare aligned dataset variants such as EN and pt-BR;
- report pairwise disagreement and exact McNemar diagnostics when ground truth is available;
- measure top-1 agreement and probability-distribution drift;
- export canonical JSON, Markdown and CSV tables.

Useful options include:

```text
--baseline
--case-policy strict|intersection
--include-cases
--format json|markdown
--export-dir <path>
--higher-is-better <metric>
--lower-is-better <metric>
```

Known metrics have explicit directionality. Unknown future metrics remain neutral unless their
direction is supplied on the command line. Legacy reports created before case manifests were
embedded remain supported when their referenced dataset can be resolved.

## Candidate integration notes

### Jev-Style

The Jev-Style family is integrated through the upstream `jev-style` runtime. The adapter translates
canonical decisions into provider requests, preserves the upstream response and normalizes the
result without using Jev-Style's own evaluation harness.

The 0.8B and 2B candidates intentionally share the same laboratory contract so model capacity can
be studied as an isolated variable. Q4 and Q8 variants keep quantization as an additional explicit
variable rather than silently replacing the full-precision candidate.

Upstream project: <https://jevstyle.com/>

### Laya

`laya` uses the upstream `convaiinnovations/laya` English checkpoint through the upstream Python
runtime. The laboratory deliberately uses the base English checkpoint rather than silently
substituting a workflow-specialized checkpoint.

Upstream model: <https://huggingface.co/convaiinnovations/laya>

### TinyJev

`tinyjev` uses the upstream TinyJev Python runtime with the model name `TinyJev-0.6B`. The public
upstream checkpoint is published as `AnkitAI/TinyJev-0.6B`. The adapter preserves the provider
response and normalizes TinyJev `choice`, `score` and `noul` answers into the laboratory contract.

Upstream source: <https://github.com/ankit-aglawe/tinyjev>

Upstream model: <https://huggingface.co/AnkitAI/TinyJev-0.6B>

### Verdict / OpenJev

`verdict` adapts the source-pinned RLCD/OpenJev `DecisionEngine`. The laboratory maps the upstream
`__insufficient_evidence__` outcome into its canonical `insufficient_evidence` label and preserves
calibration/execution telemetry in result metadata.

Upstream source: <https://github.com/Heman10x-NGU/Verdict-open-jev>

Upstream model: <https://huggingface.co/heman10x/rlcd-modernbert-151m>

## Dependency changes

Python dependencies are locked by `uv.lock`. When dependencies change, update the lockfile from
the Nix maintenance shell and rebuild the relevant image:

```bash
nix develop
uv lock
docker compose build quality
docker compose build lab
```

Source code, test, dataset and configuration changes are bind-mounted and normally do not require an
image rebuild. Dependency changes and image-owned native/runtime changes do.

## Repository layout

```text
.github/workflows/       CI quality and history-secret scanning
configs/                 experiment definitions and notes
datasets/                versioned, non-sensitive evaluation cases
requirements/            source-pinned auxiliary runtime requirements
src/decision_model_lab/  canonical schema, runners, metrics and CLI
tests/                   contract and harness tests
tools/                   offline benchmark comparison tooling
artifacts/               generated benchmark reports (gitignored)
model-cache/             persistent upstream model cache (gitignored)
```

## Quality, security and contribution policy

CI runs the lockfile check, Ruff, mypy, pytest and validation of every versioned JSONL dataset
without loading GPU models. A separate workflow scans reachable Git history for secrets with
Gitleaks.

Contribution rules and the experimental contract are documented in
[`CONTRIBUTING.md`](CONTRIBUTING.md). Security reporting guidance is in
[`SECURITY.md`](SECURITY.md).

## License and third-party software

Decision Model Lab source code is licensed under the Apache License 2.0. See [`LICENSE`](LICENSE).

Candidate models, runtimes, CUDA components and other third-party software remain governed by their
respective upstream licenses and terms. Principal upstream components are listed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Citation

Research or published evaluations using this laboratory can use the metadata in
[`CITATION.cff`](CITATION.cff).

## Project ownership

Decision Model Lab is an Orion Impact engineering and evaluation project. The laboratory is kept
independent from Orion Core so candidate research can evolve without coupling the core platform to
one model family or upstream runtime.

# Security Policy

## Supported versions

Security fixes are applied to the current `main` branch and to the latest tagged release when a release exists. Historical experimental snapshots are not maintained as separate supported branches unless explicitly stated.

## Reporting a vulnerability

Do not open a public issue for a vulnerability that could expose credentials, execute untrusted code, compromise the benchmark host or poison reproducibility evidence.

Prefer GitHub private vulnerability reporting when it is enabled for this repository. If that channel is unavailable, contact Orion Impact privately at `contato@orion-impact.com` with the subject `Decision Model Lab security report`.

Include, when possible:

- affected commit or release;
- reproduction steps;
- expected and observed behavior;
- security impact;
- whether model files, datasets, Docker build inputs or generated artifacts are involved.

Do not include live credentials or unrelated sensitive data in a report.

## Scope

Security reports about this repository may cover the harness, dataset parsing, normalization, comparison tooling, Docker/Nix build definitions and CI configuration.

Decision Model Lab also integrates third-party runtimes and model repositories. Vulnerabilities that originate exclusively in an upstream project should normally be reported to that upstream maintainer as well. The laboratory may still need a pin, mitigation or compatibility guard when an upstream issue affects reproducible execution here.

## Secrets

Local `.env` files, model caches and generated artifacts are intentionally excluded from Git. Before making a previously private repository public, scan the complete reachable Git history with a dedicated secret scanner; checking only the current working tree is not sufficient.

The repository includes a GitHub Actions history scan. The equivalent local scan is:

```bash
docker run --rm -v "$PWD:/repo" -w /repo \
  ghcr.io/gitleaks/gitleaks:v8.30.1 git --redact --no-banner .
```

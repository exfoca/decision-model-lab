# Third-Party Notices

Decision Model Lab is licensed under Apache-2.0. It integrates or downloads independent third-party software and model artifacts during laboratory builds and runs. Those components remain subject to their own licenses and upstream terms.

The repository does not vendor the model weights listed below.

## JevStyle

- Project/model family: JevStyle / Jev-Style decision models
- Upstream: Chaoliang Yan / `chaoliangUNSW`
- Project: https://jevstyle.com/
- Models used by this laboratory:
  - `chaoliangUNSW/Jev-Style-0.8B-Decision-v3`
  - `chaoliangUNSW/Jev-Style-2B-Decision-v3`
  - `chaoliangUNSW/Jev-Style-2B-Decision-v3-GGUF`
- Weight license: Apache-2.0
- Runtime package used by the laboratory: `jev-style`

JevStyle describes itself as an independent project and not as an affiliate of TypeSafe AI or the Laya authors. Decision Model Lab is likewise an independent evaluation project and does not imply endorsement by or affiliation with those projects.

## Laya

- Project/model: `convaiinnovations/laya`
- Upstream: Convai Innovations
- Model: https://huggingface.co/convaiinnovations/laya
- License: Apache-2.0
- Runtime package used by the laboratory: `laya`

## TinyJev

- Project: TinyJev
- Upstream: Ankit Aglawe
- Source: https://github.com/ankit-aglawe/tinyjev
- Model: https://huggingface.co/AnkitAI/TinyJev-0.6B
- Runtime/model license: MIT
- Runtime package used by the laboratory: `tinyjev`

TinyJev documents Qwen3-0.6B-Base and Kev as upstream lineage, both under their respective upstream terms.

## Verdict / OpenJev

- Project: Verdict / OpenJev
- Upstream: Heman10x-NGU
- Source: https://github.com/Heman10x-NGU/Verdict-open-jev
- Model: https://huggingface.co/heman10x/rlcd-modernbert-151m
- Model license: Apache-2.0
- Runtime used by the laboratory: source-pinned `rlcd` from the upstream repository

## llama.cpp

The GGUF Jev-Style scorer is built against a pinned revision of `ggerganov/llama.cpp` during the Docker build.

- Source: https://github.com/ggerganov/llama.cpp
- License: MIT

## NVIDIA CUDA container

The laboratory runtime image is based on an NVIDIA CUDA container. The container image and NVIDIA-provided components are governed by NVIDIA's applicable container and CUDA license terms, independently of the Apache-2.0 license for Decision Model Lab source code.

## Transitive dependencies

Python and system dependencies resolved by `uv.lock`, APT or upstream build systems retain their own copyright notices and license terms. This file is an attribution summary for the principal candidate/runtime components and is not a replacement for upstream license texts.

## Gitleaks

The CI security workflow uses the Gitleaks secret scanner to inspect reachable Git history.

- Source: https://github.com/gitleaks/gitleaks
- License: MIT

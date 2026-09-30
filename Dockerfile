FROM python:3.12-slim-bookworm AS quality

ARG DML_SOURCE_REVISION=unknown

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_PROJECT_ENVIRONMENT=/opt/dml-venv \
    PATH=/opt/dml-venv/bin:${PATH} \
    DML_CONTAINER_BASE_IMAGE=python:3.12-slim-bookworm \
    DML_CONTAINER_SOURCE_REVISION=${DML_SOURCE_REVISION}

LABEL org.opencontainers.image.title="Decision Model Lab quality" \
      org.opencontainers.image.revision="${DML_SOURCE_REVISION}"

WORKDIR /workspace

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir "uv==0.10.0"

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --frozen --extra dev

COPY datasets ./datasets
COPY configs ./configs
COPY tests ./tests
COPY tools ./tools

CMD ["pytest"]

FROM nvidia/cuda:13.0.3-devel-ubuntu24.04 AS runtime-base

ARG DML_SOURCE_REVISION=unknown

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_PROJECT_ENVIRONMENT=/opt/dml-venv \
    PATH=/opt/dml-venv/bin:${PATH} \
    DML_CONTAINER_BASE_IMAGE=nvidia/cuda:13.0.3-devel-ubuntu24.04 \
    DML_CONTAINER_SOURCE_REVISION=${DML_SOURCE_REVISION}

LABEL org.opencontainers.image.title="Decision Model Lab" \
      org.opencontainers.image.revision="${DML_SOURCE_REVISION}"

WORKDIR /workspace

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        ca-certificates \
        cmake \
        git \
        git-lfs \
        ninja-build \
        python3.12 \
        python3.12-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m pip install --break-system-packages --no-cache-dir "uv==0.10.0"

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --frozen --extra dev

COPY datasets ./datasets
COPY configs ./configs
COPY tests ./tests
COPY tools ./tools

FROM runtime-base AS runtime

ARG JEV_LLAMA_CPP_REVISION=441df11f65ea0b6d0c72965aaf70c8241070ddcb

COPY requirements ./requirements

RUN uv sync --frozen --extra dev --extra jev-style --extra laya --extra tinyjev \
    && uv export --frozen --all-extras --no-hashes --no-emit-project \
        --output-file /tmp/dml-base-constraints.txt \
    && uv pip install --python /opt/dml-venv/bin/python \
        --constraint /tmp/dml-base-constraints.txt \
        --exclude-newer 2026-09-27T23:59:59Z \
        --requirement requirements/verdict-runtime.txt \
    && uv pip check --python /opt/dml-venv/bin/python \
    && /opt/dml-venv/bin/python -c "import rlcd; assert hasattr(rlcd, 'DecisionEngine')" \
    && command -v ptxas \
    && ptxas --version

RUN JEV_STYLE_GGUF_REVISION="$(/opt/dml-venv/bin/python -c \
        'from decision_model_lab.jev_style_runner import MODEL_2B_GGUF_REVISION; print(MODEL_2B_GGUF_REVISION)')" \
    && git lfs install --system \
    && git init /tmp/jev-style-gguf \
    && git -C /tmp/jev-style-gguf remote add origin https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v3-GGUF \
    && GIT_LFS_SKIP_SMUDGE=1 git -C /tmp/jev-style-gguf fetch --depth 1 origin "${JEV_STYLE_GGUF_REVISION}" \
    && GIT_LFS_SKIP_SMUDGE=1 git -C /tmp/jev-style-gguf checkout --detach FETCH_HEAD \
    && git init /tmp/llama.cpp \
    && git -C /tmp/llama.cpp remote add origin https://github.com/ggerganov/llama.cpp.git \
    && git -C /tmp/llama.cpp fetch --depth 1 origin "${JEV_LLAMA_CPP_REVISION}" \
    && git -C /tmp/llama.cpp checkout --detach FETCH_HEAD \
    && cd /tmp/jev-style-gguf \
    && sh build_jev_score.sh /tmp/llama.cpp \
    && scorer="$(find /tmp/jev-style-gguf /tmp/llama.cpp -type f -name jev-score-v2 -perm /111 -print -quit)" \
    && test -n "${scorer}" \
    && install -m 0755 "${scorer}" /usr/local/bin/jev-score-v2 \
    && install -d /usr/local/lib \
    && ldd "${scorer}" \
        | awk '$3 ~ "^/tmp/(llama\\.cpp|jev-style-gguf)/" { print $1, $3 }' \
        | while read -r soname library; do install -m 0644 "${library}" "/usr/local/lib/${soname}"; done \
    && ldconfig \
    && command -v jev-score-v2 \
    && ! ldd /usr/local/bin/jev-score-v2 | grep -q "not found" \
    && rm -rf /tmp/jev-style-gguf /tmp/llama.cpp

CMD ["dml", "--help"]

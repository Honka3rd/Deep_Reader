# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEEP_READER_TOKENIZER_ASSET_DIR=/opt/deep-reader/tokenizer-assets
WORKDIR /app/Deep_Reflective_Reader

# Registry code may change without changing the OCR package set. Only this
# deterministic projection crosses into the expensive installation stage.
FROM base AS ocr-manifest
COPY Deep_Reflective_Reader/language/language_code.py Deep_Reflective_Reader/language/language_profile_registry.py ./language/
COPY Deep_Reflective_Reader/doc_loaders/pdf_ocr_language_policy.py ./doc_loaders/
RUN python -m doc_loaders.pdf_ocr_language_policy --debian-packages > /tmp/ocr-packages \
    && tr ' ' '\n' < /tmp/ocr-packages | LC_ALL=C sort -u > /ocr-packages

# Python dependency installation is independent of all OCR/application source.
FROM base AS python-dependencies
ARG TARGETARCH
ENV PATH="/opt/venv/bin:$PATH"
COPY Deep_Reflective_Reader/requirements.txt ./requirements.txt
RUN python -m venv /opt/venv
RUN --mount=type=cache,id=deep-reader-pip-py311-${TARGETARCH},target=/root/.cache/pip,sharing=locked \
    /opt/venv/bin/python -m pip install --upgrade pip \
    && /opt/venv/bin/python -m pip install -r requirements.txt

FROM base AS ocr-runtime
ARG TARGETARCH
COPY --from=ocr-manifest /ocr-packages /tmp/ocr-packages
# Debian's image cleanup hook would otherwise remove downloaded debs from the
# cache mount. Cache directories stay in BuildKit, not in the runtime image.
RUN rm -f /etc/apt/apt.conf.d/docker-clean \
    && printf 'Binary::apt::APT::Keep-Downloaded-Packages "true";\n' > /etc/apt/apt.conf.d/keep-cache
# Old trixie apt/libapt can accumulate delayed-retry warnings until OOM. Bootstrap
# without retries, then require the fixed release before the large OCR download.
RUN --mount=type=cache,id=deep-reader-apt-py311-${TARGETARCH},target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=deep-reader-apt-lists-py311-${TARGETARCH},target=/var/lib/apt,sharing=locked \
    apt-get -o Acquire::Retries=0 update \
    && apt-get -o Acquire::Retries=0 install -y --no-install-recommends --only-upgrade apt libapt-pkg7.0 \
    && for package in apt libapt-pkg7.0; do \
        version="$(dpkg-query -W -f='${Version}' "$package")" \
        && dpkg --compare-versions "$version" ge 3.0.4 \
        || { echo "apt_bootstrap_version_unsupported: $package" >&2; exit 1; }; \
    done \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        $(cat /tmp/ocr-packages) \
        poppler-utils \
    && rm /tmp/ocr-packages
ADD https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/87416418657359cb625c412a48b6e1d6d41c29bd/chi_tra_vert.traineddata /usr/share/tesseract-ocr/5/tessdata/chi_tra_vert.traineddata
ADD https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/87416418657359cb625c412a48b6e1d6d41c29bd/chi_sim_vert.traineddata /usr/share/tesseract-ocr/5/tessdata/chi_sim_vert.traineddata
RUN echo '1df02a4b210e5c217b783819538b63e9dfe6904e2b5e53b62664f1b9f7a989d0  /usr/share/tesseract-ocr/5/tessdata/chi_tra_vert.traineddata' | sha256sum -c -
RUN echo '20590de84725bab69cde93bd6e8ed360a13cc5421a7e7364ddeb93e9af53d6da  /usr/share/tesseract-ocr/5/tessdata/chi_sim_vert.traineddata' | sha256sum -c -

FROM python-dependencies AS tokenizer-assets
COPY Deep_Reflective_Reader/llm/offline_token_counter.py ./llm/offline_token_counter.py
COPY Deep_Reflective_Reader/scripts/prepare_tokenizer_assets.py ./scripts/prepare_tokenizer_assets.py
RUN python -m scripts.prepare_tokenizer_assets

FROM ocr-runtime AS runtime
COPY --from=python-dependencies /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY --from=tokenizer-assets /opt/deep-reader/tokenizer-assets /opt/deep-reader/tokenizer-assets
COPY Deep_Reflective_Reader/ ./
RUN python -m scripts.prepare_tokenizer_assets --verify-only \
    && python -m scripts.verify_ocr_assets --load-models

EXPOSE 8000

ENTRYPOINT ["python", "-m", "scripts.tokenizer_startup"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

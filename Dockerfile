FROM public.ecr.aws/lambda/python:3.12
ENV UV_PROJECT_ENVIRONMENT=/var/task/.venv UV_NO_CACHE=1 \
    PYTHONPATH=/var/task/.venv/lib/python3.12/site-packages:/var/task/src \
    PATH=/var/task/.venv/bin:$PATH HF_HOME=/tmp/huggingface TIKTOKEN_CACHE_DIR=/var/task/tiktoken_cache
WORKDIR /var/task
RUN python -m pip install --no-cache-dir uv==0.13.6
COPY pyproject.toml uv.lock .python-version ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable
COPY configs ./configs
COPY tests/fixtures ./tests/fixtures
COPY THIRD_PARTY_NOTICES.md ./
RUN rag models && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"
ENV RAG_MODEL_DIR=/var/task/models RAG_INDEX_DIR=/tmp/index RAG_REPORT_DIR=/var/task/results
CMD ["rag_lab.api.handler"]

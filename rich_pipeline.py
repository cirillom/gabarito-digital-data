"""Reliable orchestration for rich question extraction.

The low-level rich-content extractor stays focused on one question. This module
adds exam-level concerns: durable per-question checkpoints, bounded Gemini
retries, limited concurrency, and safe reuse of already-valid content.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import random
import threading
import time
from pathlib import Path
from typing import Any

import pymupdf
from dotenv import load_dotenv

from rich_content import (
    DEFAULT_MODEL_NAME,
    RICH_EXTRACTION_VERSION,
    extract_question_rich_content,
    validate_rich_content,
)


DEFAULT_RICH_WORKERS = 2
MAX_RICH_WORKERS = 4
_RETRYABLE_GEMINI_CODES = {408, 429, 500, 502, 503, 504}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomically replace one JSON file so an interrupted checkpoint stays valid."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _referenced_asset_ids(
    statement: dict[str, Any],
    options: list[dict[str, Any]],
) -> set[str]:
    referenced: set[str] = set()
    documents = [statement, *(option["content"] for option in options)]
    for document in documents:
        for block in document["blocks"]:
            referenced.update(block.get("asset_ids", []))
    return referenced


def _can_reuse_rich_content(
    content: Any,
    *,
    labels: list[str],
    pdf_sha256: str,
    use_gemini: bool,
    model_name: str,
) -> bool:
    if not isinstance(content, dict) or content.get("source_pdf_sha256") != pdf_sha256:
        return False
    if use_gemini and content.get("method") != f"gemini:{model_name}":
        return False
    try:
        validate_rich_content(content, labels)
    except ValueError:
        return False
    return True


def _trim_unused_assets(content: dict[str, Any]) -> None:
    referenced = _referenced_asset_ids(content["statement"], content["options"])
    content["assets"] = [
        asset for asset in content["assets"] if asset["id"] in referenced
    ]


def _normalize_structured_config(config: Any) -> Any:
    """Prefer explicit JSON Schema output over SDK model-schema conversion.

    Rich extraction does not use tools or function calling. Sending the JSON
    schema explicitly keeps this request on the structured-output path and
    avoids conflating Pydantic schema conversion with AFC/tool workflows.
    """
    schema = getattr(config, "response_schema", None)
    if schema is None or not hasattr(schema, "model_json_schema"):
        return config
    fields = getattr(type(config), "model_fields", {})
    if "response_json_schema" not in fields:
        return config
    normalized = config.model_copy(deep=True)
    normalized.response_schema = None
    normalized.response_json_schema = schema.model_json_schema()
    return normalized


def _error_code(error: BaseException) -> int | None:
    value = getattr(error, "code", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_timeout(error: BaseException) -> bool:
    return isinstance(error, TimeoutError) or "timeout" in type(error).__name__.lower()


def _retry_reason(error: BaseException) -> str | None:
    code = _error_code(error)
    if code not in _RETRYABLE_GEMINI_CODES and not _is_timeout(error):
        return None
    if code == 429:
        return "quota/rate limit"
    if code == 503:
        return "service unavailable/model demand"
    if code is not None:
        return f"temporary API error {code}"
    return "request timeout"


def _retry_delay(
    retry_index: int,
    *,
    base_delay: float,
    max_delay: float,
) -> float:
    exponential = min(max_delay, base_delay * (2**retry_index))
    jitter = random.uniform(0.0, min(1.0, exponential * 0.25))
    return min(max_delay, exponential + jitter)


class _RetryingModels:
    def __init__(
        self,
        models: Any,
        *,
        question_number: int,
        max_attempts: int,
        base_delay: float,
        max_delay: float,
    ) -> None:
        self._models = models
        self._question_number = question_number
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._max_delay = max_delay

    def generate_content(self, *args: Any, **kwargs: Any) -> Any:
        if "config" in kwargs:
            kwargs["config"] = _normalize_structured_config(kwargs["config"])

        for attempt in range(1, self._max_attempts + 1):
            try:
                return self._models.generate_content(*args, **kwargs)
            except Exception as error:
                reason = _retry_reason(error)
                if reason is None:
                    raise
                if attempt >= self._max_attempts:
                    raise RuntimeError(
                        f"Gemini {reason} for question {self._question_number} "
                        f"after {attempt} attempts: {error}"
                    ) from error
                delay = _retry_delay(
                    attempt - 1,
                    base_delay=self._base_delay,
                    max_delay=self._max_delay,
                )
                print(
                    f"  Question {self._question_number}: Gemini {reason}; "
                    f"retry {attempt}/{self._max_attempts - 1} in {delay:.1f}s."
                )
                time.sleep(delay)
        raise AssertionError("unreachable")


class _RetryingGeminiClient:
    def __init__(
        self,
        client: Any,
        *,
        question_number: int,
        max_attempts: int = 5,
        base_delay: float = 2.0,
        max_delay: float = 20.0,
    ) -> None:
        self.models = _RetryingModels(
            client.models,
            question_number=question_number,
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
        )


def _successful_question_count(
    questions: dict[str, Any],
    *,
    labels: list[str],
    pdf_sha256: str,
    model_name: str,
) -> int:
    successful = 0
    for question in questions.values():
        rich = (
            question.get("conteudo", {}).get("rich")
            if isinstance(question, dict)
            else None
        )
        if _can_reuse_rich_content(
            rich,
            labels=labels,
            pdf_sha256=pdf_sha256,
            use_gemini=False,
            model_name=model_name,
        ):
            successful += 1
    return successful


def _set_metadata(
    data: dict[str, Any],
    *,
    questions: dict[str, Any],
    labels: list[str],
    pdf_sha256: str,
    processed: int,
    reused: int,
    failures: dict[str, str],
    use_gemini: bool,
    model_name: str,
) -> None:
    successful = _successful_question_count(
        questions,
        labels=labels,
        pdf_sha256=pdf_sha256,
        model_name=model_name,
    )
    expected = len(questions)
    data["rich_extraction"] = {
        "version": RICH_EXTRACTION_VERSION,
        "status": "success" if successful == expected and not failures else "partial",
        "question_count": expected,
        "successful_question_count": successful,
        "processed_question_count": processed,
        "reused_question_count": reused,
        "source_pdf_sha256": pdf_sha256,
        "method": f"gemini:{model_name}" if use_gemini else "deterministic",
        **({"failures": failures} if failures else {}),
    }


def enrich_rich_data_file(
    data_path: str | Path,
    pdf_path: str | Path,
    *,
    repository_root: str | Path | None = None,
    question_numbers: set[int] | None = None,
    use_gemini: bool = False,
    model_name: str = DEFAULT_MODEL_NAME,
    assets_directory: str | Path | None = None,
    write: bool = True,
    force: bool = False,
    max_workers: int = DEFAULT_RICH_WORKERS,
) -> dict[str, Any]:
    """Extract rich content with durable resume points and bounded concurrency."""
    data_path = Path(data_path).resolve()
    pdf_path = Path(pdf_path).resolve()
    directory = data_path.parent
    root = Path(repository_root).resolve() if repository_root is not None else None
    asset_output = Path(assets_directory).resolve() if assets_directory is not None else None
    data = json.loads(data_path.read_text(encoding="utf-8"))
    labels = [str(label).strip() for label in data.get("opcoes_resposta", [])]
    if not labels:
        raise ValueError("opcoes_resposta is missing or empty.")
    questions = data.get("questoes")
    if not isinstance(questions, dict):
        raise ValueError("questoes must be an object.")

    workers = int(max_workers)
    if not 1 <= workers <= MAX_RICH_WORKERS:
        raise ValueError(f"max_workers must be between 1 and {MAX_RICH_WORKERS}.")

    pdf_digest = _sha256(pdf_path)
    previous_metadata = data.get("rich_extraction")
    previous_failures = (
        previous_metadata.get("failures", {})
        if isinstance(previous_metadata, dict)
        else {}
    )
    failures = {
        str(number): str(message)
        for number, message in previous_failures.items()
    } if isinstance(previous_failures, dict) else {}
    reused = 0
    pending: list[tuple[str, int, dict[str, Any], dict[str, Any] | None]] = []

    for raw_number, question in questions.items():
        number = int(raw_number)
        if question_numbers is not None and number not in question_numbers:
            continue
        if not isinstance(question, dict):
            failures[raw_number] = "Question is not an object."
            continue
        raw_content = question.get("conteudo")
        existing_rich = raw_content.get("rich") if isinstance(raw_content, dict) else None
        if not force and _can_reuse_rich_content(
            existing_rich,
            labels=labels,
            pdf_sha256=pdf_digest,
            use_gemini=use_gemini,
            model_name=model_name,
        ):
            _trim_unused_assets(existing_rich)
            failures.pop(raw_number, None)
            reused += 1
            continue
        fallback = existing_rich if _can_reuse_rich_content(
            existing_rich,
            labels=labels,
            pdf_sha256=pdf_digest,
            use_gemini=False,
            model_name=model_name,
        ) else None
        pending.append((raw_number, number, question, fallback))

    processed = len(pending)

    def checkpoint() -> None:
        _set_metadata(
            data,
            questions=questions,
            labels=labels,
            pdf_sha256=pdf_digest,
            processed=processed,
            reused=reused,
            failures=failures,
            use_gemini=use_gemini,
            model_name=model_name,
        )
        if write:
            _atomic_write_json(data_path, data)

    def apply_result(
        raw_number: str,
        question: dict[str, Any],
        fallback: dict[str, Any] | None,
        rich: dict[str, Any] | None,
        error: BaseException | None,
    ) -> None:
        if error is None and rich is not None:
            question.setdefault("conteudo", {})["rich"] = rich
            failures.pop(raw_number, None)
        else:
            content = question.setdefault("conteudo", {})
            if fallback is None:
                content.pop("rich", None)
            else:
                content["rich"] = fallback
            failures[raw_number] = str(error or "Unknown rich extraction failure.")
        checkpoint()

    if not pending:
        checkpoint()
        return data

    api_key: str | None = None
    if use_gemini:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required for --use-gemini.")

    if use_gemini and workers > 1:
        from google import genai

        thread_state = threading.local()

        def client_for_question(number: int) -> _RetryingGeminiClient:
            client = getattr(thread_state, "gemini_client", None)
            if client is None:
                client = genai.Client(api_key=api_key)
                thread_state.gemini_client = client
            return _RetryingGeminiClient(client, question_number=number)

        def process_one(
            raw_number: str,
            number: int,
            question: dict[str, Any],
            fallback: dict[str, Any] | None,
        ) -> tuple[str, dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
            with pymupdf.open(pdf_path) as document:
                rich = extract_question_rich_content(
                    document=document,
                    directory=directory,
                    assets_directory=asset_output,
                    repository_root=root,
                    number=number,
                    question=question,
                    labels=labels,
                    pdf_sha256=pdf_digest,
                    use_gemini=True,
                    model_name=model_name,
                    gemini_client=client_for_question(number),
                )
            return raw_number, question, fallback, rich

        futures: dict[
            Future[tuple[str, dict[str, Any], dict[str, Any] | None, dict[str, Any]]],
            tuple[str, dict[str, Any], dict[str, Any] | None],
        ] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for raw_number, number, question, fallback in pending:
                future = executor.submit(
                    process_one, raw_number, number, question, fallback
                )
                futures[future] = (raw_number, question, fallback)

            for future in as_completed(futures):
                raw_number, question, fallback = futures[future]
                try:
                    _, _, _, rich = future.result()
                except Exception as error:
                    apply_result(raw_number, question, fallback, None, error)
                else:
                    apply_result(raw_number, question, fallback, rich, None)
    else:
        gemini_client: Any | None = None
        if use_gemini:
            from google import genai

            gemini_client = genai.Client(api_key=api_key)

        with pymupdf.open(pdf_path) as document:
            for raw_number, number, question, fallback in pending:
                try:
                    client = (
                        _RetryingGeminiClient(
                            gemini_client,
                            question_number=number,
                        )
                        if use_gemini
                        else None
                    )
                    rich = extract_question_rich_content(
                        document=document,
                        directory=directory,
                        assets_directory=asset_output,
                        repository_root=root,
                        number=number,
                        question=question,
                        labels=labels,
                        pdf_sha256=pdf_digest,
                        use_gemini=use_gemini,
                        model_name=model_name,
                        gemini_client=client,
                    )
                except Exception as error:
                    apply_result(raw_number, question, fallback, None, error)
                else:
                    apply_result(raw_number, question, fallback, rich, None)

    checkpoint()
    return data

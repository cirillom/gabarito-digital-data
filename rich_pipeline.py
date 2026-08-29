"""Reliable orchestration for rich question extraction.

The low-level rich-content extractor stays focused on one question. This module
adds exam-level concerns: durable per-question checkpoints, bounded Gemini
retries, limited concurrency, progress reporting, and graceful interruption.
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
from typing import Any, Callable

import pymupdf
from dotenv import load_dotenv
from tqdm.auto import tqdm

from rich_content import (
    QuotaExceededError,
    RICH_EXTRACTION_VERSION,
    RichProvider,
    default_model_for_provider,
    extract_question_rich_content,
    is_quota_error,
    resolve_rich_provider,
    validate_rich_content,
)


DEFAULT_RICH_WORKERS = 2
MAX_RICH_WORKERS = 4
_RETRYABLE_API_CODES = {408, 500, 502, 503, 504}


class _StopRequested(Exception):
    """Internal cooperative-cancellation signal for worker threads."""


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
    provider: RichProvider | None,
    model_name: str,
) -> bool:
    if not isinstance(content, dict) or content.get("source_pdf_sha256") != pdf_sha256:
        return False
    if provider is not None and content.get("method") != f"{provider}:{model_name}":
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
    """Use explicit structured output and explicitly disable unused AFC."""
    from google.genai import types

    normalized = (
        config.model_copy(deep=True)
        if hasattr(config, "model_copy")
        else config
    )
    fields = getattr(type(normalized), "model_fields", {})
    if "automatic_function_calling" in fields:
        normalized.automatic_function_calling = types.AutomaticFunctionCallingConfig(
            disable=True
        )

    schema = getattr(normalized, "response_schema", None)
    if (
        schema is not None
        and hasattr(schema, "model_json_schema")
        and "response_json_schema" in fields
    ):
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
    if code not in _RETRYABLE_API_CODES and not _is_timeout(error):
        return None
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
        stop_event: threading.Event | None = None,
        status_callback: Callable[[str], None] | None = None,
    ) -> None:
        self._models = models
        self._question_number = question_number
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._stop_event = stop_event
        self._status_callback = status_callback

    def _check_stop(self) -> None:
        if self._stop_event is not None and self._stop_event.is_set():
            raise _StopRequested()

    def generate_content(self, *args: Any, **kwargs: Any) -> Any:
        if "config" in kwargs:
            kwargs["config"] = _normalize_structured_config(kwargs["config"])

        for attempt in range(1, self._max_attempts + 1):
            self._check_stop()
            try:
                return self._models.generate_content(*args, **kwargs)
            except Exception as error:
                if isinstance(error, _StopRequested):
                    raise
                if is_quota_error(error):
                    raise QuotaExceededError(
                        f"Gemini quota/rate limit reached while extracting "
                        f"question {self._question_number}."
                    ) from error
                reason = _retry_reason(error)
                if reason is None:
                    raise
                if attempt >= self._max_attempts:
                    raise RuntimeError(
                        f"Gemini {reason} for question {self._question_number} "
                        f"after {attempt} attempts: {error}"
                    ) from error

                self._check_stop()
                delay = _retry_delay(
                    attempt - 1,
                    base_delay=self._base_delay,
                    max_delay=self._max_delay,
                )
                if self._status_callback is not None:
                    self._status_callback(
                        f"q{self._question_number}: {reason}; retry in {delay:.1f}s"
                    )
                if self._stop_event is None:
                    time.sleep(delay)
                elif self._stop_event.wait(delay):
                    raise _StopRequested()
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
        stop_event: threading.Event | None = None,
        status_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.models = _RetryingModels(
            client.models,
            question_number=question_number,
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            stop_event=stop_event,
            status_callback=status_callback,
        )


class _RetryingResponses:
    def __init__(
        self,
        responses: Any,
        *,
        question_number: int,
        max_attempts: int,
        base_delay: float,
        max_delay: float,
        stop_event: threading.Event | None = None,
        status_callback: Callable[[str], None] | None = None,
    ) -> None:
        self._responses = responses
        self._question_number = question_number
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._stop_event = stop_event
        self._status_callback = status_callback

    def _check_stop(self) -> None:
        if self._stop_event is not None and self._stop_event.is_set():
            raise _StopRequested()

    def parse(self, *args: Any, **kwargs: Any) -> Any:
        for attempt in range(1, self._max_attempts + 1):
            self._check_stop()
            try:
                return self._responses.parse(*args, **kwargs)
            except Exception as error:
                if isinstance(error, _StopRequested):
                    raise
                if is_quota_error(error):
                    raise QuotaExceededError(
                        f"OpenAI quota/rate limit reached while extracting "
                        f"question {self._question_number}."
                    ) from error
                reason = _retry_reason(error)
                if reason is None:
                    raise
                if attempt >= self._max_attempts:
                    raise RuntimeError(
                        f"OpenAI {reason} for question {self._question_number} "
                        f"after {attempt} attempts: {error}"
                    ) from error

                self._check_stop()
                delay = _retry_delay(
                    attempt - 1,
                    base_delay=self._base_delay,
                    max_delay=self._max_delay,
                )
                if self._status_callback is not None:
                    self._status_callback(
                        f"q{self._question_number}: {reason}; retry in {delay:.1f}s"
                    )
                if self._stop_event is None:
                    time.sleep(delay)
                elif self._stop_event.wait(delay):
                    raise _StopRequested()
        raise AssertionError("unreachable")


class _RetryingOpenAIClient:
    def __init__(
        self,
        client: Any,
        *,
        question_number: int,
        max_attempts: int = 5,
        base_delay: float = 2.0,
        max_delay: float = 20.0,
        stop_event: threading.Event | None = None,
        status_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.responses = _RetryingResponses(
            client.responses,
            question_number=question_number,
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            stop_event=stop_event,
            status_callback=status_callback,
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
            provider=None,
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
    provider: RichProvider | None,
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
        "method": f"{provider}:{model_name}" if provider else "deterministic",
        **({"failures": failures} if failures else {}),
    }


def enrich_rich_data_file(
    data_path: str | Path,
    pdf_path: str | Path,
    *,
    repository_root: str | Path | None = None,
    question_numbers: set[int] | None = None,
    use_gemini: bool = False,
    provider: RichProvider | None = None,
    model_name: str | None = None,
    assets_directory: str | Path | None = None,
    write: bool = True,
    force: bool = False,
    max_workers: int = DEFAULT_RICH_WORKERS,
    progress: bool = True,
    progress_position: int = 0,
    progress_desc: str | None = None,
) -> dict[str, Any]:
    """Extract rich content with durable resume points and bounded concurrency."""
    provider = resolve_rich_provider(provider, use_gemini=use_gemini)
    model_name = model_name or default_model_for_provider(provider)
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
    failures = (
        {
            str(number): str(message)
            for number, message in previous_failures.items()
        }
        if isinstance(previous_failures, dict)
        else {}
    )
    reused = 0
    selected_keys: set[str] = set()
    pending: list[tuple[str, int, dict[str, Any], dict[str, Any] | None]] = []

    for raw_number, question in questions.items():
        number = int(raw_number)
        if question_numbers is not None and number not in question_numbers:
            continue
        selected_keys.add(raw_number)
        if not isinstance(question, dict):
            failures[raw_number] = "Question is not an object."
            continue
        raw_content = question.get("conteudo")
        existing_rich = raw_content.get("rich") if isinstance(raw_content, dict) else None
        if not force and _can_reuse_rich_content(
            existing_rich,
            labels=labels,
            pdf_sha256=pdf_digest,
            provider=provider,
            model_name=model_name,
        ):
            _trim_unused_assets(existing_rich)
            failures.pop(raw_number, None)
            reused += 1
            continue
        fallback = (
            existing_rich
            if _can_reuse_rich_content(
                existing_rich,
                labels=labels,
                pdf_sha256=pdf_digest,
                provider=None,
                model_name=model_name,
            )
            else None
        )
        pending.append((raw_number, number, question, fallback))

    processed = 0
    selected_count = len(selected_keys)
    already_complete = selected_count - len(pending)
    stop_event = threading.Event()
    progress_lock = threading.RLock()
    question_bar = tqdm(
        total=selected_count,
        initial=already_complete,
        desc=progress_desc or f"{directory.name} questions",
        unit="question",
        position=progress_position,
        leave=False,
        dynamic_ncols=True,
        disable=not progress,
    )

    def current_failure_count() -> int:
        return sum(number in failures for number in selected_keys)

    def set_progress_status(status: str | None = None) -> None:
        postfix: dict[str, Any] = {
            "reused": reused,
            "failed": current_failure_count(),
        }
        if status:
            postfix["status"] = status
        with progress_lock:
            question_bar.set_postfix(postfix, refresh=True)

    def checkpoint() -> None:
        _set_metadata(
            data,
            questions=questions,
            labels=labels,
            pdf_sha256=pdf_digest,
            processed=processed,
            reused=reused,
            failures=failures,
            provider=provider,
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
        nonlocal processed
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
        processed += 1
        checkpoint()
        with progress_lock:
            question_bar.update(1)
        set_progress_status()

    try:
        set_progress_status()
        if not pending:
            checkpoint()
            return data

        api_key: str | None = None
        if provider is not None:
            load_dotenv()
            key_name = "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"
            api_key = os.getenv(key_name)
            if not api_key:
                raise RuntimeError(f"{key_name} is required for the {provider} provider.")

        def create_api_client() -> Any:
            if provider == "gemini":
                from google import genai

                return genai.Client(api_key=api_key)
            if provider == "openai":
                from openai import OpenAI

                # The pipeline owns retries so a 429 is never retried inside the SDK.
                return OpenAI(api_key=api_key, max_retries=0)
            return None

        def retrying_client(client: Any, number: int) -> Any:
            wrapper = (
                _RetryingGeminiClient if provider == "gemini" else _RetryingOpenAIClient
            )
            return wrapper(
                client,
                question_number=number,
                stop_event=stop_event,
                status_callback=set_progress_status,
            )

        def extraction_client_arguments(client: Any) -> dict[str, Any]:
            if provider == "gemini":
                return {"gemini_client": client}
            if provider == "openai":
                return {"openai_client": client}
            return {}

        if provider is not None and workers > 1:

            thread_state = threading.local()

            def client_for_question(number: int) -> Any:
                client = getattr(thread_state, "provider_client", None)
                if client is None:
                    client = create_api_client()
                    thread_state.provider_client = client
                return retrying_client(client, number)

            def process_one(
                raw_number: str,
                number: int,
                question: dict[str, Any],
                fallback: dict[str, Any] | None,
            ) -> tuple[str, dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
                if stop_event.is_set():
                    raise _StopRequested()
                with pymupdf.open(pdf_path) as document:
                    client_arguments = extraction_client_arguments(
                        client_for_question(number)
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
                        provider=provider,
                        model_name=model_name,
                        **client_arguments,
                    )
                return raw_number, question, fallback, rich

            futures: dict[
                Future[tuple[str, dict[str, Any], dict[str, Any] | None, dict[str, Any]]],
                tuple[str, dict[str, Any], dict[str, Any] | None],
            ] = {}
            handled: set[Future[Any]] = set()
            executor = ThreadPoolExecutor(max_workers=workers)
            interrupted = False
            quota_error: QuotaExceededError | None = None
            try:
                for raw_number, number, question, fallback in pending:
                    future = executor.submit(
                        process_one, raw_number, number, question, fallback
                    )
                    futures[future] = (raw_number, question, fallback)

                try:
                    for future in as_completed(futures):
                        handled.add(future)
                        raw_number, question, fallback = futures[future]
                        if future.cancelled():
                            continue
                        try:
                            _, _, _, rich = future.result()
                        except _StopRequested:
                            continue
                        except QuotaExceededError as error:
                            quota_error = error
                            stop_event.set()
                            set_progress_status("quota reached; stopping")
                            for queued in futures:
                                if queued not in handled:
                                    queued.cancel()
                            break
                        except Exception as error:
                            apply_result(raw_number, question, fallback, None, error)
                        else:
                            apply_result(raw_number, question, fallback, rich, None)
                except KeyboardInterrupt:
                    interrupted = True
                    stop_event.set()
                    set_progress_status("stopping; waiting for in-flight requests")
                    tqdm.write(
                        "Ctrl-C received: cancelling queued questions and preserving checkpoints."
                    )
                    for future in futures:
                        if future not in handled:
                            future.cancel()

                    remaining = [
                        future
                        for future in futures
                        if future not in handled and not future.cancelled()
                    ]
                    for future in as_completed(remaining):
                        handled.add(future)
                        raw_number, question, fallback = futures[future]
                        try:
                            _, _, _, rich = future.result()
                        except _StopRequested:
                            continue
                        except Exception as error:
                            apply_result(raw_number, question, fallback, None, error)
                        else:
                            apply_result(raw_number, question, fallback, rich, None)
                    checkpoint()
            finally:
                executor.shutdown(wait=True, cancel_futures=True)

            if interrupted:
                raise KeyboardInterrupt
            if quota_error is not None:
                checkpoint()
                raise quota_error
        else:
            api_client = create_api_client()

            with pymupdf.open(pdf_path) as document:
                for raw_number, number, question, fallback in pending:
                    try:
                        client = (
                            retrying_client(api_client, number)
                            if provider is not None
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
                            provider=provider,
                            model_name=model_name,
                            **extraction_client_arguments(client),
                        )
                    except KeyboardInterrupt:
                        stop_event.set()
                        set_progress_status("stopped")
                        checkpoint()
                        tqdm.write(
                            "Ctrl-C received: saved completed rich questions; "
                            "resume with the same command."
                        )
                        raise
                    except QuotaExceededError:
                        stop_event.set()
                        set_progress_status("quota reached; stopping")
                        checkpoint()
                        raise
                    except Exception as error:
                        apply_result(raw_number, question, fallback, None, error)
                    else:
                        apply_result(raw_number, question, fallback, rich, None)

        checkpoint()
        return data
    finally:
        question_bar.close()

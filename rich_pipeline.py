"""Resumable, bounded orchestration for rich question extraction."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import hashlib
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
    DEFAULT_GEMINI_MODEL_NAME,
    QuotaExceededError,
    extract_question_rich_content,
    is_quota_error,
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


def _referenced_asset_ids(
    statement: dict[str, Any], options: list[dict[str, Any]]
) -> set[str]:
    referenced: set[str] = set()
    for document in [statement, *(option["content"] for option in options)]:
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
    content["assets"] = [asset for asset in content["assets"] if asset["id"] in referenced]


def _normalize_structured_config(config: Any) -> Any:
    """Use explicit structured output and disable unused automatic tool calls."""
    from google.genai import types

    normalized = config.model_copy(deep=True) if hasattr(config, "model_copy") else config
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
    response = getattr(error, "response", None)
    for value in (
        getattr(error, "code", None),
        getattr(error, "status_code", None),
        getattr(response, "status_code", None),
    ):
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
    return None


def _retry_reason(error: BaseException) -> str | None:
    code = _error_code(error)
    is_timeout = isinstance(error, TimeoutError) or "timeout" in type(error).__name__.lower()
    if code not in _RETRYABLE_API_CODES and not is_timeout:
        return None
    if code == 503:
        return "service unavailable/model demand"
    if code is not None:
        return f"temporary HTTP {code} error"
    return "request timeout"


def _retry_delay(retry_index: int, *, base_delay: float, max_delay: float) -> float:
    exponential = min(max_delay, base_delay * (2**retry_index))
    jitter = random.uniform(0.0, min(1.0, exponential * 0.25))
    return min(max_delay, exponential + jitter)


def retry_gemini_call(
    operation: Callable[[], Any],
    *,
    label: str,
    max_attempts: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 20.0,
    stop_event: threading.Event | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> Any:
    """Run one Gemini operation with bounded transient-error retries."""
    for attempt in range(1, max_attempts + 1):
        if stop_event is not None and stop_event.is_set():
            raise _StopRequested()
        try:
            return operation()
        except Exception as error:
            if isinstance(error, _StopRequested):
                raise
            if is_quota_error(error):
                raise QuotaExceededError(f"Gemini quota/rate limit reached for {label}.") from error
            reason = _retry_reason(error)
            if reason is None:
                raise
            if attempt >= max_attempts:
                raise RuntimeError(
                    f"Gemini {reason} for {label} after {attempt} attempts: {error}"
                ) from error
            delay = _retry_delay(
                attempt - 1, base_delay=base_delay, max_delay=max_delay
            )
            if status_callback is not None:
                status_callback(f"{label}: {reason}; retry in {delay:.1f}s")
            if stop_event is None:
                time.sleep(delay)
            elif stop_event.wait(delay):
                raise _StopRequested()
    raise AssertionError("unreachable")


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
        self._label = f"question {question_number}"
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._stop_event = stop_event
        self._status_callback = status_callback

    def generate_content(self, *args: Any, **kwargs: Any) -> Any:
        if "config" in kwargs:
            kwargs["config"] = _normalize_structured_config(kwargs["config"])
        return retry_gemini_call(
            lambda: self._models.generate_content(*args, **kwargs),
            label=self._label,
            max_attempts=self._max_attempts,
            base_delay=self._base_delay,
            max_delay=self._max_delay,
            stop_event=self._stop_event,
            status_callback=self._status_callback,
        )


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


def enrich_rich_exam(
    data: dict[str, Any],
    pdf_path: str | Path,
    *,
    directory: str | Path,
    repository_root: str | Path | None = None,
    question_numbers: set[int] | None = None,
    use_gemini: bool = False,
    model_name: str | None = None,
    assets_directory: str | Path | None = None,
    force: bool = False,
    max_workers: int = DEFAULT_RICH_WORKERS,
    progress: bool = True,
    progress_position: int = 0,
    progress_desc: str | None = None,
    checkpoint: Callable[[int, dict[str, Any] | None], None] | None = None,
    result_callback: Callable[[int, BaseException | None], None] | None = None,
) -> dict[str, Any]:
    """Extract selected questions and commit every completed result immediately."""
    model_name = model_name or DEFAULT_GEMINI_MODEL_NAME
    pdf_path = Path(pdf_path).resolve()
    directory = Path(directory).resolve()
    root = Path(repository_root).resolve() if repository_root is not None else None
    asset_output = Path(assets_directory).resolve() if assets_directory is not None else None
    labels = [str(label).strip() for label in data.get("opcoes_resposta", [])]
    questions = data.get("questoes")
    if not labels:
        raise ValueError("opcoes_resposta is missing or empty.")
    if not isinstance(questions, dict):
        raise ValueError("questoes must be an object.")
    workers = int(max_workers)
    if not 1 <= workers <= MAX_RICH_WORKERS:
        raise ValueError(f"max_workers must be between 1 and {MAX_RICH_WORKERS}.")

    pdf_digest = _sha256(pdf_path)
    reused = 0
    failures: dict[str, str] = {}
    pending: list[tuple[str, int, dict[str, Any], dict[str, Any] | None]] = []
    selected_count = 0
    for raw_number, question in questions.items():
        number = int(raw_number)
        if question_numbers is not None and number not in question_numbers:
            continue
        selected_count += 1
        if not isinstance(question, dict):
            failures[raw_number] = "Question is not an object."
            continue
        raw_content = question.get("conteudo")
        existing = raw_content.get("rich") if isinstance(raw_content, dict) else None
        if not force and _can_reuse_rich_content(
            existing,
            labels=labels,
            pdf_sha256=pdf_digest,
            use_gemini=use_gemini,
            model_name=model_name,
        ):
            _trim_unused_assets(existing)
            reused += 1
            continue
        fallback = (
            existing
            if _can_reuse_rich_content(
                existing,
                labels=labels,
                pdf_sha256=pdf_digest,
                use_gemini=False,
                model_name=model_name,
            )
            else None
        )
        pending.append((raw_number, number, question, fallback))

    processed = 0
    successful = 0
    stop_event = threading.Event()
    progress_lock = threading.RLock()
    question_bar = tqdm(
        total=selected_count,
        initial=selected_count - len(pending),
        desc=progress_desc or f"{directory.name} questions",
        unit="question",
        position=progress_position,
        leave=False,
        dynamic_ncols=True,
        disable=not progress,
    )

    def set_progress_status(status: str | None = None) -> None:
        postfix: dict[str, Any] = {"reused": reused, "failed": len(failures)}
        if status:
            postfix["status"] = status
        with progress_lock:
            question_bar.set_postfix(postfix, refresh=True)

    def apply_result(
        raw_number: str,
        number: int,
        question: dict[str, Any],
        fallback: dict[str, Any] | None,
        rich: dict[str, Any] | None,
        error: BaseException | None,
    ) -> None:
        nonlocal processed, successful
        content = question.setdefault("conteudo", {})
        if error is None and rich is not None:
            content["rich"] = rich
            failures.pop(raw_number, None)
            successful += 1
        else:
            if fallback is None:
                content.pop("rich", None)
            else:
                content["rich"] = fallback
            failures[raw_number] = str(error or "Unknown rich extraction failure.")
        if checkpoint is not None:
            checkpoint(number, content.get("rich"))
        if result_callback is not None:
            result_callback(number, error)
        processed += 1
        with progress_lock:
            question_bar.update(1)
        set_progress_status()

    try:
        set_progress_status()
        if not pending:
            return {
                "selected": selected_count,
                "processed": processed,
                "successful": successful,
                "reused": reused,
                "failures": failures,
            }

        api_key: str | None = None
        if use_gemini:
            load_dotenv()
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY is required for Gemini rich extraction.")

        def create_api_client() -> Any:
            if not use_gemini:
                return None
            from google import genai

            return genai.Client(api_key=api_key)

        def process_one(
            raw_number: str,
            number: int,
            question: dict[str, Any],
            fallback: dict[str, Any] | None,
            client: Any,
        ) -> tuple[str, int, dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
            if stop_event.is_set():
                raise _StopRequested()
            if callable(client):
                client = client(number)
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
                    use_gemini=use_gemini,
                    model_name=model_name,
                    gemini_client=client,
                )
            return raw_number, number, question, fallback, rich

        if use_gemini and workers > 1:
            thread_state = threading.local()

            def client_for_question(number: int) -> Any:
                client = getattr(thread_state, "gemini_client", None)
                if client is None:
                    client = create_api_client()
                    thread_state.gemini_client = client
                return _RetryingGeminiClient(
                    client,
                    question_number=number,
                    stop_event=stop_event,
                    status_callback=set_progress_status,
                )

            futures: dict[
                Future[tuple[str, int, dict[str, Any], dict[str, Any] | None, dict[str, Any]]],
                tuple[str, int, dict[str, Any], dict[str, Any] | None],
            ] = {}
            handled: set[Future[Any]] = set()
            executor = ThreadPoolExecutor(max_workers=workers)
            interrupted = False
            quota_error: QuotaExceededError | None = None
            try:
                for raw_number, number, question, fallback in pending:
                    future = executor.submit(
                        process_one,
                        raw_number,
                        number,
                        question,
                        fallback,
                        client_for_question,
                    )
                    futures[future] = (raw_number, number, question, fallback)
                try:
                    for future in as_completed(futures):
                        handled.add(future)
                        raw_number, number, question, fallback = futures[future]
                        if future.cancelled():
                            continue
                        try:
                            _, _, _, _, rich = future.result()
                        except _StopRequested:
                            continue
                        except QuotaExceededError as error:
                            apply_result(raw_number, number, question, fallback, None, error)
                            quota_error = error
                            stop_event.set()
                            set_progress_status("quota reached; stopping")
                            for queued in futures:
                                if queued not in handled:
                                    queued.cancel()
                            break
                        except Exception as error:
                            apply_result(raw_number, number, question, fallback, None, error)
                        else:
                            apply_result(raw_number, number, question, fallback, rich, None)
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
                        raw_number, number, question, fallback = futures[future]
                        try:
                            _, _, _, _, rich = future.result()
                        except _StopRequested:
                            continue
                        except Exception as error:
                            apply_result(raw_number, number, question, fallback, None, error)
                        else:
                            apply_result(raw_number, number, question, fallback, rich, None)
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
            if interrupted:
                raise KeyboardInterrupt
            if quota_error is not None:
                raise quota_error
        else:
            api_client = create_api_client()
            with pymupdf.open(pdf_path) as document:
                for raw_number, number, question, fallback in pending:
                    client = (
                        _RetryingGeminiClient(
                            api_client,
                            question_number=number,
                            stop_event=stop_event,
                            status_callback=set_progress_status,
                        )
                        if use_gemini
                        else None
                    )
                    try:
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
                    except KeyboardInterrupt:
                        stop_event.set()
                        set_progress_status("stopped")
                        tqdm.write(
                            "Ctrl-C received: saved completed rich questions; resume with the same command."
                        )
                        raise
                    except QuotaExceededError as error:
                        apply_result(raw_number, number, question, fallback, None, error)
                        stop_event.set()
                        set_progress_status("quota reached; stopping")
                        raise
                    except Exception as error:
                        apply_result(raw_number, number, question, fallback, None, error)
                    else:
                        apply_result(raw_number, number, question, fallback, rich, None)
        return {
            "selected": selected_count,
            "processed": processed,
            "successful": successful,
            "reused": reused,
            "failures": failures,
        }
    finally:
        question_bar.close()

"""Generate, validate, and package the SQLite exam catalog."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
from typing import Any, TextIO

from tqdm.auto import tqdm

from catalog_db import (
    CATALOG_FILENAME,
    database_stats,
    exam_id_for_directory,
    exam_label,
    load_exam,
    open_catalog,
    optimize,
    prepare_release,
    replace_exam,
    replace_layout,
    validate_catalog,
    write_rich_content,
)
from pdf_parser import parse_exam_directory
from question_layout import extract_question_layout
from rich_content import QuotaExceededError, is_quota_error, write_html_preview
from rich_pipeline import (
    DEFAULT_RICH_WORKERS,
    MAX_RICH_WORKERS,
    enrich_rich_exam,
)


ROOT_DIR = Path(__file__).resolve().parent
CATALOG_PATH = ROOT_DIR / CATALOG_FILENAME
LOG_DIR = ROOT_DIR / "logs"


def find_exam_directories(root_dir: Path = ROOT_DIR) -> list[Path]:
    """Find directories containing both required exam PDFs."""
    return sorted(
        prova.parent
        for prova in root_dir.rglob("prova.pdf")
        if (prova.parent / "gabarito.pdf").is_file()
    )


def partition_exam_directories(
    connection: Any,
    directories: list[Path],
    *,
    regenerate_all: bool,
    repository_root: Path = ROOT_DIR,
) -> tuple[list[Path], list[Path]]:
    """Return exams to refresh locally and exams to regenerate with Gemini."""
    if regenerate_all:
        return [], directories
    existing: list[Path] = []
    missing: list[Path] = []
    for directory in directories:
        target = (
            existing
            if exam_id_for_directory(
                connection, directory, repository_root=repository_root
            )
            is not None
            else missing
        )
        target.append(directory)
    return existing, missing


def _new_log() -> tuple[Path, TextIO]:
    LOG_DIR.mkdir(exist_ok=True)
    path = LOG_DIR / f"generation-{datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}.log"
    log = path.open("w", encoding="utf-8", newline="\n")
    log.write("timestamp\texam\tquestion\tstage\terror\n")
    log.flush()
    return path, log


def _record_failure(
    failures: list[dict[str, Any]],
    log: TextIO,
    *,
    exam: str,
    question: int | None,
    stage: str,
    error: BaseException | str,
) -> None:
    message = " ".join(str(error).splitlines()).strip()
    failure = {
        "exam": exam,
        "question": question,
        "stage": stage,
        "error": message,
    }
    failures.append(failure)
    log.write(
        f"{datetime.now().isoformat(timespec='seconds')}\t{exam}\t"
        f"{question if question is not None else '-'}\t{stage}\t{message}\n"
    )
    log.flush()


def _print_stats(database_path: Path) -> None:
    stats = database_stats(database_path)
    tqdm.write("Database:")
    tqdm.write(f"- size: {stats['size']} bytes")
    tqdm.write(f"- exams: {stats['exams']}")
    tqdm.write(f"- questions: {stats['questions']}")
    tqdm.write(f"- question_content rows: {stats['question_content']}")
    tqdm.write(
        f"- question_rich_content rows: {stats['question_rich_content']}"
    )


def _print_summary(
    exams: list[str],
    question_counts: dict[str, int],
    failures: list[dict[str, Any]],
    log_path: Path,
) -> None:
    tqdm.write("\nExams processed:")
    for exam in exams:
        tqdm.write(f"- {exam}")
    tqdm.write("\nQuestions:")
    tqdm.write(f"- total attempted: {question_counts['attempted']}")
    tqdm.write(f"- successfully extracted: {question_counts['successful']}")
    tqdm.write(f"- failed: {question_counts['failed']}")
    tqdm.write("\nFailed questions:")
    question_failures = [failure for failure in failures if failure["question"] is not None]
    if question_failures:
        for failure in question_failures:
            tqdm.write(
                f"- {failure['exam']} / question {failure['question']} "
                f"({failure['stage']})"
            )
    else:
        tqdm.write("- none")
    exam_failures = [failure for failure in failures if failure["question"] is None]
    if exam_failures:
        tqdm.write("\nExam-level failures:")
        for failure in exam_failures:
            tqdm.write(f"- {failure['exam']} ({failure['stage']}): {failure['error']}")
    tqdm.write(f"\nFailure log: {log_path.relative_to(ROOT_DIR)}")


def run(
    *,
    directories: list[Path] | None = None,
    database_path: Path = CATALOG_PATH,
    regenerate_all: bool = False,
    rich_content: bool = False,
    gemini_rich: bool = False,
    rich_model: str | None = None,
    rich_questions: set[int] | None = None,
    preview_dir: Path | None = None,
    rich_workers: int = DEFAULT_RICH_WORKERS,
) -> None:
    directories = directories if directories is not None else find_exam_directories()
    if not directories:
        tqdm.write("No directories containing prova.pdf and gabarito.pdf were found.")
        return

    log_path, log = _new_log()
    failures: list[dict[str, Any]] = []
    processed_exams: list[str] = []
    question_counts = {"attempted": 0, "successful": 0, "failed": 0}
    connection = open_catalog(database_path)
    terminal_error: BaseException | None = None
    existing_directories, generation_directories = partition_exam_directories(
        connection,
        directories,
        regenerate_all=regenerate_all,
        repository_root=ROOT_DIR,
    )
    overall = tqdm(
        total=len(existing_directories)
        + len(generation_directories)
        + (len(directories) if rich_content else 0),
        desc="Repository",
        unit="exam-step",
        position=0,
        dynamic_ncols=True,
    )
    try:
        for directory in tqdm(
            existing_directories,
            desc="Layouts",
            unit="exam",
            position=1,
            leave=False,
            dynamic_ncols=True,
        ):
            exam_id = exam_id_for_directory(
                connection, directory, repository_root=ROOT_DIR
            )
            assert exam_id is not None
            label = exam_label(connection, exam_id)
            if label not in processed_exams:
                processed_exams.append(label)
            try:
                data = load_exam(connection, exam_id)
                questions = data["questoes"]
                layouts = extract_question_layout(
                    directory / "prova.pdf",
                    (int(number) for number in questions),
                    raw_questions=questions,
                )
                replace_layout(connection, exam_id, layouts)
            except Exception as error:
                replace_layout(connection, exam_id, None)
                _record_failure(
                    failures,
                    log,
                    exam=label,
                    question=None,
                    stage="layout",
                    error=error,
                )
                tqdm.write(f"{label}: layout extraction failed: {error}")
            finally:
                overall.update(1)

        for directory in tqdm(
            generation_directories,
            desc="Generating base data",
            unit="exam",
            position=1,
            leave=False,
            dynamic_ncols=True,
        ):
            relative = directory.relative_to(ROOT_DIR).as_posix()
            try:
                data = parse_exam_directory(directory)
                exam_id = replace_exam(
                    connection, directory, data, repository_root=ROOT_DIR
                )
                label = exam_label(connection, exam_id)
                if label not in processed_exams:
                    processed_exams.append(label)
                if not rich_content:
                    count = len(data["questoes"])
                    question_counts["attempted"] += count
                    question_counts["successful"] += count
            except Exception as error:
                if is_quota_error(error):
                    raise QuotaExceededError(
                        f"Gemini quota/rate limit reached while generating {relative}."
                    ) from error
                _record_failure(
                    failures,
                    log,
                    exam=relative,
                    question=None,
                    stage="base extraction",
                    error=error,
                )
                tqdm.write(f"{relative}: base extraction failed: {error}")
            finally:
                overall.update(1)

        if rich_content:
            for directory in tqdm(
                directories,
                desc="Rich exams",
                unit="exam",
                position=1,
                leave=False,
                dynamic_ncols=True,
            ):
                exam_id = exam_id_for_directory(
                    connection, directory, repository_root=ROOT_DIR
                )
                if exam_id is None:
                    overall.update(1)
                    continue
                label = exam_label(connection, exam_id)
                if label not in processed_exams:
                    processed_exams.append(label)
                data = load_exam(connection, exam_id)

                def result_callback(
                    number: int,
                    error: BaseException | None,
                    *,
                    current_label: str = label,
                ) -> None:
                    question_counts["attempted"] += 1
                    if error is None:
                        question_counts["successful"] += 1
                    else:
                        question_counts["failed"] += 1
                        _record_failure(
                            failures,
                            log,
                            exam=current_label,
                            question=number,
                            stage="rich extraction",
                            error=error,
                        )

                try:
                    enriched = enrich_rich_exam(
                        data,
                        directory / "prova.pdf",
                        directory=directory,
                        repository_root=ROOT_DIR,
                        question_numbers=rich_questions,
                        use_gemini=gemini_rich,
                        model_name=rich_model,
                        force=regenerate_all,
                        max_workers=rich_workers,
                        progress=True,
                        progress_position=2,
                        progress_desc=f"{label} questions",
                        checkpoint=lambda number, rich, current_exam=exam_id: write_rich_content(
                            connection, current_exam, number, rich
                        ),
                        result_callback=result_callback,
                    )
                    if enriched["failures"]:
                        tqdm.write(
                            f"{label}: {len(enriched['failures'])} rich question(s) failed"
                        )
                    if preview_dir is not None:
                        preview_path = (
                            preview_dir / directory.relative_to(ROOT_DIR) / "index.html"
                        )
                        write_html_preview(
                            load_exam(connection, exam_id),
                            directory=directory,
                            output_path=preview_path,
                            question_numbers=rich_questions,
                        )
                except (KeyboardInterrupt, QuotaExceededError):
                    raise
                except Exception as error:
                    _record_failure(
                        failures,
                        log,
                        exam=label,
                        question=None,
                        stage="rich extraction",
                        error=error,
                    )
                    tqdm.write(f"{label}: rich extraction failed: {error}")
                finally:
                    overall.update(1)
    except (KeyboardInterrupt, QuotaExceededError) as error:
        terminal_error = error
        overall.set_postfix_str(
            "interrupted" if isinstance(error, KeyboardInterrupt) else "quota reached",
            refresh=True,
        )
        tqdm.write(
            "Generation stopped; every completed question was already committed to SQLite."
        )
    finally:
        overall.close()
        try:
            optimize(connection)
        except Exception as error:
            _record_failure(
                failures,
                log,
                exam="catalog",
                question=None,
                stage="optimization",
                error=error,
            )
        connection.close()
        try:
            validate_catalog(database_path, repository_root=ROOT_DIR)
        except Exception as error:
            _record_failure(
                failures,
                log,
                exam="catalog",
                question=None,
                stage="validation",
                error=error,
            )
        _print_summary(processed_exams, question_counts, failures, log_path)
        if database_path.is_file():
            _print_stats(database_path)
        log.close()

    if terminal_error is not None:
        raise terminal_error
    if failures:
        raise SystemExit(1)


def _scoped_directories(scopes: list[Path] | None, parser: argparse.ArgumentParser) -> list[Path] | None:
    if not scopes:
        return None
    resolved_scopes: list[Path] = []
    for scope in scopes:
        resolved = scope.resolve()
        try:
            resolved.relative_to(ROOT_DIR)
        except ValueError:
            parser.error(f"--directory must be inside {ROOT_DIR}")
        resolved_scopes.append(resolved)
    return sorted(
        {
            exam_directory
            for scope in resolved_scopes
            for exam_directory in find_exam_directories(scope)
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=CATALOG_PATH)
    parser.add_argument("--regenerate-all", action="store_true")
    parser.add_argument(
        "--directory",
        type=Path,
        action="append",
        help="Limit generation to exams beneath this path; repeatable.",
    )
    parser.add_argument(
        "--rich-content",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Build optional rich question documents (default: disabled).",
    )
    parser.add_argument(
        "--gemini-rich",
        action="store_true",
        help="Use Gemini vision to improve rich content.",
    )
    parser.add_argument("--rich-model", help="Override the Gemini rich model.")
    parser.add_argument(
        "--rich-question",
        type=int,
        action="append",
        help="Limit rich extraction to one or more question numbers.",
    )
    parser.add_argument(
        "--rich-workers",
        type=int,
        default=DEFAULT_RICH_WORKERS,
        choices=range(1, MAX_RICH_WORKERS + 1),
        metavar="1-4",
    )
    parser.add_argument("--preview-dir", type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--validate", action="store_true")
    action.add_argument("--release-version")
    parser.add_argument("--source-commit")
    parser.add_argument("--output-directory", type=Path, default=ROOT_DIR / "dist")
    args = parser.parse_args()

    database_path = args.database.resolve()
    if args.validate:
        validate_catalog(database_path, repository_root=ROOT_DIR)
        tqdm.write("Catalog validation passed.")
        _print_stats(database_path)
        return
    if args.release_version:
        if not args.source_commit:
            parser.error("--release-version requires --source-commit")
        manifest = prepare_release(
            database_path,
            args.output_directory,
            version=args.release_version,
            source_commit=args.source_commit,
            repository_root=ROOT_DIR,
        )
        tqdm.write(
            f"Prepared release {manifest['version']} ({manifest['size']} bytes) "
            f"in {args.output_directory.resolve()}"
        )
        return
    if args.gemini_rich and not args.rich_content:
        parser.error("--gemini-rich requires --rich-content")
    if args.rich_model and not args.gemini_rich:
        parser.error("--rich-model requires --gemini-rich")
    if args.rich_question and not args.rich_content:
        parser.error("--rich-question requires --rich-content")

    try:
        run(
            directories=_scoped_directories(args.directory, parser),
            database_path=database_path,
            regenerate_all=args.regenerate_all,
            rich_content=args.rich_content,
            gemini_rich=args.gemini_rich,
            rich_model=args.rich_model,
            rich_questions=set(args.rich_question) if args.rich_question else None,
            preview_dir=args.preview_dir.resolve() if args.preview_dir else None,
            rich_workers=args.rich_workers,
        )
    except KeyboardInterrupt:
        tqdm.write("Generation stopped cleanly. Re-run the same command to resume.")
        raise SystemExit(130)
    except QuotaExceededError as error:
        tqdm.write(f"{error} Re-run the same command to resume.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

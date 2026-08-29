"""Generate exam data and rebuild the repository-wide data file."""

import argparse
import hashlib
import json
from pathlib import Path

from tqdm.auto import tqdm

from folder_parser import write_gabarito_json
from pdf_parser import parse_exam_directory
from question_layout import enrich_data_file
from rich_content import (
    RICH_EXTRACTION_VERSION,
    validate_rich_content,
    write_html_preview,
)
from rich_pipeline import (
    DEFAULT_RICH_WORKERS,
    MAX_RICH_WORKERS,
    enrich_rich_data_file,
)


ROOT_DIR = Path(__file__).resolve().parent
MAIN_DATA = ROOT_DIR / "data.json"


def find_exam_directories(root_dir: Path = ROOT_DIR) -> list[Path]:
    """Find directories containing both required exam PDFs."""
    return sorted(
        prova.parent
        for prova in root_dir.rglob("prova.pdf")
        if (prova.parent / "gabarito.pdf").is_file()
    )


def partition_exam_directories(
    directories: list[Path], *, regenerate_all: bool
) -> tuple[list[Path], list[Path]]:
    """Return directories to refresh locally and directories to regenerate with AI."""
    if regenerate_all:
        return [], directories
    return (
        [directory for directory in directories if (directory / "data.json").exists()],
        [directory for directory in directories if not (directory / "data.json").exists()],
    )


def validate_complete_rich_data(data: dict, pdf_path: Path) -> list[str]:
    """Return every structural or source mismatch in one rich exam document."""
    errors: list[str] = []
    questions = data.get("questoes")
    labels = data.get("opcoes_resposta")
    metadata = data.get("rich_extraction")
    if not isinstance(questions, dict) or not isinstance(labels, list):
        return ["exam data has no valid questions or answer-option list"]

    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    successful = 0
    for number, question in questions.items():
        rich = (
            question.get("conteudo", {}).get("rich")
            if isinstance(question, dict)
            else None
        )
        if not isinstance(rich, dict):
            errors.append(f"question {number} has no rich content")
            continue
        try:
            validate_rich_content(rich, labels)
        except ValueError as error:
            errors.append(f"question {number}: {error}")
            continue
        if rich.get("source_pdf_sha256") != digest:
            errors.append(f"question {number} was extracted from a different PDF")
            continue
        successful += 1

    expected = len(questions)
    if data.get("qtd_questoes") != expected:
        errors.append("qtd_questoes does not match the question object")
    if not isinstance(metadata, dict):
        errors.append("rich_extraction metadata is missing")
    else:
        if metadata.get("version") != RICH_EXTRACTION_VERSION:
            errors.append("rich_extraction uses an unsupported version")
        if metadata.get("source_pdf_sha256") != digest:
            errors.append("rich_extraction metadata references a different PDF")
        if metadata.get("status") != "success":
            errors.append("rich_extraction status is not success")
        if metadata.get("question_count") != expected:
            errors.append("rich_extraction question_count is incorrect")
        if metadata.get("successful_question_count") != successful:
            errors.append("rich_extraction successful_question_count is incorrect")
    return errors


def check_repository_rich_content(
    directories: list[Path], *, root_dir: Path = ROOT_DIR
) -> None:
    """Exit unsuccessfully unless every discovered exam has valid rich content."""
    failures: list[tuple[Path, list[str]]] = []
    for directory in tqdm(
        directories,
        desc="Validating rich content",
        unit="exam",
        dynamic_ncols=True,
    ):
        relative_directory = directory.relative_to(root_dir)
        data_path = directory / "data.json"
        try:
            data = json.loads(data_path.read_text(encoding="utf-8"))
            errors = validate_complete_rich_data(data, directory / "prova.pdf")
            if errors:
                failures.append((relative_directory, errors))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            failures.append((relative_directory, [str(error)]))

    if failures:
        tqdm.write("Rich-content validation failed:")
        for directory, errors in failures:
            tqdm.write(f"  {directory}: {len(errors)} issue(s)")
            for error in errors[:5]:
                tqdm.write(f"    - {error}")
            if len(errors) > 5:
                tqdm.write(f"    - ... and {len(errors) - 5} more")
        raise SystemExit(1)
    tqdm.write(f"Rich content is complete and valid for all {len(directories)} exams.")


def run(
    *,
    directories: list[Path] | None = None,
    regenerate_all: bool = False,
    rich_content: bool = True,
    use_gemini_rich: bool = False,
    rich_questions: set[int] | None = None,
    preview_dir: Path | None = None,
    rich_workers: int = DEFAULT_RICH_WORKERS,
) -> None:
    directories = directories if directories is not None else find_exam_directories()

    if not directories:
        tqdm.write("No directories containing prova.pdf and gabarito.pdf were found.")
        return

    existing_directories, generation_directories = partition_exam_directories(
        directories, regenerate_all=regenerate_all
    )
    failures: list[tuple[Path, str]] = []
    total_steps = (
        len(existing_directories)
        + len(generation_directories)
        + (len(directories) if rich_content else 0)
    )

    overall = tqdm(
        total=total_steps,
        desc="Repository",
        unit="exam-step",
        position=0,
        dynamic_ncols=True,
    )
    try:
        # Refresh every existing exam before attempting network-backed Gemini work.
        # This keeps layout backfills deterministic even when one new exam fails.
        for directory in tqdm(
            existing_directories,
            desc="Layouts",
            unit="exam",
            position=1,
            leave=False,
            dynamic_ncols=True,
        ):
            relative_directory = directory.relative_to(ROOT_DIR)
            data_file = directory / "data.json"
            try:
                succeeded = enrich_data_file(data_file, directory / "prova.pdf")
                if not succeeded:
                    failures.append((relative_directory, "layout extraction failed"))
                    tqdm.write(f"{relative_directory}: layout extraction failed")
            except (OSError, ValueError) as error:
                failures.append((relative_directory, str(error)))
                tqdm.write(f"{relative_directory}: {error}")
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
            relative_directory = directory.relative_to(ROOT_DIR)
            try:
                parse_exam_directory(directory, repository_root=ROOT_DIR)
            except Exception as error:
                # Keep processing so one network/API failure cannot prevent the
                # deterministic aggregate from being rebuilt for valid exams.
                failures.append((relative_directory, str(error)))
                tqdm.write(f"{relative_directory}: generation failed: {error}")
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
                relative_directory = directory.relative_to(ROOT_DIR)
                data_file = directory / "data.json"
                try:
                    if not data_file.is_file():
                        continue
                    enriched = enrich_rich_data_file(
                        data_file,
                        directory / "prova.pdf",
                        repository_root=ROOT_DIR,
                        question_numbers=rich_questions,
                        use_gemini=use_gemini_rich,
                        force=regenerate_all,
                        max_workers=rich_workers,
                        progress=True,
                        progress_position=2,
                        progress_desc=f"{relative_directory} questions",
                    )
                    metadata = enriched["rich_extraction"]
                    if metadata["status"] != "success":
                        tqdm.write(
                            f"{relative_directory}: rich extraction "
                            f"{metadata['successful_question_count']}/"
                            f"{metadata['question_count']} ({metadata['status']})"
                        )
                    if preview_dir is not None:
                        preview_path = preview_dir / relative_directory / "index.html"
                        write_html_preview(
                            enriched,
                            directory=directory,
                            output_path=preview_path,
                            question_numbers=rich_questions,
                        )
                except (OSError, ValueError, RuntimeError) as error:
                    failures.append((relative_directory, f"rich extraction: {error}"))
                    tqdm.write(f"{relative_directory}: rich extraction failed: {error}")
                finally:
                    overall.update(1)

        write_gabarito_json(ROOT_DIR, MAIN_DATA)
        tqdm.write(f"Updated {MAIN_DATA.relative_to(ROOT_DIR)}")
    except KeyboardInterrupt:
        overall.set_postfix_str("interrupted", refresh=True)
        tqdm.write("Ctrl-C received: rebuilding the aggregate from saved checkpoints...")
        try:
            write_gabarito_json(ROOT_DIR, MAIN_DATA)
            tqdm.write(f"Saved checkpoints and refreshed {MAIN_DATA.relative_to(ROOT_DIR)}.")
        except Exception as error:
            tqdm.write(f"Could not refresh aggregate after interruption: {error}")
        raise
    finally:
        overall.close()

    if failures:
        tqdm.write("Completed with failures:")
        for directory, error in failures:
            tqdm.write(f"  {directory}: {error}")
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate exam JSON files and rebuild the aggregate catalog."
    )
    parser.add_argument(
        "--regenerate-all",
        action="store_true",
        help="Regenerate every exam data.json with Gemini, even when it already exists.",
    )
    parser.add_argument(
        "--directory",
        type=Path,
        action="append",
        help="Limit generation to exam directories found beneath this path; repeatable.",
    )
    parser.add_argument(
        "--rich-content",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build rich text, image, formula, and option content from local PDFs (default: enabled).",
    )
    parser.add_argument(
        "--use-gemini-rich",
        action="store_true",
        help="Use Gemini vision to enhance rich content (requires --rich-content).",
    )
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
        help=f"Maximum concurrent Gemini rich requests (default: {DEFAULT_RICH_WORKERS}).",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        help="Write local HTML previews beneath this directory.",
    )
    parser.add_argument(
        "--check-rich-content",
        action="store_true",
        help="Validate existing rich content for every exam without regenerating it.",
    )
    args = parser.parse_args()
    if args.use_gemini_rich and not args.rich_content:
        parser.error("--use-gemini-rich requires --rich-content")
    if args.rich_question and not args.rich_content:
        parser.error("--rich-question requires --rich-content")

    directories = None
    if args.directory:
        scopes = []
        for directory in args.directory:
            resolved = directory.resolve()
            try:
                resolved.relative_to(ROOT_DIR)
            except ValueError:
                parser.error(f"--directory must be inside {ROOT_DIR}")
            scopes.append(resolved)
        directories = sorted(
            {
                exam_directory
                for scope in scopes
                for exam_directory in find_exam_directories(scope)
            }
        )

    try:
        if args.check_rich_content:
            check_repository_rich_content(directories or find_exam_directories())
            return
        run(
            directories=directories,
            regenerate_all=args.regenerate_all,
            rich_content=args.rich_content,
            use_gemini_rich=args.use_gemini_rich,
            rich_questions=set(args.rich_question) if args.rich_question else None,
            preview_dir=args.preview_dir.resolve() if args.preview_dir else None,
            rich_workers=args.rich_workers,
        )
    except KeyboardInterrupt:
        tqdm.write("Generation stopped cleanly. Re-run the same command to resume.")
        raise SystemExit(130)


if __name__ == "__main__":
    main()

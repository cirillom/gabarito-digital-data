"""SQLite storage and validation for the generated exam catalog."""

from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path, PurePosixPath
import sqlite3
from typing import Any

import pymupdf

from rich_content import validate_rich_content


CATALOG_FILENAME = "catalog.sqlite3"
SCHEMA_VERSION = 1
APPLICATION_ID = 0x47444231  # GDB1
INSTITUTION_TYPES = {
    "ENEM": "faculdade",
    "Fuvest": "faculdade",
    "OAB": "concurso",
    "TJSP": "concurso",
}

_SCHEMA = """
CREATE TABLE institution (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    type TEXT NOT NULL
);

CREATE TABLE exam (
    id INTEGER PRIMARY KEY,
    institution_id INTEGER NOT NULL REFERENCES institution(id),
    year INTEGER NOT NULL,
    title TEXT NOT NULL,
    application_date TEXT NOT NULL,
    answer_options TEXT NOT NULL,
    pdf_path TEXT NOT NULL UNIQUE,
    UNIQUE (institution_id, year, title)
);

CREATE TABLE question (
    exam_id INTEGER NOT NULL,
    number INTEGER NOT NULL,
    answer TEXT NOT NULL,
    discipline TEXT NOT NULL,
    PRIMARY KEY (exam_id, number),
    FOREIGN KEY (exam_id) REFERENCES exam(id) ON DELETE CASCADE
) WITHOUT ROWID;

CREATE TABLE question_content (
    exam_id INTEGER NOT NULL,
    question_number INTEGER NOT NULL,
    sequence INTEGER NOT NULL,
    page INTEGER NOT NULL,
    x0 REAL NOT NULL,
    y0 REAL NOT NULL,
    x1 REAL NOT NULL,
    y1 REAL NOT NULL,
    PRIMARY KEY (exam_id, question_number, sequence),
    FOREIGN KEY (exam_id, question_number)
        REFERENCES question(exam_id, number) ON DELETE CASCADE
) WITHOUT ROWID;

CREATE TABLE question_rich_content (
    exam_id INTEGER NOT NULL,
    question_number INTEGER NOT NULL,
    format_version INTEGER NOT NULL,
    content TEXT NOT NULL,
    PRIMARY KEY (exam_id, question_number),
    FOREIGN KEY (exam_id, question_number)
        REFERENCES question(exam_id, number) ON DELETE CASCADE
) WITHOUT ROWID;
"""


def connect(database_path: str | Path, *, read_only: bool = False) -> sqlite3.Connection:
    path = Path(database_path).resolve()
    if read_only:
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize(connection: sqlite3.Connection) -> None:
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version == SCHEMA_VERSION:
        if connection.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID:
            raise ValueError("The database is not a Gabarito Digital catalog.")
        return
    if version != 0:
        raise ValueError(f"Unsupported catalog schema version: {version}.")
    connection.executescript(_SCHEMA)
    connection.execute(f"PRAGMA application_id = {APPLICATION_ID}")
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    connection.commit()


def open_catalog(database_path: str | Path) -> sqlite3.Connection:
    connection = connect(database_path)
    initialize(connection)
    return connection


def _exam_fields(
    directory: Path,
    data: dict[str, Any],
    repository_root: Path,
) -> tuple[str, str, int, str, str, str, str]:
    relative = directory.resolve().relative_to(repository_root.resolve())
    parts = relative.parts
    if len(parts) < 3 or parts[1].casefold() != "provas":
        raise ValueError(f"Exam directory must match <institution>/provas/<year>: {relative}")
    institution = parts[0]
    institution_type = INSTITUTION_TYPES.get(institution)
    if institution_type is None:
        raise ValueError(f"Unknown institution type for {institution!r}.")
    try:
        year = int(parts[2])
    except ValueError as error:
        raise ValueError(f"Exam year/edition must be numeric: {parts[2]!r}.") from error

    path_title = " / ".join(parts[3:]).strip()
    description = str(data.get("descricao") or "").strip()
    if path_title and description and path_title.casefold() != description.casefold():
        title = f"{path_title} - {description}"
    else:
        title = path_title or description or f"{institution} {year}"

    application_date = str(data.get("data") or "").strip()
    try:
        date.fromisoformat(application_date)
    except ValueError as error:
        raise ValueError(f"Invalid exam date: {application_date!r}.") from error

    raw_options = data.get("opcoes_resposta")
    if not isinstance(raw_options, list) or not raw_options:
        raise ValueError("opcoes_resposta must be a non-empty list.")
    options = [str(option).strip() for option in raw_options]
    if any(len(option) != 1 for option in options) or len(set(options)) != len(options):
        raise ValueError("Answer options must be unique single-character labels.")

    pdf_path = (relative / "prova.pdf").as_posix()
    return (
        institution,
        institution_type,
        year,
        title,
        application_date,
        "".join(options),
        pdf_path,
    )


def replace_exam(
    connection: sqlite3.Connection,
    directory: str | Path,
    data: dict[str, Any],
    *,
    repository_root: str | Path,
) -> int:
    """Atomically replace one exam and return its compact integer ID."""
    directory = Path(directory)
    root = Path(repository_root)
    institution, institution_type, year, title, exam_date, options, pdf_path = (
        _exam_fields(directory, data, root)
    )
    questions = data.get("questoes")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questoes must be a non-empty object.")
    if data.get("qtd_questoes") != len(questions):
        raise ValueError("qtd_questoes does not match questoes.")

    with connection:
        connection.execute(
            "INSERT INTO institution(name, type) VALUES (?, ?) "
            "ON CONFLICT(name) DO UPDATE SET type = excluded.type",
            (institution, institution_type),
        )
        institution_id = connection.execute(
            "SELECT id FROM institution WHERE name = ?", (institution,)
        ).fetchone()["id"]
        existing = connection.execute(
            "SELECT id FROM exam WHERE pdf_path = ?", (pdf_path,)
        ).fetchone()
        if existing is None:
            cursor = connection.execute(
                "INSERT INTO exam(institution_id, year, title, application_date, "
                "answer_options, pdf_path) VALUES (?, ?, ?, ?, ?, ?)",
                (institution_id, year, title, exam_date, options, pdf_path),
            )
            exam_id = int(cursor.lastrowid)
        else:
            exam_id = int(existing["id"])
            connection.execute(
                "UPDATE exam SET institution_id = ?, year = ?, title = ?, "
                "application_date = ?, answer_options = ? WHERE id = ?",
                (institution_id, year, title, exam_date, options, exam_id),
            )
            connection.execute("DELETE FROM question WHERE exam_id = ?", (exam_id,))

        for raw_number, raw_question in questions.items():
            if not isinstance(raw_question, dict):
                raise ValueError(f"Question {raw_number} is not an object.")
            number = int(raw_number)
            answer = str(raw_question.get("resposta") or "").strip()
            discipline = str(raw_question.get("disciplina") or "").strip()
            if answer not in options and answer != "N/A":
                raise ValueError(f"Question {number} has invalid answer {answer!r}.")
            if not discipline:
                raise ValueError(f"Question {number} has no discipline.")
            connection.execute(
                "INSERT INTO question(exam_id, number, answer, discipline) "
                "VALUES (?, ?, ?, ?)",
                (exam_id, number, answer, discipline),
            )
            content = raw_question.get("conteudo")
            if not isinstance(content, dict):
                continue
            _insert_segments(connection, exam_id, number, content.get("segments"))
            rich = content.get("rich")
            if isinstance(rich, dict):
                _insert_rich(connection, exam_id, number, rich)
    return exam_id


def _insert_segments(
    connection: sqlite3.Connection,
    exam_id: int,
    question_number: int,
    segments: Any,
) -> None:
    if segments is None:
        return
    if not isinstance(segments, list):
        raise ValueError(f"Question {question_number} segments must be a list.")
    for sequence, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"Question {question_number} has an invalid segment.")
        rect = segment.get("rect")
        if not isinstance(rect, list) or len(rect) != 4:
            raise ValueError(f"Question {question_number} has an invalid segment rectangle.")
        connection.execute(
            "INSERT INTO question_content(exam_id, question_number, sequence, page, "
            "x0, y0, x1, y1) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (exam_id, question_number, sequence, int(segment["page"]), *map(float, rect)),
        )


def _insert_rich(
    connection: sqlite3.Connection,
    exam_id: int,
    question_number: int,
    rich: dict[str, Any],
) -> None:
    version = rich.get("version")
    if not isinstance(version, int):
        raise ValueError(f"Question {question_number} rich content has no version.")
    connection.execute(
        "INSERT INTO question_rich_content(exam_id, question_number, format_version, "
        "content) VALUES (?, ?, ?, ?)",
        (
            exam_id,
            question_number,
            version,
            json.dumps(_compact_rich(rich), ensure_ascii=False, separators=(",", ":")),
        ),
    )


def _compact_rich(rich: dict[str, Any]) -> dict[str, Any]:
    """Drop extraction-only/default fields from the distributed document."""
    compact = json.loads(json.dumps(rich))
    documents = [compact.get("statement", {})]
    documents.extend(option.get("content", {}) for option in compact.get("options", []))
    for document in documents:
        for block in document.get("blocks", []):
            block.pop("source", None)
            if block.get("align") == "left":
                block.pop("align")
            if block.get("asset_ids") == []:
                block.pop("asset_ids")
            for inline in block.get("inlines", []):
                if inline.get("marks") == []:
                    inline.pop("marks")
    return compact


def exam_id_for_directory(
    connection: sqlite3.Connection,
    directory: str | Path,
    *,
    repository_root: str | Path,
) -> int | None:
    pdf_path = (
        Path(directory).resolve().relative_to(Path(repository_root).resolve()) / "prova.pdf"
    ).as_posix()
    row = connection.execute("SELECT id FROM exam WHERE pdf_path = ?", (pdf_path,)).fetchone()
    return int(row["id"]) if row is not None else None


def exam_label(connection: sqlite3.Connection, exam_id: int) -> str:
    row = connection.execute(
        "SELECT institution.name, exam.year, exam.title FROM exam "
        "JOIN institution ON institution.id = exam.institution_id WHERE exam.id = ?",
        (exam_id,),
    ).fetchone()
    if row is None:
        return f"exam {exam_id}"
    return f"{row['name']} / {row['year']} / {row['title']}"


def load_exam(connection: sqlite3.Connection, exam_id: int) -> dict[str, Any]:
    exam = connection.execute("SELECT * FROM exam WHERE id = ?", (exam_id,)).fetchone()
    if exam is None:
        raise ValueError(f"Unknown exam ID {exam_id}.")
    questions: dict[str, Any] = {}
    for row in connection.execute(
        "SELECT number, answer, discipline FROM question WHERE exam_id = ? ORDER BY number",
        (exam_id,),
    ):
        questions[str(row["number"])] = {
            "disciplina": row["discipline"],
            "resposta": row["answer"],
        }
    for row in connection.execute(
        "SELECT question_number, page, x0, y0, x1, y1 FROM question_content "
        "WHERE exam_id = ? ORDER BY question_number, sequence",
        (exam_id,),
    ):
        questions[str(row["question_number"])].setdefault("conteudo", {}).setdefault(
            "segments", []
        ).append(
            {
                "page": row["page"],
                "rect": [row["x0"], row["y0"], row["x1"], row["y1"]],
                "kind": "question",
            }
        )
    for row in connection.execute(
        "SELECT question_number, content FROM question_rich_content "
        "WHERE exam_id = ? ORDER BY question_number",
        (exam_id,),
    ):
        questions[str(row["question_number"])].setdefault("conteudo", {})["rich"] = (
            json.loads(row["content"])
        )
    return {
        "data": exam["application_date"],
        "descricao": exam["title"],
        "qtd_questoes": len(questions),
        "opcoes_resposta": list(exam["answer_options"]),
        "disciplinas": sorted({question["disciplina"] for question in questions.values()}),
        "questoes": questions,
    }


def replace_layout(
    connection: sqlite3.Connection,
    exam_id: int,
    layouts: dict[int, list[dict[str, Any]]] | None,
) -> None:
    with connection:
        connection.execute("DELETE FROM question_content WHERE exam_id = ?", (exam_id,))
        if layouts is None:
            return
        for number, segments in layouts.items():
            _insert_segments(connection, exam_id, number, segments)


def write_rich_content(
    connection: sqlite3.Connection,
    exam_id: int,
    question_number: int,
    rich: dict[str, Any] | None,
) -> None:
    """Commit one rich question checkpoint immediately."""
    with connection:
        connection.execute(
            "DELETE FROM question_rich_content WHERE exam_id = ? AND question_number = ?",
            (exam_id, question_number),
        )
        if rich is not None:
            _insert_rich(connection, exam_id, question_number, rich)


def validate_catalog(
    database_path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> None:
    """Validate SQLite integrity, foreign keys, and application invariants."""
    path = Path(database_path)
    if not path.is_file():
        raise ValueError(f"Catalog database does not exist: {path}")
    errors: list[str] = []
    connection = connect(path, read_only=True)
    try:
        integrity = [row[0] for row in connection.execute("PRAGMA integrity_check")]
        if integrity != ["ok"]:
            errors.extend(f"integrity_check: {message}" for message in integrity)
        foreign_keys = list(connection.execute("PRAGMA foreign_key_check"))
        errors.extend(f"foreign_key_check: {tuple(row)}" for row in foreign_keys)
        if connection.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID:
            errors.append("unexpected SQLite application_id")
        if connection.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
            errors.append("unsupported schema version")

        duplicates = connection.execute(
            "SELECT institution_id, year, title, COUNT(*) count FROM exam "
            "GROUP BY institution_id, year, title HAVING count > 1"
        ).fetchall()
        if duplicates:
            errors.append("duplicate exams exist")
        duplicate_questions = connection.execute(
            "SELECT exam_id, number, COUNT(*) count FROM question "
            "GROUP BY exam_id, number HAVING count > 1"
        ).fetchall()
        if duplicate_questions:
            errors.append("duplicate question numbers exist")

        page_counts: dict[int, int] = {}
        root = Path(repository_root).resolve() if repository_root is not None else None
        for exam in connection.execute(
            "SELECT id, application_date, answer_options, pdf_path FROM exam"
        ):
            try:
                date.fromisoformat(exam["application_date"])
            except ValueError:
                errors.append(f"exam {exam['id']} has an invalid application date")
            pdf_path = PurePosixPath(exam["pdf_path"])
            if pdf_path.is_absolute() or ".." in pdf_path.parts:
                errors.append(f"exam {exam['id']} has an unsafe PDF path")
            if root is not None:
                local_pdf = root.joinpath(*pdf_path.parts)
                try:
                    with pymupdf.open(local_pdf) as document:
                        page_counts[exam["id"]] = document.page_count
                except (OSError, RuntimeError) as error:
                    errors.append(f"exam {exam['id']} PDF cannot be opened: {error}")

        for question in connection.execute(
            "SELECT question.exam_id, question.number, question.answer, "
            "exam.answer_options FROM question JOIN exam ON exam.id = question.exam_id"
        ):
            if question["answer"] not in question["answer_options"] and question["answer"] != "N/A":
                errors.append(
                    f"exam {question['exam_id']} question {question['number']} has an invalid answer"
                )

        for crop in connection.execute("SELECT * FROM question_content"):
            coordinates = [crop[name] for name in ("x0", "y0", "x1", "y1")]
            if not (
                0 <= coordinates[0] < coordinates[2] <= 1
                and 0 <= coordinates[1] < coordinates[3] <= 1
            ):
                errors.append(
                    f"exam {crop['exam_id']} question {crop['question_number']} has invalid crop coordinates"
                )
            page_count = page_counts.get(crop["exam_id"])
            if crop["page"] < 1 or (page_count is not None and crop["page"] > page_count):
                errors.append(
                    f"exam {crop['exam_id']} question {crop['question_number']} references an invalid page"
                )

        for row in connection.execute(
            "SELECT rich.exam_id, rich.question_number, rich.format_version, rich.content, "
            "exam.answer_options FROM question_rich_content rich "
            "JOIN exam ON exam.id = rich.exam_id"
        ):
            try:
                rich = json.loads(row["content"])
                if row["format_version"] != rich.get("version"):
                    raise ValueError("format_version does not match content")
                validate_rich_content(rich, list(row["answer_options"]))
                page_count = page_counts.get(row["exam_id"])
                if page_count is not None and any(
                    asset["source"]["page"] > page_count for asset in rich["assets"]
                ):
                    raise ValueError("rich PDF crop references an invalid page")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                errors.append(
                    f"exam {row['exam_id']} question {row['question_number']} has invalid rich content: {error}"
                )
    except sqlite3.DatabaseError as error:
        errors.append(f"SQLite error: {error}")
    finally:
        connection.close()
    if errors:
        raise ValueError("Catalog validation failed:\n- " + "\n- ".join(errors))


def database_stats(database_path: str | Path) -> dict[str, int]:
    path = Path(database_path)
    connection = connect(path, read_only=True)
    try:
        return {
            "size": path.stat().st_size,
            "exams": connection.execute("SELECT COUNT(*) FROM exam").fetchone()[0],
            "questions": connection.execute("SELECT COUNT(*) FROM question").fetchone()[0],
            "question_content": connection.execute(
                "SELECT COUNT(*) FROM question_content"
            ).fetchone()[0],
            "question_rich_content": connection.execute(
                "SELECT COUNT(*) FROM question_rich_content"
            ).fetchone()[0],
        }
    finally:
        connection.close()


def optimize(connection: sqlite3.Connection) -> None:
    for row in connection.execute(
        "SELECT exam_id, question_number, content FROM question_rich_content"
    ).fetchall():
        content = json.dumps(
            _compact_rich(json.loads(row["content"])),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if content != row["content"]:
            connection.execute(
                "UPDATE question_rich_content SET content = ? "
                "WHERE exam_id = ? AND question_number = ?",
                (content, row["exam_id"], row["question_number"]),
            )
    connection.commit()
    connection.execute("PRAGMA optimize")
    connection.execute("VACUUM")


def prepare_release(
    database_path: str | Path,
    output_directory: str | Path,
    *,
    version: str,
    source_commit: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Create the three deliberately small catalog release artifacts."""
    source = Path(database_path).resolve()
    output = Path(output_directory).resolve()
    validate_catalog(source, repository_root=repository_root)
    output.mkdir(parents=True, exist_ok=True)
    released_database = output / CATALOG_FILENAME
    checksum_path = output / f"{CATALOG_FILENAME}.sha256"
    manifest_path = output / "catalog-manifest.json"
    for path in (released_database, checksum_path, manifest_path):
        path.unlink(missing_ok=True)

    source_connection = connect(source, read_only=True)
    release_connection = connect(released_database)
    try:
        source_connection.backup(release_connection)
        optimize(release_connection)
    finally:
        release_connection.close()
        source_connection.close()
    validate_catalog(released_database, repository_root=repository_root)

    digest = hashlib.sha256(released_database.read_bytes()).hexdigest()
    size = released_database.stat().st_size
    checksum_path.write_text(f"{digest}  {CATALOG_FILENAME}\n", encoding="ascii")
    manifest = {
        "version": version,
        "schema_version": SCHEMA_VERSION,
        "sha256": digest,
        "size": size,
        "source_commit": source_commit,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest

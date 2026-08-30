import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pymupdf

from catalog_db import (
    CATALOG_FILENAME,
    database_stats,
    load_exam,
    open_catalog,
    prepare_release,
    replace_exam,
    validate_catalog,
    write_rich_content,
)


def _rich(pdf_digest: str) -> dict:
    document = {
        "blocks": [
            {
                "type": "paragraph",
                "inlines": [{"type": "text", "text": "Question", "marks": []}],
                "asset_ids": [],
                "align": "left",
                "source": [{"page": 1, "rect": [0.1, 0.1, 0.9, 0.9]}],
            }
        ]
    }
    return {
        "version": 2,
        "status": "success",
        "source_pdf_sha256": pdf_digest,
        "method": "deterministic",
        "statement": document,
        "options": [
            {"label": label, "content": document} for label in ["A", "B"]
        ],
        "assets": [],
    }


class CatalogDatabaseTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, dict]:
        directory = root / "ENEM" / "provas" / "2026" / "1o dia"
        directory.mkdir(parents=True)
        pdf_path = directory / "prova.pdf"
        with pymupdf.open() as document:
            document.new_page().insert_text((72, 72), "1 Question")
            document.save(pdf_path)
        (directory / "gabarito.pdf").write_bytes(pdf_path.read_bytes())
        digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        data = {
            "data": "2026-08-30",
            "variante": "Caderno 1 - Azul",
            "qtd_questoes": 1,
            "opcoes_resposta": ["A", "B"],
            "questoes": {
                "1": {
                    "disciplina": "Teste",
                    "resposta": "A",
                    "conteudo": {
                        "segments": [
                            {
                                "page": 1,
                                "rect": [0.1, 0.1, 0.9, 0.9],
                                "kind": "question",
                            }
                        ],
                        "rich": _rich(digest),
                    },
                }
            },
        }
        return directory, pdf_path, data

    def test_catalog_round_trip_is_compact_and_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            directory, _, data = self._fixture(root)
            database = root / CATALOG_FILENAME
            connection = open_catalog(database)
            exam_id = replace_exam(
                connection, directory, data, repository_root=root
            )
            loaded = load_exam(connection, exam_id)
            exam_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(exam)")
            }
            connection.close()

            validate_catalog(database, repository_root=root)
            stats = database_stats(database)
            self.assertEqual(stats["exams"], 1)
            self.assertEqual(stats["questions"], 1)
            self.assertEqual(stats["question_content"], 1)
            self.assertEqual(stats["question_rich_content"], 1)
            self.assertNotIn("year", exam_columns)
            self.assertIn("description", exam_columns)
            self.assertEqual(loaded["titulo"], "2026 | 1º dia | Caderno 1 - Azul")
            self.assertIn("acesso ao ensino superior", loaded["descricao"])
            self.assertEqual(loaded["questoes"]["1"]["resposta"], "A")
            self.assertEqual(loaded["questoes"]["1"]["conteudo"]["segments"][0]["page"], 1)

    def test_each_rich_checkpoint_is_committed_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            directory, pdf_path, data = self._fixture(root)
            data["questoes"]["1"]["conteudo"].pop("rich")
            database = root / CATALOG_FILENAME
            connection = open_catalog(database)
            exam_id = replace_exam(connection, directory, data, repository_root=root)
            rich = _rich(hashlib.sha256(pdf_path.read_bytes()).hexdigest())
            write_rich_content(connection, exam_id, 1, rich)

            second_connection = open_catalog(database)
            loaded = load_exam(second_connection, exam_id)
            second_connection.close()
            connection.close()
            stored = loaded["questoes"]["1"]["conteudo"]["rich"]
            self.assertEqual(stored["source_pdf_sha256"], rich["source_pdf_sha256"])
            self.assertEqual(stored["statement"]["blocks"][0]["inlines"][0]["text"], "Question")
            self.assertNotIn("source", stored["statement"]["blocks"][0])

    def test_validation_rejects_invalid_coordinates_and_pages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            directory, _, data = self._fixture(root)
            database = root / CATALOG_FILENAME
            connection = open_catalog(database)
            replace_exam(connection, directory, data, repository_root=root)
            connection.execute("UPDATE question_content SET x1 = 2, page = 2")
            connection.commit()
            connection.close()

            with self.assertRaisesRegex(ValueError, "invalid crop coordinates"):
                validate_catalog(database, repository_root=root)
            with self.assertRaisesRegex(ValueError, "invalid page"):
                validate_catalog(database, repository_root=root)

    def test_release_contains_only_database_checksum_and_tiny_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            directory, _, data = self._fixture(root)
            database = root / CATALOG_FILENAME
            connection = open_catalog(database)
            replace_exam(connection, directory, data, repository_root=root)
            connection.close()

            manifest = prepare_release(
                database,
                root / "dist",
                version="2026.08.1",
                source_commit="abc123",
                repository_root=root,
            )
            artifacts = sorted(path.name for path in (root / "dist").iterdir())
            self.assertEqual(
                artifacts,
                ["catalog-manifest.json", "catalog.sqlite3", "catalog.sqlite3.sha256"],
            )
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(manifest["size"], (root / "dist" / CATALOG_FILENAME).stat().st_size)
            stored_manifest = json.loads(
                (root / "dist" / "catalog-manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(stored_manifest, manifest)
            self.assertLess((root / "dist" / "catalog-manifest.json").stat().st_size, 300)


if __name__ == "__main__":
    unittest.main()

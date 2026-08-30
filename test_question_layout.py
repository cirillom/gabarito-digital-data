"""Focused tests for the PDF layout extraction contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pymupdf

from catalog_db import connect, validate_catalog
from question_layout import _normalize_text, apply_layout_to_data, extract_question_layout


def _write_two_column_pdf(path: Path, *, include_second_question: bool = True) -> None:
    document = pymupdf.open()
    page = document.new_page(width=600, height=800)
    page.insert_text((35, 70), "01", fontsize=13, fontname="hebo")
    page.insert_text((35, 100), "First statement and answer choices", fontsize=11)
    if include_second_question:
        page.insert_text((315, 70), "02", fontsize=13, fontname="hebo")
        page.insert_text((315, 100), "Second statement and figure", fontsize=11)
    document.save(path)
    document.close()


def _write_high_oab_style_pdf(path: Path) -> None:
    document = pymupdf.open()
    page = document.new_page(width=600, height=800)
    page.insert_text((35, 35), "1", fontsize=11, fontname="hebo")
    page.insert_text((35, 60), "First high statement", fontsize=10)
    page.insert_text((315, 35), "2", fontsize=11, fontname="hebo")
    page.insert_text((315, 60), "Second high statement", fontsize=10)
    document.save(path)
    document.close()


def _write_oab_pdf_with_numbered_instruction_page(path: Path) -> None:
    document = pymupdf.open()
    instructions = document.new_page(width=600, height=800)
    instructions.insert_text((35, 40), "INFORMACOES GERAIS", fontsize=12)
    instructions.insert_text((35, 60), "NAO SERA PERMITIDO", fontsize=12)
    instructions.insert_text((80, 450), "1", fontsize=11, fontname="hebo")
    instructions.insert_text((110, 450), "hora antes do termino", fontsize=10)

    questions = document.new_page(width=600, height=800)
    questions.insert_text((35, 35), "1", fontsize=11, fontname="hebo")
    questions.insert_text((35, 60), "First actual question", fontsize=10)
    questions.insert_text((315, 35), "2", fontsize=11, fontname="hebo")
    questions.insert_text((315, 60), "Second actual question", fontsize=10)
    document.save(path)
    document.close()


class QuestionLayoutTest(unittest.TestCase):
    def test_decodes_symbol_font_digits_used_by_newer_oab_pdfs(self) -> None:
        self.assertEqual(_normalize_text("ϰϱ"), "45")

    def test_extracts_normalized_regions_for_every_question(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            pdf_path = Path(temporary_directory) / "prova.pdf"
            _write_two_column_pdf(pdf_path)

            layouts = extract_question_layout(pdf_path, [1, 2])

            self.assertEqual(set(layouts), {1, 2})
            self.assertTrue(layouts[1])
            self.assertTrue(layouts[2])
            for segments in layouts.values():
                for segment in segments:
                    self.assertGreaterEqual(segment["page"], 1)
                    left, top, right, bottom = segment["rect"]
                    self.assertTrue(0 <= left < right <= 1)
                    self.assertTrue(0 <= top < bottom <= 1)

    def test_retries_with_generic_profile_when_exam_margin_clips_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            pdf_path = Path(temporary_directory) / "prova.pdf"
            _write_high_oab_style_pdf(pdf_path)

            layouts = extract_question_layout(pdf_path, [1, 2])

            self.assertEqual(set(layouts), {1, 2})
            for segments in layouts.values():
                question_segment = next(
                    segment for segment in segments if segment["kind"] == "question"
                )
                self.assertLess(question_segment["rect"][1], 0.05)

    def test_ignores_numbered_exam_instruction_pages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            pdf_path = Path(temporary_directory) / "prova.pdf"
            _write_oab_pdf_with_numbered_instruction_page(pdf_path)

            layouts = extract_question_layout(pdf_path, [1, 2])

            self.assertEqual(layouts[1][0]["page"], 2)
            self.assertEqual(layouts[2][0]["page"], 2)

    def test_failure_is_explicit_and_removes_stale_regions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            pdf_path = Path(temporary_directory) / "prova.pdf"
            _write_two_column_pdf(pdf_path, include_second_question=False)
            data = {
                "qtd_questoes": 2,
                "questoes": {
                    "1": {
                        "disciplina": "Teste",
                        "resposta": "A",
                        "conteudo": {"segments": [{"page": 99, "rect": [0, 0, 1, 1]}]},
                    },
                    "2": {
                        "disciplina": "Teste",
                        "resposta": "B",
                        "conteudo": {"segments": [{"page": 99, "rect": [0, 0, 1, 1]}]},
                    },
                },
            }

            self.assertFalse(apply_layout_to_data(data, pdf_path))
            self.assertEqual(data["layout_extraction"]["status"], "failed")
            self.assertNotIn("conteudo", data["questoes"]["1"])
            self.assertNotIn("conteudo", data["questoes"]["2"])

    def test_corrupt_cached_layout_is_reextracted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            pdf_path = Path(temporary_directory) / "prova.pdf"
            _write_two_column_pdf(pdf_path)
            data = {
                "qtd_questoes": 2,
                "questoes": {
                    "1": {"disciplina": "Teste", "resposta": "A"},
                    "2": {"disciplina": "Teste", "resposta": "B"},
                },
            }

            self.assertTrue(apply_layout_to_data(data, pdf_path))
            data["questoes"]["1"]["conteudo"]["segments"][0]["rect"] = [0, 0, 2, 1]

            self.assertTrue(apply_layout_to_data(data, pdf_path))
            repaired = data["questoes"]["1"]["conteudo"]["segments"][0]["rect"]
            self.assertLessEqual(repaired[2], 1)

    def test_missing_pdf_fails_closed(self) -> None:
        data = {
            "qtd_questoes": 1,
            "layout_extraction": {"status": "success", "version": 1},
            "questoes": {
                "1": {
                    "disciplina": "Teste",
                    "resposta": "A",
                    "conteudo": {
                        "segments": [
                            {
                                "page": 1,
                                "rect": [0.0, 0.0, 0.5, 0.5],
                                "kind": "question",
                            }
                        ]
                    },
                }
            },
        }

        self.assertFalse(apply_layout_to_data(data, Path("missing-prova.pdf")))
        self.assertEqual(data["layout_extraction"]["status"], "failed")
        self.assertNotIn("conteudo", data["questoes"]["1"])

    def test_all_catalog_questions_have_pdf_content(self) -> None:
        repository_root = Path(__file__).resolve().parent
        database = repository_root / "catalog.sqlite3"
        validate_catalog(database, repository_root=repository_root)
        connection = connect(database, read_only=True)
        missing = connection.execute(
            "SELECT question.exam_id, question.number FROM question "
            "LEFT JOIN question_content content ON content.exam_id = question.exam_id "
            "AND content.question_number = question.number "
            "GROUP BY question.exam_id, question.number HAVING COUNT(content.sequence) = 0"
        ).fetchall()
        question_count = connection.execute("SELECT COUNT(*) FROM question").fetchone()[0]
        connection.close()
        self.assertGreater(question_count, 0)
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()

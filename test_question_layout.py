"""Focused tests for the PDF layout extraction contract."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pymupdf

from question_layout import apply_layout_to_data, extract_question_layout


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


class QuestionLayoutTest(unittest.TestCase):
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

    def test_all_tracked_exams_have_a_complete_success_manifest(self) -> None:
        repository_root = Path(__file__).resolve().parent
        manifests = []
        for data_path in repository_root.rglob("data.json"):
            if "provas" not in data_path.parts or not data_path.with_name("prova.pdf").is_file():
                continue
            data = json.loads(data_path.read_text(encoding="utf-8"))
            if "questoes" not in data:
                continue
            manifests.append(data_path)
            questions = data["questoes"]
            layout = data.get("layout_extraction")
            self.assertEqual(layout.get("status"), "success", data_path)
            self.assertEqual(layout.get("version"), 1, data_path)
            self.assertEqual(layout.get("question_count"), len(questions), data_path)
            self.assertEqual(data.get("qtd_questoes"), len(questions), data_path)
            for number, question in questions.items():
                segments = question.get("conteudo", {}).get("segments", [])
                self.assertTrue(segments, f"{data_path}: question {number}")
                self.assertTrue(
                    any(segment.get("kind") == "question" for segment in segments),
                    f"{data_path}: question {number}",
                )

        self.assertTrue(manifests, "No exam manifests were discovered.")


if __name__ == "__main__":
    unittest.main()

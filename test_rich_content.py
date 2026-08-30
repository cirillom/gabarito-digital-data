import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pymupdf

from rich_content import (
    GeminiBlock,
    GeminiDocument,
    GeminiInline,
    GeminiOption,
    GeminiQuestion,
    RICH_CONTENT_VERSION,
    SourceCrop,
    extract_question_rich_content,
    validate_rich_content,
    write_html_preview,
)


def _write_question_pdf(path: Path, *, include_all_options: bool = True) -> None:
    with pymupdf.open() as document:
        page = document.new_page(width=600, height=800)
        page.insert_text((40, 55), "QUESTAO 1", fontsize=12)
        page.insert_text((40, 90), "Read the introduction before the figure.", fontsize=11)
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 120, 60), False)
        pixmap.clear_with(0x5A7DD8)
        page.insert_image(pymupdf.Rect(40, 110, 220, 200), pixmap=pixmap)
        page.insert_text((40, 212), "SOURCE: Example (adapted).", fontsize=7)
        page.insert_text((40, 240), "What is shown in the figure?", fontsize=11)
        page.insert_text((40, 270), "A) First answer", fontsize=11)
        page.insert_text((40, 300), "B) Second answer", fontsize=11)
        page.insert_text((40, 330), "C) Third answer", fontsize=11)
        if include_all_options:
            page.insert_text((40, 360), "D) Fourth answer", fontsize=11)
        document.save(path)


def _write_image_option_pdf(path: Path) -> None:
    with pymupdf.open() as document:
        page = document.new_page(width=600, height=800)
        page.insert_text((40, 55), "QUESTAO 1", fontsize=12)
        page.insert_text((40, 85), "Use the reference diagram.", fontsize=11)
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 80, 40), False)
        pixmap.clear_with(0x5A7DD8)
        page.insert_image(pymupdf.Rect(40, 100, 200, 180), pixmap=pixmap)
        page.insert_text((40, 215), "Which diagram is correct?", fontsize=11)
        for index in range(4):
            option_pixmap = pymupdf.Pixmap(
                pymupdf.csRGB, pymupdf.IRect(0, 0, 100, 45), False
            )
            option_pixmap.clear_with(0x334455 + index)
            top = 245 + index * 80
            page.insert_image(
                pymupdf.Rect(60, top, 260, top + 60), pixmap=option_pixmap
            )
        document.save(path)


def _write_text_caption_pdf(path: Path) -> None:
    with pymupdf.open() as document:
        page = document.new_page(width=600, height=800)
        page.insert_text((40, 55), "QUESTAO 1", fontsize=12)
        page.insert_text((40, 90), "Read the source text.", fontsize=11)
        page.insert_text((40, 112), "SOURCE: Example (adapted).", fontsize=7)
        page.insert_text((40, 145), "What does the text show?", fontsize=11)
        for index, label in enumerate("ABCD"):
            page.insert_text((40, 180 + index * 30), f"{label}) Option {label}", fontsize=11)
        document.save(path)


def _data(*, bottom: float = 0.5) -> dict:
    return {
        "qtd_questoes": 1,
        "opcoes_resposta": ["A", "B", "C", "D"],
        "questoes": {
            "1": {
                "disciplina": "Test",
                "resposta": "A",
                "conteudo": {
                    "segments": [
                        {
                            "page": 1,
                            "rect": [0.0, 0.0, 1.0, bottom],
                            "kind": "question",
                        }
                    ]
                },
            }
        },
    }


def _extract(data: dict, pdf_path: Path, *, use_gemini: bool = False) -> dict:
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    with pymupdf.open(pdf_path) as document:
        return extract_question_rich_content(
            document=document,
            directory=pdf_path.parent,
            assets_directory=None,
            repository_root=pdf_path.parent,
            number=1,
            question=data["questoes"]["1"],
            labels=data["opcoes_resposta"],
            pdf_sha256=digest,
            use_gemini=use_gemini,
        )


class RichContentTest(unittest.TestCase):
    def test_extracts_local_text_options_images_and_preview_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            pdf_path = directory / "prova.pdf"
            _write_question_pdf(pdf_path)
            data = _data()
            rich = _extract(data, pdf_path)
            data["questoes"]["1"]["conteudo"]["rich"] = rich

            self.assertEqual(rich["version"], RICH_CONTENT_VERSION)
            self.assertEqual([option["label"] for option in rich["options"]], ["A", "B", "C", "D"])
            self.assertEqual(
                [block["type"] for block in rich["statement"]["blocks"]],
                ["paragraph", "figure", "paragraph"],
            )
            self.assertEqual(
                rich["statement"]["blocks"][1]["caption"],
                "SOURCE: Example (adapted).",
            )
            self.assertEqual(len(rich["assets"]), 1)
            self.assertFalse((directory / "assets").exists())
            validate_rich_content(rich, ["A", "B", "C", "D"])

            preview = write_html_preview(
                data,
                directory=directory,
                output_path=directory / ".rich-preview" / "question.html",
            )
            preview_text = preview.read_text(encoding="utf-8")
            self.assertIn("What is shown", preview_text)
            self.assertIn('type="radio"', preview_text)
            self.assertIn("data:image/png;base64,", preview_text)

    def test_styles_a_text_source_line_as_a_caption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            pdf_path = Path(temporary_directory) / "prova.pdf"
            _write_text_caption_pdf(pdf_path)
            rich = _extract(_data(), pdf_path)

            blocks = rich["statement"]["blocks"]
            self.assertEqual(
                [block["type"] for block in blocks],
                ["paragraph", "caption", "paragraph"],
            )
            self.assertEqual(blocks[1]["align"], "center")

    def test_enem_question_92_ignores_the_clipped_vertical_watermark(self) -> None:
        pdf_path = (
            Path(__file__).resolve().parent
            / "ENEM"
            / "provas"
            / "2024"
            / "2o dia"
            / "prova.pdf"
        )
        question = {
            "conteudo": {
                "segments": [
                    {
                        "page": 2,
                        "rect": [0.0, 0.657281, 0.5, 0.945],
                    }
                ]
            }
        }
        with pymupdf.open(pdf_path) as document:
            rich = extract_question_rich_content(
                document=document,
                directory=pdf_path.parent,
                assets_directory=None,
                repository_root=pdf_path.parent,
                number=92,
                question=question,
                labels=list("ABCDE"),
                pdf_sha256=hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
            )

        self.assertEqual(
            [option["label"] for option in rich["options"]],
            list("ABCDE"),
        )
        self.assertEqual(
            rich["options"][-1]["content"]["blocks"][0]["inlines"][0]["text"],
            "evaporação.",
        )

    def test_uses_ordered_pdf_crops_for_image_only_options(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            pdf_path = directory / "prova.pdf"
            _write_image_option_pdf(pdf_path)
            rich = _extract(_data(bottom=0.75), pdf_path)
            self.assertEqual(
                [option["content"]["blocks"][0]["type"] for option in rich["options"]],
                ["figure", "figure", "figure", "figure"],
            )
            self.assertEqual(len(rich["assets"]), 5)
            self.assertFalse((directory / "assets").exists())

    def test_fails_closed_when_an_option_cannot_be_found(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            pdf_path = Path(temporary_directory) / "prova.pdf"
            _write_question_pdf(pdf_path, include_all_options=False)
            with self.assertRaises(ValueError):
                _extract(_data(), pdf_path)

    def test_gemini_can_recover_unusable_text_and_stores_formula_as_latex(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            pdf_path = directory / "prova.pdf"
            _write_question_pdf(pdf_path, include_all_options=False)
            formula = GeminiQuestion(
                statement=GeminiDocument(
                    blocks=[
                        GeminiBlock(
                            type="paragraph",
                            inlines=[GeminiInline(type="text", text="Question text")],
                        ),
                        GeminiBlock(type="formula", latex=r"x^2"),
                    ]
                ),
                options=[
                    GeminiOption(
                        label=label,
                        content=GeminiDocument(
                            blocks=[
                                GeminiBlock(
                                    type="paragraph",
                                    inlines=[GeminiInline(type="text", text=f"Option {label}")],
                                )
                            ]
                        ),
                    )
                    for label in ["A", "B", "C", "D"]
                ],
            )
            with patch("rich_content._enrich_with_gemini", return_value=formula):
                rich = _extract(_data(), pdf_path, use_gemini=True)
            formula_block = rich["statement"]["blocks"][1]
            self.assertEqual(formula_block["latex"], r"x^2")
            self.assertNotIn("fallback_asset_id", formula_block)
            self.assertEqual(rich["assets"], [])
            validate_rich_content(rich, ["A", "B", "C", "D"])

    def test_rejects_formula_source_crops(self) -> None:
        with self.assertRaisesRegex(ValueError, "Only figure blocks"):
            GeminiBlock(
                type="formula",
                latex=r"x^2",
                source_crop=SourceCrop(
                    segment_index=0,
                    rect=[0.1, 0.1, 0.4, 0.25],
                ),
            )


if __name__ == "__main__":
    unittest.main()

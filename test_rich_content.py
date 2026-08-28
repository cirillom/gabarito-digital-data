import json
import tempfile
import unittest
from pathlib import Path
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
    enrich_rich_data_file,
    validate_rich_content,
    write_html_preview,
)


def _write_question_pdf(path: Path, *, include_all_options: bool = True) -> None:
    document = pymupdf.open()
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
    document.close()


def _write_data(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
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
                                    "rect": [0.0, 0.0, 1.0, 0.5],
                                    "kind": "question",
                                }
                            ]
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _write_image_option_pdf(path: Path) -> None:
    document = pymupdf.open()
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
    document.close()


class RichContentTest(unittest.TestCase):
    def test_extracts_local_text_options_images_and_preview_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            pdf_path = directory / "prova.pdf"
            data_path = directory / "data.json"
            _write_question_pdf(pdf_path)
            _write_data(data_path)

            data = enrich_rich_data_file(
                data_path,
                pdf_path,
                repository_root=directory,
                write=False,
            )

            rich = data["questoes"]["1"]["conteudo"]["rich"]
            self.assertEqual(rich["version"], RICH_CONTENT_VERSION)
            self.assertEqual(rich["status"], "success")
            self.assertEqual(
                [option["label"] for option in rich["options"]],
                ["A", "B", "C", "D"],
            )
            self.assertIn(
                "Read the introduction",
                rich["statement"]["blocks"][0]["inlines"][0]["text"],
            )
            self.assertEqual(
                [block["type"] for block in rich["statement"]["blocks"]],
                ["paragraph", "figure", "paragraph"],
            )
            self.assertEqual(
                rich["statement"]["blocks"][1]["caption"],
                "SOURCE: Example (adapted).",
            )
            self.assertIn(
                "What is shown",
                rich["statement"]["blocks"][2]["inlines"][0]["text"],
            )
            self.assertEqual(len(rich["assets"]), 1)
            self.assertEqual(rich["assets"][0]["kind"], "pdf_crop")
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
            self.assertIn("<figcaption>SOURCE: Example (adapted).</figcaption>", preview_text)

    def test_uses_ordered_pdf_crops_for_image_only_options(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            pdf_path = directory / "prova.pdf"
            data_path = directory / "data.json"
            _write_image_option_pdf(pdf_path)
            _write_data(data_path)
            data = json.loads(data_path.read_text(encoding="utf-8"))
            data["questoes"]["1"]["conteudo"]["segments"][0]["rect"][3] = 0.75
            data_path.write_text(json.dumps(data), encoding="utf-8")

            result = enrich_rich_data_file(data_path, pdf_path, write=False)
            rich = result["questoes"]["1"]["conteudo"]["rich"]

            self.assertEqual(
                [option["content"]["blocks"][0]["type"] for option in rich["options"]],
                ["figure", "figure", "figure", "figure"],
            )
            self.assertEqual(len(rich["assets"]), 5)
            self.assertTrue(all(asset["kind"] == "pdf_crop" for asset in rich["assets"]))
            self.assertFalse((directory / "assets").exists())

    def test_fails_closed_when_an_option_cannot_be_found(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            pdf_path = directory / "prova.pdf"
            data_path = directory / "data.json"
            _write_question_pdf(pdf_path, include_all_options=False)
            _write_data(data_path)

            data = enrich_rich_data_file(data_path, pdf_path, write=False)

            self.assertEqual(data["rich_extraction"]["status"], "partial")
            self.assertEqual(data["rich_extraction"]["successful_question_count"], 0)
            self.assertIn("1", data["rich_extraction"]["failures"])
            self.assertNotIn("rich", data["questoes"]["1"]["conteudo"])

    def test_gemini_can_recover_unusable_text_and_crops_formula_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            pdf_path = directory / "prova.pdf"
            data_path = directory / "data.json"
            _write_question_pdf(pdf_path, include_all_options=False)
            _write_data(data_path)
            formula = GeminiQuestion(
                statement=GeminiDocument(
                    blocks=[
                        GeminiBlock(
                            type="paragraph",
                            inlines=[GeminiInline(type="text", text="Question text")],
                        ),
                        GeminiBlock(
                            type="formula",
                            latex=r"x^2",
                            source_crop=SourceCrop(
                                segment_index=0,
                                rect=[0.1, 0.1, 0.4, 0.25],
                            ),
                        ),
                    ]
                ),
                options=[
                    GeminiOption(
                        label=label,
                        content=GeminiDocument(
                            blocks=[
                                GeminiBlock(
                                    type="paragraph",
                                    inlines=[
                                        GeminiInline(
                                            type="text", text=f"Option {label}"
                                        )
                                    ],
                                )
                            ]
                        ),
                    )
                    for label in ["A", "B", "C", "D"]
                ],
            )

            with (
                patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}),
                patch("google.genai.Client", return_value=MagicMock()),
                patch("rich_content._enrich_with_gemini", return_value=formula),
            ):
                data = enrich_rich_data_file(
                    data_path,
                    pdf_path,
                    repository_root=directory,
                    use_gemini=True,
                    write=False,
                )

            rich = data["questoes"]["1"]["conteudo"]["rich"]
            formula_block = rich["statement"]["blocks"][1]
            self.assertEqual(rich["status"], "success")
            self.assertIn(formula_block["fallback_asset_id"], {
                asset["id"] for asset in rich["assets"]
            })
            validate_rich_content(rich, ["A", "B", "C", "D"])

    def test_reuses_valid_content_for_the_same_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            pdf_path = directory / "prova.pdf"
            data_path = directory / "data.json"
            _write_question_pdf(pdf_path)
            _write_data(data_path)
            enrich_rich_data_file(data_path, pdf_path, write=True)

            with patch("rich_content.extract_question_rich_content") as extractor:
                data = enrich_rich_data_file(data_path, pdf_path, write=False)

            extractor.assert_not_called()
            self.assertEqual(data["rich_extraction"]["reused_question_count"], 1)
            self.assertEqual(data["rich_extraction"]["processed_question_count"], 0)


if __name__ == "__main__":
    unittest.main()

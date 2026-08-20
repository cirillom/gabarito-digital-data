"""Build versioned rich question content from local exam PDFs.

The deterministic extractor is deliberately useful without network access or
an API key. Gemini is an optional second pass for formulas and complex visual
layout; its output is validated against the same contract before being saved.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import pymupdf
from dotenv import load_dotenv
from pydantic import BaseModel, Field


RICH_CONTENT_VERSION = 1
RICH_EXTRACTION_VERSION = 1
DEFAULT_MODEL_NAME = "gemini-3.5-flash-lite"
_INLINE_OPTION_AFTER_PUNCTUATION = re.compile(
    r"(?<=[.!?;:])\s*([A-Z])(?=\s+[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ])"
)


class SourceCrop(BaseModel):
    """A model-proposed crop relative to one rendered source segment."""

    segment_index: int = Field(ge=0)
    rect: list[float] = Field(min_length=4, max_length=4)


class GeminiInline(BaseModel):
    type: Literal["text", "formula", "line_break"]
    text: str | None = None
    latex: str | None = None
    marks: list[
        Literal["bold", "italic", "underline", "superscript", "subscript"]
    ] = Field(default_factory=list)
    source_crop: SourceCrop | None = None


class GeminiBlock(BaseModel):
    type: Literal["paragraph", "formula", "figure", "quote"]
    inlines: list[GeminiInline] = Field(default_factory=list)
    latex: str | None = None
    asset_ids: list[str] = Field(default_factory=list)
    source_crop: SourceCrop | None = None
    alt: str | None = None
    align: Literal["left", "center", "right", "justify"] = "left"


class GeminiDocument(BaseModel):
    blocks: list[GeminiBlock]


class GeminiOption(BaseModel):
    label: str
    content: GeminiDocument


class GeminiQuestion(BaseModel):
    statement: GeminiDocument
    options: list[GeminiOption]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_rect(value: Any) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("A source rectangle must contain four coordinates.")
    rect = [float(coordinate) for coordinate in value]
    if not all(coordinate == coordinate for coordinate in rect):
        raise ValueError("Source rectangle coordinates must be finite.")
    left, top, right, bottom = rect
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError("Source rectangle coordinates must be normalized.")
    return [round(coordinate, 6) for coordinate in rect]


def _source_refs(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "page": int(segment["page"]),
            "rect": _normalize_rect(segment["rect"]),
        }
        for segment in segments
    ]


def _segment_clip(page: pymupdf.Page, segment: dict[str, Any]) -> pymupdf.Rect:
    left, top, right, bottom = _normalize_rect(segment["rect"])
    return pymupdf.Rect(
        left * page.rect.width,
        top * page.rect.height,
        right * page.rect.width,
        bottom * page.rect.height,
    )


def _extract_source_text(
    document: pymupdf.Document,
    segments: list[dict[str, Any]],
) -> str:
    text_parts: list[str] = []
    for segment in segments:
        page = document[int(segment["page"]) - 1]
        text_parts.append(page.get_text("text", clip=_segment_clip(page, segment), sort=True))
    text = "\n\n".join(text_parts).replace("\u00ad", "")
    text = re.sub(r"(?:E?NEM\d{4}){3,}", "", text, flags=re.IGNORECASE)
    # Some PDFs place the next ENEM alternative in the same text line.
    return _INLINE_OPTION_AFTER_PUNCTUATION.sub(r"\n\1", text)


def _option_pattern(label: str) -> re.Pattern[str]:
    escaped = re.escape(label)
    return re.compile(
        rf"(?m)^\s*\(?{escaped}\)?(?:[.)])?\s+(?=\S)",
    )


def _split_statement_and_options(
    text: str,
    labels: list[str],
) -> tuple[str, dict[str, str]]:
    """Split alternatives by selecting the last complete ordered marker path."""
    if not labels:
        raise ValueError("The exam has no answer-option labels.")
    candidates = {
        label: list(_option_pattern(label).finditer(text)) for label in labels
    }
    selected: dict[str, re.Match[str]] = {}
    next_position = len(text) + 1
    for label in reversed(labels):
        matches = [match for match in candidates[label] if match.start() < next_position]
        if not matches:
            raise ValueError(f"Could not locate option {label} in the question text.")
        selected[label] = matches[-1]
        next_position = matches[-1].start()

    ordered = [selected[label] for label in labels]
    if any(current.start() >= following.start() for current, following in zip(ordered, ordered[1:])):
        raise ValueError("Answer-option markers are not in the expected order.")

    statement = text[: ordered[0].start()].strip()
    options: dict[str, str] = {}
    for index, label in enumerate(labels):
        start = ordered[index].end()
        end = ordered[index + 1].start() if index + 1 < len(ordered) else len(text)
        option_text = text[start:end].strip()
        if not option_text:
            raise ValueError(f"Option {label} has no content.")
        options[label] = option_text
    return statement, options


def _clean_question_statement(text: str, number: int) -> str:
    lines = text.splitlines()
    cleaned: list[str] = []
    removed_number = False
    for line in lines:
        stripped = line.strip()
        normalized = "".join(
            character
            for character in unicodedata.normalize("NFKD", stripped)
            if not unicodedata.combining(character)
        ).upper()
        if re.fullmatch(r"QUESTOES\s+DE\s+\d+\s+A\s+\d+", normalized):
            continue
        header = re.fullmatch(r"(?:QUESTAO\s*)?0*(\d{1,3})[.)]?", normalized)
        if header and int(header.group(1)) == number and not removed_number:
            removed_number = True
            continue
        cleaned.append(line.rstrip())
    return "\n".join(cleaned).strip()


def _paragraph_blocks(text: str, source: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for raw_paragraph in re.split(r"\n\s*\n", text):
        lines = [line.strip() for line in raw_paragraph.splitlines() if line.strip()]
        if not lines:
            continue
        paragraph = " ".join(lines)
        blocks.append(
            {
                "type": "paragraph",
                "inlines": [{"type": "text", "text": paragraph}],
                "source": source,
            }
        )
    return blocks


def _asset_url(asset_path: Path, repository_root: Path | None) -> str:
    if repository_root is None:
        return asset_path.as_posix()
    relative = asset_path.resolve().relative_to(repository_root.resolve())
    return (
        "https://raw.githubusercontent.com/cirillom/gabarito-digital-data/"
        f"refs/heads/main/{quote(relative.as_posix())}"
    )


def _save_crop_asset(
    page: pymupdf.Page,
    clip: pymupdf.Rect,
    *,
    directory: Path,
    assets_directory: Path | None,
    repository_root: Path | None,
    prefix: str,
) -> dict[str, Any]:
    clip = clip & page.rect
    if clip.width < 2 or clip.height < 2:
        raise ValueError("Asset crop is empty.")
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), clip=clip, alpha=False)
    image_bytes = pixmap.tobytes("png")
    digest = hashlib.sha256(image_bytes).hexdigest()
    assets_directory = assets_directory or directory / "assets"
    assets_directory.mkdir(parents=True, exist_ok=True)
    asset_path = assets_directory / f"{prefix}-{digest[:20]}.png"
    if not asset_path.exists():
        asset_path.write_bytes(image_bytes)
    normalized = [
        clip.x0 / page.rect.width,
        clip.y0 / page.rect.height,
        clip.x1 / page.rect.width,
        clip.y1 / page.rect.height,
    ]
    return {
        "id": digest,
        "path": os.path.relpath(asset_path, directory).replace("\\", "/"),
        "url": _asset_url(asset_path, repository_root),
        "sha256": digest,
        "width": pixmap.width,
        "height": pixmap.height,
        "aspect_ratio": round(pixmap.width / pixmap.height, 6),
        "source": {"page": page.number + 1, "rect": _normalize_rect(normalized)},
    }


def _extract_figure_assets(
    document: pymupdf.Document,
    segments: list[dict[str, Any]],
    *,
    directory: Path,
    assets_directory: Path | None,
    repository_root: Path | None,
) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    seen: set[tuple[int, tuple[float, float, float, float]]] = set()
    for segment in segments:
        page = document[int(segment["page"]) - 1]
        segment_clip = _segment_clip(page, segment)
        for image in page.get_image_info():
            image_rect = pymupdf.Rect(image["bbox"])
            intersection = image_rect & segment_clip
            if intersection.is_empty or not segment_clip.contains(image_rect.tl):
                continue
            if intersection.width < 24 or intersection.height < 24:
                continue
            aspect = intersection.width / intersection.height
            page_coverage = intersection.get_area() / page.rect.get_area()
            if aspect < 0.12 or aspect > 8 or page_coverage > 0.7:
                continue
            key = (page.number, tuple(round(value, 2) for value in intersection))
            if key in seen:
                continue
            seen.add(key)
            assets.append(
                _save_crop_asset(
                    page,
                    intersection,
                    directory=directory,
                    assets_directory=assets_directory,
                    repository_root=repository_root,
                    prefix="figure",
                )
            )
    return assets


def _render_segment_images(
    document: pymupdf.Document,
    segments: list[dict[str, Any]],
) -> list[bytes]:
    images: list[bytes] = []
    for segment in segments:
        page = document[int(segment["page"]) - 1]
        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(2, 2),
            clip=_segment_clip(page, segment),
            alpha=False,
        )
        images.append(pixmap.tobytes("png"))
    return images


def _relative_crop_to_page(
    crop: SourceCrop,
    segments: list[dict[str, Any]],
) -> tuple[int, list[float]]:
    if crop.segment_index >= len(segments):
        raise ValueError("Gemini referenced an unknown source segment.")
    relative = _normalize_rect(crop.rect)
    segment = segments[crop.segment_index]
    segment_rect = _normalize_rect(segment["rect"])
    left, top, right, bottom = segment_rect
    width = right - left
    height = bottom - top
    return int(segment["page"]), _normalize_rect(
        [
            left + relative[0] * width,
            top + relative[1] * height,
            left + relative[2] * width,
            top + relative[3] * height,
        ]
    )


def _materialize_model_crop(
    crop: SourceCrop,
    *,
    document: pymupdf.Document,
    segments: list[dict[str, Any]],
    directory: Path,
    assets_directory: Path | None,
    repository_root: Path | None,
    prefix: str,
) -> dict[str, Any]:
    page_number, rect = _relative_crop_to_page(crop, segments)
    page = document[page_number - 1]
    clip = pymupdf.Rect(
        rect[0] * page.rect.width,
        rect[1] * page.rect.height,
        rect[2] * page.rect.width,
        rect[3] * page.rect.height,
    )
    return _save_crop_asset(
        page,
        clip,
        directory=directory,
        assets_directory=assets_directory,
        repository_root=repository_root,
        prefix=prefix,
    )


def _gemini_prompt(
    *,
    number: int,
    labels: list[str],
    deterministic_content: dict[str, Any],
    assets: list[dict[str, Any]],
    segment_count: int,
) -> str:
    available_assets = [
        {"id": asset["id"], "source": asset["source"]} for asset in assets
    ]
    return f"""
Transcribe question {number} exactly into the supplied rich-content schema.
The next {segment_count} images are ordered source segments for this question.

Rules:
- Preserve every word. Do not translate, explain, solve, summarize, or correct it.
- The option labels must be exactly {json.dumps(labels, ensure_ascii=False)} and in that order.
- Separate the statement from each selectable option.
- Use text in paragraph/quote blocks, preserving bold and italic marks when visually meaningful.
- Convert mathematical expressions to LaTeX formula nodes.
- For an inline or display formula, include source_crop relative to its segment so an exact image fallback can be generated.
- Reuse an asset_id only from AVAILABLE_ASSETS when it is the relevant figure.
- If a relevant diagram/table is missing from AVAILABLE_ASSETS, return a figure block with source_crop and useful alt text.
- source_crop.rect is [left, top, right, bottom], normalized 0..1 inside the referenced segment image.
- Do not include the printed question number or option labels inside content text.

AVAILABLE_ASSETS:
{json.dumps(available_assets, ensure_ascii=False)}

DETERMINISTIC_DRAFT:
{json.dumps(deterministic_content, ensure_ascii=False)}
""".strip()


def _enrich_with_gemini(
    *,
    number: int,
    labels: list[str],
    deterministic_content: dict[str, Any],
    assets: list[dict[str, Any]],
    segment_images: list[bytes],
    model_name: str,
    client: Any | None = None,
) -> GeminiQuestion:
    from google import genai
    from google.genai import types

    if client is None:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required for --use-gemini.")
        client = genai.Client(api_key=api_key)
    contents: list[Any] = [
        _gemini_prompt(
            number=number,
            labels=labels,
            deterministic_content=deterministic_content,
            assets=assets,
            segment_count=len(segment_images),
        )
    ]
    contents.extend(
        types.Part.from_bytes(data=image, mime_type="image/png")
        for image in segment_images
    )
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=GeminiQuestion,
        ),
    )
    parsed = response.parsed
    if isinstance(parsed, GeminiQuestion):
        return parsed
    if response.text:
        return GeminiQuestion.model_validate_json(response.text)
    raise ValueError("Gemini returned no structured rich question content.")


def _materialize_gemini_document(
    document_model: GeminiDocument,
    *,
    known_assets: dict[str, dict[str, Any]],
    pdf_document: pymupdf.Document,
    segments: list[dict[str, Any]],
    directory: Path,
    assets_directory: Path | None,
    repository_root: Path | None,
) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    default_source = _source_refs(segments)
    for block_model in document_model.blocks:
        block = block_model.model_dump(exclude_none=True)
        source_crop = block.pop("source_crop", None)
        block["source"] = default_source
        if source_crop is not None:
            crop = SourceCrop.model_validate(source_crop)
            page, rect = _relative_crop_to_page(crop, segments)
            block["source"] = [{"page": page, "rect": rect}]
            asset = _materialize_model_crop(
                crop,
                document=pdf_document,
                segments=segments,
                directory=directory,
                assets_directory=assets_directory,
                repository_root=repository_root,
                prefix="figure" if block["type"] == "figure" else "formula",
            )
            known_assets[asset["id"]] = asset
            if block["type"] == "figure":
                block["asset_ids"] = [asset["id"]]
            elif block["type"] == "formula":
                block["fallback_asset_id"] = asset["id"]

        materialized_inlines: list[dict[str, Any]] = []
        for inline_value in block.get("inlines", []):
            inline = dict(inline_value)
            inline_crop_value = inline.pop("source_crop", None)
            if inline_crop_value is not None:
                inline_crop = SourceCrop.model_validate(inline_crop_value)
                asset = _materialize_model_crop(
                    inline_crop,
                    document=pdf_document,
                    segments=segments,
                    directory=directory,
                    assets_directory=assets_directory,
                    repository_root=repository_root,
                    prefix="formula",
                )
                known_assets[asset["id"]] = asset
                inline["fallback_asset_id"] = asset["id"]
            materialized_inlines.append(inline)
        if "inlines" in block:
            block["inlines"] = materialized_inlines
        blocks.append(block)
    return {"blocks": blocks}


def _validate_document(document: Any, *, assets: dict[str, dict[str, Any]]) -> None:
    if not isinstance(document, dict) or not isinstance(document.get("blocks"), list):
        raise ValueError("A rich document must contain a block list.")
    if not document["blocks"]:
        raise ValueError("A rich document cannot be empty.")
    for block in document["blocks"]:
        if not isinstance(block, dict) or block.get("type") not in {
            "paragraph",
            "quote",
            "formula",
            "figure",
        }:
            raise ValueError("Unsupported rich-content block.")
        if block["type"] in {"paragraph", "quote"}:
            inlines = block.get("inlines")
            if not isinstance(inlines, list) or not inlines:
                raise ValueError("Paragraph and quote blocks need inline content.")
            for inline in inlines:
                if inline.get("type") == "text" and not inline.get("text", "").strip():
                    raise ValueError("Text inline content cannot be empty.")
                if inline.get("type") == "formula" and not inline.get("latex", "").strip():
                    raise ValueError("Formula inline content needs LaTeX.")
                fallback = inline.get("fallback_asset_id")
                if inline.get("type") == "formula" and fallback not in assets:
                    raise ValueError(
                        "An inline formula needs an exact source-image fallback asset."
                    )
        if block["type"] == "formula" and not block.get("latex", "").strip():
            raise ValueError("Formula blocks need LaTeX.")
        if block["type"] == "formula" and block.get("fallback_asset_id") not in assets:
            raise ValueError("A formula block needs an exact source-image fallback asset.")
        if block["type"] == "figure":
            asset_ids = block.get("asset_ids")
            if not isinstance(asset_ids, list) or not asset_ids:
                raise ValueError("Figure blocks need at least one asset.")
            if any(asset_id not in assets for asset_id in asset_ids):
                raise ValueError("A figure references an unknown asset.")


def validate_rich_content(content: dict[str, Any], labels: list[str]) -> None:
    if content.get("version") != RICH_CONTENT_VERSION:
        raise ValueError("Unsupported rich-content version.")
    if content.get("status") != "success":
        raise ValueError("Only successful rich content can be rendered.")
    assets_list = content.get("assets")
    if not isinstance(assets_list, list):
        raise ValueError("Rich content must declare its assets.")
    assets = {asset.get("id"): asset for asset in assets_list if isinstance(asset, dict)}
    if len(assets) != len(assets_list) or None in assets:
        raise ValueError("Rich-content assets need unique IDs.")
    _validate_document(content.get("statement"), assets=assets)
    options = content.get("options")
    if not isinstance(options, list) or [option.get("label") for option in options] != labels:
        raise ValueError("Rich option labels do not match the exam options.")
    for option in options:
        _validate_document(option.get("content"), assets=assets)


def _can_reuse_rich_content(
    content: Any,
    *,
    labels: list[str],
    pdf_sha256: str,
    directory: Path,
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
    return all(
        isinstance(asset.get("path"), str)
        and (directory / asset["path"]).is_file()
        for asset in content.get("assets", [])
    )


def extract_question_rich_content(
    *,
    document: pymupdf.Document,
    directory: Path,
    assets_directory: Path | None,
    repository_root: Path | None,
    number: int,
    question: dict[str, Any],
    labels: list[str],
    pdf_sha256: str,
    use_gemini: bool,
    model_name: str,
    gemini_client: Any | None = None,
) -> dict[str, Any]:
    raw_content = question.get("conteudo")
    if not isinstance(raw_content, dict) or not isinstance(raw_content.get("segments"), list):
        raise ValueError("Question has no validated PDF segments.")
    segments = [dict(segment) for segment in raw_content["segments"]]
    source = _source_refs(segments)
    assets = _extract_figure_assets(
        document,
        segments,
        directory=directory,
        assets_directory=assets_directory,
        repository_root=repository_root,
    )
    try:
        extracted_text = _extract_source_text(document, segments)
        statement_text, option_texts = _split_statement_and_options(
            extracted_text, labels
        )
        statement_text = _clean_question_statement(statement_text, number)
        if not statement_text:
            raise ValueError("Question statement is empty after removing its header.")

        statement_blocks = _paragraph_blocks(statement_text, source)
        if assets:
            statement_blocks.append(
                {
                    "type": "figure",
                    "asset_ids": [asset["id"] for asset in assets],
                    "alt": "Figura da questão",
                    "source": [asset["source"] for asset in assets],
                }
            )
        deterministic = {
            "statement": {"blocks": statement_blocks},
            "options": [
                {
                    "label": label,
                    "content": {
                        "blocks": _paragraph_blocks(option_texts[label], source)
                    },
                }
                for label in labels
            ],
        }
    except ValueError as error:
        if not use_gemini:
            raise
        deterministic = {
            "warning": f"Local text extraction was unusable: {error}",
            "statement": {"blocks": []},
            "options": [],
        }

    assets_by_id = {asset["id"]: asset for asset in assets}
    extraction_method = "deterministic"
    if use_gemini:
        model_content = _enrich_with_gemini(
            number=number,
            labels=labels,
            deterministic_content=deterministic,
            assets=assets,
            segment_images=_render_segment_images(document, segments),
            model_name=model_name,
            client=gemini_client,
        )
        statement = _materialize_gemini_document(
            model_content.statement,
            known_assets=assets_by_id,
            pdf_document=document,
            segments=segments,
            directory=directory,
            assets_directory=assets_directory,
            repository_root=repository_root,
        )
        model_options = {option.label: option for option in model_content.options}
        if list(model_options) != labels:
            raise ValueError("Gemini returned unexpected option labels or order.")
        options = [
            {
                "label": label,
                "content": _materialize_gemini_document(
                    model_options[label].content,
                    known_assets=assets_by_id,
                    pdf_document=document,
                    segments=segments,
                    directory=directory,
                    assets_directory=assets_directory,
                    repository_root=repository_root,
                ),
            }
            for label in labels
        ]
        extraction_method = f"gemini:{model_name}"
    else:
        statement = deterministic["statement"]
        options = deterministic["options"]

    content = {
        "version": RICH_CONTENT_VERSION,
        "status": "success",
        "source_pdf_sha256": pdf_sha256,
        "method": extraction_method,
        "statement": statement,
        "options": options,
        "assets": list(assets_by_id.values()),
    }
    validate_rich_content(content, labels)
    return content


def enrich_rich_data_file(
    data_path: str | Path,
    pdf_path: str | Path,
    *,
    repository_root: str | Path | None = None,
    question_numbers: set[int] | None = None,
    use_gemini: bool = False,
    model_name: str = DEFAULT_MODEL_NAME,
    assets_directory: str | Path | None = None,
    write: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Extract rich content for local questions and optionally persist it."""
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
    pdf_digest = _sha256(pdf_path)
    failures: dict[str, str] = {}
    processed = 0
    reused = 0
    gemini_client: Any | None = None
    if use_gemini:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required for --use-gemini.")
        from google import genai

        gemini_client = genai.Client(api_key=api_key)

    with pymupdf.open(pdf_path) as document:
        for raw_number, question in questions.items():
            number = int(raw_number)
            if question_numbers is not None and number not in question_numbers:
                continue
            if not isinstance(question, dict):
                failures[raw_number] = "Question is not an object."
                continue
            raw_content = question.get("conteudo")
            existing_rich = (
                raw_content.get("rich") if isinstance(raw_content, dict) else None
            )
            if not force and _can_reuse_rich_content(
                existing_rich,
                labels=labels,
                pdf_sha256=pdf_digest,
                directory=directory,
                use_gemini=use_gemini,
                model_name=model_name,
            ):
                reused += 1
                continue
            processed += 1
            try:
                question.setdefault("conteudo", {})["rich"] = extract_question_rich_content(
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
                    gemini_client=gemini_client,
                )
            except (OSError, ValueError, RuntimeError) as error:
                question.setdefault("conteudo", {}).pop("rich", None)
                failures[raw_number] = str(error)

    successful = sum(
        1
        for question in questions.values()
        if isinstance(question, dict)
        and isinstance(question.get("conteudo"), dict)
        and question["conteudo"].get("rich", {}).get("status") == "success"
    )
    expected = len(questions)
    data["rich_extraction"] = {
        "version": RICH_EXTRACTION_VERSION,
        "status": "success" if successful == expected else "partial",
        "question_count": expected,
        "successful_question_count": successful,
        "processed_question_count": processed,
        "reused_question_count": reused,
        "source_pdf_sha256": pdf_digest,
        "method": f"gemini:{model_name}" if use_gemini else "deterministic",
        **({"failures": failures} if failures else {}),
    }
    if write:
        data_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return data


def _render_inlines(inlines: list[dict[str, Any]], assets: dict[str, dict[str, Any]]) -> str:
    rendered: list[str] = []
    for inline in inlines:
        kind = inline.get("type")
        if kind == "text":
            value = html.escape(str(inline.get("text", "")))
            for mark in inline.get("marks", []):
                tag = {"bold": "strong", "italic": "em", "underline": "u"}.get(mark)
                if tag:
                    value = f"<{tag}>{value}</{tag}>"
            rendered.append(value)
        elif kind == "line_break":
            rendered.append("<br>")
        elif kind == "formula":
            fallback = assets.get(inline.get("fallback_asset_id"))
            if fallback:
                rendered.append(
                    f'<img class="formula-inline" src="{html.escape(fallback["path"])}" '
                    f'alt="{html.escape(str(inline.get("latex", "formula")))}">'
                )
            else:
                rendered.append(f'<code class="formula">{html.escape(str(inline.get("latex", "")))}</code>')
    return "".join(rendered)


def _render_document(document: dict[str, Any], assets: dict[str, dict[str, Any]]) -> str:
    rendered: list[str] = []
    for block in document.get("blocks", []):
        kind = block.get("type")
        if kind in {"paragraph", "quote"}:
            tag = "blockquote" if kind == "quote" else "p"
            rendered.append(f"<{tag}>{_render_inlines(block.get('inlines', []), assets)}</{tag}>")
        elif kind == "formula":
            fallback = assets.get(block.get("fallback_asset_id"))
            if fallback:
                rendered.append(
                    f'<img class="formula-block" src="{html.escape(fallback["path"])}" '
                    f'alt="{html.escape(str(block.get("latex", "formula")))}">'
                )
            else:
                rendered.append(f'<pre class="formula">{html.escape(str(block.get("latex", "")))}</pre>')
        elif kind == "figure":
            rendered.append('<div class="figures">')
            for asset_id in block.get("asset_ids", []):
                asset = assets[asset_id]
                rendered.append(
                    f'<img src="{html.escape(asset["path"])}" '
                    f'alt="{html.escape(str(block.get("alt", "Question figure")))}" '
                    f'width="{asset["width"]}" height="{asset["height"]}">'
                )
            rendered.append("</div>")
    return "".join(rendered)


def write_html_preview(
    data: dict[str, Any],
    *,
    directory: str | Path,
    output_path: str | Path,
    question_numbers: set[int] | None = None,
) -> Path:
    """Write a self-contained-style local preview referencing local assets."""
    directory = Path(directory).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sections: list[str] = []
    for raw_number, question in data["questoes"].items():
        number = int(raw_number)
        if question_numbers is not None and number not in question_numbers:
            continue
        rich = question.get("conteudo", {}).get("rich")
        if not isinstance(rich, dict) or rich.get("status") != "success":
            continue
        preview_rich = json.loads(json.dumps(rich))
        for asset in preview_rich.get("assets", []):
            local_asset = directory / asset["path"]
            asset["path"] = os.path.relpath(local_asset, output_path.parent).replace("\\", "/")
        assets = {asset["id"]: asset for asset in preview_rich.get("assets", [])}
        options = "".join(
            '<label class="option">'
            f'<input type="radio" name="question-{number}">'
            f'<span class="label">{html.escape(option["label"])}</span>'
            f'<span class="option-content">{_render_document(option["content"], assets)}</span>'
            "</label>"
            for option in preview_rich["options"]
        )
        sections.append(
            f'<section><h2>Questão {number}</h2>'
            f'{_render_document(preview_rich["statement"], assets)}'
            f'<div class="options">{options}</div></section>'
        )

    page = """<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Gabarito Digital — rich content preview</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#19162d;color:#fff}
body{max-width:920px;margin:auto;padding:24px}section{background:#24203d;padding:24px;border-radius:16px;margin:0 0 24px}
p,blockquote{font-size:18px;line-height:1.55}.figures{display:flex;gap:16px;flex-wrap:wrap;justify-content:center}
.figures img{max-width:100%;height:auto;background:white;border-radius:8px}.options{display:grid;gap:12px;margin-top:24px}
.option{display:flex;align-items:flex-start;gap:14px;padding:16px;border:2px solid #6d64ba;border-radius:12px;cursor:pointer}
.option:has(input:checked){background:#4a3ba7;border-color:#a99fff}.option input{margin-top:7px}.label{font-size:20px;font-weight:700}
.option-content{flex:1}.option-content p{margin:0}.formula{font-family:serif;background:#151225;padding:2px 5px;border-radius:4px}
.formula-inline{max-height:1.7em;vertical-align:middle}.formula-block{display:block;max-width:100%;margin:16px auto}
</style></head><body><h1>Rich question preview</h1>""" + "".join(sections) + "</body></html>"
    output_path.write_text(page, encoding="utf-8", newline="\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and preview rich question content from local PDFs."
    )
    parser.add_argument("--directory", "-d", type=Path, required=True)
    parser.add_argument("--question", "-q", type=int, action="append")
    parser.add_argument("--use-gemini", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--write", action="store_true", help="Update the exam data.json.")
    parser.add_argument("--preview", type=Path, help="Write a local HTML preview.")
    args = parser.parse_args()

    directory = args.directory.resolve()
    data = enrich_rich_data_file(
        directory / "data.json",
        directory / "prova.pdf",
        repository_root=Path(__file__).resolve().parent,
        question_numbers=set(args.question) if args.question else None,
        use_gemini=args.use_gemini,
        model_name=args.model,
        assets_directory=(
            args.preview.resolve().parent / "assets"
            if args.preview is not None and not args.write
            else None
        ),
        write=args.write,
    )
    if args.preview:
        preview = write_html_preview(
            data,
            directory=directory,
            output_path=args.preview,
            question_numbers=set(args.question) if args.question else None,
        )
        print(f"Preview written to {preview}")
    metadata = data["rich_extraction"]
    print(
        f"Rich extraction: {metadata['successful_question_count']}/"
        f"{metadata['question_count']} successful ({metadata['status']})."
    )


if __name__ == "__main__":
    main()

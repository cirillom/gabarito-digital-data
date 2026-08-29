# Gabarito Digital data

Each exam directory contains `prova.pdf`, `gabarito.pdf`, and a generated
`data.json`. Besides the answer and discipline, the generator detects the
original PDF region occupied by every question. It also reconstructs a
versioned rich-content tree containing text, LaTeX formulas, figures, and the
complete selectable alternatives. Figures remain normalized references into
`prova.pdf`; formulas are never stored as image crops, and generation does not
add PNG/JPG files to the repository. The Flutter app prefers that tree and
falls back question-by-question to the original PDF crop.

Run the complete repository update with:

```powershell
uv sync --locked
uv run main.py
```

The default mode generates missing exams, refreshes PDF layout metadata, and
builds deterministic rich content for missing or stale questions. Existing
rich content is reused when the source PDF digest is unchanged. To add the
Gemini vision pass for faithful formulas and complex layouts, run:

```powershell
uv run main.py --use-gemini-rich
```

To regenerate every exam with Gemini—including answers, official subject
research, question classification, and rich content—run:

```powershell
uv run main.py --regenerate-all --use-gemini-rich
uv run main.py --check-rich-content
```

The second command performs no API calls. It fails unless every question in
every discovered exam has structurally valid rich content generated from the
current `prova.pdf`, with the complete expected alternative list.

On Windows, the checked-in helper installs locked dependencies, runs the test
suite, and then generates the catalog:

```powershell
.\scripts\generate-data.ps1
.\scripts\generate-data.ps1 -RegenerateAll -UseGeminiRich
```

## Generate data from GitHub

The **Generate exam data** workflow can run the same process without a local
checkout. Add `prova.pdf` and `gabarito.pdf` under the desired exam directory,
then open **Actions > Generate exam data > Run workflow**. The workflow tests
the extractor, rebuilds the root `data.json`, and commits the generated files
to the selected branch. Enable **Regenerate every exam data.json with Gemini**
to replace all per-exam JSON files instead of generating only missing exams.
Rich reconstruction and **Use Gemini vision** are enabled by default. For a
full refresh, enable **Regenerate every exam data.json with Gemini**. Before
committing, the workflow validates every rich question against its current
PDF and stops without publishing if any exam is incomplete.

Before its first run, create an Actions repository secret named
`GEMINI_API_KEY` under **Settings > Secrets and variables > Actions**. The
workflow's `contents: write` permission is used only to commit generated JSON
files back to the branch.

To add or refresh layout metadata without calling Gemini again:

```powershell
uv run question_layout.py --directory "path\to\exam"
```

An exam receives `layout_extraction.status: "success"` only when every
question has a validated, normalized `conteudo.segments` region. On failure,
the JSON is still usable for answer-only modes and the PDF mode remains
disabled in the app.

Each segment stores a 1-based PDF page, a normalized
`[left, top, right, bottom]` rectangle, and a `kind` of `question` or
`shared`. Shared passages are attached to every question that references
them, so a question remains complete even when its source text appears in a
different column or on a preceding page.

## Preview without GitHub

Generate a browser preview from the checked-out PDFs without changing the
exam JSON or downloading the catalog from GitHub:

```powershell
uv run rich_content.py --directory "ENEM\provas\2024\2o dia" --question 91 --preview ".rich-preview\enem-91.html"
```

Add `--use-gemini` to inspect the AI-enhanced result, or `--write` when the
validated result should be persisted. The preview contains selectable answer
cards, renders LaTeX as browser-native MathML, and embeds temporary figure
renderings inside the HTML file; it does not create repository image assets.

Useful ENEM 2024 day-two samples are question 91 for figure/caption ordering,
question 109 for image-only alternatives, and question 150 for equations.

For end-to-end Flutter testing against the checkout, first generate the local
catalog, then serve it with CORS:

```powershell
uv run folder_parser.py --output data.json
uv run scripts\serve-local.py
```

In a second terminal, run Flutter with the local catalog and repository origin:

```powershell
flutter run -d chrome --dart-define=CATALOG_URL=http://127.0.0.1:8765/data.json --dart-define=DATA_REPOSITORY_BASE_URL=http://127.0.0.1:8765/ --dart-define=FORCE_CATALOG_REFRESH=true
```

This path reads both JSON and source PDFs from the local checkout. It does not
depend on `raw.githubusercontent.com`. The legacy `DATA_ASSET_BASE_URL` define
remains accepted for existing local scripts.

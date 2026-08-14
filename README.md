# Gabarito Digital data

Each exam directory contains `prova.pdf`, `gabarito.pdf`, and a generated
`data.json`. Besides the answer and discipline, the generator detects the
original PDF region occupied by every question. The Flutter app uses those
regions to render the statement, figures, tables, and equations directly from
the source PDF.

Run the complete repository update with:

```powershell
uv sync --locked
uv run main.py
```

The default mode generates missing exams and refreshes PDF layout metadata for
existing JSON files. To regenerate every exam with Gemini—including answers,
official subject research, and question classification—run:

```powershell
uv run main.py --regenerate-all
```

On Windows, the checked-in helper installs locked dependencies, runs the test
suite, and then generates the catalog:

```powershell
.\scripts\generate-data.ps1
.\scripts\generate-data.ps1 -RegenerateAll
```

## Generate data from GitHub

The **Generate exam data** workflow can run the same process without a local
checkout. Add `prova.pdf` and `gabarito.pdf` under the desired exam directory,
then open **Actions > Generate exam data > Run workflow**. The workflow tests
the extractor, rebuilds the root `data.json`, and commits the generated files
to the selected branch. Enable **Regenerate every exam data.json with Gemini**
to replace all per-exam JSON files instead of generating only missing exams.

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

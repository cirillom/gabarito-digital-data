# Gabarito Digital data

`catalog.sqlite3` is the canonical generated exam catalog. PDFs remain in their
exam directories and the database stores only relative PDF paths, compact
question metadata, normalized PDF crops, and optional rich documents.

The tables are deliberately separated so answer-only modes can query
`question` without loading `question_content` or `question_rich_content`.

## Generate

```powershell
uv sync --locked
uv run main.py
```

The default run:

- refreshes PDF crops for exams already in the database;
- sends only missing exams to Gemini for answer and discipline extraction;
- commits every completed exam directly to SQLite;
- validates the final database and prints its size and row counts.

Set `GEMINI_API_KEY` to generate missing base data. Rich reconstruction is
optional and opt-in; completed rich questions whose PDF is unchanged are
skipped:

```powershell
uv run main.py --rich-content
uv run main.py --rich-content --gemini-rich
uv run main.py --rich-content --force-rich
```

The first command generates every rich question that is still missing, using
the local deterministic extractor. `--gemini-rich` asks Gemini vision to improve
those results. `--force-rich` rebuilds already completed rich questions without
also regenerating the base answer data. `--regenerate-all` is the expensive
option that explicitly regenerates both base and rich data.

Gemini calls use limited concurrency, stop immediately on quota/rate limits,
and retry timeouts and temporary 408/500/502/503/504 errors with bounded
backoff.

The PowerShell helper installs locked dependencies, tests, generates, and
validates:

```powershell
.\scripts\generate-data.ps1
.\scripts\generate-data.ps1 -RichContent -GeminiRich
.\scripts\generate-data.ps1 -RegenerateAll -RichContent -GeminiRich
```

## Generate rich exams by scope

Run these commands from `gabarito-digital-data`. A directory may identify one
exam or any parent scope, such as an institution. Repeat `--directory` to combine
scopes.

```powershell
# Every question that does not have rich content yet
uv run main.py --rich-content

# Every remaining OAB question
uv run main.py --rich-content --directory OAB

# One exact exam
uv run main.py --rich-content --directory "OAB\provas\46"

# Selected questions from one exam
uv run main.py --rich-content --directory "ENEM\provas\2024\2o dia" --rich-question 92 --rich-question 93

# Two institutions in one resumable run
uv run main.py --rich-content --directory OAB --directory Fuvest

# Improve missing questions with Gemini vision and write a browser preview
$env:GEMINI_API_KEY = "your-key"
uv run main.py --rich-content --gemini-rich --directory Fuvest --rich-workers 2 --preview-dir .rich-preview

# Rebuild a completed question after fixing the extractor
uv run main.py --rich-content --force-rich --directory "ENEM\provas\2024\2o dia" --rich-question 92
```

`--rich-question` is repeatable and requires `--rich-content`. Successful
questions are committed immediately and skipped on the next run, so an
interrupted or quota-limited command can be resumed unchanged. Use
`--rich-model MODEL_NAME` with `--gemini-rich` to override the model, and keep
`--rich-workers` between 1 and 4.

To add a completely new exam, create
`<institution>\provas\<title>\prova.pdf` and `gabarito.pdf`, set
`GEMINI_API_KEY`, and run `uv run main.py --directory "<exam-directory>"` once
for answers, disciplines, metadata, and PDF crops. Then run the same scope with
`--rich-content` for the rich representation.

Every generation run prints the processed exams, attempted/successful/failed
question counts, and the exact failed-question list. Failure-only diagnostic
logs are written beneath `logs/`; the path is printed in the summary.

## Validate

```powershell
uv run main.py --validate
```

Validation includes `PRAGMA integrity_check`, `PRAGMA foreign_key_check`,
natural exam and question uniqueness, answer labels, normalized coordinates,
PDF page bounds, rich-document structure, and orphan checks.

## Release

The deliberately triggered **Release catalog** workflow runs tests, validates
the checked-in database, prepares checksums, and publishes exactly:

```text
catalog.sqlite3
catalog.sqlite3.sha256
catalog-manifest.json
```

The release workflow never invokes Gemini. A local release bundle can be built
with:

```powershell
uv run main.py --release-version 2026.08.1 --source-commit <commit-id>
```

The tiny manifest contains only `version`, `schema_version`, `sha256`, `size`,
and `source_commit`.

## Local serving

The existing static server exposes the SQLite catalog and source PDFs for local
application development:

```powershell
uv run scripts\serve-local.py
```

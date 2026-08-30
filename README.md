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
uv run main.py --regenerate-all --rich-content --gemini-rich
```

`--regenerate-all` explicitly forces all base and rich extraction. Gemini calls
use limited concurrency, stop immediately on quota/rate limits, and retry
timeouts and temporary 408/500/502/503/504 errors with bounded backoff.

The PowerShell helper installs locked dependencies, tests, generates, and
validates:

```powershell
.\scripts\generate-data.ps1
.\scripts\generate-data.ps1 -RichContent -GeminiRich
.\scripts\generate-data.ps1 -RegenerateAll -RichContent -GeminiRich
```

## Retry failures

Limit a run to one exam with `--directory`. Repeat `--rich-question` to retry
several failed questions; completed questions are skipped unless
`--regenerate-all` is present.

```powershell
uv run main.py --rich-content --directory "OAB\provas\46" --rich-question 17
uv run main.py --rich-content --directory "ENEM\provas\2024\2o dia" --rich-question 17 --rich-question 42 --gemini-rich
```

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

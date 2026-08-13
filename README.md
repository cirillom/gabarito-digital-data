# Gabarito Digital data

Each exam directory contains `prova.pdf`, `gabarito.pdf`, and a generated
`data.json`. Besides the answer and discipline, the generator detects the
original PDF region occupied by every question. The Flutter app uses those
regions to render the statement, figures, tables, and equations directly from
the source PDF.

Run the complete repository update with:

```powershell
.\.venv\Scripts\python.exe main.py
```

To add or refresh layout metadata without calling Gemini again:

```powershell
.\.venv\Scripts\python.exe question_layout.py --directory "path\to\exam"
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

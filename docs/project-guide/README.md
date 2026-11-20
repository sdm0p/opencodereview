# AI engineer interview guide

This folder contains the OpenCodeReview architecture guide in DOCX and PDF formats.
The guide explains the project in plain language and adds technical details for AI
engineering interviews.

## Contents

- `OpenCodeReview_AI_Engineering_Architecture_Guide.docx` — editable document.
- `OpenCodeReview_AI_Engineering_Architecture_Guide.pdf` — shareable PDF.
- `assets/` — source diagrams used by the document builder.
- `tools/` — scripts that create the DOCX, HTML preview, PDF, and page contact sheets.
- `working/` — generated HTML/PDF previews and rendered page images; kept local and ignored by Git.

## Rebuild

From the repository root, run the DOCX builder with Python packages `python-docx`
and `Pillow` available:

```powershell
py -3 docs/project-guide/tools/build_opencodereview_guide.py
```

To create an HTML preview, run:

```powershell
py -3 docs/project-guide/tools/docx_to_html_preview.py
```

The PDF renderer uses Playwright. Set `BROWSER_EXECUTABLE_PATH` if you want it to
use a specific installed browser; otherwise Playwright uses its configured browser:

```powershell
node docs/project-guide/tools/render_preview.cjs
```

Rendered PDF pages and contact sheets stay in `working/` so they do not clutter the
project's source folders.

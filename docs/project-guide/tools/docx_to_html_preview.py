from __future__ import annotations

import base64
import html
from pathlib import Path

from docx import Document
from docx.document import Document as DocumentType
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph


ROOT = Path(__file__).resolve().parents[3]
GUIDE_DIR = ROOT / "docs" / "project-guide"
DOCX = GUIDE_DIR / "OpenCodeReview_AI_Engineering_Architecture_Guide.docx"
OUT = GUIDE_DIR / "working" / "preview.html"


def iter_blocks(parent):
    parent_elm = parent.element.body if isinstance(parent, DocumentType) else parent._tc
    for child in parent_elm.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, parent)
        elif child.tag == qn("w:tbl"):
            yield Table(child, parent)


def paragraph_image_data(paragraph: Paragraph):
    blips = paragraph._p.xpath(".//a:blip")
    if not blips:
        return None
    rid = blips[0].get(qn("r:embed"))
    if not rid:
        return None
    part = paragraph.part.related_parts[rid]
    content_type = part.content_type or "image/png"
    return f"data:{content_type};base64,{base64.b64encode(part.blob).decode('ascii')}"


def para_html(paragraph: Paragraph, number_state: dict[str, int]):
    text = html.escape(paragraph.text)
    style = paragraph.style.name if paragraph.style else "Normal"
    image = paragraph_image_data(paragraph)
    if image:
        return f'<div class="figure"><img src="{image}" alt="Embedded diagram"></div>'
    if not text:
        return '<div class="spacer"></div>'
    if style == "Title":
        return f"<h1 class='title'>{text}</h1>"
    if style.startswith("Heading 1"):
        return f"<h1 class='section'>{text}</h1>"
    if style.startswith("Heading 2"):
        return f"<h2>{text}</h2>"
    if style.startswith("Heading 3"):
        return f"<h3>{text}</h3>"
    if style.startswith("List Bullet"):
        return f"<p class='bullet'>• {text}</p>"
    if style.startswith("List Number"):
        number_state["n"] += 1
        return f"<p class='number'><span>{number_state['n']}.</span> {text}</p>"
    number_state["n"] = 0
    is_code = any(run.font.name == "Consolas" for run in paragraph.runs)
    if is_code:
        return f"<pre>{text}</pre>"
    is_caption = bool(paragraph.runs) and all(bool(run.italic) for run in paragraph.runs if run.text)
    if is_caption:
        return f"<p class='caption'>{text}</p>"
    return f"<p>{text}</p>"


def table_html(table: Table):
    rows = []
    for r_idx, row in enumerate(table.rows):
        cells = []
        tag = "th" if r_idx == 0 else "td"
        for cell in row.cells:
            value = "<br>".join(html.escape(p.text) for p in cell.paragraphs)
            cells.append(f"<{tag}>{value}</{tag}>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<table>" + "".join(rows) + "</table>"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = Document(DOCX)
    number_state = {"n": 0}
    body = []
    for block in iter_blocks(doc):
        if isinstance(block, Paragraph):
            fragment = para_html(block, number_state)
            if fragment.startswith("<h1 class='section'"):
                fragment = fragment.replace("class='section'", "class='section pagebreak'", 1)
            body.append(fragment)
        else:
            body.append(table_html(block))
    css = """
      @page { size: Letter; margin: 0.72in 0.78in 0.68in; }
      * { box-sizing: border-box; }
      body { font-family: Arial, sans-serif; color: #1F2937; font-size: 11pt; line-height: 1.18; margin: 0; }
      h1.title { font-size: 29pt; line-height: 1.08; margin: 0.9in 0 0.18in; }
      h1.section { font-size: 19pt; margin: 0.08in 0 0.08in; break-after: avoid; }
      h1.pagebreak { break-before: page; }
      h2 { font-size: 14pt; color: #0B4F59; margin: 0.12in 0 0.05in; break-after: avoid; }
      h3 { font-size: 11.5pt; margin: 0.1in 0 0.04in; break-after: avoid; }
      p { margin: 0 0 0.075in; }
      p.bullet { margin-left: 0.25in; text-indent: -0.16in; }
      p.number { margin-left: 0.3in; text-indent: -0.2in; }
      p.caption { text-align: center; color: #52606D; font-size: 9pt; font-style: italic; }
      pre { font-family: Consolas, monospace; color: #0B4F59; font-size: 9.4pt; white-space: pre-wrap; margin: 0.03in 0 0.08in 0.28in; }
      .figure { text-align: center; break-inside: avoid; }
      .figure img { max-width: 6.8in; height: auto; }
      .spacer { height: 0.04in; }
      table { width: 100%; border-collapse: collapse; margin: 0.06in 0 0.12in; font-size: 9pt; break-inside: auto; }
      tr { break-inside: avoid; }
      th, td { border: 1px solid #D9D9D9; padding: 0.06in 0.075in; vertical-align: top; }
      th { background: #0B4F59; color: white; text-align: left; }
      tr:nth-child(odd) td { background: #F6F8FA; }
    """
    OUT.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>OpenCodeReview AI Engineering Architecture Guide</title>"
        "<style>" + css + "</style></head><body>" + "".join(body) + "</body></html>",
        encoding="utf-8",
    )
    print(OUT)


if __name__ == "__main__":
    main()

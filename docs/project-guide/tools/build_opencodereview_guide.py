from __future__ import annotations

from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[3]
GUIDE_DIR = ROOT / "docs" / "project-guide"
WORK = GUIDE_DIR / "working"
ASSETS = GUIDE_DIR / "assets"
OUTPUT = GUIDE_DIR / "OpenCodeReview_AI_Engineering_Architecture_Guide.docx"
ARCH_DIAGRAM = ASSETS / "architecture.png"
FLOW_DIAGRAM = ASSETS / "review_workflow.png"

BLACK = "1F2937"
MUTED = "52606D"
ACCENT = "0F6B78"
ACCENT_DARK = "0B4F59"
ACCENT_LIGHT = "DDEFF2"
LIGHT = "F6F8FA"
MID = "E7ECF0"
WHITE = "FFFFFF"
GREEN = "2F855A"
AMBER = "B7791F"
RED = "B23A48"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def draw_centered_text(draw: ImageDraw.ImageDraw, box, text: str, size: int, bold: bool = False, fill="#1F2937"):
    x1, y1, x2, y2 = box
    fnt = font(size, bold)
    max_width = x2 - x1 - 30
    words = text.split()
    lines = []
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=fnt)[2] <= max_width:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    line_h = size + 7
    total_h = len(lines) * line_h
    y = y1 + ((y2 - y1) - total_h) / 2
    for item in lines:
        bounds = draw.textbbox((0, 0), item, font=fnt)
        width = bounds[2] - bounds[0]
        draw.text((x1 + ((x2 - x1) - width) / 2, y), item, font=fnt, fill=fill)
        y += line_h


def rounded_box(draw, box, fill, outline="#8AA2AE", radius=18, width=3):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw, start, end, fill="#52606D", width=5):
    draw.line([start, end], fill=fill, width=width)
    x2, y2 = end
    x1, y1 = start
    if abs(x2 - x1) >= abs(y2 - y1):
        sign = 1 if x2 > x1 else -1
        pts = [(x2, y2), (x2 - 15 * sign, y2 - 10), (x2 - 15 * sign, y2 + 10)]
    else:
        sign = 1 if y2 > y1 else -1
        pts = [(x2, y2), (x2 - 10, y2 - 15 * sign), (x2 + 10, y2 - 15 * sign)]
    draw.polygon(pts, fill=fill)


def create_architecture_diagram():
    img = Image.new("RGB", (1800, 1050), "white")
    d = ImageDraw.Draw(img)
    title = font(42, True)
    d.text((70, 42), "OpenCodeReview deployment and runtime architecture", font=title, fill="#1F2937")
    d.text((72, 98), "Two deployments with one modular backend process", font=font(25), fill="#52606D")

    # User and frontend
    rounded_box(d, (70, 210, 340, 380), "#F6F8FA")
    draw_centered_text(d, (70, 210, 340, 380), "Reviewer in browser", 29, True)
    arrow(d, (340, 295), (430, 295))
    rounded_box(d, (430, 180, 780, 410), "#DDEFF2", "#0F6B78")
    draw_centered_text(d, (430, 195, 780, 270), "Vercel frontend", 31, True, "#0B4F59")
    draw_centered_text(d, (450, 270, 760, 390), "React and Vite\nStatic UI\nProvider test function", 24)

    # API and graph
    arrow(d, (780, 295), (880, 295))
    rounded_box(d, (880, 150, 1710, 720), "#F6F8FA", "#7C8B96")
    draw_centered_text(d, (900, 165, 1690, 235), "Hugging Face Docker Space", 31, True)
    rounded_box(d, (930, 265, 1220, 435), "#E8F4EA", "#2F855A")
    draw_centered_text(d, (930, 265, 1220, 435), "FastAPI\nAuth and job API", 27, True)
    arrow(d, (1220, 350), (1300, 350))
    rounded_box(d, (1300, 250, 1650, 460), "#FFF4D6", "#B7791F")
    draw_centered_text(d, (1300, 250, 1650, 460), "In process workers\nLangGraph review\nRAGAS evaluation", 26, True)
    rounded_box(d, (930, 505, 1220, 655), "#F1EAFE", "#7559A8")
    draw_centered_text(d, (930, 505, 1220, 655), "Legacy Gradio UI\nMounted at root", 25, True)
    rounded_box(d, (1300, 520, 1650, 665), "#EAF2FF", "#3D6FB4")
    draw_centered_text(d, (1300, 520, 1650, 665), "Job records  SQLite or Supabase\nGraph checkpoints  SQLite", 23, True)
    arrow(d, (1475, 460), (1475, 520))

    # External systems
    rounded_box(d, (120, 820, 450, 980), "#F6F8FA")
    draw_centered_text(d, (120, 820, 450, 980), "GitHub API\nPR data and comments", 26, True)
    rounded_box(d, (570, 820, 930, 980), "#F6F8FA")
    draw_centered_text(d, (570, 820, 930, 980), "Model provider APIs\nGroq Gemini or custom", 26, True)
    rounded_box(d, (1050, 820, 1410, 980), "#F6F8FA")
    draw_centered_text(d, (1050, 820, 1410, 980), "Chroma vector store\nOptional local retrieval", 26, True)
    rounded_box(d, (1510, 820, 1730, 980), "#F6F8FA")
    draw_centered_text(d, (1510, 820, 1730, 980), "LangSmith\nand Langfuse", 24, True)
    arrow(d, (1120, 720), (365, 820))
    arrow(d, (1380, 720), (760, 820))
    arrow(d, (1490, 720), (1230, 820))
    arrow(d, (1600, 720), (1620, 820))
    img.save(ARCH_DIAGRAM, quality=95)


def create_flow_diagram():
    img = Image.new("RGB", (1900, 990), "white")
    d = ImageDraw.Draw(img)
    d.text((70, 40), "LangGraph review workflow", font=font(42, True), fill="#1F2937")
    d.text((72, 96), "Specialists run in parallel, then a human decides whether findings can be posted", font=font(25), fill="#52606D")

    boxes = {
        "ingestion": (90, 240, 350, 390),
        "retrieval": (430, 240, 690, 390),
        "correctness": (805, 175, 1095, 300),
        "security": (805, 355, 1095, 480),
        "tests": (805, 535, 1095, 660),
        "aggregate": (1230, 320, 1510, 490),
        "approval": (1600, 320, 1840, 490),
        "executor": (1230, 700, 1510, 850),
    }
    rounded_box(d, boxes["ingestion"], "#DDEFF2", "#0F6B78")
    draw_centered_text(d, boxes["ingestion"], "Ingestion\nPR diff and files", 27, True)
    rounded_box(d, boxes["retrieval"], "#EAF2FF", "#3D6FB4")
    draw_centered_text(d, boxes["retrieval"], "Retrieval\nRelevant repo context", 27, True)
    for key, label in [("correctness", "Correctness reviewer"), ("security", "Security reviewer"), ("tests", "Test coverage reviewer")]:
        rounded_box(d, boxes[key], "#FFF4D6", "#B7791F")
        draw_centered_text(d, boxes[key], label, 25, True)
    rounded_box(d, boxes["aggregate"], "#E8F4EA", "#2F855A")
    draw_centered_text(d, boxes["aggregate"], "Aggregator\nGround deduplicate score", 26, True)
    rounded_box(d, boxes["approval"], "#F1EAFE", "#7559A8")
    draw_centered_text(d, boxes["approval"], "Human approval\nInterrupt and resume", 26, True)
    rounded_box(d, boxes["executor"], "#FCE8EA", "#B23A48")
    draw_centered_text(d, boxes["executor"], "Executor\nPost to GitHub", 27, True)

    arrow(d, (350, 315), (430, 315))
    for y in (237, 417, 597):
        arrow(d, (690, 315), (805, y))
    for y in (237, 417, 597):
        arrow(d, (1095, y), (1230, 405))
    arrow(d, (1510, 405), (1600, 405))
    arrow(d, (1720, 490), (1450, 700))
    d.text((1530, 610), "approved", font=font(23, True), fill="#2F855A")
    d.text((1600, 535), "rejected ends without posting", font=font(22), fill="#B23A48")

    rounded_box(d, (90, 760, 930, 900), "#F6F8FA", "#A7B3BC")
    draw_centered_text(d, (90, 760, 930, 900), "State travels through every node\nChanged files  Retrieved context  Findings  Verdict  Review stages", 24)
    img.save(FLOW_DIAGRAM, quality=95)


def set_font(run, name="Arial", size=None, bold=None, color=None, italic=None):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)


def shade_cell(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=90, start=110, bottom=90, end=110):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def prevent_row_split(row):
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    tr_pr.append(cant_split)


def add_alt_text(inline_shape, title, description):
    doc_pr = inline_shape._inline.docPr
    doc_pr.set("title", title)
    doc_pr.set("descr", description)


def add_page_number(paragraph):
    run = paragraph.add_run()
    fld_char_1 = OxmlElement("w:fldChar")
    fld_char_1.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = " PAGE "
    fld_char_2 = OxmlElement("w:fldChar")
    fld_char_2.set(qn("w:fldCharType"), "end")
    run._r.extend([fld_char_1, instr_text, fld_char_2])
    set_font(run, size=9, color=MUTED)


def add_body(doc, text, before=0, after=6, keep_with_next=False):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.12
    p.paragraph_format.keep_with_next = keep_with_next
    r = p.add_run(text)
    set_font(r, size=11, color=BLACK)
    return p


def add_lead(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(10)
    p.paragraph_format.line_spacing = 1.18
    r = p.add_run(text)
    set_font(r, size=12.5, color=BLACK)
    return p


def add_bullets(doc, items, level=0):
    for item in items:
        p = doc.add_paragraph(style="List Bullet" if level == 0 else "List Bullet 2")
        p.paragraph_format.left_indent = Inches(0.28 + 0.2 * level)
        p.paragraph_format.first_line_indent = Inches(-0.18)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.08
        r = p.add_run(item)
        set_font(r, size=10.7, color=BLACK)


def add_numbered(doc, items):
    for item in items:
        p = doc.add_paragraph(style="List Number")
        p.paragraph_format.left_indent = Inches(0.3)
        p.paragraph_format.first_line_indent = Inches(-0.2)
        p.paragraph_format.space_after = Pt(4)
        r = p.add_run(item)
        set_font(r, size=10.7, color=BLACK)


def add_code_line(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.28)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    set_font(r, name="Consolas", size=9.4, color=ACCENT_DARK)
    return p


def add_table(doc, headers, rows, widths=None, font_size=9.2):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.style = "Table Grid"
    hdr = table.rows[0]
    set_repeat_table_header(hdr)
    prevent_row_split(hdr)
    for i, header in enumerate(headers):
        cell = hdr.cells[i]
        cell.text = ""
        shade_cell(cell, ACCENT_DARK)
        set_cell_margins(cell)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(header)
        set_font(r, size=font_size, bold=True, color=WHITE)
        if widths:
            cell.width = Inches(widths[i])
    for ridx, row_data in enumerate(rows):
        row = table.add_row()
        prevent_row_split(row)
        for i, value in enumerate(row_data):
            cell = row.cells[i]
            cell.text = ""
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP
            if ridx % 2 == 1:
                shade_cell(cell, LIGHT)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.03
            r = p.add_run(str(value))
            set_font(r, size=font_size, color=BLACK)
            if widths:
                cell.width = Inches(widths[i])
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return table


def add_heading(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.page_break_before = level == 1
    p.paragraph_format.space_before = Pt(8 if level == 1 else 7)
    p.paragraph_format.space_after = Pt(5)
    return p


def add_figure(doc, path, width, caption, title, alt):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    shape = p.add_run().add_picture(str(path), width=Inches(width))
    add_alt_text(shape, title, alt)
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(8)
    r = cap.add_run(caption)
    set_font(r, size=9, italic=True, color=MUTED)


def configure_document(doc):
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.72)
    section.bottom_margin = Inches(0.68)
    section.left_margin = Inches(0.78)
    section.right_margin = Inches(0.78)

    normal = doc.styles["Normal"]
    normal.font.name = "Arial"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Arial")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Arial")
    normal.font.size = Pt(11)
    normal.font.color.rgb = RGBColor.from_string(BLACK)

    title = doc.styles["Title"]
    title.font.name = "Arial"
    title._element.rPr.rFonts.set(qn("w:ascii"), "Arial")
    title._element.rPr.rFonts.set(qn("w:hAnsi"), "Arial")
    title.font.size = Pt(29)
    title.font.bold = True
    title.font.color.rgb = RGBColor.from_string(BLACK)

    for style_name, size, color in (("Heading 1", 19, BLACK), ("Heading 2", 14, ACCENT_DARK), ("Heading 3", 11.5, BLACK)):
        style = doc.styles[style_name]
        style.font.name = "Arial"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Arial")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Arial")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)

    header = section.header
    hp = header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    hr = hp.add_run("OpenCodeReview AI Engineering Guide")
    set_font(hr, size=8.5, color=MUTED)

    footer = section.footer
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fr = fp.add_run("OpenCodeReview   |   Repository snapshot d939059   |   Page ")
    set_font(fr, size=8.5, color=MUTED)
    add_page_number(fp)


def build_document():
    WORK.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    create_architecture_diagram()
    create_flow_diagram()

    doc = Document()
    configure_document(doc)

    # Cover
    p = doc.add_paragraph(style="Title")
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(86)
    p.paragraph_format.space_after = Pt(14)
    p.add_run("OpenCodeReview AI Engineering Architecture Guide")
    sub = doc.add_paragraph()
    sub.paragraph_format.space_after = Pt(22)
    r = sub.add_run("A simple technical guide to the complete project and how to explain it in an AI engineer interview")
    set_font(r, size=16, color=ACCENT_DARK)
    add_body(doc, "Repository snapshot d939059", after=2)
    add_body(doc, "Prepared 30 October 2026", after=2)
    add_body(doc, "Scope: frontend, API, LangGraph workflow, RAG, prompting, evaluation, persistence, deployment, security, limitations, and interview answers", after=20)
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(130)
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run("Accuracy note")
    set_font(r, size=10.5, bold=True, color=BLACK)
    add_body(doc, "This guide describes the code that exists in the repository. It separates implemented behavior from optional features, deployment assumptions, and recommended production improvements.", after=0)
    add_heading(doc, "How to use this guide", 1)
    add_lead(doc, "Read the first four sections for a clear project story. Use the middle sections to explain technical choices. Use the final sections to prepare for interview questions.")
    add_table(doc, ["If you have", "Read"], [
        ("Two minutes", "Project overview and the 30 second interview answer"),
        ("Ten minutes", "Architecture, end to end flow, RAG, prompting, and evaluation"),
        ("A technical interview", "LangGraph state, grounding, persistence, security, limitations, and tradeoffs"),
        ("A system design round", "Deployment boundaries, failure handling, scaling plan, and production roadmap"),
    ], widths=[1.5, 5.2])
    add_heading(doc, "Guide map", 2)
    add_bullets(doc, [
        "Sections 1 through 4 explain the product, architecture, and complete request flow.",
        "Sections 5 through 15 explain the AI pipeline, RAG, prompts, reviewers, grounding, approval, and execution.",
        "Sections 16 through 22 explain storage, authentication, observability, evaluation, deployment, and failure handling.",
        "Sections 23 through 27 explain limitations, design tradeoffs, source files, interview answers, and terminology.",
        "Appendices list API routes, job states, configuration, and verification evidence.",
    ])

    add_heading(doc, "1 Project overview", 1)
    add_lead(doc, "OpenCodeReview is an AI assisted GitHub pull request reviewer. It collects a pull request, asks three specialist reviewers to inspect it, combines and grounds their findings, pauses for human approval, and posts approved comments back to GitHub.")
    add_heading(doc, "The problem it solves", 2)
    add_body(doc, "A large pull request can contain correctness defects, security risks, and missing tests. A single general prompt can miss important details or mix unrelated concerns. OpenCodeReview divides the review into specialist tasks and keeps a human in control of publication.")
    add_heading(doc, "The main engineering idea", 2)
    add_body(doc, "The project is a modular monolith with a static web frontend. The backend is one Python process containing an API, an in process worker pool, a LangGraph workflow, optional retrieval, evaluation, and GitHub execution. The modules are separated in code, but they are not independently deployed microservices.")
    add_table(doc, ["Question", "Short answer"], [
        ("What kind of AI system is it", "A multi stage LLM workflow for pull request review with optional RAG and human approval"),
        ("What is the orchestration framework", "LangGraph"),
        ("What are the specialist roles", "Correctness, security, and test coverage"),
        ("What is the frontend", "React and Vite deployed on Vercel"),
        ("What is the backend", "FastAPI and Gradio in one Docker deployment on Hugging Face Spaces"),
        ("What is the vector database", "ChromaDB with local MiniLM embeddings when explicitly enabled"),
        ("What controls publishing", "A LangGraph interrupt that requires a human decision"),
    ], widths=[2.2, 4.5])

    add_heading(doc, "2 System architecture", 1)
    add_lead(doc, "The system has two deployment surfaces and several external dependencies. Vercel serves the user interface. A Docker Space runs the Python backend. GitHub and model providers remain external services.")
    add_figure(doc, ARCH_DIAGRAM, 6.8, "Figure 1  Deployment and runtime architecture", "OpenCodeReview architecture", "Architecture diagram showing a browser using a Vercel React frontend, a Hugging Face Docker backend containing FastAPI and LangGraph workers, and integrations with GitHub, model providers, ChromaDB, Supabase or SQLite, LangSmith, and Langfuse.")
    add_heading(doc, "Deployment boundaries", 2)
    add_table(doc, ["Boundary", "Responsibility", "Important detail"], [
        ("Vercel", "Serves the React application and the provider test function", "Most API routes are rewritten to the Hugging Face backend"),
        ("Hugging Face Docker Space", "Runs Uvicorn, FastAPI, worker threads, LangGraph, RAGAS, and the mounted Gradio UI", "These components share one process and failure domain"),
        ("GitHub", "Supplies pull request data and accepts review comments", "Posting requires a valid token and an unchanged pull request head"),
        ("Model providers", "Generate structured reviewer findings", "Built in and custom provider configurations are supported"),
        ("Storage", "Stores job records, graph checkpoints, and optional embeddings", "The project uses separate storage mechanisms for separate concerns"),
    ], widths=[1.35, 2.6, 2.75])
    add_heading(doc, "Why this is a modular monolith", 2)
    add_body(doc, "FastAPI, the worker pool, LangGraph, and RAGAS run inside the same Python service. The code has clear modules and nodes, but there is no external queue and no independently scalable reviewer service. Calling it a microservice architecture would overstate the current implementation.")

    add_heading(doc, "3 End to end review flow", 1)
    add_lead(doc, "A review moves from a browser request to a persisted job, then through the graph, approval, and GitHub posting. Each stage adds data or enforces a safety condition.")
    add_numbered(doc, [
        "The user signs in or supplies the configured API credential, enters a repository and pull request number, selects a review mode and model provider, and submits the job.",
        "FastAPI validates the request, checks rate limits, resolves the current head and base commits when possible, creates a pipeline fingerprint, and reuses an equivalent active job when appropriate.",
        "The API persists a job record and submits work to a process local ThreadPoolExecutor. The default maximum worker count is two.",
        "The ingestion node downloads pull request metadata, the diff, changed files, and current file content from GitHub.",
        "The retrieval node optionally builds or loads a Chroma collection for unchanged repository context and returns the fifteen most similar chunks.",
        "Correctness, security, and test coverage reviewers run in parallel. Each reviewer receives the diff, changed file context, retrieved context, and a specialist prompt.",
        "The aggregator grounds findings against changed lines, removes duplicates, calculates a risk oriented score, and creates the verdict.",
        "LangGraph interrupts at the approval node. The frontend displays the result and waits for the user to approve or reject it.",
        "After approval, the API verifies that the pull request commits have not changed. The executor posts valid inline comments and a summary to GitHub.",
        "RAGAS can run in the background and attach evaluation metrics. The frontend polls the job endpoint every three seconds until it reaches a final state.",
    ])
    add_heading(doc, "Why the fingerprint matters", 2)
    add_body(doc, "The job fingerprint combines the owner, request, pull request commits, selected configuration, prompts, and relevant node source. This prevents accidental duplicate work and makes a changed prompt or pipeline version produce a new review.")

    add_heading(doc, "4 Frontend architecture", 1)
    add_lead(doc, "The frontend is a lightweight React application. It creates jobs, polls status, renders review results, manages provider settings in the browser, and sends approval or rejection decisions.")
    add_table(doc, ["File", "Responsibility"], [
        ("frontend/src/main.jsx", "Page composition, form state, status polling, results, approval controls, saved providers, and user feedback"),
        ("frontend/src/review-client.js", "HTTP client functions for authentication, reviews, decisions, cancellation, feedback, and provider tests"),
        ("frontend/src/styles.css", "Responsive visual styling"),
        ("frontend/api/provider-test.js", "Vercel serverless function that tests a user supplied model endpoint without starting a review"),
        ("frontend/vercel.json", "Build output, API rewrites, and single page application routing"),
    ], widths=[2.3, 4.4])
    add_heading(doc, "Polling model", 2)
    add_body(doc, "The browser asks for the latest job state every three seconds. Polling is simple and works well for a portfolio scale deployment. At higher volume, Server Sent Events or WebSockets would reduce repeated requests and deliver progress faster.")
    add_heading(doc, "Provider settings", 2)
    add_body(doc, "The UI keeps saved custom provider configurations in browser local storage and sends the selected configuration with the review request. This is convenient, but long lived API keys in local storage increase exposure if the site ever has a cross site scripting defect. A production design should use a server side encrypted secret store and short references from the browser.")

    add_heading(doc, "5 FastAPI and job lifecycle", 1)
    add_lead(doc, "FastAPI is the control plane. It authenticates the caller, creates and persists jobs, starts worker tasks, exposes progress, and accepts decisions and feedback.")
    add_heading(doc, "Job creation", 2)
    add_body(doc, "The create endpoint validates repository input, pull request number, review mode, provider choice, and authentication. It applies an optional per minute rate limit, derives a stable pipeline version, and records enough request data to reproduce the intended review configuration.")
    add_heading(doc, "Worker execution", 2)
    add_body(doc, "The worker pool is a Python ThreadPoolExecutor inside the API process. This avoids extra infrastructure and is appropriate for a small deployment. It is not a durable queue: queued work can be lost during a process restart, and another replica cannot see the process local queue.")
    add_heading(doc, "Job states", 2)
    add_table(doc, ["State", "Meaning"], [
        ("queued", "The API accepted the request and scheduled the worker"),
        ("processing", "The graph is ingesting, retrieving, reviewing, or aggregating"),
        ("awaiting_approval", "The graph interrupted and is waiting for a human decision"),
        ("processing_decision", "The graph resumed after approval and is preparing to post"),
        ("posted", "The executor posted the review successfully"),
        ("partially_posted", "Some comments posted and some failed or fell back"),
        ("post_failed", "The review completed but GitHub publication failed"),
        ("rejected", "The user rejected the review and nothing was posted"),
        ("cancelled", "The user requested cancellation"),
        ("incomplete or failed", "A reviewer or pipeline failure prevented a reliable result"),
    ], widths=[1.65, 5.05])

    add_heading(doc, "6 LangGraph state and orchestration", 1)
    add_lead(doc, "LangGraph models the review as a state machine. Nodes read the shared state and return updates. The graph controls sequencing, parallel branches, interruption, and resume behavior.")
    add_figure(doc, FLOW_DIAGRAM, 6.8, "Figure 2  LangGraph review workflow", "LangGraph review workflow", "Workflow diagram showing ingestion, retrieval, three parallel reviewers, aggregation, a human approval interrupt, and GitHub execution. Shared state contains changed files, retrieved context, findings, verdict, and stage metadata.")
    add_heading(doc, "Graph topology", 2)
    add_code_line(doc, "START -> ingestion -> retrieval -> {correctness | security | test coverage} -> aggregator -> human approval -> executor -> END")
    add_heading(doc, "Shared state", 2)
    add_body(doc, "OpenCodeReviewState is a Pydantic model that carries repository identifiers, commit SHAs, the diff, changed files, retrieved context, findings, reviewer stage results, the verdict, approval data, and execution receipts. List reducers append context chunks, findings, and stage metadata from parallel branches instead of overwriting them.")
    add_table(doc, ["State object", "Purpose"], [
        ("ChangedFile", "Path, status, patch, content, and line information for a changed file"),
        ("ContextChunk", "Retrieved repository or pull request context with source metadata and similarity"),
        ("Finding", "Structured issue with file, line, severity, category, explanation, confidence, and grounding flags"),
        ("Verdict", "Aggregated recommendation, risk score, summary, and completeness information"),
        ("Review stage metadata", "Timing, provider, status, and failure classification for each specialist"),
    ], widths=[1.8, 4.9])
    add_heading(doc, "Why use a graph", 2)
    add_body(doc, "A graph makes parallel review, human interruption, and explicit state transitions visible. A simple chain could call the models, but it would make the approval pause and partial failure handling harder to reason about.")

    add_heading(doc, "7 Ingestion and GitHub context", 1)
    add_lead(doc, "Ingestion turns a pull request URL and number into the concrete evidence the reviewers need.")
    add_bullets(doc, [
        "Pull request metadata including head and base commits",
        "The complete pull request diff where GitHub provides it",
        "Changed file metadata and patches with pagination",
        "Current file content for changed paths",
        "Fork aware repository identifiers",
        "Optional baseline comparison data for incremental review",
    ])
    add_heading(doc, "Incremental review", 2)
    add_body(doc, "The system can compare a newer head against a previous baseline when the baseline is an ancestor and the comparison is not truncated. An incremental result is deliberately blocked from posting because line positions and coverage may not represent the full pull request.")
    add_heading(doc, "Rate limits and failures", 2)
    add_body(doc, "GitHub API calls use timeouts and surface authentication or rate limit problems. A missing token sharply reduces the public API allowance. The system should treat incomplete GitHub data as an incomplete review rather than silently presenting a confident verdict.")

    add_heading(doc, "8 Retrieval augmented generation", 1)
    add_lead(doc, "RAG supplies repository context that is not present in the pull request diff. It is optional and runs only when local embeddings are explicitly enabled.")
    add_heading(doc, "What is indexed", 2)
    add_table(doc, ["Indexed source", "What enters Chroma", "Why"], [
        ("Unchanged repository files", "Eligible source, configuration, SQL, shell, markup, style, and documentation files from the base commit", "Gives reviewers definitions, conventions, and neighboring behavior outside the diff"),
        ("Important repository documents", "README, CONTRIBUTING, code of conduct, style guide, and selected GitHub contribution docs", "Adds project rules and intended usage"),
        ("Recent merged pull requests", "Title and body of up to twenty recently updated merged pull requests", "Adds historical design intent and related change descriptions"),
    ], widths=[1.45, 3.15, 2.1])
    add_heading(doc, "What is not indexed", 2)
    add_bullets(doc, [
        "Files changed by the current pull request because their diff and current content are already supplied directly to the reviewers",
        "Tests and spec files selected by path naming rules",
        "Hidden paths, dependency folders, virtual environments, vendored code, build output, and distribution output",
        "Binary files and unsupported extensions",
        "Full review comment discussions from old pull requests; the current implementation indexes only merged pull request title and body text",
    ])
    add_heading(doc, "Why the entire repository is not embedded", 2)
    add_body(doc, "The goal is useful context, not a byte for byte archive. Embedding changed files would duplicate evidence already included in the prompt. Embedding dependencies, generated artifacts, and binary files would add cost and noise. Filtering also lowers indexing time and reduces irrelevant retrieval.")
    add_body(doc, "A simple interview answer is: the system indexes eligible unchanged code and project documentation, then retrieves only the small set most relevant to the current diff. It does not send the entire repository to the model.")

    add_heading(doc, "9 Chunking and vector search", 1)
    add_lead(doc, "Chunking is structure aware for Python and paragraph based for other supported files. The current design favors simple, explainable rules over a language parser for every programming language.")
    add_table(doc, ["Content", "Chunking method", "Technical detail"], [
        ("Python", "Split at top level class, function, and async function definitions", "Keeps a logical unit together; preamble code becomes its own chunk"),
        ("Other source and documents", "Split at blank lines and merge consecutive paragraphs", "Targets about 1,200 characters; a single long paragraph is not force split"),
        ("Merged pull request text", "Use the same paragraph chunker", "Title and body are combined before chunking"),
    ], widths=[1.35, 2.75, 2.6])
    add_heading(doc, "Embedding model and store", 2)
    add_body(doc, "ChromaDB uses a persistent collection for each repository. The preferred embedding function is ONNX MiniLM L6 V2. The collection uses cosine distance and records the base commit SHA as a freshness key. A matching SHA reuses the collection; a changed base SHA rebuilds it.")
    add_heading(doc, "Retrieval query", 2)
    add_body(doc, "The current pull request diff becomes the semantic query. It is truncated to 8,000 characters to stay within embedding limits. Chroma returns the top fifteen chunks. Each ContextChunk keeps the source type, file path, chunk type, text, and similarity information for later prompting and evaluation.")
    add_heading(doc, "Chunking tradeoffs", 2)
    add_body(doc, "The Python rule preserves semantic units better than fixed token windows. The generic rule is cheap and language independent, but it can combine unrelated paragraphs or keep an unusually long block intact. A future version could use tree sitter parsers, token aware limits, and small overlaps for more languages.")

    add_heading(doc, "10 Prompt engineering", 1)
    add_lead(doc, "The project uses specialist, zero shot, retrieval contextual prompts with structured outputs. Each reviewer gets a role specific system prompt and a user message containing the actual pull request evidence.")
    add_table(doc, ["Prompting choice", "How it is used"], [
        ("Zero shot", "The reviewer is given instructions and evidence without worked examples"),
        ("Role specialization", "Separate prompts define correctness, security, and test coverage responsibilities"),
        ("System and user separation", "Stable reviewer rules stay in the system message; repository and pull request data stay in the user message"),
        ("RAG contextualization", "Retrieved chunks are included as supporting context when retrieval is enabled"),
        ("Structured output", "Pydantic schemas require machine readable findings instead of free form prose"),
        ("Repair prompting", "One retry asks the model to fix invalid output formatting"),
        ("Token cut retry", "If output is truncated, a retry requests a shorter list focused on the top five findings"),
    ], widths=[1.75, 4.95])
    add_heading(doc, "Prompt inputs", 2)
    add_bullets(doc, [
        "Pull request metadata and review mode",
        "The diff and changed file content",
        "Retrieved repository context when available",
        "Rules for severity, file and line references, confidence, and evidence",
        "A strict output schema for findings",
    ])
    add_heading(doc, "Fast and thorough modes", 2)
    add_body(doc, "Fast mode narrows changed file content to windows of about twelve lines around diff hunks. Thorough mode supplies more surrounding content. This is a latency and token tradeoff, not a different model architecture.")
    add_heading(doc, "Why structured output matters", 2)
    add_body(doc, "The aggregator and executor need fields such as path, line, severity, category, confidence, and comment text. A prose response is difficult to validate and post safely. Structured output makes the model response testable and lets later nodes reject malformed or ungrounded findings.")

    add_heading(doc, "11 Specialist reviewer nodes", 1)
    add_lead(doc, "The three reviewers are specialized LLM calls inside graph nodes. They are not autonomous services with separate memory or independent goals.")
    add_table(doc, ["Reviewer", "Primary questions", "Typical output"], [
        ("Correctness", "Can this change produce wrong behavior, exceptions, data loss, race conditions, or broken edge cases", "Defect findings tied to changed files and lines"),
        ("Security", "Can untrusted input, authentication, authorization, secrets, injection, or unsafe execution be exploited", "Security findings with severity and evidence"),
        ("Test coverage", "Which changed behaviors lack useful tests or assertions", "Test suggestions that do not directly increase the defect risk score"),
    ], widths=[1.45, 3.45, 1.8])
    add_heading(doc, "Parallel execution", 2)
    add_body(doc, "LangGraph fans out from retrieval to all three reviewers. Parallel execution reduces total latency because the calls are independent. List reducers combine their outputs when the branches meet at the aggregator.")
    add_heading(doc, "Runtime controls", 2)
    add_body(doc, "Each reviewer runs with a timeout. The runtime records completion, provider details, token usage when available, and classified failures such as authentication, rate limiting, timeout, or generic model error. A failed specialist makes the review incomplete instead of pretending that all review dimensions were covered.")

    add_heading(doc, "12 Grounding and hallucination controls", 1)
    add_lead(doc, "The model can propose a plausible issue at the wrong location. OpenCodeReview therefore validates findings against the actual diff before scoring or posting them.")
    add_heading(doc, "Line grounding", 2)
    add_body(doc, "finding_quality.py checks whether a finding points to a real changed file and overlaps an added or changed line. It marks findings as source grounded, outside the diff, or needing verification. The executor posts only source grounded findings.")
    add_heading(doc, "Claim calibration", 2)
    add_body(doc, "Test suggestions are kept at low or informational severity and do not increase the defect score. Compiler or type checker claims require a reproducible diagnostic before they affect risk. These rules reduce confident comments that are not supported by source evidence.")
    add_heading(doc, "Defense in depth", 2)
    add_numbered(doc, [
        "Prompts ask for concrete file and line evidence.",
        "Pydantic validates the response shape.",
        "The finding quality layer validates source location.",
        "The aggregator excludes ungrounded items from scoring.",
        "The human reviews the result before publication.",
        "The executor checks the latest pull request head and GitHub line validity.",
    ])

    add_heading(doc, "13 Aggregation and risk scoring", 1)
    add_lead(doc, "The aggregator converts three specialist outputs into one review result. It calibrates, deduplicates, scores, and summarizes findings while preserving failure information.")
    add_heading(doc, "Deduplication", 2)
    add_body(doc, "Findings are considered duplicates when they refer to the same file, are within five lines, have the same kind or category, and have the same normalized comment. A default confidence threshold of 0.25 removes very weak items.")
    add_heading(doc, "Risk score", 2)
    add_code_line(doc, "score = clamp(10 - 4 x critical - 2 x high - 0.5 x medium, 0, 10)")
    add_body(doc, "The score is a review heuristic, not a probability that the code is correct. Only grounded defect findings contribute. Test suggestions and unsupported claims do not. If a specialist fails, the recommendation becomes incomplete and the system avoids presenting a normal overall score.")
    add_heading(doc, "Optional critic", 2)
    add_body(doc, "The default verdict uses deterministic rules. An optional LLM critic can be enabled with AGGREGATOR_USE_LLM_CRITIC. Keeping the rule based path as the default reduces cost and makes the final score easier to explain.")

    add_heading(doc, "14 Human approval and checkpointing", 1)
    add_lead(doc, "The human approval node calls LangGraph interrupt. This stops the graph after analysis and returns a review payload to the application. A later decision resumes the same graph with Command resume.")
    add_heading(doc, "Approval safeguards", 2)
    add_bullets(doc, [
        "Rejection ends the workflow without posting comments",
        "Approval triggers a fresh head and base commit check",
        "A changed pull request is treated as stale and must be reviewed again",
        "Incremental baseline reviews cannot be posted",
        "Only grounded findings are eligible for execution",
    ])
    add_heading(doc, "Checkpoint limitation", 2)
    add_body(doc, "LangGraph checkpoints use a separate SQLite database. A durable database path is required for approval to survive a backend restart. Supabase job persistence alone does not make graph checkpoints durable because the two stores serve different purposes.")

    add_heading(doc, "15 GitHub executor", 1)
    add_lead(doc, "The executor is the side effect boundary. It turns approved findings into GitHub review comments and records a posting receipt.")
    add_heading(doc, "Posting sequence", 2)
    add_numbered(doc, [
        "Fetch the latest pull request head commit.",
        "Stop if the reviewed head no longer matches.",
        "Build right side inline comments for grounded locations.",
        "Post multi line comments when the finding spans a valid range.",
        "If GitHub rejects a line with HTTP 422, fall back to an issue level comment.",
        "Post the verdict summary and persist a receipt with successful and failed comment counts.",
    ])
    add_heading(doc, "Possible receipts", 2)
    add_body(doc, "The receipt can record posted, partially posted, post failed, dry run, or skipped outcomes. Keeping analysis separate from execution makes it possible to inspect a good review even if GitHub publication fails.")

    add_heading(doc, "16 Model provider layer", 1)
    add_lead(doc, "The LLM factory hides provider specific client setup from the reviewer nodes. Reviewers ask for a chat model and structured output without owning provider credentials or request formatting.")
    add_table(doc, ["Provider path", "Current behavior"], [
        ("Groq", "Preferred by the automatic selection path when GROQ_API_KEY is available; default model is openai/gpt-oss-20b"),
        ("Google Gemini", "Used when configured; current default model is gemini-3.1-flash-lite"),
        ("OpenAI compatible", "Custom base URL, model name, and API key can be registered for the job"),
        ("Anthropic compatible", "Custom endpoint configuration supported by the endpoint layer"),
        ("Google compatible", "Custom Google endpoint configuration supported"),
    ], widths=[1.7, 5.0])
    add_heading(doc, "Bring your own provider", 2)
    add_body(doc, "The frontend can test and save endpoint configurations. The backend registers a temporary endpoint for a review and removes it after review and background evaluation work. A production system should avoid sending raw long lived keys in ordinary job payloads and should use encrypted server side secret references.")
    add_heading(doc, "Provider naming detail", 2)
    add_body(doc, "The user interface includes a label written as Grok while the implementation uses Groq. This should be corrected because Groq is the inference provider used by the code and Grok refers to a different model family.")

    add_heading(doc, "17 Persistence and data ownership", 1)
    add_lead(doc, "The project uses several stores. Understanding their boundaries is important because persistence is only as strong as the least durable state needed to resume a workflow.")
    add_table(doc, ["Data", "Current store", "Durability note"], [
        ("Job records", "Supabase Postgres when SUPABASE_DB_URL is set, otherwise SQLite", "Supabase is suitable for durable job status; local SQLite depends on the mounted filesystem"),
        ("LangGraph checkpoints", "SQLite through SqliteSaver", "Must use a persistent path to survive restarts"),
        ("Live queue and worker futures", "Process memory", "Lost when the process stops"),
        ("OAuth sessions", "Process memory", "Lost during restart and not shared across replicas"),
        ("Temporary provider registrations", "Process memory", "Removed after the job"),
        ("Vector collections", "Chroma persistent directory", "Durable only when the configured directory is on persistent storage"),
        ("Saved browser providers", "Browser local storage", "Specific to one browser profile and exposed to frontend script context"),
    ], widths=[1.55, 2.1, 3.05])
    add_heading(doc, "Supabase security", 2)
    add_body(doc, "The migration enables row level security, revokes anonymous and authenticated access, and grants the service role access to review_jobs. The backend should hold the service credential. Browser code should never receive it.")
    add_heading(doc, "Production improvement", 2)
    add_body(doc, "Move the work queue to a durable service, store graph checkpoints in durable shared storage, and move OAuth sessions to a signed server side store. This would allow safe restarts and multiple backend replicas.")

    add_heading(doc, "18 Authentication and security", 1)
    add_lead(doc, "The web API supports GitHub OAuth and an optional API key. Job ownership checks prevent one signed in user from reading or controlling another user's job.")
    add_heading(doc, "Authentication paths", 2)
    add_table(doc, ["Path", "Use"], [
        ("GitHub OAuth authorization code", "Browser login for the hosted application"),
        ("OCR_API_KEY", "Optional API credential for controlled clients"),
        ("GitHub device flow and keyring", "Command line authentication path"),
        ("GITHUB_TOKEN", "Service credential used by core GitHub ingestion and execution when configured"),
    ], widths=[2.2, 4.5])
    add_heading(doc, "Important token boundary", 2)
    add_body(doc, "The web API can identify a signed in user through an OAuth session, but core graph ingestion and execution resolve GitHub credentials through the shared GitHub client and environment or keyring. In the hosted architecture this behaves more like a service credential than full per user token propagation. Explain this honestly in an interview and propose explicit per job credential injection as the improvement.")
    add_heading(doc, "Security controls", 2)
    add_bullets(doc, [
        "Configured CORS origins rather than a universal browser origin",
        "Optional request authentication and per minute rate limiting",
        "Owner checks on job reads, decisions, cancellation, and feedback",
        "No posting before human approval",
        "Stale commit checks before side effects",
        "Structured output and source grounding before GitHub comments",
        "Environment variables and OS keyring for long lived server or command line secrets",
    ])

    add_heading(doc, "19 Observability", 1)
    add_lead(doc, "Observability covers model traces, token usage, latency, and stage failures. It helps separate a model quality problem from a network, provider, or orchestration problem.")
    add_table(doc, ["Capability", "Implementation"], [
        ("Tracing", "LangSmith is enabled when a LangSmith or legacy LangChain key is configured"),
        ("LLM telemetry", "Langfuse callbacks can record model calls and scores"),
        ("Cost and tokens", "TokenCostCallback records usage, estimated cost, and time to first token where available"),
        ("Stage status", "Each reviewer records success, timing, provider, and failure type"),
        ("Health", "The API health endpoint reports configuration and important runtime capabilities"),
        ("External smoke checks", "A scheduled GitHub Actions workflow can alert through Slack or Discord webhooks"),
    ], widths=[1.55, 5.15])
    add_heading(doc, "What to monitor in production", 2)
    add_bullets(doc, [
        "Job queue wait time and end to end review latency",
        "Reviewer timeout, authentication, rate limit, and invalid output rates",
        "Grounded findings per review and human rejection rate",
        "GitHub 422 fallback rate and stale pull request rate",
        "Token usage and cost by provider, model, reviewer, and mode",
        "Checkpoint resume success after deploys or restarts",
    ])

    add_heading(doc, "20 Evaluation", 1)
    add_lead(doc, "The repository contains both online RAG evaluation and offline review evaluation. These tools measure different parts of the system and should not be combined into a single quality claim.")
    add_heading(doc, "RAGAS evaluation", 2)
    add_table(doc, ["Metric", "Meaning", "Condition"], [
        ("Context precision", "How much retrieved context is relevant to the answer", "Runs when evaluation dependencies and provider access are available"),
        ("Faithfulness", "Whether the answer is supported by supplied context", "Runs against generated findings and available context"),
        ("Context recall", "Whether retrieved context covers a supplied reference answer", "Requires ground truth"),
        ("Answer relevancy and diversity", "Additional hosted quality signals", "May be skipped in constrained hosted mode"),
    ], widths=[1.4, 3.0, 2.3])
    add_heading(doc, "Offline review evaluation", 2)
    add_body(doc, "evaluate.py compares predicted findings with human review comments. A match uses the same file path and a line within about three lines, with optional keyword checks. The script reports precision, recall, and F1. Additional fixtures support located diff checks, regression tests, and CodeReviewBench style evaluation.")
    add_heading(doc, "Current evidence", 2)
    add_body(doc, "The committed evaluation snapshots do not demonstrate strong model quality. The current eval_results.json snapshot contains one pull request with zero predictions and zero precision, recall, and F1. offline_baseline.json records 136 gold items and zero predictions. The correct claim is that the project has evaluation infrastructure, not that it has proven high accuracy.")
    add_heading(doc, "How to improve the evaluation", 2)
    add_numbered(doc, [
        "Create a versioned gold dataset with clearly located defects and reviewer rationale.",
        "Evaluate each specialist separately before evaluating the aggregate verdict.",
        "Measure grounded precision, recall, severity calibration, duplicate rate, latency, and cost.",
        "Run the same dataset across prompt, model, and retrieval changes.",
        "Require a regression threshold before deployment and report confidence intervals when the dataset is large enough.",
    ])

    add_heading(doc, "21 Deployment and continuous delivery", 1)
    add_lead(doc, "The frontend and backend deploy separately. The frontend is static. The backend uses a Docker image because the Python stack includes FastAPI, Gradio, LangGraph, evaluation libraries, and optional local retrieval dependencies.")
    add_table(doc, ["Artifact", "Deployment"], [
        ("frontend", "Vercel builds the Vite application and serves the static output"),
        ("frontend provider test", "Vercel serverless function under api/provider-test"),
        ("backend", "Dockerfile.api starts Uvicorn with combined_server:app on Hugging Face Spaces"),
        ("combined application", "combined_server.py exposes FastAPI routes and mounts the Gradio demo at the root"),
        ("CI deployment", "A GitHub Actions workflow deploys the Hugging Face Space from the main branch"),
        ("monitoring workflow", "A scheduled smoke workflow runs every six hours and can send alerts"),
    ], widths=[1.9, 4.8])
    add_heading(doc, "Vercel routing", 2)
    add_body(doc, "The provider test route stays on Vercel. Other API requests are rewritten to the Hugging Face backend. The final fallback returns the React application for client side navigation.")
    add_heading(doc, "Local deployment", 2)
    add_body(doc, "Docker Compose supports a command line profile and a named Chroma volume. Durable local behavior still depends on mounting the checkpoint and job database paths, not only the vector directory.")

    add_heading(doc, "22 Failure handling", 1)
    add_lead(doc, "The system records partial failures instead of hiding them. This is essential for an AI review product because a missing reviewer changes the meaning of the result.")
    add_table(doc, ["Failure", "Current response", "Recommended production action"], [
        ("Model timeout", "Classify the reviewer stage as timed out and mark the review incomplete", "Retry through a durable queue with a bounded attempt count"),
        ("Invalid model output", "Run one formatting repair attempt", "Track schema failure rate by model and prompt version"),
        ("Token truncation", "Retry with a higher budget and request the top five findings", "Use adaptive input reduction and provider specific limits"),
        ("GitHub rate limit or authentication", "Surface the error and avoid a normal verdict", "Refresh credentials or defer until reset"),
        ("Stale pull request", "Block posting", "Start a fresh review for the new head"),
        ("Inline comment rejected", "Fall back to an issue level comment", "Record the line mapping failure for analysis"),
        ("Backend restart", "Process local queue and sessions are lost", "Use durable queue, session store, and shared checkpoints"),
    ], widths=[1.35, 2.7, 2.65], font_size=9.0)

    add_heading(doc, "23 Current limitations", 1)
    add_lead(doc, "The project demonstrates a complete architecture, but several parts are portfolio scale rather than production scale. Naming these limits makes the design explanation stronger.")
    add_bullets(doc, [
        "The worker queue is process local and not restart safe.",
        "OAuth sessions and temporary endpoint registrations are process local and not replica safe.",
        "LangGraph checkpoints use SQLite and require explicitly durable storage.",
        "Optional local retrieval is disabled by default on hosted deployments, so the hosted review may operate without vector context.",
        "The generic chunker is not syntax aware for most languages and has no overlap.",
        "The retrieval query uses only the first 8,000 characters of a long diff.",
        "Core GitHub operations use a service style credential path rather than complete per user token propagation.",
        "Saved provider keys in browser local storage are convenient but not ideal for production secret handling.",
        "The committed evaluation files currently provide no positive accuracy result.",
        "The UI label Grok does not match the Groq provider used by the implementation.",
    ])
    add_heading(doc, "Production roadmap", 2)
    add_numbered(doc, [
        "Replace the in process executor with a durable queue and idempotent workers.",
        "Move checkpoints and OAuth sessions to durable shared stores.",
        "Pass scoped per user GitHub credentials explicitly through each job.",
        "Store provider secrets encrypted on the server and expose only secret references to the browser.",
        "Use language aware chunking, diff hunk queries, reranking, and retrieval evaluation.",
        "Build a versioned gold dataset and enforce regression gates for grounded precision and recall.",
        "Add streaming progress, distributed tracing, quotas, abuse controls, and load tests.",
    ])

    add_heading(doc, "24 AI engineering decisions and tradeoffs", 1)
    add_lead(doc, "The strongest interview explanation connects each technology to a concrete problem and admits the cost of that choice.")
    add_table(doc, ["Decision", "Benefit", "Tradeoff"], [
        ("Three specialist reviewers", "Better task focus and parallel latency", "More model calls, cost, and duplicate findings"),
        ("LangGraph", "Explicit branches, shared state, interrupt, and resume", "More orchestration code than a simple sequential script"),
        ("Structured output", "Validation, aggregation, and safe posting", "Schema failures require repair logic"),
        ("Optional RAG", "Repository context beyond the diff", "Indexing cost, retrieval noise, and hosted resource requirements"),
        ("Human approval", "Prevents automatic publication of weak findings", "Adds delay and requires durable checkpoints"),
        ("Rule based score", "Cheap, consistent, and explainable", "Cannot capture every contextual risk"),
        ("Process local worker pool", "Simple deployment and low infrastructure cost", "Not durable or horizontally scalable"),
        ("Vercel plus Docker Space", "Simple frontend delivery and a flexible Python backend", "Two deployment surfaces and cross origin operational complexity"),
    ], widths=[1.7, 2.5, 2.5], font_size=9.0)
    add_heading(doc, "The architecture in one technical paragraph", 2)
    add_body(doc, "OpenCodeReview uses a React and Vite frontend on Vercel and a FastAPI modular monolith on a Hugging Face Docker Space. FastAPI persists review jobs and starts an in process worker. A LangGraph state machine ingests GitHub pull request evidence, optionally retrieves base branch context from ChromaDB, fans out to three structured output LLM reviewers, grounds and aggregates their findings, interrupts for human approval, and resumes to post source grounded comments through the GitHub API. Supabase or SQLite stores job state, a separate SQLite saver stores graph checkpoints, and LangSmith, Langfuse, RAGAS, and offline fixtures support observability and evaluation.")

    add_heading(doc, "25 Source code map", 1)
    add_lead(doc, "Use this map when an interviewer asks where a behavior is implemented.")
    add_table(doc, ["Area", "Files"], [
        ("Web API and Gradio app", "backend/api.py and backend/app.py"),
        ("Combined backend server", "backend/combined_server.py"),
        ("Graph definition and shared state", "backend/graph.py and backend/state.py"),
        ("CLI", "backend/main.py"),
        ("GitHub access", "backend/integrations/github_client.py and auth.py"),
        ("Model and endpoint setup", "backend/integrations/llm_factory.py, endpoints.py, and config.py"),
        ("Graph nodes", "nodes/ingestion.py, retrieval.py, correctness_reviewer.py, security_reviewer.py, test_coverage_reviewer.py, finding_quality.py, aggregator.py, human_approval.py, executor.py"),
        ("Prompt versions", "prompts/correctness_v1.txt, security_v1.txt, test_coverage_v1.txt, aggregator_v1.txt"),
        ("Observability", "backend/observability.py"),
        ("Evaluation", "eval_data/ragas_eval.py, evaluate.py, offline_regression.py, code_review_bench_eval.py, and fixtures"),
        ("Frontend", "frontend/src/main.jsx, review-client.js, styles.css, frontend/api/provider-test.js"),
        ("Deployment", "Dockerfile.api, docker-compose.yml, frontend/vercel.json, GitHub Actions workflows"),
        ("Database migration", "supabase/migrations/20261020_create_review_jobs.sql"),
        ("Project documentation", "docs/architecture, docs/runbooks, docs/evaluation, and docs/project-guide"),
        ("Backward compatibility", "Root Python modules are small shims for existing imports and launch commands"),
    ], widths=[1.65, 5.05], font_size=9.0)

    add_heading(doc, "26 Interview answers", 1)
    add_heading(doc, "Thirty second version", 2)
    add_body(doc, "I built OpenCodeReview, an AI assisted pull request reviewer. A React frontend creates a review job in FastAPI. LangGraph then ingests the GitHub diff, optionally retrieves repository context from ChromaDB, and runs correctness, security, and test coverage reviewers in parallel. Their structured findings are grounded against changed lines, deduplicated, and scored. The graph pauses for human approval before an executor posts valid comments to GitHub. I also added tracing and both online and offline evaluation, while keeping the current quality claims evidence based.")
    add_heading(doc, "Ninety second version", 2)
    add_body(doc, "The frontend is a Vite React app on Vercel and the backend is a FastAPI modular monolith in a Hugging Face Docker Space. When a user submits a pull request, the API validates the request, fingerprints the pipeline, persists a job, and starts an in process worker. LangGraph orchestrates ingestion, optional retrieval, three parallel reviewer nodes, aggregation, human approval, and execution. RAG indexes eligible unchanged base branch code, important documentation, and recent merged pull request descriptions. Python is chunked at top level definitions, while other files use paragraph chunks around 1,200 characters. Each reviewer uses a specialist zero shot prompt with structured Pydantic output. A grounding layer checks that findings refer to real changed lines, and only grounded findings affect risk or get posted. The main production gaps are the process local queue and sessions, SQLite checkpoints, and the lack of a positive committed evaluation baseline. My next step would be durable workers, shared checkpoints, per user GitHub credentials, and regression gated evaluation.")
    add_heading(doc, "Five minute structure", 2)
    add_numbered(doc, [
        "State the problem: pull requests need multiple review perspectives, but automated comments must remain grounded and human controlled.",
        "Draw the system boundary: Vercel frontend, FastAPI Docker backend, GitHub, model providers, and storage.",
        "Walk through the LangGraph nodes from ingestion to executor.",
        "Explain RAG precisely, including what is indexed, chunking, top fifteen retrieval, and why changed files are excluded.",
        "Explain prompting and structured output, then the repair and timeout paths.",
        "Explain hallucination controls, scoring, and human approval.",
        "Describe evaluation and state the current evidence honestly.",
        "Finish with one production limitation and the concrete upgrade you would make.",
    ])

    add_heading(doc, "27 Interview questions and answers", 1)
    qa = [
        ("Why did you use multiple reviewers instead of one prompt", "Correctness, security, and test coverage require different attention. Separate prompts improve focus and allow parallel execution. The aggregator handles overlap and the extra cost."),
        ("Is this really a multi agent system", "It is a graph of specialist LLM nodes with shared state. I call it a multi agent workflow in product language, but technically the specialists are controlled graph nodes, not independent long running agents."),
        ("What type of chunking did you use", "Python uses structure aware top level class and function chunks. Other files use blank line paragraph chunks merged toward about 1,200 characters. There is currently no overlap."),
        ("What is indexed in the vector database", "Eligible unchanged base branch code and documentation, selected project documents, and the title and body of up to twenty recent merged pull requests. Changed files, dependencies, build output, tests selected by path rules, and binaries are excluded."),
        ("Why not index every file", "Changed files already enter the prompt directly, and dependencies or generated output add noise. Filtering reduces embedding time, storage, and irrelevant retrieval."),
        ("What type of prompting did you use", "Specialist zero shot system and user prompts, augmented with retrieved context, plus structured Pydantic outputs. A repair prompt handles invalid formatting and a shorter retry handles truncated output."),
        ("How do you reduce hallucinations", "I require structured file and line evidence, validate findings against changed lines, exclude unsupported items from scoring, require human approval, and recheck the pull request head before posting."),
        ("How do you evaluate it", "RAGAS evaluates context related behavior, while offline fixtures compare findings with human comments using path and line matching. The infrastructure exists, but the committed snapshots currently show zero predictions, so I do not claim proven accuracy."),
        ("What happens if one reviewer fails", "The stage records the failure and the aggregate recommendation becomes incomplete. The system should not present a normal score when one review dimension is missing."),
        ("What would break at scale", "The in process queue, process local sessions, and SQLite checkpoints. I would replace them with a durable queue, shared session store, and durable graph checkpoint store before horizontal scaling."),
        ("Why human approval", "LLM findings can be wrong even after grounding. Approval creates a clear control point before the system performs an external side effect."),
        ("Why is the score not a quality guarantee", "It is a deterministic severity based heuristic over grounded findings. It does not prove that the model found every defect or that the code is correct."),
    ]
    add_table(doc, ["Question", "Answer"], qa, widths=[2.25, 4.45], font_size=9.0)

    add_heading(doc, "28 Glossary", 1)
    glossary = [
        ("Agent node", "A graph step that asks an LLM to perform a specialized task"),
        ("Base SHA", "Commit identifier for the target branch version used as review context"),
        ("Checkpoint", "Saved graph state that allows an interrupted workflow to resume"),
        ("Chunk", "A smaller unit of code or text stored and retrieved from the vector database"),
        ("Cosine similarity", "A measure used to rank how close two embeddings are"),
        ("Embedding", "A numeric vector representing the semantic meaning of text"),
        ("Grounded finding", "A finding tied to a real changed file and line in the pull request"),
        ("Head SHA", "Commit identifier for the pull request version being reviewed"),
        ("Human in the loop", "A workflow that requires a human decision before it continues"),
        ("LangGraph", "A framework for stateful graph based LLM workflows"),
        ("Modular monolith", "One deployable backend process organized into well separated modules"),
        ("RAG", "Retrieval augmented generation, where relevant stored context is supplied to an LLM"),
        ("RAGAS", "A framework for measuring retrieval and answer quality"),
        ("Structured output", "Model output constrained to a machine readable schema"),
        ("Vector database", "A store that searches embeddings by semantic similarity"),
    ]
    add_table(doc, ["Term", "Meaning"], glossary, widths=[1.6, 5.1])

    add_heading(doc, "Appendix A API routes", 1)
    add_table(doc, ["Method and route", "Purpose"], [
        ("GET /api/health", "Report backend health and selected capability configuration"),
        ("GET /api/auth/github/start", "Begin GitHub OAuth"),
        ("GET /api/auth/github/callback", "Complete OAuth and create the browser session"),
        ("GET /api/auth/me", "Return the current signed in identity"),
        ("POST /api/auth/logout", "Remove the current session"),
        ("POST /api/reviews", "Create or reuse a review job"),
        ("GET /api/reviews/{job_id}", "Read job progress and results"),
        ("POST /api/reviews/{job_id}/cancel", "Request cancellation"),
        ("POST /api/reviews/{job_id}/feedback", "Record finding feedback"),
        ("POST /api/reviews/{job_id}/decision", "Approve or reject an interrupted review"),
        ("POST /api/provider-test", "Test a custom provider from the Vercel frontend function"),
    ], widths=[2.65, 4.05])

    add_heading(doc, "Appendix B Important configuration", 1)
    add_table(doc, ["Variable", "Purpose"], [
        ("GITHUB_TOKEN", "GitHub service credential for ingestion and execution"),
        ("GITHUB_OAUTH_CLIENT_ID and GITHUB_OAUTH_CLIENT_SECRET", "Browser OAuth configuration"),
        ("GITHUB_OAUTH_CALLBACK_URL", "OAuth callback location"),
        ("OCR_REQUIRE_AUTH and OCR_API_KEY", "API authentication policy"),
        ("OCR_ALLOWED_ORIGINS", "CORS allowlist"),
        ("OCR_FRONTEND_URL", "Frontend redirect target"),
        ("OCR_MAX_WORKERS", "In process review worker count, default two"),
        ("OCR_RATE_LIMIT_PER_MINUTE", "Optional API rate limit"),
        ("SUPABASE_DB_URL", "Postgres job persistence"),
        ("OPENCODEREVIEW_DB_PATH", "Local SQLite and graph checkpoint path used by application components"),
        ("OPENCODEREVIEW_ALLOW_LOCAL_EMBEDDINGS", "Enables Chroma retrieval"),
        ("OPENCODEREVIEW_VECTOR_DIR", "Persistent Chroma directory"),
        ("GROQ_API_KEY and GROQ_MODEL", "Groq access and model selection"),
        ("GEMINI_API_KEY", "Google Gemini access"),
        ("OCR_MODEL_TIMEOUT_SECONDS", "Model request timeout"),
        ("OCR_MODEL_MAX_OUTPUT_TOKENS", "Default model output budget"),
        ("OCR_REVIEWER_TIMEOUT_SECONDS", "Per reviewer timeout"),
        ("OCR_RAGAS_TIMEOUT_SECONDS", "Evaluation time budget"),
        ("LANGSMITH_API_KEY", "LangSmith tracing"),
        ("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY", "Langfuse telemetry"),
        ("AGGREGATOR_USE_LLM_CRITIC", "Optional LLM based aggregation critic"),
    ], widths=[2.75, 3.95], font_size=9.0)

    add_heading(doc, "Appendix C Verification and evidence limits", 1)
    add_lead(doc, "The following checks were completed while preparing this guide. They confirm buildability and syntax, not full production correctness.")
    add_table(doc, ["Check", "Result", "What it proves"], [
        ("Frontend production build", "Passed with Vite 8.3.4", "The current frontend source bundles successfully"),
        ("Backend Python compilation", "Passed for backend, nodes, eval_data, scripts, and tests", "The project Python modules have valid syntax"),
        ("Automated Python tests", "Passed: 43 tests on Python 3.12", "The current automated test suite passed"),
        ("Committed offline evaluation", "Zero predictions in the current snapshots", "Evaluation infrastructure exists, but quality is not established"),
        ("End to end hosted review", "Not used as proof for this guide", "A reachable frontend alone does not prove durable jobs, posting, or evaluation quality"),
    ], widths=[1.7, 2.0, 3.0], font_size=9.0)
    add_heading(doc, "A final interview principle", 2)
    add_body(doc, "Explain what the system does, then state what is optional, what is process local, what has been evaluated, and what you would change for production. Clear boundaries make the architecture credible.")

    # Document properties
    props = doc.core_properties
    props.title = "OpenCodeReview AI Engineering Architecture Guide"
    props.subject = "Complete architecture and interview guide for the OpenCodeReview project"
    props.author = "OpenCodeReview project documentation"
    props.keywords = "AI engineering, LangGraph, RAG, code review, architecture, prompting, evaluation"
    props.comments = "Prepared from repository snapshot d939059"

    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build_document()

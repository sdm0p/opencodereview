from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3] / "docs" / "project-guide" / "working"
PAGES = sorted((ROOT / "preview_pages").glob("page-*.png"), key=lambda p: int(p.stem.split("-")[-1]))
FONT_PATH = Path(r"C:\Windows\Fonts\arial.ttf")
FONT = ImageFont.truetype(str(FONT_PATH), 20) if FONT_PATH.exists() else ImageFont.load_default()

for sheet_index in range(0, len(PAGES), 16):
    selected = PAGES[sheet_index:sheet_index + 16]
    canvas = Image.new("RGB", (1320, 1760), "#D9D9D9")
    draw = ImageDraw.Draw(canvas)
    for idx, page_path in enumerate(selected):
        page = Image.open(page_path).convert("RGB")
        page.thumbnail((300, 395))
        col, row = idx % 4, idx // 4
        x, y = 18 + col * 326, 35 + row * 430
        canvas.paste(page, (x, y))
        draw.text((x, 8 + row * 430), f"Page {sheet_index + idx + 1}", fill="#1F2937", font=FONT)
    canvas.save(ROOT / f"contact-{sheet_index // 16 + 1}.png")

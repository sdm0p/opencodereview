from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3] / "docs" / "project-guide" / "working" / "pdf-review"
PAGES = sorted(ROOT.glob("page-*.png"), key=lambda p: int(p.stem.split("-")[-1]))
FONT_PATH = Path(r"C:\Windows\Fonts\arial.ttf")
FONT = ImageFont.truetype(str(FONT_PATH), 20) if FONT_PATH.exists() else ImageFont.load_default()

for start in range(0, len(PAGES), 16):
    canvas = Image.new("RGB", (1320, 1760), "#D9D9D9")
    draw = ImageDraw.Draw(canvas)
    for offset, page_path in enumerate(PAGES[start:start + 16]):
        page = Image.open(page_path).convert("RGB")
        page.thumbnail((300, 395))
        col, row = offset % 4, offset // 4
        x, y = 18 + col * 326, 35 + row * 430
        canvas.paste(page, (x, y))
        draw.text((x, 8 + row * 430), f"Page {start + offset + 1}", fill="#1F2937", font=FONT)
    canvas.save(ROOT / f"contact-final-{start // 16 + 1}.png")

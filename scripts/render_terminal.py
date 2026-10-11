"""Render a captured console log (e.g. logs/pytest.log) as a terminal-style PNG
so it can be placed on a slide as a screenshot of the real output.

    python scripts/render_terminal.py logs/pytest.log figures/pytest.png "$ pytest -v tests/"
"""
import sys

from PIL import Image, ImageDraw, ImageFont

src, dst = sys.argv[1], sys.argv[2]
cmd = sys.argv[3] if len(sys.argv) > 3 else ""
max_lines = int(sys.argv[4]) if len(sys.argv) > 4 else 30
lines = [ln.rstrip("\n") for ln in open(src, encoding="utf-8", errors="replace")]
lines = [ln for ln in lines if ln.strip() and "Powered by Stella" not in ln and "A.L.E:" not in ln]
lines = ([cmd] if cmd else []) + lines[-max_lines:]

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
font = ImageFont.truetype(FONT, 22)
cw, lh = font.getbbox("M")[2], 32
width = max(1100, min(2200, 60 + cw * max(len(ln) for ln in lines)))
height = 70 + lh * len(lines) + 30
img = Image.new("RGB", (width, height), "#141A33")
d = ImageDraw.Draw(img)
d.rectangle([0, 0, width, 44], fill="#222B52")
for i, c in enumerate(["#C8462B", "#F2A01D", "#2E7D4F"]):
    d.ellipse([20 + i * 30, 14, 36 + i * 30, 30], fill=c)
y = 64
for ln in lines:
    col = "#F2A01D" if ln.startswith("$") else ("#7FD19B" if "PASSED" in ln or "passed" in ln else "#E6E8F0")
    d.text((30, y), ln, font=font, fill=col)
    y += lh
img.save(dst)
print("saved", dst)

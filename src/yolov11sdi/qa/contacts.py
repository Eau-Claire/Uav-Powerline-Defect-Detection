"""Contact sheets: image + boxes + title for visual review."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..io_utils import atomic_path


def make_contact(image_path: Path, boxes: Sequence[Sequence[float]], labels: Sequence[str],
                 out_path: Path, title: str = "", max_side: int = 1200) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    im = Image.open(image_path).convert("RGB")
    scale = min(1.0, max_side / max(im.size))
    if scale < 1:
        im = im.resize((int(im.width * scale), int(im.height * scale)))
    W0, H0 = im.size
    head = 55
    canvas = Image.new("RGB", (W0, H0 + head), "white")
    canvas.paste(im, (0, head))
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    d.text((8, 8), title[:160], fill="black", font=font)
    for (x1, y1, x2, y2), label in zip(boxes, labels):
        rect = [x1 * W0, y1 * H0 + head, x2 * W0, y2 * H0 + head]
        d.rectangle(rect, outline="red", width=4)
        d.text((rect[0] + 2, max(head, rect[1] - 18)), str(label), fill="red", font=font)
    with atomic_path(out_path) as tmp:
        canvas.save(tmp, format="JPEG", quality=90)
    return out_path

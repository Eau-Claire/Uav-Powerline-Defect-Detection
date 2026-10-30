from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
from core import load_jsonl, read_yolo_boxes, draw_review

root = Path('artifacts/roboflow_relabel/full_train')
rows = [r for r in load_jsonl(root / 'proposals.jsonl') if any(a['status'] == 'add' for a in r.get('additions', []))]
selected = [rows[int(i * (len(rows)-1) / 7)] for i in range(8)]
canvas = Image.new('RGB', (1200, 1000), 'white')
for index, record in enumerate(selected):
    path = Path(record['image_path'])
    with Image.open(path) as im:
        boxes = read_yolo_boxes(Path(record['label_path']), im.size)
    preview = draw_review(path, boxes, additions=record['additions'])
    tile = ImageOps.contain(preview, (590, 210))
    x, y = (index % 2) * 600, (index // 2) * 250
    canvas.paste(tile, (x, y + 25))
    ImageDraw.Draw(canvas).text((x+5, y+5), f'Sample {index+1}: red=broken, green=added insulator', fill='black')
canvas.save(root / 'preview_additions.jpg')
print(root / 'preview_additions.jpg')

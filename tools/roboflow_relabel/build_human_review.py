"""Build a dependency-light browser reviewer for cross-label proposals."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts/roboflow_relabel/curated_combined_serverless"
DATASET = ROOT / "data/working/curated_combined_cross_labeled"
OUTPUT = ARTIFACT / "human_review"
CLASSES = [
    "insulator", "insulator_broken", "insulator_flashover",
    "bird_nest", "foreign_object", "broken_strand",
]


def stem_from_foreign_path(value: str) -> str:
    return Path(value.replace("\\", "/")).stem


def context_boxes(label_path: Path, width: int, height: int) -> list[dict]:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, cx, cy, bw, bh = map(float, line.split()[:5])
        boxes.append({
            "class_id": int(class_id),
            "class_name": CLASSES[int(class_id)],
            "x1": (cx - bw / 2) * width,
            "y1": (cy - bh / 2) * height,
            "x2": (cx + bw / 2) * width,
            "y2": (cy + bh / 2) * height,
        })
    return boxes


def main() -> None:
    changes = json.loads((ARTIFACT / "changes.json").read_text(encoding="utf-8"))
    conflicts = json.loads((ARTIFACT / "conflicts.json").read_text(encoding="utf-8"))
    grouped: dict[str, list[dict]] = {}
    for row in changes:
        for box in row.get("additions", []):
            grouped.setdefault(stem_from_foreign_path(row["image"]), []).append({
                "kind": "addition", "source": "accepted_by_filter", "box": box,
            })
    for row in conflicts:
        grouped.setdefault(stem_from_foreign_path(row["image"]), []).append({
            "kind": "conflict", "source": row.get("source", "unknown"), "box": row["box"],
        })

    image_index = {path.stem: path for path in (DATASET / "images/train").iterdir() if path.is_file()}
    items = []
    for stem in sorted(grouped):
        image = image_index.get(stem)
        if image is None:
            raise RuntimeError(f"Missing image for review item: {stem}")
        with Image.open(image) as opened:
            width, height = opened.size
        context = context_boxes(DATASET / "labels/train" / f"{stem}.txt", width, height)
        image_url = "../../../../data/working/curated_combined_cross_labeled/images/train/" + image.name
        for index, proposal in enumerate(grouped[stem]):
            box = proposal["box"]
            items.append({
                "id": f"{stem}:{proposal['kind']}:{index}",
                "image": image_url,
                "image_name": image.name,
                "width": width,
                "height": height,
                "kind": proposal["kind"],
                "source": proposal["source"],
                "class_id": box["class_id"],
                "class_name": box.get("class_name", CLASSES[box["class_id"]]),
                "confidence": box.get("confidence"),
                "box": {key: box[key] for key in ("x1", "y1", "x2", "y2")},
                "context_boxes": context,
            })

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "review_items.json").write_text(json.dumps(items), encoding="utf-8")
    (OUTPUT / "index.html").write_text(HTML, encoding="utf-8")
    (OUTPUT / "README.txt").write_text(
        "Run from the repository root:\n\n"
        "  python -m http.server 8765 --bind 127.0.0.1\n\n"
        "Then open:\n\n"
        "  http://127.0.0.1:8765/artifacts/roboflow_relabel/"
        "curated_combined_serverless/human_review/\n\n"
        "Review decisions stay in this browser's local storage until you click "
        "'Xuat decisions.json'. Keep the downloaded human_review_decisions.json.\n",
        encoding="utf-8",
    )
    print(f"Built {len(items)} box-review items in {OUTPUT}")


HTML = r'''<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Cross-label human review</title><style>
body{margin:0;background:#111827;color:#e5e7eb;font:15px system-ui}header,main{max-width:1200px;margin:auto;padding:14px}
header{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.grow{flex:1}button,select,input{padding:9px;border:1px solid #475569;border-radius:7px;background:#1e293b;color:#fff}
button{cursor:pointer}.accept{background:#166534}.reject{background:#991b1b}.unsure{background:#92400e}.active{outline:3px solid #f8fafc}
.panel{display:grid;grid-template-columns:minmax(0,3fr) minmax(280px,1fr);gap:14px}.canvas-wrap{background:#020617;border-radius:10px;overflow:auto;text-align:center}canvas{max-width:100%;height:auto}
.info{background:#1e293b;padding:14px;border-radius:10px}.muted{color:#94a3b8}.progress{height:8px;background:#334155;border-radius:4px}.bar{height:100%;background:#22c55e;border-radius:4px}
textarea{width:100%;min-height:90px;box-sizing:border-box;background:#0f172a;color:#fff;border:1px solid #475569;border-radius:7px;padding:8px}
@media(max-width:800px){.panel{grid-template-columns:1fr}}
</style></head><body><header><b>Cross-label human review</b><span id="counter"></span><div class="grow"></div>
<select id="filter"><option value="all">Tất cả</option><option value="pending">Chưa review</option><option value="addition">Addition</option><option value="conflict">Conflict</option></select>
<button onclick="exportDecisions()">Xuất decisions.json</button></header><main><div class="progress"><div id="bar" class="bar"></div></div><p class="muted">Phím: ←/→ chuyển box · A accept · R reject · U unsure. Khung xám = nhãn hiện có/final; khung xanh lá = proposal addition; đỏ = conflict.</p>
<div class="panel"><div class="canvas-wrap"><canvas id="canvas"></canvas></div><div class="info">
<h3 id="name"></h3><div id="meta"></div><p><button class="accept" onclick="decide('accept')">A · Accept</button> <button class="reject" onclick="decide('reject')">R · Reject</button> <button class="unsure" onclick="decide('unsure')">U · Unsure</button></p>
<label>Ghi chú</label><textarea id="note"></textarea><p><button onclick="move(-1)">← Trước</button> <button onclick="move(1)">Sau →</button></p><div id="saved" class="muted"></div>
</div></div></main><script>
let all=[],view=[],pos=0;const key='cross-review-20261031';let decisions=JSON.parse(localStorage.getItem(key)||'{}');
const el=id=>document.getElementById(id);fetch('review_items.json').then(r=>r.json()).then(x=>{all=x;applyFilter()});
function applyFilter(){let f=el('filter').value;view=all.filter(x=>f==='all'||(f==='pending'&&!decisions[x.id])||x.kind===f);pos=Math.min(pos,Math.max(0,view.length-1));render()}
el('filter').onchange=applyFilter;el('note').onchange=()=>{if(view[pos]){let d=decisions[view[pos].id]||{decision:'unsure'};d.note=el('note').value;decisions[view[pos].id]=d;save()}};
function save(){localStorage.setItem(key,JSON.stringify(decisions));el('saved').textContent=`Đã lưu ${Object.keys(decisions).length}/${all.length} quyết định trong trình duyệt`;}
function decide(value){let x=view[pos];if(!x)return;decisions[x.id]={decision:value,note:el('note').value,reviewed_at:new Date().toISOString(),image:x.image_name,class_name:x.class_name,kind:x.kind,box:x.box};save();render();if(el('filter').value==='pending')applyFilter();else move(1)}
function move(delta){if(!view.length)return;pos=Math.max(0,Math.min(view.length-1,pos+delta));render()}
function render(){el('counter').textContent=view.length?`${pos+1}/${view.length}`:'0/0';el('bar').style.width=(all.length?100*Object.keys(decisions).length/all.length:0)+'%';if(!view.length)return;let x=view[pos],d=decisions[x.id];el('name').textContent=x.image_name;el('meta').innerHTML=`<p><b>${x.kind}</b> · ${x.source}</p><p>Lớp: <b>${x.class_name}</b><br>Confidence: ${x.confidence==null?'n/a':x.confidence.toFixed(4)}</p><p>ID: <span class=muted>${x.id}</span></p>`;el('note').value=d?.note||'';document.querySelectorAll('.accept,.reject,.unsure').forEach(b=>b.classList.remove('active'));if(d)document.querySelector('.'+d.decision)?.classList.add('active');draw(x);save()}
function rect(ctx,b,color,width=2){ctx.strokeStyle=color;ctx.lineWidth=width;ctx.strokeRect(b.x1,b.y1,b.x2-b.x1,b.y2-b.y1)}
function draw(x){let img=new Image();img.onload=()=>{let c=el('canvas'),ctx=c.getContext('2d');c.width=x.width;c.height=x.height;ctx.drawImage(img,0,0);x.context_boxes.forEach(b=>rect(ctx,b,'rgba(203,213,225,.65)',2));let color=x.kind==='conflict'?'#ef4444':'#22c55e';rect(ctx,x.box,color,Math.max(4,x.width/180));ctx.fillStyle=color;ctx.font=`bold ${Math.max(16,x.width/40)}px sans-serif`;ctx.fillText(`${x.class_name} ${x.confidence?.toFixed(2)||''}`,Math.max(2,x.box.x1),Math.max(22,x.box.y1-5))};img.src=x.image}
function exportDecisions(){let payload={package:'curated_combined_cross_labeled_review_20261031',exported_at:new Date().toISOString(),total_items:all.length,reviewed:Object.keys(decisions).length,decisions};let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));a.download='human_review_decisions.json';a.click();URL.revokeObjectURL(a.href)}
document.onkeydown=e=>{if(e.target.tagName==='TEXTAREA')return;if(e.key==='ArrowLeft')move(-1);if(e.key==='ArrowRight')move(1);if(e.key.toLowerCase()==='a')decide('accept');if(e.key.toLowerCase()==='r')decide('reject');if(e.key.toLowerCase()==='u')decide('unsure')};
</script></body></html>'''


if __name__ == "__main__":
    main()

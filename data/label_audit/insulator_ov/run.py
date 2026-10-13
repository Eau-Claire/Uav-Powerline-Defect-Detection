"""Open-vocabulary insulator proposals (Grounding DINO base + OWLv2 base) for every v6p5_v4 image without an
insulator label, plus 1,000 train images that have one (benchmark vs GT). -> pred_{split}.jsonl (xyxy normalised)."""
import json, random, sys, torch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, Owlv2Processor, Owlv2ForObjectDetection
R = Path('/teamspace/studios/this_studio/datasets/v6p5_v4')
gp = AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-base')
gm = AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-base').cuda().eval().half()
op = Owlv2Processor.from_pretrained('google/owlv2-base-patch16-ensemble')
om = Owlv2ForObjectDetection.from_pretrained('google/owlv2-base-patch16-ensemble').cuda().eval().half()
def run(ims):
    sizes = [im.size for im in ims]
    gi = gp(images=ims, text=[["insulator"]] * len(ims), return_tensors='pt', device='cuda').to('cuda')  # preprocess on GPU (6x faster)
    gi['pixel_values'] = gi['pixel_values'].half()
    with torch.no_grad(): go = gm(**gi)
    go.logits, go.pred_boxes = go.logits.float(), go.pred_boxes.float()
    gr = gp.post_process_grounded_object_detection(go, gi.input_ids, threshold=0.2, text_threshold=0.2, target_sizes=[s[::-1] for s in sizes])
    oi = op(text=[["a photo of an insulator"]] * len(ims), images=ims, return_tensors='pt', device='cuda').to('cuda')
    oi['pixel_values'] = oi['pixel_values'].half()
    with torch.no_grad(): oo = om(**oi)
    oo.logits, oo.pred_boxes = oo.logits.float(), oo.pred_boxes.float()
    orr = op.post_process_grounded_object_detection(oo, threshold=0.1, target_sizes=torch.tensor([[max(s), max(s)] for s in sizes]))
    out = []
    for (W, H), g, o in zip(sizes, gr, orr):
        n = lambda b: [round(b[0]/W, 5), round(b[1]/H, 5), round(b[2]/W, 5), round(b[3]/H, 5)]
        out.append({'wh': [W, H], 'gdino': [n(b) for b in g['boxes'].tolist()], 'gdino_s': [round(s, 4) for s in g['scores'].tolist()],
                    'owl': [n(b) for b in o['boxes'].tolist()], 'owl_s': [round(s, 4) for s in o['scores'].tolist()]})
    return out
if __name__ == "__main__":
    jobs = {}
    for split in ('train', 'val'):
        coco = json.load(open(R/'annotations'/f'instances_{split}.json'))  # one file instead of 20k label reads
        ins = {a['image_id'] for a in coco['annotations'] if a['category_id'] == 0}
        no = sorted(Path(i['file_name']).stem for i in coco['images'] if i['id'] not in ins)
        has = sorted(Path(i['file_name']).stem for i in coco['images'] if i['id'] in ins)
        random.seed(0)
        jobs[split] = [(s, False) for s in no] + ([(s, True) for s in random.sample(has, 1000)] if split == 'train' else [])
        print(split, 'no-insulator', len(no), 'benchmark', len(jobs[split]) - len(no), flush=True)
    for split, items in jobs.items():
        done = set()
        f = Path(f'pred_{split}.jsonl')
        if f.exists(): done = {json.loads(l)['image'] for l in f.open()}
        items = [x for x in items if x[0] + '.jpg' not in done]
        load = lambda s: Image.open(R/'images'/split/(s+'.jpg')).convert('RGB')
        with f.open('a') as fh, ThreadPoolExecutor(8) as ex:
            for i in range(0, len(items), 8):
                chunk = items[i:i+8]
                ims = list(ex.map(load, [s for s, _ in chunk]))
                for (s, bench), r in zip(chunk, run(ims)):
                    fh.write(json.dumps({'image': s + '.jpg', 'benchmark': bench, **r}) + '\n')
                fh.flush()
                if i % 800 == 0: print(split, i, len(items), flush=True)
        print(split, 'done', flush=True)

"""Visual review decisions (2026-10-15) for items.csv -> decisions.csv. One line per category; IDs from the sheets.
G (labelled FO): keep | person / bird_nest (relabel box as class 6 / 3) | drop (remove box, image kept) | syn_image (drop
whole v65 synthetic image). User answers (2026-10-15): hornet nests = FO, branches/vines lying on a line or structure = FO
(user 2026-10-15: vegetation on a pole / line / equipment is FO too;
only plants growing on the ground are not), workers -> person; the 53 'unsure' boxes were re-checked at 400 px. F (teacher FO, unlabelled): add_fo | neg (not FO) | ignore
(box overlaps an existing label / synthetic image) | unsure."""
import pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
ids = lambda s, p: [f"{p}{int(x):03d}" for x in s.split()]
G = {
 'person': "1 2 3 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 21 22 23 25 27 28 29 30 32 33 37 42 84 177 235 430",
 'bird_nest': "47 169 256 280 323 366 374",
 'drop:merged into one nest box': "352 471",
 'drop:group of people': "4",
 'drop:vegetation growing on the ground': "90 100 106 119 142",
 'drop:ambiguous': "45 126 386 452",
 'drop:empty/equipment': "49 50 56 64 66 69 76 92 171 266 322 445",
 'drop:drawing': "358",
 'keep:branch on line': "87 109 155 173 516 51 72 80 83 88 95 102 116 118 168 195 199 226 268 284 329 332 68 364 392 416 459 468 500 536 670 715 725 727",
 'keep:hornet nest': "152 194 196 215 338 354 432 447 495 526 527 627 679",
 'keep:vegetation on structure': "59 209 55 61 62 65 73 79 107 113 129 130 149 175 180 220 227 233 244 249 304 313 456",
 'keep:rechecked': "34 40 48 77 99 120 141 170 176 205 236 263 331 351 513",
}
F = {
 'add_fo': "8 15 16 22 31 56",
 'ignore:ambiguous': "3 37",
 'neg': "12 33 6 10 13 14 19 20 21 25 27 28 32 36 39 40 42 44 46 48 49 50 53 59 61 62 64 65",
}
# second pass (user, full-image view): box fixes and boxes the review adds (A = added; box = x1 y1 x2 y2 normalised)
BOX = {'G002': '0.3279 0.4859 0.4729 0.9939'}   # was upper body only -> COCO yolo26s@1280 q0 0.91, whole person
ADD = [('G002', 'person', '0.3350 0.4659 0.4016 0.6576'),   # second worker behind, COCO q1 0.61
       ('G006', 'person', '0.41372 0.55337 0.52362 0.90912'),  # COCO p1 0.506
       ('G018', 'person', '0.33174 0.17933 0.54567 0.51802'),  # COCO p1 0.371
       ('G019', 'person', '0.44267 0.36818 0.72858 0.67148'),  # COCO p0 0.858
       ('G025', 'person', '0.51656 0.38869 0.66291 0.63345'),  # COCO p1 0.484
       ('G027', 'person', '0.68123 0.40937 0.90906 0.86108'),  # COCO p0 0.825
       ('G037', 'person', '0.46466 0.29444 0.53380 0.50231'),  # COCO p0 0.619
       ('G034', 'person', '0.5700 0.1600 0.7234 0.5729'),      # man behind the kite (G034 box = the kite, kept FO); hand box
       ('G352', 'bird_nest', '0.11220 0.26157 0.80317 0.91782')]  # one nest split in two FO boxes (G352 + G471) -> union
items = pd.read_csv(HERE/'items.csv'); dec = {}
for groups, p in [(G, 'G'), (F, 'F')]:
    for k, s in groups.items():
        for i in ids(s, p): assert i not in dec, i; dec[i] = k
for k, b in BOX.items(): items.loc[items.id == k, 'box'] = b
items['decision'] = [('syn_image' if r.syn else dec.get(r.id, 'keep' if r.group == 'G' else 'ignore')) for r in items.itertuples()]
bad = [i for i in dec if items.loc[items.id == i, 'syn'].any()]
assert not bad, f"decided on synthetic ids {bad}"
img = items.set_index('id').img
add = pd.DataFrame([dict(id=f"A{i + 1:03d}", group='A', img=img[k], box=b, teacher=0.0, person=0.0, syn=False, decision=f"add_{c}",
                         note=f"from {k}") for i, (k, c, b) in enumerate(ADD)])
items = pd.concat([items, add], ignore_index=True)
items[['id', 'group', 'img', 'box', 'teacher', 'person', 'syn', 'decision']].to_csv(HERE/'decisions.csv', index=False)
print(items.groupby(['group', 'decision']).size().to_string())

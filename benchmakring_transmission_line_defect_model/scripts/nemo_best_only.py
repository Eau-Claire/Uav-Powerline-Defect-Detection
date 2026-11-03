import csv
from pathlib import Path

root=Path(__file__).resolve().parents[1]/'results/finding_nemo/final_comparison'
with (root/'finding_nemo_final_best.csv').open(encoding='utf-8-sig',newline='') as handle:
    rows=list(csv.DictReader(handle))
columns=['model','checkpoint','best_epoch_candidates','val_precision','val_recall','val_coco_AR100','val_mAP50','val_mAP50_95','test_precision','test_recall','test_f1','test_mAP50','test_mAP50_95','test_MEAN5_AP50']
columns += [k for k in rows[0] if k.startswith('test_') and (k.endswith('_AP50') or k.endswith('_AP50_95')) and k not in columns]
columns += ['latency_ms','fps','checkpoint_stash']
output=root/'finding_nemo_best_only.csv'
with output.open('w',encoding='utf-8-sig',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=columns,extrasaction='ignore')
    writer.writeheader(); writer.writerows(rows)
print(output)
print('models',len(rows))

"""Join best validation epochs with supplied best.pt test evaluation, without test-based selection."""
from pathlib import Path
import csv,json,hashlib

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'benchmakring_transmission_line_defect_model/results/finding_nemo'
REMOTE=ROOT/'artifacts/finding_nemo_stash'
OUT=BASE/'final_comparison'
MODELS=['yolo11n','yolo11m','yolo26n','yolo26m','rtdetr50','rtdetr101']

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return [{k.strip():float(v) for k,v in row.items()} for row in csv.DictReader(f)]

def main():
    OUT.mkdir(exist_ok=True)
    evaluations=json.loads((BASE/'test_evaluation/benchmark_test_results.json').read_text())
    state=json.loads((BASE/'training_state.json').read_text())
    output=[]; histories=[]; notes=[]
    for model in MODELS:
        local=read(BASE/model/'results.csv'); remote_path=REMOTE/f'{model}_results.csv'; rows=read(remote_path)
        equivalent=local==rows
        best=max(rows,key=lambda r:r['metrics/mAP50-95(B)'])
        maximum=best['metrics/mAP50-95(B)']
        ties=[int(r['epoch']) for r in rows if r['metrics/mAP50-95(B)']==maximum]
        # YOLO overwrites on equal fitness; RT-DETR saves only on strict improvement.
        if model.startswith('yolo'): best=[r for r in rows if int(r['epoch'])==ties[-1]][0]
        key='nemo_rtdetr100' if model=='rtdetr101' else 'nemo_'+model
        test=evaluations[key]; accuracy=test['accuracy']; hw=test['hardware']; is_yolo=model.startswith('yolo')
        remote_run=f'stash://bunpmc/projects/finding_nemo/{model}' if is_yolo else f'stash://bunpmc/projects/finding_nemo/rtdetr_training/finding_nemo_human_reviewed_6cls_20261101/{"r50vd" if model=="rtdetr50" else "r101vd"}'
        row={'model':model,'checkpoint':'best.pt','checkpoint_stash':remote_run+('/weights/best.pt' if is_yolo else '/checkpoints/best.pt'),
             'best_epoch_from_log':int(best['epoch']),'best_epoch_candidates':';'.join(map(str,ties)),
             'best_epoch_method':'max_val_map50_95_from_rounded_csv','epochs_logged':len(rows),'last_epoch_logged':int(rows[-1]['epoch']),
             'local_epochs_logged':len(local),'local_last_epoch_logged':int(local[-1]['epoch']),
             'local_matches_stash_numeric':equivalent,'selection_metric':'val_mAP50_95',
             'val_precision':best['metrics/precision(B)'] if is_yolo else '',
             'val_recall':best['metrics/recall(B)'] if is_yolo else '',
             'val_coco_AR100':best['metrics/recall(B)'] if not is_yolo else '',
             'val_mAP50':best['metrics/mAP50(B)'],'val_mAP50_95':maximum,
             'test_precision':accuracy['precision'],'test_recall':accuracy['recall'],'test_f1':accuracy['f1_score'],
             'test_mAP50':accuracy['mAP50'],'test_mAP50_95':accuracy['mAP50_95'],'test_MEAN5_AP50':accuracy['MEAN_5_defect_ap50'],
             'test_minus_val_mAP50_pp':round(100*(accuracy['mAP50']-best['metrics/mAP50(B)']),4),
             'test_minus_val_mAP50_95_pp':round(100*(accuracy['mAP50_95']-maximum),4),
             'latency_ms':hw['latency_ms'],'fps':hw['fps'],'params_m':hw['parameters_m'],'gflops':hw['gflops'],'vram_mb':hw['inference_peak_vram_mb'],
             'checkpoint_sha256_recorded':state['runs'].get(model,{}).get('best_sha256',''),
             'test_split':test['evaluation_split'],'test_dataset_yaml_recorded':test['evaluation_dataset_yaml'],
             'test_checkpoint_path_recorded':test['checkpoint_path'],'test_evaluated_at':test['evaluated_at'],
             'train_log_stash':remote_run+('/results.csv' if is_yolo else '/state/result.csv'),
             'train_log_sha256':hashlib.sha256(remote_path.read_bytes()).hexdigest()}
        for name,metrics in accuracy['per_class'].items():
            row[f'test_{name}_AP50']=metrics['ap50']; row[f'test_{name}_AP50_95']=metrics['ap50_95']
        row['notes']='' if is_yolo else 'Training precision is AP50_95 proxy, not P; training recall is COCO AR100. Validation uses COCOeval; test uses custom evaluator.'
        if len(ties)>1: row['notes']+=' Best epoch tied at CSV precision; candidate epochs retained, checkpoint epoch not independently verified.'
        output.append(row)
        histories.extend({'model':model,**r,'selected_best_row':int(r['epoch'])==int(best['epoch'])} for r in rows)
    for filename,records in [('finding_nemo_final_best.csv',output),('finding_nemo_all_epochs.csv',histories)]:
        with (OUT/filename).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in records for k in r)))
            writer.writeheader(); writer.writerows(records)
    report=['# Finding Nemo: best-validation vs test','',
        'One row per model in finding_nemo_final_best.csv. All metrics are fractions (0–1), gaps are percentage points.',
        'Best epoch is inferred from validation mAP50–95 in CSV, not chosen using test. CSV rounding can hide ties; candidates are preserved.',
        'Test metrics are the supplied evaluations of best.pt; checkpoint bytes were not downloaded or re-evaluated.',
        'RT-DETR training precision is a mislabeled COCO AP proxy; it is excluded from val_precision. Training recall is COCO AR@100.',
        'RT-DETR val COCOeval and test custom matching protocols differ; their gaps are descriptive, not a pure generalization estimate.',
        'nemo_rtdetr100 is normalized to rtdetr101 using checkpoint path and local test_metrics.csv.',
        'Training CSVs downloaded from stash://bunpmc/projects/finding_nemo/. Local vs remote numerical equality is recorded per model.','',
        'R101 local CSV stops at epoch 32 while stash reaches epoch 37; both have their best AP50–95 at epoch 7. Final table uses stash logs.',
        '| Model | Best epoch candidates | Val AP50 | Val AP50–95 | Test AP50 | Test AP50–95 | Test MEAN5 | Δ95 pp |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in output:
        report.append(f"| {r['model']} | {r['best_epoch_candidates']} | {r['val_mAP50']:.4f} | {r['val_mAP50_95']:.4f} | {r['test_mAP50']:.4f} | {r['test_mAP50_95']:.4f} | {r['test_MEAN5_AP50']:.4f} | {r['test_minus_val_mAP50_95_pp']:+.2f} |")
    (OUT/'README.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    assert len(output)==6 and len({r['model'] for r in output})==6
    for r in output:
        assert abs(sum(r[f'test_{c}_AP50'] for c in ['insulator_broken','insulator_flashover','bird_nest','foreign_object','broken_strand'])/5-r['test_MEAN5_AP50'])<0.00011
    print('\n'.join(report[-8:]))
    print('rows',len(output),'epochs',len(histories),'all_local_match',all(r['local_matches_stash_numeric'] for r in output))

if __name__=='__main__': main()

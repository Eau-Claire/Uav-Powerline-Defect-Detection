Backend storage vs model output (perf/mf04_db_checks.py), 22/11/2026.
Inputs: jobs of 1254_v2_concurrent, 1257_v2_categories, 1259_v2_large, 1330_def014 (64 images);
DB dump of mission db33eddc-... taken after the be_flows run.
Result: 63/64 images match (category per label, IoU 1.0); the mismatch is the media re-analysed in
be_flows RE-1 (duplicate active detection). insulator: 35 model boxes, 0 stored defects. No test image
has EXIF GPS: all stored CaptureLocation NULL. Positive GPS case: run *_gps (EXIF 10.845678, 106.801234
stored exactly, SRID 4326). DetectedAnomalies.Location NULL (bridge sends no GPS).

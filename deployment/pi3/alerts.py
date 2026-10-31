"""Local alert selection and durable SQLite outbox; no network in inference."""
import json
import math
import sqlite3
import time
from uuid import uuid4
from datetime import datetime, timezone
from pathlib import Path

CLASSES = {'insulator', 'insulator_broken', 'insulator_flashover', 'bird_nest', 'foreign_object', 'broken_strand'}

class AlertOutbox:
    def __init__(self, config_path, database):
        self.config = json.loads(Path(config_path).read_text())
        if not self.config.get('drone_id') or self.config['drone_id'].startswith('REPLACE_'):
            raise ValueError('Configure drone_id with a registered BE UAV code')
        for name, rule in self.config['rules'].items():
            if name not in CLASSES or not 0 <= rule['min_confidence'] <= 1:
                raise ValueError(f'Invalid alert rule: {name}')
            if not rule.get('category_code') or rule.get('cooldown_seconds', 30) < 0:
                raise ValueError(f'Invalid category/cooldown: {name}')
        self.db = sqlite3.connect(database)
        self.db.execute('CREATE TABLE IF NOT EXISTS alerts (id INTEGER PRIMARY KEY, class_name TEXT, created REAL, payload TEXT, sent INTEGER DEFAULT 0)')
        columns={row[1] for row in self.db.execute('PRAGMA table_info(alerts)')}
        if 'event_id' not in columns:
            self.db.execute('ALTER TABLE alerts ADD COLUMN event_id TEXT')
        for row in self.db.execute('SELECT id FROM alerts WHERE event_id IS NULL').fetchall():
            self.db.execute('UPDATE alerts SET event_id=? WHERE id=?',(str(uuid4()),row[0]))
        self.db.execute('CREATE UNIQUE INDEX IF NOT EXISTS alert_event_id ON alerts(event_id)')
        self.db.commit()

    def enqueue(self, detections):
        now = time.time()
        count = 0
        # One highest-confidence detection per class per cooldown, not object tracking.
        for det in sorted(detections, key=lambda d: d['confidence'], reverse=True):
            name = det['class_name']
            rule = self.config['rules'].get(name)
            if not rule or not rule.get('enabled', False):
                continue
            if not math.isfinite(det['confidence']) or det['confidence'] < rule['min_confidence']:
                continue
            previous = self.db.execute('SELECT MAX(created) FROM alerts WHERE class_name=?', (name,)).fetchone()[0]
            if previous is not None and now - previous < rule.get('cooldown_seconds', 30):
                continue
            payload = {'DroneId': self.config['drone_id'], 'ClassName': rule['category_code'],
                       'Confidence': det['confidence'], 'Timestamp': datetime.now(timezone.utc).isoformat(),
                       'Latitude': 0, 'Longitude': 0, 'TrackId': 0,
                       'BoundingBox': [round(x) for x in det['bbox_xyxy']]}
            self.db.execute('INSERT INTO alerts(class_name,created,payload,event_id) VALUES(?,?,?,?)',
                            (name, now, json.dumps(payload, allow_nan=False), str(uuid4())))
            self.db.commit()
            count += 1
        return count

    def close(self):
        self.db.close()

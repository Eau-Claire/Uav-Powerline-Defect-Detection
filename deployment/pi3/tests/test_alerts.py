import json
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alerts import AlertOutbox
from send_alerts import deliver

class AlertTests(unittest.TestCase):
    def test_threshold_cooldown_and_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);config=root/'config.json';db=root/'outbox.db'
            config.write_text(json.dumps({'drone_id':'UAV01','rules':{'broken_strand':{'enabled':True,'min_confidence':.8,'cooldown_seconds':30,'category_code':'STRAND'}}}))
            detection={'class_name':'broken_strand','confidence':.79,'bbox_xyxy':[1,2,3,4]}
            outbox=AlertOutbox(config,db)
            self.assertEqual(outbox.enqueue([detection]),0)
            detection['confidence']=.9
            self.assertEqual(outbox.enqueue([detection,detection]),1)
            event_id=outbox.db.execute('SELECT event_id FROM alerts').fetchone()[0]
            outbox.close()
            outbox=AlertOutbox(config,db)
            self.assertEqual(outbox.enqueue([detection]),0)
            self.assertEqual(outbox.db.execute('SELECT event_id FROM alerts').fetchone()[0],event_id)
            payload=json.loads(outbox.db.execute('SELECT payload FROM alerts').fetchone()[0])
            self.assertEqual(payload['ClassName'],'STRAND')
            self.assertEqual(payload['DroneId'],'UAV01')
            self.assertEqual(outbox.db.execute('SELECT sent FROM alerts').fetchone()[0],0)
            outbox.close()

class DeliveryTests(unittest.TestCase):
    def test_device_headers_remove_claimed_drone(self):
        response=MagicMock()
        response.__enter__.return_value.read.return_value=b'{"success":true}'
        with patch('urllib.request.build_opener') as opener:
            opener.return_value.open.return_value=response
            deliver('https://example.test/api','device-secret','{"DroneId":"spoofed","ClassName":"STRAND"}',
                    'stable-id','device-key')
            request=opener.return_value.open.call_args.args[0]
            headers={k.lower():v for k,v in request.header_items()}
            self.assertEqual(headers['x-device-key'],'device-secret')
            self.assertEqual(headers['idempotency-key'],'stable-id')
            self.assertNotIn('authorization',headers)
            self.assertNotIn('DroneId',json.loads(request.data))

    def test_http_success_with_business_failure_is_rejected(self):
        response=MagicMock()
        response.__enter__.return_value.read.return_value=b'{"success":false}'
        with patch('urllib.request.build_opener') as opener:
            opener.return_value.open.return_value=response
            with self.assertRaises(RuntimeError):
                deliver('https://example.test/api','test-token','{}')

    def test_business_success_is_accepted(self):
        response=MagicMock()
        response.__enter__.return_value.read.return_value=b'{"success":true}'
        with patch('urllib.request.build_opener') as opener:
            opener.return_value.open.return_value=response
            deliver('https://example.test/api','test-token','{}')

if __name__=='__main__':unittest.main()

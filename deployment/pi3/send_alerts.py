"""Drain the local outbox to the existing authenticated BE Vision Bridge."""
import argparse
import json
import os
import sqlite3
import time
import urllib.request
from urllib.parse import urlparse


def deliver(url, token, payload, event_id=None, auth_mode='jwt'):
    headers={'Content-Type':'application/json'}
    if auth_mode == 'device-key':
        if not event_id: raise ValueError('Missing stable event ID')
        headers['X-Device-Key']=token
        headers['Idempotency-Key']=event_id
        data=json.loads(payload)
        data.pop('DroneId',None)  # BE derives UAV from authenticated device
        payload=json.dumps(data)
    elif auth_mode == 'jwt': headers['Authorization']=f'Bearer {token}'
    else: raise ValueError('Unknown auth mode')
    request = urllib.request.Request(url, data=payload.encode(), method='POST',
        headers=headers)
    # Do not forward credentials through redirects.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None
    with urllib.request.build_opener(NoRedirect).open(request, timeout=15) as response:
        result = json.load(response)
    # BE may return HTTP 201 with Success=false (unknown UAV/no executing mission).
    if result.get('success', result.get('Success')) is not True:
        raise RuntimeError('BE did not accept detection; check UAV and executing mission')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outbox', default='alerts.sqlite3')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    url = os.environ['POWERLINE_BE_ALERT_URL']
    if urlparse(url).scheme != 'https':
        parser.error('POWERLINE_BE_ALERT_URL must use HTTPS')
    auth_mode=os.environ.get('POWERLINE_BE_AUTH_MODE','device-key')
    token_file = os.environ['POWERLINE_DEVICE_KEY_FILE' if auth_mode=='device-key' else 'POWERLINE_BE_TOKEN_FILE']
    db = sqlite3.connect(args.outbox)
    try:
        while True:
            try:
                rows = db.execute('SELECT id,payload,event_id FROM alerts WHERE sent=0 ORDER BY id LIMIT 20').fetchall()
                for identifier, payload, event_id in rows:
                    from pathlib import Path
                    token = Path(token_file).read_text().strip()
                    if not token:
                        raise ValueError('Empty token file')
                    deliver(url, token, payload, event_id, auth_mode)
                    db.execute('UPDATE alerts SET sent=1 WHERE id=?', (identifier,))
                    db.commit()
                    print(json.dumps({'delivered': identifier}), flush=True)
                if args.once:
                    return
                time.sleep(1 if rows else 2)
            except Exception as exc:
                print(json.dumps({'delivery_error': type(exc).__name__, 'pending_kept': True}), flush=True)
                if args.once:
                    raise SystemExit(1)
                time.sleep(10)
    finally:
        db.close()

if __name__ == '__main__':
    main()

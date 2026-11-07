"""Offline, append-only sample journal with idempotent IDs and an upload outbox.

No carbon-stock/flux estimation or cloud endpoint is inferred from sample data.
"""
from contextlib import closing
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3

REQUIRED={'sample_id','plot_id','observed_at','kind','value','unit',
          'calibration_version','localization_source','quality'}


class SampleJournal:
    def __init__(self, path):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db:
            db.execute('CREATE TABLE IF NOT EXISTS samples (id TEXT PRIMARY KEY, body TEXT NOT NULL, sha256 TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS uploaded (id TEXT PRIMARY KEY REFERENCES samples(id), receipt TEXT NOT NULL)')
            db.commit()

    def append(self, record):
        if REQUIRED-record.keys():
            raise ValueError(f'Missing fields: {sorted(REQUIRED-record.keys())}')
        for key in REQUIRED-{'value','quality'}:
            if not isinstance(record[key],str) or not record[key].strip():
                raise ValueError(f'{key} must be a nonempty string')
        if record['quality'] not in ('raw','checked','rejected'):
            raise ValueError('Invalid quality')
        date=datetime.fromisoformat(record['observed_at'].replace('Z','+00:00'))
        if date.utcoffset() is None:
            raise ValueError('Measurement timestamps must include timezone')
        body=json.dumps(record,ensure_ascii=False,sort_keys=True,allow_nan=False,separators=(',',':'))
        digest=hashlib.sha256(body.encode('utf-8')).hexdigest()
        with closing(sqlite3.connect(self.path)) as db:
            db.execute('BEGIN IMMEDIATE')
            old=db.execute('SELECT sha256 FROM samples WHERE id=?',(record['sample_id'],)).fetchone()
            if old:
                if old[0]!=digest:
                    raise ValueError('Sample ID conflict; original evidence is immutable')
                return False
            db.execute('INSERT INTO samples VALUES (?,?,?)',(record['sample_id'],body,digest))
            db.commit()
        return True

    def pending(self, limit=100):
        if not isinstance(limit,int) or limit<=0:
            raise ValueError('Positive limit required')
        with closing(sqlite3.connect(self.path)) as db:
            rows=db.execute('SELECT s.body,s.sha256 FROM samples s LEFT JOIN uploaded u ON s.id=u.id WHERE u.id IS NULL ORDER BY s.rowid LIMIT ?', (limit,)).fetchall()
        return [{'record':json.loads(body),'sha256':digest} for body,digest in rows]

    def acknowledge(self, sample_id, sha256, receipt):
        if not isinstance(receipt,str) or not receipt.strip():
            raise ValueError('Server receipt required')
        with closing(sqlite3.connect(self.path)) as db:
            db.execute('BEGIN IMMEDIATE')
            record=db.execute('SELECT sha256 FROM samples WHERE id=?',(sample_id,)).fetchone()
            if not record or record[0]!=sha256:
                raise ValueError('Upload acknowledgment must match stored evidence')
            db.execute('INSERT OR IGNORE INTO uploaded VALUES (?,?)',(sample_id,receipt))
            db.commit()

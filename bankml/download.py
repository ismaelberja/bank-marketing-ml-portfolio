"""Download the ORIGINAL full UCI dataset; does not redistribute the raw data."""
from __future__ import annotations

import csv
import hashlib
import json
import io
from pathlib import Path
import urllib.request
import zipfile

URL = 'https://archive.ics.uci.edu/static/public/222/bank+marketing.zip'
FILE = 'bank-additional-full.csv'


def download_data(target_dir: str | Path = 'data/raw') -> Path:
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / FILE
    provenance = target_dir / 'source_provenance.json'
    if dest.exists() and provenance.is_file():
        meta = json.loads(provenance.read_text(encoding='utf-8'))
        if (meta.get('source_url') == URL and meta.get('sha256') == hashlib.sha256(dest.read_bytes()).hexdigest()
                and meta.get('records') == 41_188):
            print(f'Using previously verified download: {dest}')
            return dest
    req = urllib.request.Request(URL, headers={'User-Agent': 'bank-ml-portfolio/1.0'})
    with urllib.request.urlopen(req, timeout=90) as res:
        data = res.read()
    with zipfile.ZipFile(io.BytesIO(data)) as outer:
        candidates = [(name, outer.read(name)) for name in outer.namelist()
                      if name.lower().endswith('.zip')]
        csv_contents = None
        for _, nested_bytes in candidates:
            with zipfile.ZipFile(io.BytesIO(nested_bytes)) as nested:
                options = [name for name in nested.namelist() if name.endswith('/'+FILE) or name == FILE]
                if options:
                    csv_contents = nested.read(options[0])
                    break
        if csv_contents is None:
            options = [name for name in outer.namelist() if name.endswith('/'+FILE) or name == FILE]
            if options:
                csv_contents = outer.read(options[0])
        if csv_contents is None:
            raise FileNotFoundError('Expected full additional CSV not found in UCI archive')
    if not csv_contents.startswith(b'"age";'):
        raise ValueError('Unexpected dataset header')
    rows = sum(1 for _ in csv.reader(io.StringIO(csv_contents.decode('utf-8-sig')), delimiter=';')) - 1
    if rows != 41_188:
        raise ValueError(f'Expected exactly 41188 rows from UCI, found {rows}')
    dest.write_bytes(csv_contents)
    digest = hashlib.sha256(csv_contents).hexdigest()
    provenance.write_text(json.dumps({'source_url': URL, 'records': rows, 'sha256': digest},
                                     indent=2) + '\n', encoding='utf-8')
    print(f'Downloaded {rows:,} original UCI rows to {dest}; SHA256: {digest}')
    return dest


if __name__ == '__main__':
    download_data()

"""Git-shared JSONL shards for sequential train production.

Shard files are the collaboration boundary; SQLite remains private to each machine.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_sha(value) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode())


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def manifest_path(shard: Path) -> Path:
    return shard.with_suffix(shard.suffix + '.manifest.json')


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def export_shard(folder: Path, output: Path, producer: str, start: int, end: int):
    """Export accepted rows for a 1-based inclusive contiguous range."""
    from workflow import read_json, Store, verify
    manifest, _, plan = verify(folder)
    if start < 1 or end < start or end > len(plan):
        raise ValueError('Geçersiz 1-based train aralığı')
    expected = [s['slot_id'] for s in plan[start - 1:end]]
    store = Store(folder / 'state.sqlite3')
    try:
        rows = []
        for slot_id in expected:
            result = store.execute('SELECT result,status FROM jobs WHERE id=?', (slot_id,))
            if not result or result[0][1] != 'accepted':
                raise ValueError(f'{slot_id} henüz accepted değil; shard çıkarılamaz')
            rows.append(json.loads(result[0][0]))
    finally:
        store.close()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + '.tmp')
    tmp.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    tmp.replace(output)
    contract = {'contract_sha256': manifest['contract_sha256'], 'config': manifest['config'].get('version')}
    shard_manifest = {'manifest_version': 1, 'run_id': folder.name, 'producer': producer,
                      'from': start, 'to': end, 'family_count': len(rows), 'shard_file': output.name,
                      'slot_ids': expected, 'shard_sha256': sha256_bytes(output.read_bytes()),
                      'run_contract': contract, 'run_contract_sha256': json_sha(contract)}
    write_json(manifest_path(output), shard_manifest)
    return shard_manifest


def validate_shards(folder: Path, shard_dir: Path):
    from workflow import read_json, verify
    manifest, _, plan = verify(folder)
    shard_dir = Path(shard_dir)
    files = sorted(shard_dir.glob('*.jsonl'))
    if not files:
        return {'run_id': folder.name, 'shards': 0, 'covered': 0, 'next': 1}
    contracts = set(); ranges = []
    for path in files:
        sidecar = manifest_path(path)
        if not sidecar.exists():
            raise ValueError(f'Shard manifest eksik: {path.name}')
        meta = read_json(sidecar); rows = read_jsonl(path)
        if meta.get('run_id') != folder.name or meta.get('shard_file') != path.name:
            raise ValueError(f'{path.name}: run_id veya dosya adı uyuşmuyor')
        start, end = meta.get('from'), meta.get('to')
        expected = [s['slot_id'] for s in plan[start - 1:end]]
        actual = [r.get('slot_id') for r in rows]
        if meta.get('slot_ids') != expected or actual != expected or len(rows) != end - start + 1:
            raise ValueError(f'{path.name}: sıra veya slot kapsamı hatalı')
        if meta.get('shard_sha256') != sha256_bytes(path.read_bytes()):
            raise ValueError(f'{path.name}: checksum hatalı')
        if meta.get('run_contract_sha256') != json_sha(meta.get('run_contract')):
            raise ValueError(f'{path.name}: contract checksum hatalı')
        if meta['run_contract'].get('contract_sha256') != manifest['contract_sha256']:
            raise ValueError(f'{path.name}: farklı üretim sözleşmesi')
        contracts.add(meta['run_contract_sha256']); ranges.append((start, end, path.name))
    ranges.sort()
    expected_start = 1
    for start, end, name in ranges:
        if start != expected_start:
            raise ValueError(f'Shard aralığı boşluklu/çakışık: {name}, beklenen {expected_start}')
        expected_start = end + 1
    return {'run_id': folder.name, 'shards': len(ranges), 'covered': expected_start - 1,
            'next': expected_start, 'contract': next(iter(contracts))}


def sync_shards(folder: Path, shard_dir: Path):
    from workflow import Store
    status = validate_shards(folder, shard_dir)
    store = Store(folder / 'state.sqlite3'); imported = 0
    try:
        for path in sorted(Path(shard_dir).glob('*.jsonl')):
            for row in read_jsonl(path):
                slot_id = row['slot_id']
                current = store.execute('SELECT status FROM jobs WHERE id=?', (slot_id,))
                if not current:
                    raise ValueError(f'Plan dışı slot: {slot_id}')
                if current[0][0] != 'accepted':
                    store.execute("UPDATE jobs SET status='accepted',result=? WHERE id=?", (json.dumps(row, ensure_ascii=False), slot_id))
                    imported += 1
    finally:
        store.close()
    return {**status, 'imported': imported}

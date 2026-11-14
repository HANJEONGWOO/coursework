"""Compare a student's own native runs under exactly the same conditions."""
import hashlib
import json


def measurement_contract(root):
    digest = hashlib.sha256()
    files = [root / 'tp_api.py', root / 'tp_bridge.py', root / 'tp_worker.py', root / 'score_rubric.py']
    files += sorted((root / '_engine').rglob('*.py'))
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def comparison_key(result):
    fields = ('case', 'case_sha256', 'measurement_contract_sha256', 'native_binary_sha256')
    if result.get('status') != 'complete' or any(not result.get(key) for key in fields):
        return None
    return tuple(result[key] for key in fields)


def compact(result, run):
    score = result['learning_score']
    return {'run': run, 'policy_sha256': result['policy_sha256'],
            'score': score['automated_score_100'],
            'drain_s': score['actual_drain_ns'] / 1e9,
            'classes': score['per_class']}


def attach_progress(root, run_dir, result):
    key = comparison_key(result)
    if key is None:
        return
    previous = []
    for folder in (root / 'results').glob('run-*'):
        if folder == run_dir:
            continue
        try:
            saved = json.loads((folder / result['case'] / 'summary.json').read_text(encoding='utf-8'))
            meta = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
            if comparison_key(saved) == key:
                previous.append((meta['created_utc'], compact(saved, folder.name)))
        except (OSError, ValueError, KeyError, TypeError):
            # Partial/old runs remain saved, but are not valid comparisons.
            continue
    previous.sort(key=lambda item: item[0])
    rows = [item[1] for item in previous]
    current = compact(result, run_dir.name)
    first = rows[0] if rows else current
    best = max([*rows, current], key=lambda row: row['score'])
    shown = rows[-5:]
    for special in (first, best):
        if special['run'] != run_dir.name and special not in shown:
            shown.insert(0, special)
    result['progress'] = {
        'same_input_and_engine_only': True, 'previous_count': len(rows),
        'delta_previous': current['score'] - rows[-1]['score'] if rows else None,
        'delta_first': current['score'] - first['score'],
        'best_score': best['score'], 'first_run': first['run'],
        'best_run': best['run'], 'rows': [*shown, current],
    }

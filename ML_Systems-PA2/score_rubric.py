"""TP assignment rubric, shared by the student runner and instructor grader.

The caller must first run the official lab_score.score_record(record), which
validates complete request evidence and computes the optimistic resource bound
and E. Its returned dictionary is passed as ``validated_baseline``. This module
recomputes the request timing and checks that its per-class SLO pass counts
agree with the baseline. It does not invoke or approximate ASTRA.

An expected case manifest is mandatory: missing workloads cannot silently
improve a grade. Zero-count classes may be absent from a sampled input. The
caller supplies actual seed IDs, including one ID for public ``--all``.
"""
from collections import defaultdict
import math


class ScoreError(ValueError):
    """Malformed scoring evidence or incomplete evaluation set."""


def _identity(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ScoreError(f'{field} must be a nonempty string')
    return value


def _integer(value, field, minimum=0):
    if type(value) is not int or value < minimum:
        raise ScoreError(f'{field} must be an integer >= {minimum}')
    return value


def _finite(value, field, lower, upper):
    if type(value) not in (float, int) or not math.isfinite(value) or not lower <= value <= upper:
        raise ScoreError(f'{field} must be finite and within [{lower}, {upper}]')
    return float(value)


def _p95(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * .95
    left = int(pos)
    right = min(left + 1, len(ordered) - 1)
    return ordered[left] + (ordered[right] - ordered[left]) * (pos - left)


def score_case(record, validated_baseline, *, model, traffic, seed):
    """Score all requests in a complete, *already officially validated* record.

    Returns per-class additive score sums and counts, not averages of seed
    averages, so aggregate() can weight seeds by their number of requests.
    A failed request gets exactly 0. A successful request gets
    70 + 20*(1-max(TTFT/target, ITL95/target)) + 10*E.
    """
    model, traffic, seed = (_identity(model, 'model'), _identity(traffic, 'traffic'),
                            _identity(seed, 'seed'))
    if not isinstance(record, dict) or record.get('metrics', {}).get('complete') is not True:
        raise ScoreError('A completed validated record is required')
    if not isinstance(validated_baseline, dict) or validated_baseline.get('schema') != 'pa2-service-drain-score-v1':
        raise ScoreError('A validated baseline score_record output is required')
    efficiency = _finite(validated_baseline.get('drain_efficiency'), 'drain_efficiency', 0, 1)
    base_classes = validated_baseline.get('per_class')
    if not isinstance(base_classes, dict) or not base_classes:
        raise ScoreError('Baseline must contain per-class pass counts')
    requests = record.get('requests')
    if not isinstance(requests, list) or not requests:
        raise ScoreError('At least one completed request is required')
    classes = defaultdict(lambda: {'count': 0, 'passed': 0, 'score_sum': 0.0,
                                   'slack_sum': 0.0})
    seen = set()
    for request in requests:
        if not isinstance(request, dict):
            raise ScoreError('Each request must be a dictionary')
        try:
            identity = _identity(request['id'], 'request ID')
            name = _identity(request['class'], 'request class')
            arrival = _integer(request['arrival_ns'], 'arrival_ns')
            goal_ttft = _integer(request['ttft_goal_ns'], 'ttft_goal_ns', 1)
            goal_itl = _integer(request['itl_goal_ns'], 'itl_goal_ns', 1)
            outputs = _integer(request['output_tokens'], 'output_tokens', 1)
            stamps = request['token_times_ns']
        except KeyError as exc:
            raise ScoreError(f'Missing request field {exc.args[0]}') from exc
        if identity in seen:
            raise ScoreError(f'Duplicate request ID {identity}')
        seen.add(identity)
        if not isinstance(stamps, list) or len(stamps) != outputs:
            raise ScoreError(f'{identity}: incomplete token timestamps')
        previous = arrival
        for timestamp in stamps:
            _integer(timestamp, 'token timestamp')
            if timestamp <= previous:
                raise ScoreError(f'{identity}: token timestamps must strictly increase')
            previous = timestamp
        ttft_ratio = (stamps[0] - arrival) / goal_ttft
        itl_ratio = _p95([b - a for a, b in zip(stamps, stamps[1:])]) / goal_itl
        passed = ttft_ratio <= 1 and itl_ratio <= 1
        slack = max(0.0, 1.0 - max(ttft_ratio, itl_ratio)) if passed else 0.0
        score = 70.0 + 20.0 * slack + 10.0 * efficiency if passed else 0.0
        row = classes[name]
        row['count'] += 1
        row['passed'] += int(passed)
        row['score_sum'] += score
        row['slack_sum'] += slack
    if set(classes) != set(base_classes):
        raise ScoreError('Per-class request categories disagree with validated baseline')
    for name, row in classes.items():
        expected = base_classes[name]
        if not isinstance(expected, dict) or expected.get('count') != row['count'] or expected.get('passed') != row['passed']:
            raise ScoreError(f'{name}: request pass counts disagree with validated baseline')
    return {'schema': 'pa2-tp-case-v1', 'model': model, 'traffic': traffic,
            'seed': seed, 'efficiency': efficiency, 'classes': dict(sorted(classes.items()))}


def _unique_labels(values, field):
    if not isinstance(values, (list, tuple)) or not values:
        raise ScoreError(f'{field} must be a nonempty sequence')
    labels = [_identity(v, field) for v in values]
    if len(set(labels)) != len(labels):
        raise ScoreError(f'{field} contains duplicates')
    return labels


def aggregate(cases, *, expected_seeds, expected_models=('small', 'medium', 'large'),
              expected_traffic=('steady', 'shift'), expected_classes=('chat', 'rag', 'long')):
    """Aggregate complete cases using specified model/traffic/seed manifest.

    Within a model, traffic, and class, scores of *requests* are pooled across
    the listed seeds (equivalently, seed means weighted by that seed's request
    count). Only classes appearing in that model/traffic stratum are averaged;
    those classes have equal weight, as do traffic conditions.
    Final raw score is 75% of the mean model score plus 25% of the worst model.
    The public suite uses the identical path with exactly one expected seed.
    """
    models = _unique_labels(expected_models, 'expected_models')
    traffic = _unique_labels(expected_traffic, 'expected_traffic')
    seeds = _unique_labels(expected_seeds, 'expected_seeds')
    names = _unique_labels(expected_classes, 'expected_classes')
    if not isinstance(cases, (list, tuple)):
        raise ScoreError('cases must be a list or tuple')
    expected = {(m, t, s) for m in models for t in traffic for s in seeds}
    found = {}
    for case in cases:
        if not isinstance(case, dict) or case.get('schema') != 'pa2-tp-case-v1':
            raise ScoreError('Invalid case score schema')
        identity = tuple(_identity(case.get(field), field) for field in ('model', 'traffic', 'seed'))
        if identity in found:
            raise ScoreError(f'Duplicate case {identity}')
        if identity not in expected:
            raise ScoreError(f'Unexpected case {identity}')
        efficiency = _finite(case.get('efficiency'), f'{identity}.efficiency', 0, 1)
        rows = case.get('classes')
        if not isinstance(rows, dict) or not rows or not set(rows) <= set(names):
            raise ScoreError(f'{identity}: empty or unexpected request class')
        for name, row in rows.items():
            if not isinstance(row, dict):
                raise ScoreError(f'{identity}/{name}: invalid class evidence')
            n = _integer(row.get('count'), f'{name}.count', 1)
            passed = _integer(row.get('passed'), f'{name}.passed')
            if passed > n:
                raise ScoreError(f'{name}.passed cannot exceed count')
            total = _finite(row.get('score_sum'), f'{name}.score_sum', 0, 100 * passed)
            slack = _finite(row.get('slack_sum'), f'{name}.slack_sum', 0, passed)
            expected_total = (70 + 10 * efficiency) * passed + 20 * slack
            if not math.isclose(total, expected_total, rel_tol=1e-12, abs_tol=1e-8):
                raise ScoreError(f'{identity}/{name}: score components disagree')
        found[identity] = case
    missing = expected - set(found)
    if missing:
        raise ScoreError('Missing cases: ' + ', '.join(map(str, sorted(missing))))
    detail = {}
    model_scores = {}
    for model in models:
        by_traffic = {}
        for condition in traffic:
            by_class = {}
            for name in names:
                rows = [found[(model, condition, seed)]['classes'][name] for seed in seeds
                        if name in found[(model, condition, seed)]['classes']]
                if not rows:
                    continue
                count = sum(row['count'] for row in rows)
                score_sum = math.fsum(row['score_sum'] for row in rows)
                by_class[name] = {'count': count, 'passed': sum(row['passed'] for row in rows),
                                  'score_sum': score_sum, 'score': score_sum / count}
            if not by_class:
                raise ScoreError(f'{model}/{condition}: no requests across all seeds')
            by_traffic[condition] = {'classes': by_class,
                                     'score': math.fsum(x['score'] for x in by_class.values()) / len(by_class)}
        model_scores[model] = math.fsum(x['score'] for x in by_traffic.values()) / len(traffic)
        detail[model] = by_traffic
    mean_model = math.fsum(model_scores.values()) / len(models)
    worst_model = min(model_scores.values())
    raw = .75 * mean_model + .25 * worst_model
    return {'schema': 'pa2-tp-overall-v1', 'raw_score': raw,
            'model_mean': mean_model, 'worst_model_score': worst_model,
            'model_scores': model_scores, 'details': detail,
            'expected_seeds': seeds, 'formula': '0.75 * mean(model scores) + 0.25 * min(model scores)'}


def case_from_summary(learning_score, *, model, traffic, seed):
    """Use the exact per-class sums saved by the shared local/grader scorer."""
    if learning_score.get('schema') != 'pa2-tp-service-quality-v1':
        raise ScoreError('The saved score uses a different rubric; rerun this answer.')
    return {'schema': 'pa2-tp-case-v1', 'model': model, 'traffic': traffic,
            'seed': seed, 'efficiency': learning_score['drain_efficiency'],
            'classes': learning_score['per_class']}


def invalid_case(requests, *, model, traffic, seed):
    """Assign zero only for an invalid student answer, using the offered requests."""
    counts = defaultdict(int)
    for request in requests:
        counts[_identity(request.get('class'), 'request class')] += 1
    if not counts:
        raise ScoreError('An invalid answer still requires the offered workload.')
    return {'schema': 'pa2-tp-case-v1', 'model': model, 'traffic': traffic,
            'seed': seed, 'efficiency': 0.0,
            'classes': {name: {'count': count, 'passed': 0, 'score_sum': 0.0, 'slack_sum': 0.0}
                        for name, count in counts.items()}}

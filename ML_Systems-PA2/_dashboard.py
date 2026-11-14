"""Portable, self-contained result pages for native PA2 experiments."""
from html import escape
from urllib.parse import quote
from score_rubric import aggregate, case_from_summary, invalid_case

STYLE = """:root{color-scheme:light;--ink:#173047;--muted:#576c7e;--teal:#007f86;--line:#dce5ec}
*{box-sizing:border-box}body{margin:0;background:#edf3f7;color:var(--ink);font:16px/1.55 system-ui,sans-serif}
main{max-width:1280px;margin:auto;padding:36px 28px 70px}h1{font-size:36px;line-height:1.2;margin:6px 0 12px}h2{font-size:23px;margin:0 0 14px}h3{font-size:18px}
p{max-width:92ch}a{color:#006e79}code{font:14px ui-monospace,Consolas,monospace;background:#eef3f7;padding:2px 5px;border-radius:4px}.kicker{font-weight:700;color:var(--teal);letter-spacing:.1em;font-size:13px}.muted,small{color:var(--muted)}
section{background:white;border:1px solid var(--line);border-radius:14px;padding:24px;margin-top:22px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:24px}.metric{padding:18px;background:white;border:1px solid var(--line);border-radius:12px}.metric strong{display:block;font-size:29px;line-height:1.3}.metric span{font-size:13px;color:var(--muted)}
.badge{display:inline-block;border-radius:15px;background:#d8f0e9;color:#006449;font-size:13px;font-weight:700;padding:4px 11px}.error{border-left:5px solid #bc3e35}.bad{color:#ac342f;font-weight:700}.good{color:#006f52;font-weight:700}.links{display:flex;gap:10px;flex-wrap:wrap}.links a{display:inline-block;border:1px solid var(--line);border-radius:7px;padding:7px 13px;text-decoration:none;background:#f7fafc}
.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:14px}th{text-align:left;color:#536c7e;background:#f2f6f9;font-weight:600;white-space:nowrap}th,td{padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:top}td.num{text-align:right;font-variant-numeric:tabular-nums}.progress{height:9px;background:#f2d9d3;border-radius:5px;overflow:hidden;margin-top:6px}.progress i{display:block;height:100%;background:#008b88}.node{display:grid;grid-template-columns:80px repeat(8,1fr);gap:6px;margin:10px 0;align-items:center}.gpu{border-radius:6px;padding:9px 3px;text-align:center;font-size:12px;background:#dae8f5}.gpu b{display:block;font-size:14px}.columns{display:grid;grid-template-columns:1fr 1fr;gap:22px}pre{background:#f2f6f9;padding:14px;overflow:auto;border-radius:7px;white-space:pre-wrap}.notice{background:#fff5d9;padding:12px 16px;border-radius:8px}.timeline{height:14px;background:#eef1f5;border-radius:4px;position:relative;min-width:200px}.timeline i{position:absolute;height:14px;border-radius:4px;background:#51a8af}.timeline b{position:absolute;height:14px;width:2px;background:#12334e}input{font:inherit;border:1px solid #bdcbd5;border-radius:6px;padding:8px 12px;min-width:240px;margin-bottom:14px}
.route-row>td{padding:0 12px 10px;background:#fafcfd}.route-details{border:1px solid #dce5ec;border-radius:6px;background:#fff}.route-details>summary{cursor:pointer;color:#006e79;padding:8px 12px;font-size:13px;font-weight:600}.route-details[open]>summary{background:#edf6f7}.route-content{padding:2px 14px 14px}.route-content p{font-size:13px}.route-content .selected{background:#e1f3eb}.resident{margin:3px 0;font-size:12px}.request-group[hidden]{display:none}
@media(max-width:800px){main{padding:20px 12px}.metrics{grid-template-columns:repeat(2,1fr)}.columns{grid-template-columns:1fr}.node{grid-template-columns:54px repeat(8,1fr);gap:3px}.gpu{font-size:10px}section{padding:16px}h1{font-size:28px}}
"""


def text(value):
    return escape(str(value), quote=True)


def link(base, name, label):
    target = quote((base.rstrip("/") + "/" + name).lstrip("/"), safe="/.-_")
    return f'<a href="{target}">{text(label)}</a>'


def page(title, body):
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{text(title)}</title><style>{STYLE}</style></head>'
            f'<body><main>{body}</main></body></html>')


def metric(label, value, note):
    return f'<div class="metric"><span>{text(label)}</span><strong>{text(value)}</strong><span>{text(note)}</span></div>'


def progress_panel(result, base):
    progress = result.get('progress')
    if not progress:
        return ''
    body = '<section><h2>Your progress on this input</h2>'
    if not progress['previous_count']:
        return body + '<p>This is your first saved run under these conditions. Change one part of student.py and run the same command again. This page will show the score change and your saved answers.</p></section>'
    body += (f'<p>Score change: <strong>{progress["delta_previous"]:+.2f}</strong> since your previous run; '
             f'<strong>{progress["delta_first"]:+.2f}</strong> since your first run. '
             f'Your best saved score: <strong>{progress["best_score"]:.2f}</strong>.</p>')
    body += '<p>Only runs with the same input, simulator binary, and measurement code are compared. Each saved answer remains available, so an unsuccessful edit does not lose your earlier work.</p>'
    body += '<div class="scroll"><table><thead><tr><th>Run</th><th>Score</th><th>Drain</th><th>Targets passed by type</th><th>Saved answer</th></tr></thead><tbody>'
    for index, row in enumerate(progress['rows']):
        current = index == len(progress['rows']) - 1
        label = 'This run' if current else row['run']
        if row['run'] == progress['first_run']:
            label += ' (first)'
        if row['run'] == progress['best_run']:
            label += ' (best)'
        target = base.rstrip('/') + '/../../' + row['run']
        passed = '; '.join(f'{name}: {counts["passed"]}/{counts["count"]}'
                           for name, counts in row['classes'].items())
        body += (f'<tr><td>{link(target, result["case"] + "/index.html", label)}</td>'
                 f'<td>{row["score"]:.2f}</td><td>{row["drain_s"]:.3f} s</td><td>{text(passed)}</td>'
                 f'<td>{link(target, "student.py", "Python code")}</td></tr>')
    return body + '</tbody></table></div><p>A higher score on this input is one result. Also try the matching shift input before keeping an edit.</p></section>'


def routing_details(decisions):
    """Show only fields available through the TP student interface."""
    if not decisions:
        return '<details class="route-details"><summary>Routing snapshot</summary><div class="route-content"><p>No admission snapshot was saved for this request.</p></div></details>'
    parts = []
    for decision in decisions:
        snapshot = decision["snapshot"]
        chosen = int(decision["replica_id"])
        now_ms = snapshot["now_ns"] / 1e6
        candidates = sorted(snapshot["candidates"], key=lambda row: int(row["replica"]["id"]))
        header = f'Routing at {now_ms:,.3f} ms · selected ID {chosen} · {len(candidates)} candidates'
        part = f'<details class="route-details"><summary>{text(header)}</summary><div class="route-content">'
        part += '<p>This is the state before admission. Candidates already passed the memory check. Later arrivals can also affect latency; this snapshot is not the full GPU schedule.</p>'
        part += '<div class="scroll"><table><thead><tr><th>Candidate ID</th><th>TP</th><th>Resident count</th><th>Residents: prompt / MAX / generated</th><th>Choice</th></tr></thead><tbody>'
        for candidate in candidates:
            replica = candidate["replica"]
            identity = int(replica["id"])
            residents = candidate["residents"]
            resident_text = "".join(
                f'<div class="resident"><strong>{text(resident["id"])}</strong>: '
                f'{resident["prompt_tokens"]:,} / {resident["max_output_tokens"]:,} / {resident["generated_tokens"]:,}</div>'
                for resident in residents)
            if not residents:
                resident_text = '<span class="muted">No residents</span>'
            selected = identity == chosen
            part += (f'<tr class="{"selected" if selected else ""}"><td>{identity}</td><td>{replica["tp"]}</td>'
                     f'<td>{len(residents)}</td><td>{resident_text}</td>'
                     f'<td>{"Selected" if selected else ""}</td></tr>')
        part += '</tbody></table></div></div></details>'
        parts.append(part)
    return "".join(parts)


def result_page(result, record, base=".", policy_link="../student.py", comparison_link=None):
    """Render real result fields only. No policy predictions become measurements."""
    case_name = result["case"]
    if result["status"] != "complete":
        return error_page(result, base)
    metrics = result["metrics"]
    score = result["learning_score"]
    requests = record["requests"]
    layout = result["layout"]
    content = f'<div class="kicker">PA2 · TENSOR PARALLELISM LAB</div><h1>{text(case_name)} · {text(result["model"])}</h1>'
    content += f'<span class="badge">Complete · native ASTRA verified</span><p>Policy: <code>{text(result["policy"])}</code> · {len(requests)} requests · {len(layout)} replicas on 32 GPUs.</p>'
    content += '<p class="muted">Your Python code chose the layout and destinations. The course wrapper sent compute and communication tasks to ASTRA, then used its events to measure token times. No timing cache was used.</p>'
    content += '<div class="metrics">'
    content += metric("Score", f'{score["automated_score_100"]:.2f} / 100', "Target pass, latency margin, and drain")
    content += metric("Target pass rate", f'{100*metrics["slo_pass_fraction"]:.1f}%', "Both TTFT and request P95 ITL pass")
    content += metric("Drain time", f'{score["actual_drain_ns"]/1e9:.3f} s', "Last arrival to final completion; lower is better")
    content += metric("Output throughput", f'{metrics["throughput_output_tokens_per_s"]:.1f} tokens/s', "Output tokens over the whole run")
    content += '</div>' + progress_panel(result, base)
    content += '<section><h2>Latency results by request type</h2><div class="scroll"><table><thead><tr><th>Request class</th><th>Pass / total</th><th>Pass fraction</th><th>P95 TTFT</th><th>P95 of request P95 ITL</th></tr></thead><tbody>'
    for name, group in score["per_class"].items():
        observed = metrics["classes"][name]
        fraction = group["pass_fraction"]
        content += (f'<tr><td>{text(name)}</td><td>{group["passed"]} / {group["count"]}</td>'
                    f'<td>{100*fraction:.1f}%<div class="progress"><i style="width:{100*fraction:.3f}%"></i></div></td>'
                    f'<td>{observed["p95_ttft_ns"]/1e6:,.2f} ms</td><td>{observed["p95_request_itl_ns"]/1e6:,.2f} ms</td></tr>')
    content += '</tbody></table></div><p>TTFT = time from arrival to first token. ITL = time between tokens. The class ITL column is a summary across requests; each request is tested against its own P95 ITL goal.</p>'
    content += f'<p class="notice">Target pass: {score["service_points"]:.2f}/70. Latency margin: {score["latency_points"]:.2f}/20. Drain: {score["efficiency_points"]:.2f}/10. A request that misses either target contributes zero to all three components.</p></section>'
    owner = {gpu: (index, replica) for index, replica in enumerate(layout) for gpu in replica["gpus"]}
    content += '<section><h2>Your GPU groups</h2><p>One colored group is one replica. Its GPUs jointly process each request sent to that replica. Separate replicas can progress independently.</p>'
    for node in range(4):
        content += f'<div class="node"><strong>Node {node}</strong>'
        for gpu in range(node*8, node*8+8):
            if gpu in owner:
                index, replica = owner[gpu]
                # A different hue for each ID; IDs remain visible even if two colors look similar.
                hue = (index * 137.508) % 360
                content += f'<div class="gpu" style="background:hsl({hue:.3f} 65% 84%)">GPU {gpu}<b>Replica {text(replica["id"])}</b></div>'
            else:
                content += f'<div class="gpu">GPU {gpu}<b>Unused</b></div>'
        content += '</div>'
    assignments = {}
    for request in requests:
        identity = str(request["prefill_worker"])
        assignments[identity] = assignments.get(identity, 0) + 1
    content += '<div class="scroll"><table><thead><tr><th>Replica</th><th>TP</th><th>GPU IDs</th><th>Assigned requests</th></tr></thead><tbody>'
    for replica in layout:
        content += f'<tr><td>{text(replica["id"])}</td><td>{replica["tp"]}</td><td>{text(", ".join(map(str, replica["gpus"])))}</td><td>{assignments.get(str(replica["id"]),0)}</td></tr>'
    content += '</tbody></table></div></section><section><h2>Results for each request</h2>'
    content += '<p>Prompt and MAX output are known to your router. Actual output and measured times are revealed after execution. Admission reserves prompt + MAX KV space until EOS.</p>'
    content += '<p>Each request has an expandable routing snapshot below its result. It shows the candidate replicas and their residents at the decision time.</p>'
    content += '<input id="filter" aria-label="Filter requests" placeholder="Filter by request, class, or replica"><div class="scroll"><table id="requests"><thead><tr><th>Request</th><th>Class</th><th>Replica</th><th>Prompt / MAX / actual</th><th>Admission wait</th><th>TTFT / goal</th><th>P95 ITL / goal</th><th>Targets</th></tr></thead>'
    by_request = {}
    for decision in record.get("routing_decisions", []):
        by_request.setdefault(decision["request_id"], []).append(decision)
    for request in requests:
        passed = request["slo_met"]
        ttft_ok = request["ttft_ns"] <= request["ttft_goal_ns"]
        itl_ok = request["p95_itl_ns"] <= request["itl_goal_ns"]
        filter_text = f'{request["id"]} {request["class"]} replica {request["prefill_worker"]}'
        content += f'<tbody class="request-group" data-filter="{text(filter_text)}">'
        content += (f'<tr><td>{text(request["id"])}</td><td>{text(request["class"])}</td><td>{text(request["prefill_worker"])}</td>'
                    f'<td>{request["prompt_tokens"]:,} / {request["max_output_tokens"]:,} / {request["output_tokens"]:,}</td>'
                    f'<td>{(request["admission_ns"]-request["arrival_ns"])/1e6:,.2f} ms</td>'
                    f'<td class="{"good" if ttft_ok else "bad"}">{request["ttft_ns"]/1e6:,.2f} / {request["ttft_goal_ns"]/1e6:,.0f} ms</td>'
                    f'<td class="{"good" if itl_ok else "bad"}">{request["p95_itl_ns"]/1e6:,.2f} / {request["itl_goal_ns"]/1e6:,.0f} ms</td>'
                    f'<td class="{"good" if passed else "bad"}">{"PASS" if passed else "MISS"}</td></tr>')
        content += '<tr class="route-row"><td colspan="8">' + routing_details(by_request.get(request["id"], [])) + '</td></tr></tbody>'
    content += '</table></div></section><section><h2>Comparing saved runs</h2><p>Changing only the layout or only the router helps identify its effect. The saved input stays the same for a direct comparison. Results on the matching shift case show whether the idea still works when traffic changes.</p>'
    content += '<div class="links">' + link(base, "summary.json", "Summary JSON") + link(base, "record.json", "Full request and event record")
    content += f'<a href="{text(policy_link)}">Exact policy used</a></div>'
    content += f'<p class="muted">CPU experiment time: {result.get("wall_time_s",0):.2f} s. This is different from simulated request latency. ASTRA binary SHA256: <code>{text(result["native_binary_sha256"])}</code>.</p></section>'
    content += '<script>document.getElementById("filter").addEventListener("input",function(){const q=this.value.toLowerCase();document.querySelectorAll("#requests > tbody.request-group").forEach(r=>r.hidden=!r.dataset.filter.toLowerCase().includes(q));});</script>'
    return page("PA2 result: " + case_name, content)


def error_page(result, base="."):
    issue = result["issue"]
    location = issue.get("location", {})
    body = '<div class="kicker">PA2 · FIX AND TRY AGAIN</div><h1>This run has no score</h1>'
    body += f'<section class="error"><h2>{text(issue.get("code","ERROR"))}</h2><p>{text(issue["message"])}</p>'
    if location:
        body += f'<p><strong>Where:</strong> {text(location)}</p>'
    body += f'<p><strong>Next step:</strong> {text(issue.get("hint", "Correct the policy and run again."))}</p>'
    if result.get('status') == 'invalid_submission':
        body += '<p>This input counts as zero for grading until the answer is fixed. No performance was measured.</p>'
    else:
        body += '<p>The overall score is pending because this failure needs an environment check.</p>'
    body += '<div class="links">' + link(base, "diagnostic.json", "Full diagnostic") + '</div></section>'
    return page("PA2: fix this run", body)


def suite_score(results):
    """The same aggregate is used by the student suite and instructor grader."""
    if len(results) != 6 or any(row['status'] not in ('complete', 'invalid_submission') for row in results):
        return None
    case_rows = []
    for row in results:
        identity = row['case']
        options = {'model':identity.removesuffix('-shift'),
                   'traffic':'shift' if identity.endswith('-shift') else 'steady', 'seed':'public'}
        if row['status'] == 'invalid_submission':
            counts = row.get('offered_class_counts')
            if not counts:
                return None
            offered = [{'class': name} for name, count in counts.items() for _ in range(count)]
            case_rows.append(invalid_case(offered, **options))
        else:
            case_rows.append(case_from_summary(row['learning_score'], **options))
    return aggregate(case_rows, expected_seeds=['public'])


def suite_page(results, run_name, base="."):
    body = '<div class="kicker">PA2 · SIX PUBLIC CASES</div><h1>Compare all three models</h1><p>Each row is a separate run with one model and a fresh policy process. The same file handles all six cases.</p>'
    total = suite_score(results)
    if total is not None:
        body += (f'<p><strong>Overall score: {total["raw_score"]:.2f} / 100.</strong> '
                 f'Model average: {total["model_mean"]:.2f}; lowest model: {total["worst_model_score"]:.2f}. '
                 'Overall score = 75% of the model average + 25% of the lowest model. '
                 'Each model averages its two traffic patterns; each pattern averages request types equally. '
                 'Invalid student answers count as zero on the affected input; infrastructure failures keep the overall score pending.</p>')
    else:
        body += '<p>No overall score is available. Check missing inputs or infrastructure errors before comparing results.</p>'
    body += '<section><div class="scroll"><table><thead><tr><th>Case</th><th>Model</th><th>Status</th><th>Score</th><th>Target pass</th><th>Drain</th><th>Details</th></tr></thead><tbody>'
    for result in results:
        complete = result["status"] == "complete"
        score = f'{result["learning_score"]["automated_score_100"]:.2f}' if complete else "No score"
        rate = f'{100*result["metrics"]["slo_pass_fraction"]:.1f}%' if complete else "—"
        drain = f'{result["learning_score"]["actual_drain_ns"]/1e9:.3f} s' if complete else "—"
        body += (f'<tr><td>{text(result["case"])}</td><td>{text(result["model"])}</td><td>{text(result["status"])}</td>'
                 f'<td>{score}</td><td>{rate}</td><td>{drain}</td><td>{link(base, result["case"]+"/index.html", "Open result")}</td></tr>')
    body += '</tbody></table></div><p>Read per-class and per-request results before interpreting the overall score. A failed run has no performance measurement.</p></section>'
    body += '<p class="muted">Saved run: ' + text(run_name) + '</p>'
    return page("PA2 public-case results", body)

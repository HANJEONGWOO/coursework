/**
 * Overview charts and the requests behind a selected chart region.
 *
 * Reading route: studyIndex -> studySeries -> studyChart -> studyInspect.
 * studyIndex caches lookups by request ID so a redraw need not rescan events.
 * The graph inspector uses the SAME definitions and time bins as the charts.
 *
 * State shared with visualizer.js and explorer.js:
 *   studyWindow: null for the full recording, otherwise {start, end}.
 *   studySelection: graph type, selected interval, and related request rows.
 * Changing these values only changes replay, never the recorded simulation.
 *
 * Use r for a run, f for one recorded iteration, and q for a request/metric row.
 * These short names stay inside a small function or callback.
 */

/* Study overview and chart-to-request context. No network or external libraries. */
var studyWindow = null;
var studySelection = null;
const studyIndexCache = new WeakMap();
/**
 * Build and cache request-ID lookups for an entire immutable recording.
 * A WeakMap keys each cache by the run object, allowing old runs to be released.
 * Output events are a fallback when an incomplete run has no final metrics.
 */
function studyIndex(r) {
  if (studyIndexCache.has(r)) {
    return studyIndexCache.get(r);
  }
  const inputs = new Map(r.requests.map((q) => [q.id, q])),
    metrics = new Map((r.metrics?.requests || []).map((q) => [q.id, q]));
  const admitted = new Map(),
    completed = new Map(),
    actions = new Map(),
    outputs = new Map();
  for (const f of r.frames) {
    for (const a of f.plan) {
      if (!actions.has(a.request_id)) {
        actions.set(a.request_id, []);
      }
      actions.get(a.request_id).push({ frame: f, action: a });
    }
    for (const e of f.events) {
      if (e.kind === "admit") {
        admitted.set(e.request_id, f.start);
      }
      if (e.kind === "free") {
        completed.set(e.request_id, f.end);
      }
      if (e.kind === "output") {
        if (!outputs.has(e.request_id)) {
          outputs.set(e.request_id, []);
        }
        outputs.get(e.request_id).push(e.time);
      }
    }
  }
  const index = { inputs, metrics, admitted, completed, actions, outputs };
  studyIndexCache.set(r, index);
  return index;
}

/**
 * Explain which final latency goal failed, or why no result is available.
 */
function studyFailure(q) {
  if (!q) {
    return "Final measurements unavailable";
  }
  const reasons = [];
  if (q.ttft > q.ttft_slo) {
    reasons.push("TTFT missed");
  }
  if (q.p95_itl > q.itl_slo) {
    reasons.push("P95 ITL missed");
  }
  return reasons.join(" + ") || "Both SLOs met";
}

/**
 * Connect a graph position to the exact requests represented there.
 * Arrivals and attainment use arrival-time bins. Queue/memory use iteration
 * starts. These are different meanings of the horizontal time coordinate.
 */
function studyContext(r, key, time) {
  const index = studyIndex(r),
    end = endTime(r) || 1;
  time = Math.max(0, Math.min(end, time));
  let rows = [],
    start = time,
    stop = time,
    frame = null,
    heading = "",
    explain = "";
  if (key === "arrivals" || key === "attainment") {
    const bin = Math.min(49, Math.floor((time / end) * 50));
    start = (bin * end) / 50;
    stop = ((bin + 1) * end) / 50;
    rows = r.requests
      .filter(
        (q) =>
          q.arrival >= start && (q.arrival < stop || (bin === 49 && q.arrival <= stop)),
      )
      .map((q) => ({
        id: q.id,
        detail: "Arrived at " + fmt(q.arrival),
        metric: index.metrics.get(q.id),
      }));
    rows.sort(
      (a, b) =>
        Number(a.metric?.slo_met ?? true) - Number(b.metric?.slo_met ?? true) ||
        (b.metric?.ttft || 0) - (a.metric?.ttft || 0) ||
        index.inputs.get(a.id).arrival - index.inputs.get(b.id).arrival,
    );
    heading =
      key === "arrivals"
        ? "Requests arriving in this bin"
        : "Final outcomes for this arrival bin";
    explain =
      key === "arrivals"
        ? "These requests arrived in the selected time bin. Failed " +
          "requests are listed first."
        : "This chart groups final outcomes by ARRIVAL time. It does " +
          "not locate the time of a failure. Failed requests are listed " +
          "first.";
  } else {
    frame = r.frames.reduce(
      (best, f) =>
        !best || Math.abs(f.start - time) < Math.abs(best.start - time) ? f : best,
      null,
    );
    if (frame) {
      start = stop = time = frame.start;
      if (key === "waiting") {
        rows = r.requests
          .filter(
            (q) => q.arrival <= time && (index.admitted.get(q.id) ?? Infinity) > time,
          )
          .map((q) => ({
            id: q.id,
            detail: "Waiting " + fmt(time - q.arrival) + "; not admitted",
            metric: index.metrics.get(q.id),
          }));
        rows.sort(
          (a, b) => index.inputs.get(a.id).arrival - index.inputs.get(b.id).arrival,
        );
        heading = "Requests waiting at " + frame.id;
        explain =
          "These requests have arrived but have not entered memory at " +
          "this iteration start. The earliest is listed first. They " +
          "have no scheduled action yet.";
      } else {
        // The graph samples memory after admission, before decode growth.
        // The stored during snapshot is later: subtract growth events to
        // recover the SAME ownership count that produced the plotted point.
        const growth = new Map();
        for (const e of frame.events) {
          if (e.kind === "grow") {
            growth.set(
              e.request_id,
              (growth.get(e.request_id) || 0) + e.kv_after - e.kv_before,
            );
          }
        }
        rows = (frame.during || frame.admitted_state || [])
          .filter((q) => q.reserved_blocks > 0)
          .map((q) => ({
            id: q.id,
            slots: (q.reserved_blocks - (growth.get(q.id) || 0)) * r.config.block_size,
            metric: index.metrics.get(q.id),
          }));
        rows.sort((a, b) => b.slots - a.slots);
        rows.forEach((q) => (q.detail = q.slots + " slots reserved after admission"));
        heading = "Memory owners at " + frame.id;
        explain =
          "These are the reservations represented by the chart: after " +
          "admission, before this iteration grows memory for decode. " +
          "Largest owners are listed first.";
      }
    }
  }
  return { key, time, start, end: stop, frameId: frame?.id, heading, explain, rows };
}

/**
 * Build the related-request table and its Trace buttons.
 */
function studyInspector(r) {
  const s = studySelection;
  if (!s) {
    return (
      '<section id="chart-inspector"><h3>Chart → related requests → ' +
      "recorded actions</h3><p>Click a peak or low pass-rate " +
      "region. A request list will appear here. Keyboard: focus a " +
      "chart, use Left/Right to choose a time, then " +
      "Enter.</p></section>"
    );
  }
  const failures = s.rows.filter((q) => q.metric && !q.metric.slo_met).length;
  return /* HTML */ `<section
    id="chart-inspector"
    tabindex="-1"
    aria-label="Requests related to selected chart region"
  >
    <h3>${esc(s.heading)}</h3>
    <p>${esc(s.explain)}</p>
    <p>
      <b>${s.rows.length} related requests</b> ·
      ${s.start === s.end
        ? "Iteration start " + fmt(s.start)
        : "Arrival interval " + fmt(s.start) + "–" + fmt(s.end)}
      · ${failures} with a final SLO miss
    </p>
    <p>
      Each <b>Trace</b> button selects that request and opens its lane and recorded
      actions. Waiting/memory views keep the selected iteration; arrival views go to
      arrival, and final-outcome views go to the relevant output.
    </p>
    <div class="related-scroll">
      <table id="related-requests">
        <thead>
          <tr>
            <th>Request</th>
            <th>Connection to this region</th>
            <th>Final outcome</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          ${s.rows
            .map(
              (q) =>
                /* HTML */ `<tr
                  class="${q.id === $("request").value ? "related-selected" : ""}"
                >
                  <td>${esc(q.id)}</td>
                  <td>${esc(q.detail)}</td>
                  <td class="${q.metric && !q.metric.slo_met ? "error" : ""}">
                    ${esc(studyFailure(q.metric))}
                  </td>
                  <td>
                    <button class="trace-related" data-id="${esc(q.id)}">
                      Trace ${esc(q.id)}
                    </button>
                  </td>
                </tr>`,
            )
            .join("") ||
          '<tr><td colspan="4">No requests belong to this region. ' +
            "Select another time.</td></tr>"}
        </tbody>
      </table>
    </div>
  </section>`;
}

/**
 * Select graph context, make its time visible, and reveal related requests.
 */
function studyInspect(key, time) {
  stop();
  const r = primary();
  studySelection = studyContext(r, key, time);
  clock = studySelection.time;
  if (!studyWindow) {
    explorerSetWindow(clock - 30, 90);
  } else if (clock < studyWindow.start || clock > studyWindow.end) {
    explorerSetWindow(
      clock - (studyWindow.end - studyWindow.start) / 2,
      studyWindow.end - studyWindow.start,
    );
  }
  if (studySelection.rows.length) {
    $("request").value = studySelection.rows[0].id;
  }
  render();
  $("chart-inspector").focus({ preventScroll: true });
  $("chart-inspector").scrollIntoView({ block: "center" });
}

/**
 * Follow a related request from a graph to its recorded work.
 * For a failed first response, inspect its first output. For an ITL failure,
 * inspect the largest recorded gap as a diagnostic, not as the P95 itself.
 */
function studyTraceRequest(id) {
  stop();
  const r = primary(),
    index = studyIndex(r),
    input = index.inputs.get(id),
    metric = index.metrics.get(id);
  $("request").value = id;
  if (studySelection?.key === "arrivals") {
    clock = input.arrival;
  } else if (studySelection?.key === "attainment" && metric) {
    const ts = metric.token_times;
    if (metric.ttft > metric.ttft_slo) {
      clock = ts[0];
    } else if (metric.p95_itl > metric.itl_slo && ts.length > 1) {
      let end = 1;
      for (let i = 2; i < ts.length; i++) {
        if (ts[i] - ts[i - 1] > ts[end] - ts[end - 1]) {
          end = i;
        }
      }
      clock = ts[end];
    } else {
      clock = ts[0];
    }
    // Select the iteration that PRODUCED this output, not the next iteration.
    const f = r.frames.find((f) =>
      f.events.some(
        (e) =>
          e.kind === "output" && e.request_id === id && Math.abs(e.time - clock) < 1e-7,
      ),
    );
    if (f) {
      clock = Math.max(f.start, f.end - 1e-7);
    }
  } else if (studySelection) {
    clock = studySelection.time;
  }
  explorerSelect(id, { time: clock, scroll: true });
}

/**
 * Use the stored policy name as the display label.
 */
function studyLabel(r) {
  return r.policy;
}

/**
 * Derive points from the full recording using each chart's definition.
 * An empty arrival bin has no pass fraction, so it is omitted from attainment.
 * Failed/incomplete recordings can still show arrivals, queue, and memory.
 */
function studySeries(r, key, bins = 50) {
  const end = endTime(r) || 1;
  if (key === "arrivals") {
    const v = Array(bins).fill(0);
    for (const q of r.requests) {
      v[Math.min(bins - 1, Math.floor((q.arrival / end) * bins))]++;
    }
    return v.map((y, i) => ({ x: ((i + 0.5) * end) / bins, y }));
  }
  if (key === "attainment") {
    // Group FINAL outcomes by ARRIVAL time, not by completion/failure time.
    const groups = Array.from({ length: bins }, () => []);
    for (const q of r.metrics?.requests || []) {
      const req = r.requests.find((x) => x.id === q.id);
      groups[Math.min(bins - 1, Math.floor((req.arrival / end) * bins))].push(
        q.slo_met,
      );
    }
    return groups
      .map((a, i) => ({
        x: ((i + 0.5) * end) / bins,
        y: a.length ? (100 * a.filter(Boolean).length) / a.length : null,
      }))
      .filter((x) => x.y !== null);
  }
  return r.frames.map((f) => ({
    x: f.start,
    y:
      key === "waiting"
        ? f.system?.waiting || 0
        : key === "active"
          ? f.system?.active || 0
          : f.system?.reserved_slots || 0,
  }));
}

/**
 * Bound SVG path size while preserving the maximum of each local group.
 * This retains queue spikes that simple every-Nth-point sampling could hide.
 */
function studyReduced(pts) {
  const stride = Math.max(1, Math.ceil(pts.length / 1200)),
    reduced = [];
  for (let i = 0; i < pts.length; i += stride) {
    const group = pts.slice(i, i + stride);
    reduced.push(group.reduce((a, b) => (a.y > b.y ? a : b)));
  }
  return reduced;
}

/**
 * Draw axes, series, cursor, visible-window band, and selected graph region.
 */
function studyChart(r, key, title) {
  const pts = studySeries(r, key),
    maxX = endTime(r) || 1,
    maxY = Math.max(1, ...pts.map((p) => p.y));
  const w = 560,
    h = 170,
    left = 45,
    right = 12,
    top = 12,
    bottom = 28;
  const x = (v) => left + (v / maxX) * (w - left - right),
    y = (v) => h - bottom - (v / maxY) * (h - top - bottom);
  // Preserve queue spikes while bounding DOM/path size.
  const reduced = studyReduced(pts);
  const line = reduced.map((p, i) => (i ? "L" : "M") + x(p.x) + "," + y(p.y)).join(" ");
  return /* HTML */ `<div>
    <h3>${esc(title)}</h3>
    <svg
      class="study-chart"
      data-key="${key}"
      data-max-y="${maxY}"
      data-end="${maxX}"
      tabindex="0"
      viewBox="0 0 ${w} ${h}"
      role="button"
      aria-label="${esc(
        title,
      )}; click a region or use arrow keys and Enter to list related requests"
    >
      ${studyWindow
        ? /* HTML */ `<rect
            class="visible-window-band"
            x="${x(studyWindow.start)}"
            y="${top}"
            width="${Math.max(1, x(studyWindow.end) - x(studyWindow.start))}"
            height="${h - top - bottom}"
            fill="var(--dec)"
            opacity="0.12"
          />`
        : ""}${studySelection?.key === key
        ? /* HTML */ `<rect
            x="${x(studySelection.start)}"
            y="${top}"
            width="${Math.max(2, x(studySelection.end) - x(studySelection.start))}"
            height="${h - top - bottom}"
            fill="var(--pre)"
            opacity="0.22"
          />`
        : ""}
      <path
        d="M${left},${top}V${h - bottom}H${w - right}"
        fill="none"
        stroke="var(--line)"
      />
      <path d="${line}" fill="none" stroke="var(--dec)" stroke-width="2" />
      <text x="2" y="20">${maxY.toFixed(key === "attainment" ? 0 : 0)}</text>
      <text x="20" y="${h - bottom}">0</text>
      <text x="${left}" y="${h - 5}">0</text>
      <text x="${w - 65}" y="${h - 5}">${maxX.toFixed(0)}</text>
      <text x="${w / 2}" y="${h - 5}" text-anchor="middle">Synthetic time</text>
      <line
        x1="${x(clock)}"
        x2="${x(clock)}"
        y1="${top}"
        y2="${h - bottom}"
        stroke="var(--ink)"
        stroke-dasharray="3 3"
      />
    </svg>
  </div>`;
}

/**
 * Render traffic overview and bind graph inspection controls.
 * These charts use the primary run. The result comparison table uses only
 * matching input fingerprints; class labels are post-run display information.
 */
function renderStudy(chosen) {
  const el = $("study-overview"),
    r = chosen[0];
  el.hidden = !r?.traffic;
  if (!r?.traffic) {
    return;
  }
  const m = r.metrics,
    meta = r.traffic,
    group = m?.groups || {};
  const comparable = runs.filter((x) => x.input_sha256 === r.input_sha256);
  el.innerHTML = /* HTML */ `<h2>
      ${esc(meta.workload)}: ${r.requests.length} requests, seed ${meta.seed}, load
      ×${meta.load}
    </h2>
    <p>
      Input fingerprint ${esc(r.input_sha256.slice(0, 12))}. Compare settings on
      identical requests. All values use synthetic time.
    </p>
    <p>
      Selected configuration: <b>${esc(r.policy)}</b>. Budget ${r.config.budget}, active
      limit ${r.config.max_active}. Admission protects room for up to
      ${r.config.output_bound} output tokens per request. That headroom is not allocated
      KV memory.
    </p>
    <div class="study-charts" id="study-charts">
      ${studyChart(r, "arrivals", "Arrivals per time bin")}${studyChart(
        r,
        "waiting",
        "Waiting requests at iteration start",
      )}${studyChart(r, "memory", "Allocated KV slots after admission")}${studyChart(
        r,
        "attainment",
        "Requests meeting both SLOs by arrival time (%)",
      )}
    </div>
    <p>
      Click a chart region to list its requests. Each chart describes the result on the
      left; the pass-rate chart groups final outcomes by arrival time.
    </p>
    ${studyInspector(r)}
    <div class="toolbar">
      <label
        >Inspect time<input
          id="study-seek"
          type="number"
          min="0"
          max="${endTime(r)}"
          step="0.1"
          value="${clock.toFixed(1)}" /></label
      ><button id="study-full">Show full time range</button
      ><button onclick="$('request-explorer').scrollIntoView({block:'start'})">
        Browse all requests</button
      ><button onclick="$('timeline-tools').scrollIntoView({block:'start'})">
        Go to timeline / zoom
      </button>
    </div>
    <div id="settings-table" style="overflow-x:auto">
      <table>
        <tr>
          <th>Configuration</th>
          <th>Outputs/unit</th>
          <th>Good requests/unit</th>
          <th>Peak queue</th>
          <th>P95 TTFT</th>
          <th>Recovery delay</th>
        </tr>
        ${comparable
          .map(
            (x) =>
              /* HTML */ `<tr>
                <td>${esc(x.policy)}</td>
                <td>${fmt(x.metrics?.throughput)}</td>
                <td>${fmt(x.metrics?.goodput)}</td>
                <td>${x.metrics?.peak_waiting ?? "—"}</td>
                <td>${fmt(x.metrics?.p95_ttft)}</td>
                <td>
                  ${x.traffic.burst_end === null
                    ? "N/A"
                    : fmt(x.metrics?.recovery_delay)}
                </td>
              </tr>`,
          )
          .join("")}
      </table>
    </div>
    <p>
      Recovery delay: time after the burst ends until the sampled queue remains at most
      2 for the rest of this finite run. A dash means no observed sustained recovery.
    </p>
    <table id="class-table">
      <tr>
        <th>Request class</th>
        <th>Count</th>
        <th>Both SLOs met</th>
        <th>P95 TTFT</th>
        <th>P95 of request P95 ITL</th>
      </tr>
      ${Object.entries(group)
        .map(
          ([k, v]) =>
            /* HTML */ `<tr>
              <td>${esc(k)}</td>
              <td>${v.count}</td>
              <td>${(100 * v.attainment).toFixed(1)}%</td>
              <td>${fmt(v.p95_ttft)}</td>
              <td>${fmt(v.p95_request_itl)}</td>
            </tr>`,
        )
        .join("")}
    </table>
    <p>
      Slowest first responses (click to trace):
      ${(m?.requests || [])
        .slice()
        .sort((a, b) => b.ttft - a.ttft)
        .slice(0, 8)
        .map(
          (q) =>
            /* HTML */ `<button class="study-request" data-id="${esc(q.id)}">
              ${esc(q.id)}: ${fmt(q.ttft)}
            </button>`,
        )
        .join(" ")}
    </p>`;
  if (r.competition) {
    el.querySelector("h2").nextElementSibling.textContent =
      "Selected competition case. All values use synthetic time.";
    el.querySelectorAll("#settings-table tr").forEach((row) =>
      row.lastElementChild.remove(),
    );
    el.querySelector("#settings-table").nextElementSibling.remove();
  }
  el.querySelectorAll(".study-chart").forEach(
    (svg) =>
      (svg.onclick = (e) => {
        const rect = svg.getBoundingClientRect(),
          px = ((e.clientX - rect.left) / rect.width) * 560,
          py = ((e.clientY - rect.top) / rect.height) * 170;
        const fraction = Math.max(0, Math.min(1, (px - 45) / 503)),
          key = svg.dataset.key;
        let time = fraction * Number(svg.dataset.end);
        if (key === "waiting" || key === "memory") {
          const points = studyReduced(studySeries(r, key)),
            maxY = Number(svg.dataset.maxY),
            end = Number(svg.dataset.end);
          const distance = (p) =>
            Math.hypot(45 + (p.x / end) * 503 - px, 142 - (p.y / maxY) * 130 - py);
          const nearest = points.reduce(
            (a, b) => (!a || distance(b) < distance(a) ? b : a),
            null,
          );
          if (nearest) {
            time = nearest.x;
          }
        }
        studyInspect(key, time);
      }),
  );
  el.querySelectorAll(".study-chart").forEach(
    (svg) =>
      (svg.onkeydown = (e) => {
        const step = Number(svg.dataset.end) / 50;
        if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
          e.preventDefault();
          const t = Number(svg.dataset.cursor ?? clock);
          svg.dataset.cursor = Math.max(
            0,
            Math.min(
              Number(svg.dataset.end),
              t + (e.key === "ArrowRight" ? step : -step),
            ),
          );
          svg.setAttribute(
            "aria-label",
            svg.dataset.key +
              "; selected time " +
              Number(svg.dataset.cursor).toFixed(1) +
              "; press Enter",
          );
        }
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          studyInspect(svg.dataset.key, Number(svg.dataset.cursor ?? clock));
        }
      }),
  );
  el.querySelectorAll(".trace-related").forEach(
    (button) => (button.onclick = () => studyTraceRequest(button.dataset.id)),
  );
  $("study-seek").onchange = (e) => studySeek(Number(e.target.value));
  $("study-full").onclick = () => {
    studyWindow = null;
    render();
  };
  el.querySelectorAll(".study-request").forEach(
    (button) =>
      (button.onclick = () => {
        const id = button.dataset.id;
        studySelection = null;
        explorerFitRequest(id);
        explorerSelect(id, {
          time: studyIndex(r).metrics.get(id)?.token_times?.[0] ?? clock,
          scroll: true,
        });
      }),
  );
}

/**
 * Treat typed inspection time as a queue-graph selection.
 */
function studySeek(value) {
  studyInspect("waiting", Number.isFinite(value) ? value : 0);
}

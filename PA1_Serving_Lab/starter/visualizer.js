/**
 * Offline dashboard controller. Start reading at load(), then render().
 *
 * Python has already executed the policy and written a recording. This script
 * reads that local file, selects a result, and replays recorded observations.
 * It never runs Python, advances the simulator, or predicts future performance.
 *
 * Shared state used by the other classic scripts:
 *   runs: loaded results for ONE input case.
 *   clock: replay cursor in synthetic time (not the simulator's live clock).
 *   request select element: one request ID shared by every panel.
 *   studyWindow / studySelection: visible time range / selected graph context.
 *
 * render() coordinates panels. study_view.js draws overview charts;
 * explorer.js draws request lanes and navigation; arena_view.js shows score.
 * All recording-provided text must pass through esc() before entering HTML.
 */

"use strict";
const $ = (id) => document.getElementById(id);
// These are replay controls. Changing them never changes a saved result.
let runs = [],
  timer = null,
  clock = 0;
// File contents are data, not markup. Escape them before building HTML strings.
const esc = (x) =>
  String(x ?? "").replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c],
  );
// A dash means no measurement, not a measured latency of zero.
const fmt = (x) => (Number.isFinite(x) ? x.toFixed(3) : "—");
/**
 * Stop automatic playback without changing the selected replay time.
 */
function stop() {
  clearInterval(timer);
  timer = null;
  $("play").textContent = "Play";
}

/**
 * Return the result chosen for the left-hand panel.
 */
function primary() {
  return runs[Number($("run").value)];
}

/**
 * Return the primary result and, optionally, one eligible comparison.
 */
function peers() {
  const a = primary();
  if (!a) {
    return [];
  }
  const b = $("compare-run").value === "" ? null : runs[Number($("compare-run").value)];
  return $("compare").checked && b && b !== a ? [a, b] : [a];
}

/**
 * Offer only runs with compatible inputs/resources for comparison.
 * The one-slot demo differs in block layout but preserves total token capacity.
 */
function comparisons() {
  const a = primary();
  const eligible = runs
    .map((r, i) => ({ r, i }))
    .filter(
      ({ r }) =>
        r !== a &&
        r.scenario === a.scenario &&
        (a.traffic
          ? r.input_sha256 === a.input_sha256
          : JSON.stringify(r.requests) === JSON.stringify(a.requests) &&
            r.config.capacity_blocks * r.config.block_size ===
              a.config.capacity_blocks * a.config.block_size &&
            [
              "budget",
              "prefill_cap",
              "prefill_rate",
              "decode_base",
              "decode_per_request",
            ].every((k) => r.config[k] === a.config[k])),
    );
  $("compare-run").innerHTML =
    '<option value="">None</option>' +
    eligible
      .map(({ r, i }) => /* HTML */ `<option value="${i}">${esc(r.policy)}</option>`)
      .join("");
  if (eligible.length) {
    $("compare-run").value = eligible[0].i;
  }
}

/**
 * Find the final recorded time. Legacy measurements convert seconds to ms.
 */
function endTime(r) {
  return r.kind === "measurement"
    ? Math.max(0, ...r.trace.token_timestamps_s.map((t) => t * 1000))
    : Math.max(0, ...r.frames.map((f) => f.end));
}

/**
 * Find the latest iteration that has started at the replay cursor.
 * At a shared end/start boundary this selects the NEXT iteration. Output-trace
 * buttons subtract a tiny epsilon when they need the producing iteration.
 */
function frameAt(r, t) {
  return r.frames.filter((f) => f.start <= t + 1e-9).at(-1) || null;
}

/**
 * Check the recording envelope and fields needed for safe rendering.
 * One loaded file must describe one input case, though it may compare policies.
 * This checks data shape; it does not recompute the policy or certify a score.
 */
function validate(bundle) {
  if (
    bundle.schema !== "serving-lab-case-v1" ||
    typeof bundle.case_id !== "string" ||
    !Array.isArray(bundle.runs) ||
    !bundle.runs.length
  ) {
    throw Error("Open one serving-lab-case-v1 recording.");
  }
  if (bundle.runs.some((r) => r.scenario !== bundle.case_id)) {
    throw Error("One recording must contain exactly one test case.");
  }
  for (const r of bundle.runs) {
    if (
      typeof r.scenario !== "string" ||
      typeof r.policy !== "string" ||
      typeof r.status !== "string"
    ) {
      throw Error("Missing run metadata");
    }
    if (r.kind === "measurement") {
      if (
        !Array.isArray(r.trace?.token_timestamps_s) ||
        !r.trace.token_timestamps_s.every(Number.isFinite)
      ) {
        throw Error("Invalid measured timestamps");
      }
    } else {
      if (
        !Array.isArray(r.frames) ||
        !Array.isArray(r.requests) ||
        !r.config ||
        !Number.isInteger(r.config.capacity_blocks) ||
        r.config.capacity_blocks < 1 ||
        r.config.capacity_blocks > 4096
      ) {
        throw Error("Invalid simulator structure");
      }
      for (const f of r.frames) {
        if (
          !Number.isFinite(f.start) ||
          !Number.isFinite(f.end) ||
          !Array.isArray(f.events) ||
          !Array.isArray(f.plan)
        ) {
          throw Error("Invalid frame");
        }
      }
    }
  }
  return bundle.runs;
}

/**
 * Replace loaded results, reset the cursor, then initialize the selectors.
 */
function load(data) {
  stop();
  runs = data;
  clock = 0;
  $("case-name").value = runs[0].scenario;
  selectScenario();
}

/**
 * Fill the result selector from this single-case recording.
 */
function selectScenario() {
  $("run").innerHTML = runs
    .map(
      (r, i) =>
        /* HTML */ `<option value="${i}">${esc(r.policy)} · ${esc(r.status)}</option>`,
    )
    .join("");
  selectRun();
}

/**
 * Reset request filters and zoom when the primary policy result changes.
 */
function selectRun() {
  stop();
  clock = 0;
  studyWindow = null;
  studySelection = null;
  explorerQuery = "";
  explorerFilter = "all";
  explorerSort = "arrival";
  explorerFollow = true;
  comparisons();
  const r = primary();
  $("request").innerHTML = (r.kind === "measurement" ? [{ id: "request" }] : r.requests)
    .map((r) => /* HTML */ `<option>${esc(r.id)}</option>`)
    .join("");
  render();
}

/**
 * Delegate simulator lanes to explorer.js.
 * The remaining branch supports older single-request measurement recordings.
 */
function timeline(r, t, width, max) {
  if (r.kind !== "measurement") {
    return explorerTimeline(r, t, width, max);
  }
  const left = 48,
    right = 15,
    plot = Math.max(150, width - left - right),
    x = (v) => left + (v / Math.max(max, 0.001)) * plot;
  const reqs = r.kind === "measurement" ? [{ id: "request", arrival: 0 }] : r.requests;
  const height = 52 + reqs.length * 40;
  // Open the SVG here and close it after appending its drawn children below.
  let svg = `<svg
    class="timeline"
    role="img"
    aria-label="Request execution lanes with time in ${esc(r.unit)}"
    viewBox="0 0 ${width} ${height}"
  >`;
  for (let i = 0; i <= 4; i++) {
    let time = (max * i) / 4;
    svg += /* HTML */ `<line
        x1="${x(time)}"
        y1="20"
        x2="${x(time)}"
        y2="${height - 22}"
        stroke="var(--line)"
      /><text
        x="${x(time)}"
        y="14"
        text-anchor="${i === 0 ? "start" : i === 4 ? "end" : "middle"}"
        >${time.toFixed(1)}</text
      >`;
  }
  reqs.forEach((q, i) => {
    const y = 32 + i * 40;
    if (q.id === $("request").value) {
      svg += /* HTML */ `<rect
        x="0"
        y="${y - 2}"
        width="${width}"
        height="32"
        fill="var(--free)"
        stroke="var(--dec)"
      />`;
    }
    svg += /* HTML */ `<text x="2" y="${y + 15}">${esc(q.id)}</text>`;
    if (q.arrival <= t) {
      svg += /* HTML */ `<line
        x1="${x(q.arrival)}"
        y1="${y + 22}"
        x2="${x(Math.min(t, max))}"
        y2="${y + 22}"
        stroke="var(--line)"
      />`;
    }
    if (r.kind !== "measurement") {
      for (const f of r.frames) {
        const a = f.plan.find((a) => a.request_id === q.id);
        if (a && f.start < t) {
          svg += /* HTML */ `<rect
            x="${x(f.start)}"
            y="${y}"
            width="${Math.max(0, x(Math.min(t, f.end)) - x(f.start))}"
            height="18"
            fill="${a.phase === "prefill" ? "var(--pre)" : "var(--dec)"}"
            ><title>
              ${esc(f.id)} ${esc(a.phase)} ${a.tokens} tokens;
              ${fmt(f.start)}–${fmt(f.end)}
            </title></rect
          >`;
        }
      }
    }
    const ts =
      r.kind === "measurement"
        ? r.trace.token_timestamps_s.map((t) => 1000 * t)
        : r.frames.flatMap((f) =>
            f.events
              .filter((e) => e.kind === "output" && e.request_id === q.id)
              .map((e) => e.time),
          );
    for (const time of ts.filter((v) => v <= t + 1e-9)) {
      svg += /* HTML */ `<circle cx="${x(time)}" cy="${y + 22}" r="3" fill="var(--ink)"
        ><title>Output at ${fmt(time)}</title></circle
      >`;
    }
  });
  svg += /* HTML */ `<line
      x1="${x(t)}"
      x2="${x(t)}"
      y1="20"
      y2="${height - 22}"
      stroke="var(--ink)"
      stroke-dasharray="3 3"
    />
    <text x="${width / 2}" y="${height - 2}" text-anchor="middle">
      Elapsed ${esc(r.unit)}
    </text>`;
  svg += "</svg>";
  return svg;
}

/**
 * Render one result's iteration details, block ownership, and events.
 * The memory grid uses the during snapshot: after growth, before release.
 * Final metrics describe the completed run and do not change with the cursor.
 */
function body(r, t) {
  if (r.kind === "measurement") {
    const m = r.metrics;
    return /* HTML */ `<h3>Measured request · complete run</h3>
      <p>
        Prompt ${r.trace.prompt_tokens}; output ${r.trace.output_token_ids.length}; KV
        cache ${r.trace.use_cache ? "on" : "off"}
      </p>
      <table>
        <tr>
          <th>Metric</th>
          <th>Value</th>
        </tr>
        ${[
          "ttft_ms",
          "prefill_ms",
          "mean_tpot_ms",
          "p95_itl_ms",
          "kv_after_prefill_bytes",
          "kv_final_bytes",
        ]
          .map(
            (k) =>
              /* HTML */ `<tr>
                <td>${esc(k)}</td>
                <td>${fmt(m[k])}</td>
              </tr>`,
          )
          .join("")}
      </table>
      <p>
        Retained KV bytes are measured only after prefill and at termination.
        Intermediate memory is not recorded.
      </p>`;
  }
  const f = frameAt(r, t);
  const current = f?.during || f?.admitted_state || f?.after || [];
  const used = current.reduce((sum, s) => sum + s.reserved_blocks, 0);
  const blocks = current.flatMap((s) =>
    Array.from({ length: s.reserved_blocks }, () => s.id),
  );
  const capacity = r.config.capacity_blocks;
  const req = $("request").value;
  const row = r.metrics?.requests.find((s) => s.id === req);
  const state = f?.after?.find((s) => s.id === req);
  let text = /* HTML */ `<div class="iteration-details">
    <p>
      Memory: ${r.config.capacity_blocks} blocks × ${r.config.block_size} slots =
      ${r.config.capacity_blocks * r.config.block_size} slots. Active limit:
      ${r.config.max_active}.
    </p>
    <h3>${f ? esc(f.id) + " · iteration " + f.iteration : "Before first iteration"}</h3>
    <p>
      Selected iteration: ${fmt(f?.start)} to ${fmt(f?.end)}. The plan and events below
      show this whole recorded iteration.
    </p>
    <p>
      Global budget: ${f?.budget_used ?? 0}/${r.config.budget} tokens · duration
      ${fmt(f?.duration)}
    </p>
    <div class="plan">
      ${f?.plan
        .map(
          (a) =>
            /* HTML */ `<button
              data-select-request="${esc(a.request_id)}"
              class="pill ${a.phase === "prefill" ? "pre" : "dec"} ${a.request_id ===
              req
                ? "selected-action"
                : ""}"
            >
              ${esc(a.request_id)}: ${esc(a.phase)} ${a.tokens}
            </button>`,
        )
        .join("") || "No selected work"}
    </div>
  </div>`;
  text += /* HTML */ `<div class="memory-details">
    <h3>KV reservation · recorded iteration end, before release</h3>
    <p>
      ${used}/${capacity} blocks · ${used * r.config.block_size} reserved slots ·
      ${current.reduce((s, r) => s + r.live_positions, 0)} useful positions
    </p>
    <div class="blocks">
      ${Array.from(
        { length: capacity },
        (_, i) =>
          /* HTML */ `<span
            class="block ${i < used ? "used" : ""} ${blocks[i] === req
              ? "selected-block"
              : ""}"
            ${i < used
              ? `data-select-request="${esc(blocks[i])}" tabindex="0" role="button"`
              : ""}
            title="Block slot ${i}: ${esc(blocks[i] || "free")}"
            >${esc(blocks[i] || "·")}</span
          >`,
      ).join("")}
    </div>
    <p class="muted">
      Ownership counts only; tile position is not a physical page address.
    </p>
  </div>`;
  if (r.status === "complete" && row) {
    text += /* HTML */ `<h3>${esc(req)} · final-run metrics</h3>
      <table>
        <tr>
          <th>Metric</th>
          <th>Value</th>
          <th>Goal</th>
        </tr>
        <tr>
          <td>Admission time</td>
          <td>${fmt(row.admission_time)}</td>
          <td>Absolute time</td>
        </tr>
        <tr>
          <td>Admission wait</td>
          <td>${fmt(row.admission_wait)}</td>
          <td>Admission minus arrival</td>
        </tr>
        <tr>
          <td>TTFT</td>
          <td>${fmt(row.ttft)}</td>
          <td>≤ ${fmt(row.ttft_slo)}</td>
        </tr>
        <tr>
          <td>P95 ITL</td>
          <td>${fmt(row.p95_itl)}</td>
          <td>≤ ${fmt(row.itl_slo)}</td>
        </tr>
        <tr>
          <td>Maximum gap</td>
          <td>${fmt(row.max_itl)}</td>
          <td>Diagnostic</td>
        </tr>
      </table>
      <p>Both SLOs: <b>${row.slo_met ? "met" : "not met"}</b></p>
      <details>
        <summary>Output times and gaps</summary>
        <p>
          ${row.token_times
            .map((v, i) => {
              const gap = i ? " (gap " + fmt(v - row.token_times[i - 1]) + ")" : "";
              return `${i + 1}: ${fmt(v)}${gap}`;
            })
            .join(" · ")}
        </p>
      </details>`;
  }
  if (r.status === "complete") {
    text += /* HTML */ `<p>
      Peak resident requests: ${r.metrics.peak_active}. Peak reserved slots:
      ${r.metrics.peak_reserved_slots}.<br />Final throughput:
      ${fmt(r.metrics.throughput)} tokens/unit · goodput: ${fmt(r.metrics.goodput)}
      requests/unit
    </p>`;
  }
  text += /* HTML */ `<details open>
    <summary>Events · current iteration</summary>
    <div class="events">
      ${(f?.events || [])
        .map(
          (e, i) =>
            /* HTML */ `<button
              ${e.request_id
                ? `data-select-request="${esc(e.request_id)}"`
                : "disabled"}
              class="${e.request_id === req ? "selected-event" : ""}"
            >
              <code>${esc(f.id)}:e${i + 1}</code> ${esc(e.kind)}
              ${esc(e.request_id || "")}
              ${esc(e.reason || e.message || "")}${e.kv_before !== undefined
                ? " · blocks " + e.kv_before + " → " + e.kv_after
                : ""}${e.time !== undefined ? " · time " + fmt(e.time) : ""}
            </button>`,
        )
        .join("") || "—"}
    </div>
  </details>`;
  return text;
}

/**
 * Compare final measurements for the SAME selected request across runs.
 */
function renderRequestSummary(chosen) {
  const id = $("request").value;
  $("request-summary").innerHTML = /* HTML */ `<h2>
      Request ${esc(id)}: final measurements
    </h2>
    <p>
      Changing “Request to inspect” changes this table and highlights that request's
      lane. Moving time changes the recorded actions below, but keeps these final
      measurements.
    </p>
    <table>
      <tr>
        <th>Result</th>
        <th>Admission time</th>
        <th>Admission wait</th>
        <th>Time to first token</th>
        <th>P95 output gap</th>
      </tr>
      ${chosen
        .map((r) => {
          const q = r.metrics?.requests.find((x) => x.id === id);
          return /* HTML */ `<tr>
            <td>${esc(r.policy)}</td>
            <td>${fmt(q?.admission_time)}</td>
            <td>${fmt(q?.admission_wait)}</td>
            <td>${fmt(q?.ttft)}</td>
            <td>${fmt(q?.p95_itl)}</td>
          </tr>`;
        })
        .join("")}
    </table>
    <p>Times use synthetic units. A dash means no final measurement is available.</p>`;
}

/**
 * Explain the selected request at the cursor using recorded events.
 * The state label is cursor-dependent; its final SLO outcome is not.
 */
function requestAtTime(r, t) {
  const id = $("request").value,
    q = r.requests?.find((q) => q.id === id);
  if (!q) {
    return "";
  }
  const idx = studyIndex(r),
    a = idx.admitted.get(id) ?? Infinity,
    c = idx.completed.get(id) ?? Infinity,
    f = frameAt(r, t);
  const action = f?.plan.find((a) => a.request_id === id);
  const state =
    t < q.arrival
      ? "Not arrived"
      : t < a
        ? "Waiting for admission"
        : t >= c
          ? "Completed"
          : action
            ? "Selected for " + action.phase + " " + action.tokens
            : "Active, not selected in this iteration";
  const m = idx.metrics.get(id);
  return (
    esc(state) +
    " · " +
    esc(studyFailure(m)) +
    "<br>Arrival " +
    fmt(q.arrival) +
    " · admission " +
    fmt(a) +
    " · final TTFT " +
    fmt(m?.ttft) +
    " / goal " +
    fmt(q.ttft_slo) +
    " · final P95 ITL " +
    fmt(m?.p95_itl) +
    " / goal " +
    fmt(q.itl_slo)
  );
}

/**
 * Redraw every panel from the shared selection, clock, and window.
 * Saving scroll, expansion, and focus avoids jumping back to the top when a
 * student clicks an event. DOM replacement requires rebinding event handlers.
 * DOM means the browser's document object model, i.e. the page's element tree.
 */
function render() {
  // 1. Remember UI position before replacing generated DOM elements.
  const active = document.activeElement,
    focusId = active?.id,
    focusChart = active?.closest?.('[id^="chart-"]')?.id,
    focusKind = active?.classList?.contains("timeline-ruler")
      ? ".timeline-ruler"
      : ".timeline",
    focusGraph = active?.classList?.contains("study-chart") ? active.dataset.key : null;
  const savedScroll = Array.from(
    document.querySelectorAll(".lane-scroll,.events,.related-scroll"),
  ).map((e) => e.scrollTop);
  const expanded = Array.from(document.querySelectorAll("#panels details")).map(
    (e) => e.open,
  );
  const chosen = peers();
  if (!chosen.length) {
    return;
  }
  // 2. Clamp the shared replay cursor and synchronize the time controls.
  const max = Math.max(...chosen.map(endTime), 0.001);
  clock = Math.max(0, Math.min(clock, max));
  explorerEnsureCursor();
  $("seek").max = max;
  $("seek").value = clock.toFixed(3);
  $("time").max = max;
  $("time").value = clock;
  $("clock").textContent = `Elapsed ${fmt(clock)} / ${fmt(max)} ${chosen[0].unit}`;
  // 3. Redraw summaries, graph context, and the request directory.
  renderArena(runs);
  renderRequestSummary(chosen);
  renderStudy(chosen);
  renderExplorer();
  // 4. Rebuild the result panels, then draw lanes at their actual screen width.
  $("panels").innerHTML = chosen
    .map(
      (r, i) =>
        /* HTML */ `<section class="panel">
          <h2>${esc(r.policy)}</h2>
          <div class="${r.status === "error" ? "error" : "muted"}">
            ${esc(
              r.status === "complete"
                ? "Execution completed · resource validation passed"
                : r.error || r.status,
            )}
          </div>
          <div class="request-focus">
            <b>Selected request: ${esc($("request").value)}</b><br />${requestAtTime(
              r,
              clock,
            )}
          </div>
          ${studySelection
            ? '<button class="chart-return">Back to related requests</button>'
            : ""}
          <div id="chart-${i}"></div>
          ${body(r, clock)}
        </section>`,
    )
    .join("");
  chosen.forEach((r, i) => {
    const el = $("chart-" + i);
    el.innerHTML = timeline(r, clock, Math.max(240, el.clientWidth), max);
  });
  document
    .querySelectorAll(".chart-return")
    .forEach(
      (b) =>
        (b.onclick = () => $("chart-inspector").scrollIntoView({ block: "center" })),
    );
  document.querySelectorAll("[data-trace-request]").forEach((el) => {
    const activate = () => {
      stop();
      $("request").value = el.dataset.traceRequest;
      clock = Number(el.dataset.traceTime);
      render();
    };
    el.onclick = activate;
    el.onkeydown = (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        activate();
      }
    };
  });
  // 5. Bind new elements, then restore scroll positions and keyboard focus.
  explorerBind();
  document
    .querySelectorAll(".lane-scroll,.events,.related-scroll")
    .forEach((e, i) => (e.scrollTop = savedScroll[i] || 0));
  document.querySelectorAll("#panels details").forEach((e, i) => {
    if (expanded[i] !== undefined) {
      e.open = expanded[i];
    }
  });
  const focusTarget = focusId
    ? $(focusId)
    : focusChart
      ? $(focusChart)?.querySelector(focusKind)
      : focusGraph
        ? document.querySelector('.study-chart[data-key="' + focusGraph + '"]')
        : null;
  focusTarget?.focus({ preventScroll: true });
  const f = chosen[0].kind === "measurement" ? null : frameAt(chosen[0], clock);
  $("bookmark").value =
    `${chosen[0].scenario} | ${chosen[0].policy} | ${f?.id || "measurement"}` +
    ` | elapsed=${fmt(clock)} | request=${$("request").value}`;
}

/**
 * Move to the next/previous recorded boundary, skipping empty time spans.
 */
function step(direction) {
  const boundaries = [
    ...new Set(
      peers().flatMap((r) =>
        r.kind === "measurement"
          ? [0, ...r.trace.token_timestamps_s.map((t) => 1000 * t)]
          : [0, ...r.frames.flatMap((f) => [f.start, f.end])],
      ),
    ),
  ].sort((a, b) => a - b);
  const next =
    direction > 0
      ? boundaries.find((t) => t > clock + 1e-7)
      : boundaries.filter((t) => t < clock - 1e-7).at(-1);
  if (next === undefined) {
    stop();
    return;
  }
  clock = next;
  render();
}
$("files").addEventListener("change", async (e) => {
  stop();
  try {
    const list = [];
    for (const file of Array.from(e.target.files).slice(0, 1)) {
      if (file.size > 100e6) {
        throw Error("File exceeds 100 MB. Load fewer configurations per recording.");
      }
      list.push(...validate(JSON.parse(await file.text())));
    }
    load(list);
    $("load-status").textContent =
      `${e.target.files[0].name}: ${list.length} simulation results loaded. ` +
      "Select a result above. One file contains one test case.";
  } catch (err) {
    $("load-status").textContent = "Could not load: " + err.message;
  }
});
$("run").onchange = selectRun;
$("request").onchange = () => explorerSelect($("request").value, { fit: true });
$("compare").onchange = render;
$("compare-run").onchange = () => {
  stop();
  render();
};
$("time").oninput = (e) => {
  stop();
  clock = Number(e.target.value);
  render();
};
$("next").onclick = () => {
  stop();
  step(1);
};
$("prev").onclick = () => {
  stop();
  step(-1);
};
$("play").onclick = () => {
  if (timer) {
    stop();
    return;
  }
  $("play").textContent = "Pause";
  timer = setInterval(() => step(1), 800);
};
window.addEventListener("resize", render);
$("load-status").textContent =
  "No recording loaded. Open results/demo/tutorial.json above.";
$("seek").onchange = (e) => {
  stop();
  clock = Math.max(0, Number(e.target.value) || 0);
  render();
};

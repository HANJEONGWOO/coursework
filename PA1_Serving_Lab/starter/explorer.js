/**
 * Request directory and timeline navigation for the offline dashboard.
 *
 * Search and filters determine visible request rows AND matching timeline lanes.
 * Every matching request is retained; vertical scrolling handles large inputs.
 * The selected request can remain visible even when a filter excludes it.
 *
 * clock is the replay cursor. studyWindow is the independent visible time range.
 * Zoom/pan changes the window; selecting a bar also changes request and cursor.
 * All views share these values and update through visualizer.js render().
 *
 * Reading route: explorerRequests -> explorerTimeline -> explorerBind.
 * The bind function connects rendered SVG elements to pointer/keyboard actions.
 * SVG means Scalable Vector Graphics: coordinates here describe drawn lanes.
 */

/* Shared selection and time-window controls. All processing stays in the browser. */
var explorerQuery = "",
  explorerFilter = "all",
  explorerSort = "arrival",
  explorerFollow = true;
/**
 * Use a common full-run extent so comparison panes share one time window.
 */
function explorerEnd() {
  return Math.max(0.001, ...peers().map(endTime));
}

/**
 * Resolve null to the full extent without changing stored window state.
 */
function explorerWindow() {
  return studyWindow || { start: 0, end: explorerEnd() };
}

/**
 * Clamp both width and start so the visible window stays inside the run.
 */
function explorerSetWindow(start, span) {
  const end = explorerEnd();
  span = Math.max(Math.min(0.01, end), Math.min(end, span));
  start = Math.max(0, Math.min(end - span, start));
  studyWindow = { start, end: start + span };
}

/**
 * Scale the visible duration around an anchor, preserving its relative spot.
 * A factor below one zooms in. Manual navigation disables cursor following.
 */
function explorerZoom(factor, anchor = clock) {
  explorerFollow = false;
  const w = explorerWindow(),
    span = w.end - w.start;
  anchor = Math.max(w.start, Math.min(w.end, anchor));
  explorerSetWindow(anchor - (anchor - w.start) * factor, span * factor);
  render();
}

/**
 * Shift by a fraction of the current window width, then redraw.
 */
function explorerPan(fraction) {
  explorerFollow = false;
  const w = explorerWindow();
  explorerSetWindow(w.start + (w.end - w.start) * fraction, w.end - w.start);
  render();
}

/**
 * Recenter only when following is enabled and the cursor leaves the window.
 */
function explorerEnsureCursor() {
  if (!studyWindow || !explorerFollow) {
    return;
  }
  const w = explorerWindow();
  if (clock < w.start || clock > w.end) {
    explorerSetWindow(clock - (w.end - w.start) / 2, w.end - w.start);
  }
}

/**
 * Fit arrival through completion, with a small margin on both sides.
 */
function explorerFitRequest(id = $("request").value) {
  explorerFollow = false;
  const r = primary(),
    idx = studyIndex(r),
    q = idx.inputs.get(id);
  if (!q) {
    return;
  }
  const end =
    idx.completed.get(id) ?? idx.metrics.get(id)?.token_times?.at(-1) ?? endTime(r);
  const span = Math.max(1, end - q.arrival),
    pad = span * 0.12;
  explorerSetWindow(q.arrival - pad, span + pad * 2);
}

/**
 * Select one request everywhere, optionally seeking or fitting its lifetime.
 * Call this from row, bar, block, and event clicks to keep views synchronized.
 */
function explorerSelect(id, options = {}) {
  const r = primary();
  if (!r?.requests?.some((q) => q.id === id)) {
    return;
  }
  stop();
  $("request").value = id;
  if (Number.isFinite(options.time)) {
    clock = options.time;
  }
  if (options.fit) {
    explorerFitRequest(id);
    clock = studyIndex(r).inputs.get(id).arrival;
  }
  render();
  explorerReveal(id);
  if (options.scroll) {
    $("timeline-tools").scrollIntoView({ block: "start", behavior: "smooth" });
  }
}

/**
 * Scroll the selected lane and table row into their own visible containers.
 */
function explorerReveal(id) {
  document.querySelectorAll(".lane-scroll").forEach((el) => {
    const row = Array.from(el.querySelectorAll("[data-lane-id]")).find(
      (e) => e.dataset.laneId === id,
    );
    if (row) {
      const y = Number(row.dataset.row) * 32;
      if (y < el.scrollTop || y + 32 > el.scrollTop + el.clientHeight) {
        el.scrollTop = Math.max(0, y - el.clientHeight / 3);
      }
    }
  });
  const row = Array.from(document.querySelectorAll("#request-list tbody tr")).find(
      (e) => e.dataset.id === id,
    ),
    sc = $("request-list-scroll");
  if (row && sc) {
    const top = row.offsetTop;
    if (top < sc.scrollTop || top + row.offsetHeight > sc.scrollTop + sc.clientHeight) {
      sc.scrollTop = Math.max(0, top - sc.clientHeight / 3);
    }
  }
}

/**
 * Determine request state at the cursor from arrival/admission/release times.
 */
function explorerState(q, idx) {
  if (clock < q.arrival) {
    return "Not arrived";
  }
  if (clock < (idx.admitted.get(q.id) ?? Infinity)) {
    return "Waiting";
  }
  if (clock >= (idx.completed.get(q.id) ?? Infinity)) {
    return "Completed";
  }
  return "Active";
}

/**
 * Apply shared search/filter/sort rules without truncating the request list.
 */
function explorerRequests() {
  const r = primary(),
    idx = studyIndex(r),
    w = explorerWindow(),
    related = new Set(studySelection?.rows.map((q) => q.id) || []);
  const rows = r.requests.filter((q) => {
    if (!q.id.toLowerCase().includes(explorerQuery.toLowerCase())) {
      return false;
    }
    const state = explorerState(q, idx),
      m = idx.metrics.get(q.id);
    return (
      explorerFilter === "all" ||
      (explorerFilter === "window" &&
        q.arrival <= w.end &&
        (idx.completed.get(q.id) ?? Infinity) >= w.start) ||
      (explorerFilter === "failed" && m && !m.slo_met) ||
      (explorerFilter === "related" && related.has(q.id)) ||
      explorerFilter === state.toLowerCase()
    );
  });
  rows.sort((a, b) =>
    explorerSort === "id"
      ? a.id.localeCompare(b.id)
      : explorerSort === "ttft"
        ? (idx.metrics.get(b.id)?.ttft ?? -1) - (idx.metrics.get(a.id)?.ttft ?? -1)
        : explorerSort === "itl"
          ? (idx.metrics.get(b.id)?.p95_itl ?? -1) -
            (idx.metrics.get(a.id)?.p95_itl ?? -1)
          : a.arrival - b.arrival || a.id.localeCompare(b.id),
  );
  return rows;
}

/**
 * Populate all matching rows with cursor state and separate final metrics.
 */
function explorerRows() {
  const r = primary(),
    idx = studyIndex(r),
    rows = explorerRequests();
  $("request-count").textContent =
    `${rows.length} of ${r.requests.length} requests · ` +
    "every matching request is listed and has a timeline lane";
  $("request-list-body").innerHTML =
    rows
      .map((q) => {
        const m = idx.metrics.get(q.id);
        return /* HTML */ `<tr
          data-id="${esc(q.id)}"
          class="${q.id === $("request").value ? "related-selected" : ""}"
        >
          <td><button data-pick-request="${esc(q.id)}">${esc(q.id)}</button></td>
          <td>${explorerState(q, idx)}</td>
          <td>${fmt(q.arrival)}</td>
          <td>${fmt(m?.ttft)}</td>
          <td>${fmt(m?.p95_itl)}</td>
          <td class="${m && !m.slo_met ? "error" : ""}">${esc(studyFailure(m))}</td>
        </tr>`;
      })
      .join("") ||
    '<tr><td colspan="6">No matching requests. Clear the search ' +
      "or choose All requests.</td></tr>";
  $("request-list-body")
    .querySelectorAll("[data-pick-request]")
    .forEach(
      (b) =>
        (b.onclick = () =>
          explorerSelect(b.dataset.pickRequest, { fit: true, scroll: true })),
    );
}

/**
 * Render request controls and time-window navigation.
 * Keep the search input element alive while typing. Recreate time controls
 * from the current window so their values agree with zoom and pan.
 */
function renderExplorer() {
  const r = primary(),
    el = $("request-explorer"),
    nav = $("timeline-tools");
  el.hidden = nav.hidden = r.kind === "measurement";
  if (el.hidden) {
    return;
  }
  const w = explorerWindow(),
    total = explorerEnd(),
    span = w.end - w.start;
  if (!el.children.length) {
    el.innerHTML = /* HTML */ `<h2>All requests</h2>
      <p>
        Select a request to fit its full lifetime in the timeline. Search and filters
        also control which lanes appear below.
      </p>
      <div class="toolbar">
        <label
          >Find request<input
            id="request-search"
            type="search"
            placeholder="Request ID" /></label
        ><label
          >Show<select id="request-filter">
            <option value="all">All requests</option>
            <option value="window">Overlapping visible time</option>
            <option value="waiting">Waiting at cursor</option>
            <option value="active">Active at cursor</option>
            <option value="completed">Completed at cursor</option>
            <option value="failed">Final SLO missed</option>
            <option value="related">Related to selected graph</option>
          </select></label
        ><label
          >Sort<select id="request-sort">
            <option value="arrival">Arrival time</option>
            <option value="id">Request ID</option>
            <option value="ttft">Highest TTFT first</option>
            <option value="itl">Highest P95 ITL first</option>
          </select></label
        ><button id="request-clear">Show all / clear search</button>
      </div>
      <p id="request-count" role="status"></p>
      <div id="request-list-scroll" class="request-table-scroll">
        <table id="request-list">
          <thead>
            <tr>
              <th>Request</th>
              <th>At cursor</th>
              <th>Arrival</th>
              <th>Final TTFT</th>
              <th>Final P95 ITL</th>
              <th>Final outcome</th>
            </tr>
          </thead>
          <tbody id="request-list-body"></tbody>
        </table>
      </div>`;
    $("request-search").oninput = (e) => {
      explorerQuery = e.target.value;
      render();
    };
    $("request-filter").onchange = (e) => {
      explorerFilter = e.target.value;
      render();
    };
    $("request-sort").onchange = (e) => {
      explorerSort = e.target.value;
      render();
    };
    $("request-clear").onclick = () => {
      explorerQuery = "";
      explorerFilter = "all";
      render();
    };
  }
  $("request-search").value = explorerQuery;
  $("request-filter").value = explorerFilter;
  $("request-sort").value = explorerSort;
  explorerRows();
  nav.innerHTML = /* HTML */ `<h2>
      Timeline navigation
      <span class="muted"
        >${fmt(w.start)}–${fmt(w.end)} · ${(total / span).toFixed(1)}×</span
      >
    </h2>
    <div class="toolbar">
      <button id="zoom-in">+ Zoom in</button><button id="zoom-out">− Zoom out</button
      ><button id="pan-left">← Earlier</button><button id="pan-right">Later →</button
      ><button id="fit-run">Fit full run</button
      ><button id="fit-request">Fit selected request</button
      ><button id="center-cursor">Center cursor</button
      ><label
        ><span
          ><input
            id="follow-cursor"
            type="checkbox"
            ${explorerFollow ? "checked" : ""}
          />
          Follow cursor</span
        ></label
      >
    </div>
    <div class="toolbar">
      <button id="local-prev">Previous boundary</button
      ><button id="local-next">Next boundary</button
      ><button data-jump="arrival">Arrival</button
      ><button data-jump="first">First output</button
      ><button data-jump="gap">Largest output gap</button
      ><button data-jump="completion">Completion</button>
    </div>
    <div class="toolbar">
      <label
        >Window start<input
          id="window-start"
          type="number"
          step="0.001"
          min="0"
          value="${w.start.toFixed(3)}" /></label
      ><label
        >Window end<input
          id="window-end"
          type="number"
          step="0.001"
          max="${total}"
          value="${w.end.toFixed(3)}" /></label
      ><button id="apply-window">Apply range</button
      ><span id="window-message" role="status"></span>
    </div>
    <label
      >Move visible window across the full recording<input
        id="window-pan"
        type="range"
        min="0"
        max="${Math.max(0, total - span)}"
        step="any"
        value="${w.start}"
        ${span >= total ? "disabled" : ""}
    /></label>
    <p class="muted">
      All recorded actions remain visible. The dashed cursor selects the iteration
      details. Click a bar, token, request name, memory block, or event to connect them.
      Drag a lane to pan; Ctrl + wheel zooms at the pointer; Shift + wheel pans. Use + /
      − and arrow keys when a timeline has focus.
    </p>`;
  $("zoom-in").onclick = () => explorerZoom(0.5);
  $("zoom-out").onclick = () => explorerZoom(2);
  $("pan-left").onclick = () => explorerPan(-0.5);
  $("pan-right").onclick = () => explorerPan(0.5);
  $("fit-run").onclick = () => {
    studyWindow = null;
    render();
  };
  $("fit-request").onclick = () => {
    explorerFitRequest();
    render();
    explorerReveal($("request").value);
  };
  $("center-cursor").onclick = () => {
    const v = explorerWindow();
    explorerSetWindow(clock - (v.end - v.start) / 2, v.end - v.start);
    render();
  };
  $("follow-cursor").onchange = (e) => (explorerFollow = e.target.checked);
  $("local-prev").onclick = () => {
    stop();
    step(-1);
  };
  $("local-next").onclick = () => {
    stop();
    step(1);
  };
  $("window-pan").onchange = (e) => {
    explorerSetWindow(Number(e.target.value), span);
    explorerFollow = false;
    render();
  };
  $("apply-window").onclick = () => {
    const a = Number($("window-start").value),
      b = Number($("window-end").value);
    if (!Number.isFinite(a) || !Number.isFinite(b) || b <= a || a < 0 || b > total) {
      $("window-message").textContent = "Use 0 ≤ start < end ≤ " + fmt(total);
      return;
    }
    explorerSetWindow(a, b - a);
    explorerFollow = false;
    render();
  };
  nav
    .querySelectorAll("[data-jump]")
    .forEach((b) => (b.onclick = () => explorerJump(b.dataset.jump)));
}

/**
 * Seek to one recorded milestone for the selected request.
 * For output/completion, select the iteration producing the event. For a gap,
 * also fit the interval between its two outputs so the delay is visible.
 */
function explorerJump(kind) {
  const r = primary(),
    id = $("request").value,
    idx = studyIndex(r),
    q = idx.inputs.get(id),
    times = idx.metrics.get(id)?.token_times || idx.outputs.get(id) || [];
  if (!q) {
    return;
  }
  let time =
    kind === "arrival"
      ? q.arrival
      : kind === "completion"
        ? idx.completed.get(id)
        : times[0];
  if (kind === "gap" && times.length > 1) {
    let i = 1;
    for (let j = 2; j < times.length; j++) {
      if (times[j] - times[j - 1] > times[i] - times[i - 1]) {
        i = j;
      }
    }
    explorerSetWindow(
      times[i - 1] - (times[i] - times[i - 1]) * 0.2,
      Math.max(0.1, (times[i] - times[i - 1]) * 1.4),
    );
    time = times[i];
  }
  if (!Number.isFinite(time)) {
    $("window-message").textContent = "That event was not recorded for this request.";
    return;
  }
  if (kind === "first" || kind === "gap" || kind === "completion") {
    const f = r.frames.find((f) =>
      f.events.some(
        (e) =>
          e.request_id === id &&
          ((e.kind === "output" && Math.abs(e.time - time) < 1e-7) ||
            (kind === "completion" &&
              e.kind === "free" &&
              Math.abs(f.end - time) < 1e-7)),
      ),
    );
    if (f) {
      time = Math.max(f.start, f.end - 1e-7);
    }
  }
  clock = time;
  explorerEnsureCursor();
  explorerSelect(id, { time });
}

/**
 * Draw all filtered request lanes over the visible time interval.
 * Clip work to window boundaries, not to the replay cursor. This makes later
 * recorded work visible while the cursor controls the detail panels.
 */
function explorerTimeline(r, t, width, max) {
  const w = explorerWindow(),
    lo = w.start,
    hi = w.end,
    left = 126,
    right = 12,
    plot = width - left - right,
    x = (v) => left + ((v - lo) / (hi - lo)) * plot,
    idx = studyIndex(r);
  const ids = explorerRequests().map((q) => q.id),
    selected = $("request").value;
  // A selected request remains inspectable even when a filter excludes it.
  if (!ids.includes(selected) && idx.inputs.has(selected)) {
    ids.unshift(selected);
  }
  const reqs = ids.map((id) => idx.inputs.get(id)).filter(Boolean),
    h = Math.max(40, reqs.length * 32);
  const ruler = /* HTML */ `<svg
    class="timeline-ruler"
    viewBox="0 0 ${width} 30"
    data-left="${left}"
    data-start="${lo}"
    data-end="${hi}"
    data-plot="${plot}"
    tabindex="0"
    role="group"
    aria-label="Time ruler. Click to seek."
  >
    ${[0, 1, 2, 3, 4]
      .map(
        (i) =>
          /* HTML */ `<text
            x="${x(lo + ((hi - lo) * i) / 4)}"
            y="20"
            text-anchor="${i === 0 ? "start" : i === 4 ? "end" : "middle"}"
            >${(lo + ((hi - lo) * i) / 4).toFixed(2)}</text
          >`,
      )
      .join("")}
  </svg>`;
  // This fragment deliberately stays open until every lane has been added.
  // Do not mark it as embedded HTML: formatters may auto-close an open SVG.
  let out = `<svg
    class="timeline"
    style="height:${h}px;min-height:0"
    preserveAspectRatio="none"
    viewBox="0 0 ${width} ${h}"
    data-left="${left}"
    data-start="${lo}"
    data-end="${hi}"
    data-plot="${plot}"
    tabindex="0"
    role="group"
    aria-label="All matching request lanes. Drag to pan, plus or minus to zoom."
  >`;
  reqs.forEach((q, i) => {
    const y = i * 32;
    // Keep the group open so bars and output dots belong to this request row.
    out += `<g data-lane-id="${esc(q.id)}" data-row="${i}"
      ><rect
        x="0"
        y="${y}"
        width="${width}"
        height="31"
        fill="${q.id === selected ? "var(--free)" : "var(--panel)"}"
        ${q.id === selected ? 'stroke="var(--dec)"' : ""}
      /><text
        class="lane-name"
        data-select-request="${esc(q.id)}"
        x="4"
        y="${y + 18}"
        tabindex="0"
        role="button"
        >${esc(q.id)}</text
      >`;
    const arrival = Math.max(lo, q.arrival),
      end = Math.min(hi, idx.completed.get(q.id) ?? endTime(r));
    if (end >= arrival) {
      out += /* HTML */ `<line
        x1="${x(arrival)}"
        x2="${x(end)}"
        y1="${y + 26}"
        y2="${y + 26}"
        stroke="var(--line)"
        stroke-width="2"
      />`;
    }
    for (const { frame: f, action: a } of idx.actions.get(q.id) || []) {
      if (f.end <= lo || f.start >= hi) {
        continue;
      }
      const a0 = Math.max(lo, f.start),
        a1 = Math.min(hi, f.end);
      out += /* HTML */ `<rect
        x="${x(a0)}"
        y="${y + 3}"
        width="${Math.max(0.7, x(a1) - x(a0))}"
        height="15"
        data-trace-request="${esc(q.id)}"
        data-trace-time="${(a0 + a1) / 2}"
        tabindex="0"
        role="button"
        aria-label="${esc(q.id)} ${esc(a.phase)} ${fmt(f.start)} to ${fmt(f.end)}"
        fill="${a.phase === "prefill" ? "var(--pre)" : "var(--dec)"}"
        ><title>
          ${esc(q.id)} · ${esc(f.id)} · ${esc(a.phase)} ${a.tokens} ·
          ${fmt(f.start)}–${fmt(f.end)}
        </title></rect
      >`;
    }
    for (const v of idx.metrics.get(q.id)?.token_times || idx.outputs.get(q.id) || []) {
      if (v >= lo && v <= hi) {
        out += /* HTML */ `<circle
          cx="${x(v)}"
          cy="${y + 25}"
          r="3"
          fill="var(--ink)"
          data-trace-request="${esc(q.id)}"
          data-trace-time="${Math.max(0, v - 1e-7)}"
          tabindex="0"
          role="button"
          aria-label="${esc(q.id)} output at ${fmt(v)}"
          ><title>Output at ${fmt(v)}</title></circle
        >`;
      }
    }
    out += "</g>";
  });
  if (t >= lo && t <= hi) {
    out += /* HTML */ `<line
      x1="${x(t)}"
      x2="${x(t)}"
      y1="0"
      y2="${h}"
      stroke="var(--ink)"
      stroke-dasharray="4 3"
      pointer-events="none"
    />`;
  }
  out += "</svg>";
  const retained = ids.length > explorerRequests().length;
  return /* HTML */ `<p class="lane-count">
      ${reqs.length} lanes · scroll vertically to see every
      request${retained ? " · selected request retained outside filter" : ""}
    </p>
    ${ruler}
    <div class="lane-scroll">${out}</div>`;
}

/**
 * Connect the current SVG/HTML elements to selection and navigation.
 * SVG coordinates differ from CSS pixels; convert pointer positions through
 * the viewBox before computing a time. A drag pans; a short click selects.
 */
function explorerBind() {
  document.querySelectorAll(".timeline-ruler").forEach((el) => {
    el.style.width = el.parentElement.querySelector(".lane-scroll").clientWidth + "px";
  });
  const activate = (el) => {
    const id = el.dataset.selectRequest || el.dataset.traceRequest;
    explorerSelect(id, {
      time: el.hasAttribute("data-trace-time") ? Number(el.dataset.traceTime) : clock,
    });
  };
  document
    .querySelectorAll("[data-trace-request],[data-select-request]")
    .forEach((el) => {
      el.onclick = (e) => {
        e.stopPropagation();
        activate(el);
      };
      el.onkeydown = (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          e.stopPropagation();
          activate(el);
        }
      };
    });
  document
    .querySelectorAll("svg.timeline[data-start],.timeline-ruler")
    .forEach((svg) => {
      const toTime = (e) => {
        const b = svg.getBoundingClientRect(),
          px = ((e.clientX - b.left) / b.width) * svg.viewBox.baseVal.width;
        return Math.max(
          Number(svg.dataset.start),
          Math.min(
            Number(svg.dataset.end),
            Number(svg.dataset.start) +
              ((px - Number(svg.dataset.left)) / Number(svg.dataset.plot)) *
                (Number(svg.dataset.end) - Number(svg.dataset.start)),
          ),
        );
      };
      svg.onwheel = (e) => {
        if (!e.ctrlKey && !e.shiftKey) {
          return;
        }
        e.preventDefault();
        explorerFollow = false;
        if (e.ctrlKey) {
          explorerZoom(e.deltaY > 0 ? 1.3 : 1 / 1.3, toTime(e));
        } else {
          explorerPan((e.deltaY || e.deltaX) > 0 ? 0.2 : -0.2);
        }
      };
      svg.onkeydown = (e) => {
        if (["+", "=", "-", "ArrowLeft", "ArrowRight"].includes(e.key)) {
          e.preventDefault();
          explorerFollow = false;
          if (e.key === "-") {
            explorerZoom(2);
          } else if (e.key === "+" || e.key === "=") {
            explorerZoom(0.5);
          } else {
            explorerPan(e.key === "ArrowLeft" ? -0.25 : 0.25);
          }
        }
      };
      let drag = null;
      svg.onpointerdown = (e) => {
        if (e.button !== 0) {
          return;
        }
        drag = {
          x: e.clientX,
          start: explorerWindow().start,
          span: explorerWindow().end - explorerWindow().start,
          target: e.target,
          time: toTime(e),
        };
        svg.setPointerCapture(e.pointerId);
      };
      svg.onpointerup = (e) => {
        if (!drag) {
          return;
        }
        const d = drag;
        drag = null;
        const dx = e.clientX - d.x;
        if (Math.abs(dx) > 5) {
          e.preventDefault();
          explorerFollow = false;
          explorerSetWindow(
            d.start -
              (((dx / svg.getBoundingClientRect().width) * svg.viewBox.baseVal.width) /
                Number(svg.dataset.plot)) *
                d.span,
            d.span,
          );
          render();
        } else if (!d.target.closest("[data-trace-request],[data-select-request]")) {
          stop();
          clock = d.time;
          const lane = d.target.closest("[data-lane-id]");
          if (lane) {
            $("request").value = lane.dataset.laneId;
          }
          render();
        } else {
          activate(d.target.closest("[data-trace-request],[data-select-request]"));
        }
      };
      svg.onpointercancel = () => {
        drag = null;
      };
    });
}

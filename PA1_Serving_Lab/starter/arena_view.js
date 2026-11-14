/**
 * Case-score panel for the currently loaded recording.
 *
 * This is a display of the saved score, not a second scoring implementation.
 * The Python runner writes full-practice totals to a separate scoreboard.html.
 * The panel stays hidden when a basic-mission recording has no competition data.
 */

/* One case per recording. Overall practice totals live in scoreboard.html. */
/**
 * Show the saved case score when competition metadata is present.
 */
function renderArena(all) {
  const records = all.filter((r) => r.competition),
    el = $("arena-board");
  el.hidden = !records.length;
  if (!records.length) {
    return;
  }
  const r = records[0],
    c = r.competition;
  el.innerHTML = /* HTML */ `<div class="panel">
    <h2>Case score: ${c.score.toFixed(2)} / 200</h2>
    <p><b>${esc(r.scenario)}</b> · ${esc(r.status)}. Reference: 100.00.</p>
    <div style="overflow-x:auto">
      <table>
        <tr>
          <th>Your balanced goodput</th>
          <th>Reference goodput</th>
          <th>Case score</th>
        </tr>
        <tr>
          <td>${fmt(c.balanced_goodput)}</td>
          <td>${fmt(c.reference_goodput)}</td>
          <td>${c.score.toFixed(2)}</td>
        </tr>
      </table>
    </div>
    <p>
      This file contains one test case. Open scoreboard.html in the results folder for
      the full practice score. Load a different JSON to inspect another case.
    </p>
  </div>`;
}

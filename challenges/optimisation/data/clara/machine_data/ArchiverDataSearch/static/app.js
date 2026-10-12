"use strict";

/*
 * Single-page front end for the search API.
 *
 * The UI holds no query state of its own: it builds the same flat parameter
 * form that GET ./api/search accepts, puts it in the URL hash, and reads it back
 * on load. That makes every view a shareable link and keeps the browser's back
 * button working, with no framework involved.
 */

const OPS = [
  { value: "between", label: "between", arity: 2 },
  { value: "outside", label: "outside", arity: 2 },
  { value: "gt", label: ">", arity: 1 },
  { value: "gte", label: "≥", arity: 1 },
  { value: "lt", label: "<", arity: 1 },
  { value: "lte", label: "≤", arity: 1 },
  { value: "eq", label: "=", arity: 1 },
  { value: "ne", label: "≠", arity: 1 },
  { value: "in", label: "one of", arity: -1 },
  { value: "isnan", label: "is missing", arity: 0 },
  { value: "notnan", label: "is present", arity: 0 },
];

const PALETTE = [
  "#4f9cf9", "#f2a65a", "#68d391", "#f16d9a",
  "#b58cf5", "#57c7d4", "#e5d95f", "#ff8d6b",
];

/** What the time inputs sit at until the user says otherwise. */
const MIDNIGHT = "00:00";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  pvs: [],
  pvByName: new Map(),
  cameras: [],
  selectPvs: [],
  plotPvs: [],
  offset: 0,
  sort: "timestamp",
  order: "asc",
  lastRows: [],
  incompleteFilters: [],
  incompleteTimes: [],
  chart: null,
};

// --- helpers -----------------------------------------------------------------

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (value !== null && value !== undefined) node.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined) continue;
    node.append(child.nodeType ? child : document.createTextNode(child));
  }
  return node;
}

/** Read a `<date> <time>` pair as an ISO instant, or null if no date is set.
 *
 * The inputs have no timezone and the archive is UTC throughout. The time
 * shows 00:00 from the start, so a date on its own is already a usable bound;
 * the fallback here is only for a user who clears the field outright.
 */
function readInstant(prefix) {
  const date = $(`#${prefix}-date`).value;
  if (!date) return null;
  const time = $(`#${prefix}-time`).value || MIDNIGHT;
  // The control works to the minute; a link carrying seconds keeps them.
  return `${date}T${time.length === 5 ? `${time}:00` : time}Z`;
}

function writeInstant(prefix, iso) {
  const [date, clock = ""] = (iso || "").replace("Z", "").split("T");
  $(`#${prefix}-date`).value = date || "";
  // Never leave the control blank: the time half is always a real value, so
  // the form always shows exactly the instant a search would use.
  $(`#${prefix}-time`).value = clock.slice(0, 8) || MIDNIGHT;
}

/** True when the user has half-typed a date — the browser then reports "". */
function dateIsHalfTyped(prefix) {
  const input = $(`#${prefix}-date`);
  return !input.value && input.validity.badInput;
}

function formatNumber(value) {
  if (value === null || value === undefined) return "—";
  if (!Number.isFinite(value)) return "—";
  return Math.abs(value) >= 1e5 || (value !== 0 && Math.abs(value) < 1e-3)
    ? value.toExponential(4)
    : value.toFixed(4).replace(/\.?0+$/, "");
}

// How images are addressed inside a capture file: 1 -> "Capture000001".
function imageDataset(index) {
  return `Capture${String(index).padStart(6, "0")}`;
}

async function getJson(url) {
  const response = await fetch(url);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `HTTP ${response.status}`);
  return body;
}

// --- query <-> URL -----------------------------------------------------------

/** Collect the whole form into URLSearchParams, matching GET ./api/search. */
function buildParams({ includePaging = true } = {}) {
  const params = new URLSearchParams();
  const mode = $("#mode").value;
  // A date the browser cannot parse yet is flagged for the same reason an
  // incomplete filter is: searching all time while the user is looking at a
  // date they typed is the wrong kind of silence.
  state.incompleteTimes = [];
  for (const [prefix, label] of [["from", "From"], ["to", "To"]]) {
    const bound = $(`#${prefix}-date`).closest(".field");
    const bad = mode !== "all" && (mode === "absolute" || prefix === "from")
      && dateIsHalfTyped(prefix);
    bound.classList.toggle("incomplete", bad);
    if (bad) state.incompleteTimes.push(label);
  }
  if (mode !== "all") {
    const from = readInstant("from");
    if (from) params.set("from", from);
    if (mode === "absolute") {
      const to = readInstant("to");
      if (to) params.set("to", to);
    } else if ($("#days").value) {
      params.set("days", $("#days").value);
    }
  }

  for (const box of $$("#cameras input:checked")) params.append("camera", box.value);

  // A row with no PV is an empty template and is ignored; a row naming a PV but
  // missing values is flagged, because dropping it quietly would show the user
  // an unfiltered result set that looks like a filtered one.
  state.incompleteFilters = [];
  for (const row of $$(".filter-row")) {
    const pv = row.querySelector(".pv").value.trim();
    row.classList.remove("incomplete");
    if (!pv) continue;
    const op = row.querySelector(".op").value;
    const arity = OPS.find((o) => o.value === op).arity;
    const values = Array.from(row.querySelectorAll(".values input"))
      .map((input) => input.value.trim())
      .filter((v) => v !== "");
    if ((arity > 0 && values.length < arity) || (arity === -1 && !values.length)) {
      row.classList.add("incomplete");
      state.incompleteFilters.push(pv);
      continue;
    }
    const key = row.querySelector(".match")?.value === "all" ? "fall" : "f";
    params.append(key, [pv, op, ...values].join("|"));
  }

  params.set("combine", $("#combine").value);
  for (const pv of state.selectPvs) params.append("select", pv);

  if (includePaging) {
    params.set("sort", state.sort);
    params.set("order", state.order);
    params.set("limit", $("#limit").value);
    params.set("offset", String(state.offset));
  }
  return params;
}

function syncUrl(params) {
  const next = `#${params.toString()}`;
  if (location.hash !== next) history.replaceState(null, "", next);
}

/** Rebuild the form from a hash produced by buildParams. */
function applyParams(params) {
  const from = params.get("from");
  const to = params.get("to");
  const days = params.get("days");
  $("#mode").value = days ? "following" : to || from ? "absolute" : "all";
  writeInstant("from", from);
  writeInstant("to", to);
  if (days) $("#days").value = days;
  updateModeVisibility();

  const cameras = new Set(params.getAll("camera"));
  for (const box of $$("#cameras input")) box.checked = cameras.has(box.value);

  $("#filters").replaceChildren();
  for (const token of params.getAll("f")) addFilterRow(parseToken(token, "any"));
  for (const token of params.getAll("fall")) addFilterRow(parseToken(token, "all"));
  if (!$$(".filter-row").length) addFilterRow();

  $("#combine").value = params.get("combine") || "and";
  state.selectPvs = params.getAll("select");
  renderSelectChips();

  state.sort = params.get("sort") || "timestamp";
  state.order = params.get("order") || "asc";
  state.offset = Number(params.get("offset") || 0);
  if (params.get("limit")) $("#limit").value = params.get("limit");
}

function parseToken(token, match) {
  const [pv, op, ...values] = token.split("|");
  return { pv, op, values, match };
}

// --- filter rows -------------------------------------------------------------

function addFilterRow(initial = {}) {
  const pv = el("input", {
    class: "pv",
    list: "pv-list",
    placeholder: "PV name or glob, e.g. CLA-S04-MAG-QUAD-01:SETI",
    value: initial.pv || "",
  });
  const op = el(
    "select",
    { class: "op" },
    ...OPS.map((o) =>
      el("option", { value: o.value, ...(initial.op === o.value ? { selected: "" } : {}) }, o.label),
    ),
  );
  const values = el("div", { class: "values" });
  const match = el(
    "select",
    { class: "match", title: "how a glob matching several PVs is combined" },
    el("option", { value: "any" }, "any PV"),
    el("option", { value: "all", ...(initial.match === "all" ? { selected: "" } : {}) }, "every PV"),
  );
  const hint = el("div", { class: "hint" });
  const row = el(
    "div",
    { class: "filter-row" },
    pv,
    el("button", {
      class: "remove",
      type: "button",
      title: "remove",
      onclick: () => row.remove(),
    }, "×"),
    op,
    values,
    match,
    hint,
  );

  const refresh = () => {
    const arity = OPS.find((o) => o.value === op.value).arity;
    const existing = Array.from(values.querySelectorAll("input")).map((i) => i.value);
    const wanted = arity === 0 ? 0 : arity === -1 ? Math.max(existing.length, 1) : arity;
    values.replaceChildren();
    for (let i = 0; i < wanted; i += 1) {
      values.append(
        el("input", {
          type: "text",
          inputmode: "decimal",
          placeholder: arity === 2 ? (i === 0 ? "min" : "max") : "value",
          value: existing[i] ?? initial.values?.[i] ?? "",
        }),
      );
    }
    if (arity === -1) {
      values.append(
        el("button", {
          type: "button",
          class: "secondary",
          title: "add another value",
          onclick: () => { values.append(el("input", { type: "text", inputmode: "decimal" })); },
        }, "+"),
      );
    }
    const info = state.pvByName.get(pv.value.trim());
    const isGlob = /[*?[]/.test(pv.value);
    match.classList.toggle("hidden", !isGlob);
    hint.textContent = info
      ? `range in index: ${formatNumber(info.min)} … ${formatNumber(info.max)}`
      : isGlob
        ? "glob — matches several PVs"
        : "";
  };

  op.addEventListener("change", refresh);
  pv.addEventListener("change", refresh);
  pv.addEventListener("input", refresh);
  $("#filters").append(row);
  refresh();
  return row;
}

// --- rendering ---------------------------------------------------------------

function renderChips(target, items, onRemove) {
  const host = $(target);
  host.replaceChildren(
    ...items.map((item) =>
      el("span", { class: "chip" }, item,
        el("button", { type: "button", title: "remove", onclick: () => onRemove(item) }, "×")),
    ),
  );
}

function renderSelectChips() {
  renderChips("#select-pvs", state.selectPvs, (pv) => {
    state.selectPvs = state.selectPvs.filter((p) => p !== pv);
    renderSelectChips();
  });
}

function renderPlotChips() {
  renderChips("#plot-pvs", state.plotPvs, (pv) => {
    state.plotPvs = state.plotPvs.filter((p) => p !== pv);
    renderPlotChips();
  });
}

function renderCameras(cameras) {
  const byArea = new Map();
  for (const camera of cameras) {
    const area = camera.area || "other";
    if (!byArea.has(area)) byArea.set(area, []);
    byArea.get(area).push(camera);
  }
  const host = $("#cameras");
  host.replaceChildren();
  for (const [area, group] of [...byArea].sort((a, b) => a[0].localeCompare(b[0]))) {
    host.append(
      el("div", { class: "area-head" },
        `${area} `,
        el("button", {
          type: "button", class: "link",
          onclick: () => {
            const boxes = group.map((c) => host.querySelector(`input[value="${c.name}"]`));
            const turnOn = boxes.some((b) => !b.checked);
            boxes.forEach((b) => { b.checked = turnOn; });
          },
        }, "toggle all"),
      ),
    );
    for (const camera of group) {
      host.append(
        el("label", {},
          el("input", { type: "checkbox", value: camera.name }),
          `${camera.name} `,
          el("span", { class: "muted small" }, `(${camera.n_rows})`)),
      );
    }
  }
}

const BASE_COLUMNS = [
  { key: "timestamp", label: "Timestamp (UTC)", sortable: true },
  { key: "camera", label: "Camera", sortable: true },
  { key: "n_images", label: "Images", numeric: true },
  // Relative to whatever the reader has the capture archive mounted on; the
  // acquisition host's own root is not stored and not shown.
  { key: "capture_path", label: "Capture file" },
];

function renderResults(data) {
  const head = $("#results thead tr");
  const pvColumns = state.selectPvs.map((pv) => state.pvByName.get(pv)?.name || pv);
  head.replaceChildren(
    ...BASE_COLUMNS.map((col) => {
      const active = state.sort === col.key;
      const th = el("th", {
        class: col.sortable ? "sortable" : "",
        title: col.sortable ? "sort by this column" : null,
      }, col.label + (active ? (state.order === "asc" ? " ▲" : " ▼") : ""));
      if (col.sortable) {
        th.addEventListener("click", () => {
          state.order = state.sort === col.key && state.order === "asc" ? "desc" : "asc";
          state.sort = col.key;
          state.offset = 0;
          runSearch();
        });
      }
      return th;
    }),
    ...pvColumns.map((pv) => el("th", {}, pv)),
    el("th", {}, ""),
  );

  const body = $("#results tbody");
  body.replaceChildren(
    ...data.rows.map((row) =>
      el("tr", {},
        el("td", {}, row.timestamp),
        el("td", {}, row.camera),
        el("td", { class: "num" }, String(row.n_images)),
        el("td", { class: "path", title: row.capture_path }, row.capture_path),
        ...pvColumns.map((pv) => el("td", { class: "num" }, formatNumber(row.values[pv]))),
        el("td", {}, el("button", {
          type: "button", class: "link", onclick: () => showRow(row.id),
        }, "details")),
      ),
    ),
  );

  if (!data.rows.length) {
    const span = BASE_COLUMNS.length + pvColumns.length + 1;
    body.append(el("tr", {}, el("td", { colspan: String(span), class: "muted" },
      data.total ? "no rows on this page" : "no rows match this query")));
  }

  const limit = Number($("#limit").value);
  const first = data.total ? data.offset + 1 : 0;
  const last = Math.min(data.offset + data.rows.length, data.total);
  $("#page-info").textContent = `${first}–${last} of ${data.total}`;
  $("#pager").classList.toggle("hidden", data.total === 0);
  $("#prev").disabled = data.offset === 0;
  $("#next").disabled = data.offset + limit >= data.total;
  $("#status").textContent =
    `${data.total} row(s) · scanned ${data.scanned_rows} · ${data.pv_reads} PV read(s) · ${data.elapsed_ms} ms`;
}

async function showRow(rowId) {
  const row = await getJson(`./api/rows/${rowId}`);
  $("#row-title").textContent = `${row.camera} · ${row.timestamp}`;
  const entries = Object.entries(row.values).sort(([a], [b]) => a.localeCompare(b));

  const parts = [
    el("p", { class: "muted small" },
      `${row.n_images} image(s) in this capture · source ${row.source_file}`),
  ];
  // Only when the server found the capture file: has_image means the render
  // will succeed, so there is no broken-image flash to guard against.
  if (row.has_image) {
    parts.push(el("figure", { class: "capture" },
      el("img", { src: `./api/rows/${rowId}/image.png`, alt: `${row.camera} capture` }),
      el("figcaption", { class: "muted small" }, imageDataset(1)),
    ));
  }
  parts.push(el("table", {},
    el("tbody", {}, ...entries.map(([name, value]) =>
      el("tr", {},
        el("td", {}, name),
        el("td", { class: "num" }, formatNumber(value)))))));

  $("#row-body").replaceChildren(...parts);
  $("#row-dialog").showModal();
}

// --- actions -----------------------------------------------------------------

function showError(selector, message, level = "error") {
  const node = $(selector);
  node.textContent = message;
  node.classList.toggle("hidden", !message);
  node.classList.toggle("warn", Boolean(message) && level === "warn");
}

function incompleteMessage() {
  const parts = [];
  const pvs = state.incompleteFilters;
  if (pvs.length) {
    parts.push(
      `Ignoring ${pvs.length} incomplete filter (${pvs.join(", ")}) — fill in every value.`);
  }
  if (state.incompleteTimes.length) {
    parts.push(
      `Ignoring an unfinished date (${state.incompleteTimes.join(", ")}) — ` +
      "complete it, or clear it to search all time.");
  }
  return parts.join(" ");
}

async function runSearch() {
  const params = buildParams();
  syncUrl(params);
  showError("#error", incompleteMessage(), "warn");
  $("#status").textContent = "searching…";
  try {
    const data = await getJson(`./api/search?${params}`);
    state.lastRows = data.rows;
    renderResults(data);
  } catch (err) {
    showError("#error", err.message);
    $("#status").textContent = "";
  }
}

async function runPlot() {
  const params = buildParams({ includePaging: false });
  for (const pv of state.plotPvs) params.append("pv", pv);
  const pattern = $("#plot-pattern").value.trim();
  if (pattern) params.set("pv_pattern", pattern);
  if (!state.plotPvs.length && !pattern) {
    showError("#plot-error", "Add at least one PV or a pattern to plot.");
    return;
  }
  showError("#plot-error", incompleteMessage(), "warn");
  try {
    const data = await getJson(`./api/series?${params}`);
    drawPlot(data.series);
  } catch (err) {
    showError("#plot-error", err.message);
  }
}

function drawPlot(series) {
  const host = $("#plot");
  host.replaceChildren();
  state.chart = null;
  if (!series.length || !series[0].t.length) {
    host.append(el("p", { class: "muted" }, "No data in this window."));
    $("#plot-legend").replaceChildren();
    return;
  }

  const xs = series[0].t.map((iso) => Date.parse(iso) / 1000);
  const data = [xs, ...series.map((s) => s.v)];
  const options = {
    width: host.clientWidth || 800,
    height: Math.max(host.clientHeight || 420, 320),
    legend: { show: false },
    scales: { x: { time: true } },
    axes: [
      { stroke: "#8c9aad", grid: { stroke: "#2b3542" }, ticks: { stroke: "#2b3542" } },
      { stroke: "#8c9aad", grid: { stroke: "#2b3542" }, ticks: { stroke: "#2b3542" } },
    ],
    series: [
      {},
      ...series.map((s, i) => ({
        label: s.pv,
        stroke: PALETTE[i % PALETTE.length],
        width: 1.5,
        points: { show: xs.length < 120 },
      })),
    ],
  };
  state.chart = new uPlot(options, data, host);

  $("#plot-legend").replaceChildren(
    ...series.map((s, i) =>
      el("span", {},
        el("span", { class: "swatch", style: `background:${PALETTE[i % PALETTE.length]}` }),
        s.pv + (s.decimated ? " (decimated)" : ""))),
  );
}

// --- wiring ------------------------------------------------------------------

function updateModeVisibility() {
  const mode = $("#mode").value;
  $$(".range-field").forEach((n) => n.classList.toggle("hidden", mode === "all"));
  $$(".absolute-field").forEach((n) => n.classList.toggle("hidden", mode !== "absolute"));
  $$(".following-field").forEach((n) => n.classList.toggle("hidden", mode !== "following"));
}

function wire() {
  $("#mode").addEventListener("change", updateModeVisibility);
  $("#add-filter").addEventListener("click", () => addFilterRow());
  $("#run").addEventListener("click", () => { state.offset = 0; runSearch(); });
  $("#reset").addEventListener("click", () => {
    location.hash = "";
    location.reload();
  });
  $("#limit").addEventListener("change", () => { state.offset = 0; runSearch(); });
  $("#prev").addEventListener("click", () => {
    state.offset = Math.max(0, state.offset - Number($("#limit").value));
    runSearch();
  });
  $("#next").addEventListener("click", () => {
    state.offset += Number($("#limit").value);
    runSearch();
  });
  $("#export").addEventListener("click", () => {
    const params = buildParams({ includePaging: false });
    window.location = `./api/export.csv?${params}`;
  });
  $("#share").addEventListener("click", async () => {
    syncUrl(buildParams());
    await navigator.clipboard.writeText(location.href);
    $("#status").textContent = "link copied";
  });

  $("#select-input").addEventListener("change", (event) => {
    const value = event.target.value.trim();
    if (value && !state.selectPvs.includes(value)) {
      state.selectPvs.push(value);
      renderSelectChips();
    }
    event.target.value = "";
  });
  $("#plot-input").addEventListener("change", (event) => {
    const value = event.target.value.trim();
    if (value && !state.plotPvs.includes(value)) {
      state.plotPvs.push(value);
      renderPlotChips();
    }
    event.target.value = "";
  });
  $("#plot-run").addEventListener("click", runPlot);

  for (const tab of $$(".tab")) {
    tab.addEventListener("click", () => {
      $$(".tab").forEach((t) => t.classList.toggle("active", t === tab));
      $("#panel-results").classList.toggle("hidden", tab.dataset.tab !== "results");
      $("#panel-plot").classList.toggle("hidden", tab.dataset.tab !== "plot");
      if (tab.dataset.tab === "plot" && state.chart) {
        state.chart.setSize({ width: $("#plot").clientWidth, height: $("#plot").clientHeight });
      }
    });
  }

  window.addEventListener("resize", () => {
    if (state.chart && !$("#panel-plot").classList.contains("hidden")) {
      state.chart.setSize({
        width: $("#plot").clientWidth,
        height: Math.max($("#plot").clientHeight, 320),
      });
    }
  });
}

async function init() {
  wire();
  try {
    const [meta, pvs] = await Promise.all([getJson("./api/meta"), getJson("./api/pvs?limit=5000")]);
    state.pvs = pvs;
    state.pvByName = new Map(pvs.map((pv) => [pv.name, pv]));
    state.cameras = meta.cameras;

    $("#meta-summary").textContent =
      `${meta.n_rows.toLocaleString()} captures · ${meta.pv_count} PVs · ` +
      `${(meta.time_from || "?").slice(0, 10)} → ${(meta.time_to || "?").slice(0, 10)} · ${meta.domain}`;

    $("#pv-list").replaceChildren(...pvs.map((pv) => el("option", { value: pv.name })));
    renderCameras(meta.cameras);

    applyParams(new URLSearchParams(location.hash.slice(1)));
    await runSearch();
  } catch (err) {
    showError("#error", `Could not load the index: ${err.message}`);
  }
}

init();

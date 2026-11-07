"use strict";

// Fact text is untrusted. Everything reaches the page through h(), which only ever sets text nodes
// and attributes, so a planted statement cannot become markup.

const SVG_NS = "http://www.w3.org/2000/svg";
const GRAPH_LIMIT = 300;
const PAGE = 100;
const REFRESH_MS = 30000;

const S = {
  token: "",
  scopes: [],
  truncated: false,
  scope: null,
  tab: "facts",
  health: null,
  offline: null,
  factState: "current",
  filter: "",
  facts: [],
  factsDone: false,
  graph: null,
  timeline: null,
  review: null,
  reviewDenied: false,
  notes: null,
  ask: null,
  drawer: null,
  selectedNode: null,
  signature: "",
};

// ---- Small helpers ----------------------------------------------------------------------------

function h(tag, props, ...children) {
  const el = document.createElement(tag);
  applyProps(el, props);
  append(el, children);
  return el;
}

function s(tag, props, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  applyProps(el, props);
  append(el, children);
  return el;
}

function applyProps(el, props) {
  for (const [key, value] of Object.entries(props || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else if (key === "class") el.setAttribute("class", value);
    else if (key === "css") for (const [p, v] of Object.entries(value)) el.style.setProperty(p, v);
    else if (key === "value") el.value = value;
    else el.setAttribute(key, value === true ? "true" : String(value));
  }
}

function append(el, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

function icon(path) {
  return s("svg", { viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", "stroke-width": 2, "stroke-linecap": "round", "stroke-linejoin": "round", "aria-hidden": "true" },
    path.map((d) => s("path", { d })));
}

const ICONS = {
  search: ["M21 21l-4.3-4.3", "M11 18a7 7 0 1 0 0-14 7 7 0 0 0 0 14z"],
  refresh: ["M21 12a9 9 0 1 1-2.6-6.4L21 8", "M21 3v5h-5"],
  theme: ["M12 3a9 9 0 1 0 9 9 7 7 0 0 1-9-9z"],
  close: ["M18 6 6 18", "M6 6l12 12"],
};

function storageGet(store, key) {
  try { return store.getItem(key); } catch { return null; }
}

function storageSet(store, key, value) {
  try { store.setItem(key, value); } catch { /* the page works without it */ }
}

function toast(message, bad) {
  const el = h("div", { class: bad ? "toast bad" : "toast" }, message);
  document.getElementById("toasts").append(el);
  setTimeout(() => el.remove(), bad ? 6000 : 3000);
}

const dateFormat = new Intl.DateTimeFormat(undefined, { year: "numeric", month: "short", day: "numeric" });

function date(text) {
  return text ? dateFormat.format(new Date(text)) : "";
}

function ago(text) {
  if (!text) return "";
  const seconds = (Date.now() - new Date(text).getTime()) / 1000;
  const units = [[31536000, "year"], [2592000, "month"], [86400, "day"], [3600, "hour"], [60, "minute"]];
  for (const [size, name] of units) {
    if (Math.abs(seconds) >= size) {
      const n = Math.round(Math.abs(seconds) / size);
      const word = `${n} ${name}${n === 1 ? "" : "s"}`;
      return seconds >= 0 ? `${word} ago` : `in ${word}`;
    }
  }
  return "just now";
}

const humanize = (text) => String(text || "").replace(/[_-]+/g, " ");

const TRUST = {
  system: ["System", "Set by the system. The highest level of trust."],
  user: ["Confirmed", "Came from a person, or a person approved it."],
  agent: ["Agent", "Your coding agent decided to remember this. Its model also reads outside content, so writes about people and shared knowledge wait for review."],
  external: ["Outside source", "Came from outside content. The lowest level of trust."],
};

function factStatus(fact) {
  const now = Date.now();
  if (fact.status === "quarantined") return { key: "waiting", label: "Waiting for review", kind: "waiting" };
  if (fact.status === "rejected") return { key: "rejected", label: "Rejected", kind: "bad" };
  if (fact.invalidated_at && new Date(fact.invalidated_at).getTime() <= now) return { key: "retracted", label: "Retracted", kind: "bad" };
  if (new Date(fact.valid_from).getTime() > now) return { key: "later", label: "Starts later", kind: "past" };
  if (fact.valid_to && new Date(fact.valid_to).getTime() <= now) {
    return { key: "past", label: fact.superseded_by ? "Replaced" : "Ended", kind: "past" };
  }
  return { key: "current", label: "True now", kind: "current" };
}

function pill(fact) {
  const status = factStatus(fact);
  return h("span", { class: `pill ${status.kind}` }, status.label);
}

function scopeParts(scope) {
  const at = scope.indexOf(":");
  return { kind: scope.slice(0, at), name: scope.slice(at + 1) };
}

const current = () => S.scopes.find((item) => item.scope === S.scope) || null;

// ---- API --------------------------------------------------------------------------------------

async function api(path, options) {
  let response;
  try {
    response = await fetch(`/api/${path}`, {
      method: options?.method || "GET",
      headers: { "X-MemoryOS-UI": S.token, "Content-Type": "application/json" },
      body: options?.body ? JSON.stringify(options.body) : undefined,
    });
  } catch {
    throw Object.assign(new Error("This page lost its connection. Is `memoryos ui` still running?"), { status: 0 });
  }
  const text = await response.text();
  let body = null;
  try { body = text ? JSON.parse(text) : null; } catch { /* reported below */ }
  if (!response.ok) {
    throw Object.assign(new Error(body?.error?.message || `Request failed with status ${response.status}`), {
      status: response.status,
      code: body?.error?.code,
      body,
    });
  }
  return body;
}

const scopeQuery = (extra) => new URLSearchParams({ scope: S.scope, ...extra }).toString();

// `memoryos setup` keeps an image it already built, so a newer command can meet an older server.
async function loadScopes() {
  const body = await api("scopes").catch((error) => {
    if (error.status !== 404) throw error;
    throw Object.assign(new Error("Your server is older than this page. Update it with: docker compose -f ~/.config/memoryos/docker-compose.yml up -d --build"), { status: 0 });
  });
  S.scopes = body.scopes;
  S.truncated = body.truncated;
  if (!S.scope || !current()) S.scope = S.scopes[0]?.scope ?? null;
}

async function loadHealth() {
  try {
    const response = await fetch("/api/status", { headers: { "X-MemoryOS-UI": S.token } });
    S.health = await response.json();
  } catch {
    S.health = null;
  }
}

async function loadFacts(reset) {
  const params = { state: S.factState, limit: PAGE };
  if (!reset && S.facts.length) params.before = S.facts[S.facts.length - 1].id;
  const body = await api(`facts?${scopeQuery(params)}`);
  S.facts = reset ? body.facts : S.facts.concat(body.facts);
  S.factsDone = body.facts.length < PAGE;
}

async function loadTab() {
  if (!S.scope) return;
  if (S.tab === "facts") await loadFacts(true);
  if (S.tab === "map") {
    const body = await api(`facts?${scopeQuery({ state: "current", limit: GRAPH_LIMIT })}`);
    S.graph = buildGraph(body.facts);
  }
  if (S.tab === "timeline") {
    S.timeline = (await api(`facts?${scopeQuery({ state: "all", limit: GRAPH_LIMIT })}`)).facts;
  }
  if (S.tab === "review") await loadReview();
  if (S.tab === "notes") {
    S.notes = (await api(`agents/${encodeURIComponent(scopeParts(S.scope).name)}/blocks`)).blocks;
  }
}

async function loadReview() {
  S.reviewDenied = false;
  try {
    const body = await api(`quarantine?${new URLSearchParams({ scopes: S.scope, limit: 200 })}`);
    S.review = await Promise.all(body.facts.map(async (fact) => {
      const [replaces, source] = await Promise.all([
        fact.pending_supersede_of ? api(`facts/${fact.pending_supersede_of}`).catch(() => null) : null,
        fact.source_episode_ids[0] ? api(`episodes/${fact.source_episode_ids[0]}`).catch(() => null) : null,
      ]);
      return { fact, replaces, source };
    }));
  } catch (error) {
    if (error.status !== 403) throw error;
    S.review = [];
    S.reviewDenied = true;
  }
}

// ---- Actions ----------------------------------------------------------------------------------

function guarded(work) {
  return work().catch((error) => {
    if (error.status === 0 || error.code === "server_unreachable" || error.status === 401) {
      S.offline = error.message;
      render();
    } else {
      toast(error.message, true);
    }
  });
}

function selectScope(scope) {
  S.scope = scope;
  S.drawer = null;
  S.ask = null;
  S.filter = "";
  S.selectedNode = null;
  S.facts = [];
  S.graph = S.timeline = S.review = S.notes = null;
  const summary = current();
  if (S.tab === "notes" && !summary?.blocks) S.tab = "facts";
  render();
  return guarded(async () => { await loadTab(); S.offline = null; render(); });
}

function selectTab(tab) {
  S.tab = tab;
  S.ask = null;
  render();
  return guarded(async () => { await loadTab(); render(); });
}

function ask(query) {
  const text = query.trim();
  if (!text) { S.ask = null; render(); return Promise.resolve(); }
  S.ask = { query: text, results: null };
  render();
  return guarded(async () => {
    const body = await api("recall", { method: "POST", body: { scopes: [S.scope], query: text, limit: 10 } });
    if (S.ask?.query === text) { S.ask.results = body.facts; render(); }
  });
}

async function openFact(id) {
  S.drawer = { type: "fact", id, fact: null, history: null, sources: null };
  renderDrawer();
  await guarded(async () => {
    const [fact, history] = await Promise.all([api(`facts/${id}`), api(`facts/${id}/history`)]);
    const sources = await Promise.all(fact.source_episode_ids.slice(0, 3).map((e) => api(`episodes/${e}`).catch(() => null)));
    if (S.drawer?.id !== id) return;
    S.drawer.fact = fact;
    S.drawer.history = [...history.facts].sort((a, b) => new Date(a.valid_from) - new Date(b.valid_from));
    S.drawer.sources = sources.filter(Boolean);
    renderDrawer();
  });
}

function closeDrawer() {
  S.drawer = null;
  S.selectedNode = null;
  render();
}

async function decide(id, action) {
  await guarded(async () => {
    await api(`quarantine/${id}/${action}`, { method: "POST" });
    toast(action === "approve" ? "Approved. Your agent can now recall this." : "Rejected. It will not be recalled.");
    if (S.drawer?.id === id) S.drawer = null;
    await Promise.all([loadScopes(), loadReview()]);
  });
  // Also after a failure, so the buttons are enabled again.
  render();
}

async function refresh(quiet) {
  await guarded(async () => {
    await Promise.all([loadScopes(), loadHealth()]);
    const recovered = Boolean(S.offline);
    S.offline = null;
    const signature = JSON.stringify(S.scopes);
    const changed = recovered || signature !== S.signature;
    S.signature = signature;
    if (!quiet || changed) {
      await loadTab();
      render();
    } else {
      renderRail();
    }
    if (!quiet) toast("Up to date");
  });
}

// ---- Layout -----------------------------------------------------------------------------------

function render() {
  const app = document.getElementById("app");
  if (!S.token) {
    app.replaceChildren(h("div", { class: "empty" },
      h("h2", {}, "This page needs its address"),
      h("p", {}, "Run ", h("code", {}, "memoryos ui"), " and open the full address it prints.")));
    return;
  }
  app.replaceChildren(h("div", { class: "shell" },
    h("aside", { class: "rail", id: "rail" }),
    h("main", { class: "main" },
      S.offline && h("div", { class: "banner", role: "alert" }, S.offline),
      renderHead(),
      h("div", { class: `body${S.tab === "map" && !S.ask && S.scope ? " flush" : ""}`, id: "body" }))));
  renderRail();
  renderBody();
  renderDrawer();
}

function renderRail() {
  const rail = document.getElementById("rail");
  if (!rail) return;
  const groups = [["user", "People"], ["agent", "Agents"], ["shared", "Shared"]];
  rail.replaceChildren();
  append(rail, [
    h("div", { class: "brand" },
      h("img", { class: "brand-mark", alt: "", src: document.querySelector("link[rel=icon]").href }), "MemoryOS"),
    groups.map(([kind, title]) => {
      const items = S.scopes.filter((item) => scopeParts(item.scope).kind === kind);
      if (!items.length) return null;
      return h("div", { class: "rail-group" }, h("h2", {}, title), items.map(renderSpace));
    }),
    S.truncated && h("p", { class: "help" }, "Showing the 500 most recently used spaces."),
    h("div", { class: "rail-foot" }, healthDot(), h("span", {}, healthLabel()), h("span", { class: "spacer" }),
      h("button", { class: "icon-btn", title: "Refresh", "aria-label": "Refresh", onclick: () => refresh(false) }, icon(ICONS.refresh)),
      h("button", { class: "icon-btn", title: "Switch light and dark", "aria-label": "Switch light and dark", onclick: toggleTheme }, icon(ICONS.theme))),
  ]);
}

function renderSpace(item) {
  const { name } = scopeParts(item.scope);
  return h("button", {
    class: "space", title: item.scope, "aria-current": item.scope === S.scope ? "true" : null,
    onclick: () => selectScope(item.scope),
  },
    h("span", { class: "space-avatar" }, name.slice(0, 1)),
    h("span", { class: "space-name" }, name),
    item.quarantined > 0 && h("span", { class: "review-dot", title: `${item.quarantined} waiting for review` }),
    h("span", { class: "space-count" }, item.current));
}

function healthDot() {
  const ready = S.health?.status === "ready";
  const degraded = S.health && !ready;
  return h("span", { class: `status-dot${S.offline || !S.health ? " bad" : degraded ? " warn" : ""}` });
}

function healthLabel() {
  if (S.offline || !S.health) return "Server offline";
  if (S.health.status === "ready") return "Connected";
  return S.health.database === "ok" ? "Model loading" : "Database offline";
}

function renderHead() {
  const summary = current();
  if (!summary) {
    return h("header", { class: "head" }, h("div", { class: "head-top" }, h("h1", {}, "MemoryOS")));
  }
  const { name, kind } = scopeParts(summary.scope);
  const tabs = [["facts", "Facts"], ["map", "Map"], ["timeline", "Timeline"], ["review", "Review"]];
  if (summary.blocks > 0) tabs.push(["notes", "Agent notes"]);
  return h("header", { class: "head" },
    h("select", {
      class: "mobile-select", "aria-label": "Memory space",
      onchange: (event) => selectScope(event.target.value),
    }, S.scopes.map((item) => h("option", { value: item.scope, selected: item.scope === S.scope }, item.scope))),
    h("div", { class: "head-top" }, h("h1", {}, name), h("span", { class: "scope-code" }, `${kind} space`)),
    h("div", { class: "stats" },
      stat(summary.current, "Known now", () => showFacts("current")),
      stat(summary.past, "Replaced or ended", () => showFacts("past")),
      stat(summary.quarantined, "Waiting for review", () => selectTab("review"), summary.quarantined > 0),
      stat(summary.entities, "People and things", () => selectTab("map"))),
    h("div", { class: "ask" }, icon(ICONS.search),
      h("input", {
        id: "ask", type: "search", placeholder: "Ask what this memory knows, the way your agent would", "aria-label": "Ask this memory",
        value: S.ask?.query || "",
        onkeydown: (event) => {
          if (event.key === "Enter") ask(event.target.value);
          if (event.key === "Escape") { event.target.value = ""; ask(""); event.target.blur(); }
        },
      }),
      h("kbd", {}, "/")),
    h("nav", { class: "tabs", role: "tablist" }, tabs.map(([key, label]) => h("button", {
      class: "tab", role: "tab", "aria-selected": !S.ask && S.tab === key ? "true" : "false",
      onclick: () => selectTab(key),
    }, label, key === "review" && summary.quarantined > 0 && h("span", { class: "tab-badge" }, summary.quarantined)))));
}

function stat(value, label, onclick, attention) {
  return h("button", { class: `stat${attention ? " attention" : ""}`, onclick },
    h("div", { class: "stat-value" }, value.toLocaleString()),
    h("div", { class: "stat-label" }, label));
}

function showFacts(state) {
  S.factState = state;
  S.tab = "facts";
  S.ask = null;
  render();
  return guarded(async () => { await loadFacts(true); render(); });
}

function renderBody() {
  const body = document.getElementById("body");
  if (!body) return;
  if (!S.scope) { body.replaceChildren(welcome()); return; }
  if (S.ask) { body.replaceChildren(renderAsk()); return; }
  const views = { facts: renderFacts, map: renderMap, timeline: renderTimeline, review: renderReview, notes: renderNotes };
  body.replaceChildren(views[S.tab]());
}

function welcome() {
  if (S.offline) {
    return h("div", { class: "empty" }, h("h2", {}, "Cannot reach MemoryOS"),
      h("p", {}, "Start the server with ", h("code", {}, "memoryos setup"), ", then refresh this page."));
  }
  return h("div", { class: "empty" }, h("h2", {}, "Nothing remembered yet"),
    h("p", {}, "Work with your agent and ask it to remember something. It will show up here."));
}

const loading = () => h("div", { class: "loading" }, "Loading");

function empty(title, text) {
  return h("div", { class: "empty" }, h("h2", {}, title), h("p", {}, text));
}

// ---- Facts ------------------------------------------------------------------------------------

function renderFacts() {
  const toolbar = h("div", { class: "toolbar" },
    h("div", { class: "segmented", role: "group", "aria-label": "Which facts" },
      [["current", "Now"], ["past", "Past"], ["all", "All"]].map(([key, label]) => h("button", {
        "aria-pressed": S.factState === key ? "true" : "false",
        onclick: () => showFacts(key),
      }, label))),
    h("input", {
      class: "filter", id: "filter", type: "search", placeholder: "Filter this list", "aria-label": "Filter this list", value: S.filter,
      oninput: (event) => {
        S.filter = event.target.value;
        const results = document.getElementById("fact-results");
        if (results) results.replaceChildren(renderFactGroups());
      },
    }),
    h("span", { class: "toolbar-note" }, `${S.facts.length.toLocaleString()} loaded`));
  return h("div", {}, toolbar, h("div", { id: "fact-results" }, renderFactGroups()));

  function renderFactGroups() {
    const needle = S.filter.toLowerCase();
    const list = S.facts.filter((f) => !needle
      || f.statement.toLowerCase().includes(needle)
      || f.subject_name.toLowerCase().includes(needle)
      || humanize(f.predicate).toLowerCase().includes(needle));
    if (!list.length) {
      if (S.filter) return empty("No match", "Nothing in this list matches. Clear the filter, or ask above to search the whole space.");
      const words = { current: "Nothing is known now", past: "Nothing has been replaced yet", all: "No facts yet" };
      return empty(words[S.factState], "Facts your agent saves here will show up in this list.");
    }
    const groups = new Map();
    for (const fact of list) {
      if (!groups.has(fact.subject_id)) groups.set(fact.subject_id, { name: fact.subject_name, kind: fact.subject_kind, facts: [] });
      groups.get(fact.subject_id).facts.push(fact);
    }
    return h("div", {},
      [...groups.values()].map((group) => h("section", { class: "group" },
        h("div", { class: "group-title" }, group.name, h("span", { class: "kind-tag" }, humanize(group.kind))),
        h("div", { class: "list" }, group.facts.map(factRow)))),
      !S.factsDone && !S.filter && h("button", {
        class: "more",
        onclick: () => guarded(async () => { await loadFacts(false); render(); }),
      }, "Load more"));
  }
}

function factRow(fact) {
  const status = factStatus(fact);
  return h("button", {
    class: `row${status.key === "current" ? "" : " is-past"}`,
    "aria-current": S.drawer?.id === fact.id ? "true" : null,
    onclick: () => openFact(fact.id),
  },
    h("div", { class: "row-text" }, fact.statement),
    h("div", { class: "meta" },
      status.key !== "current" && pill(fact),
      h("span", {}, humanize(fact.predicate)),
      h("span", {}, status.key === "current" ? `since ${date(fact.valid_from)}` : `${date(fact.valid_from)} to ${date(fact.valid_to || fact.invalidated_at) || "now"}`),
      h("span", {}, TRUST[fact.trust][0])));
}

function renderAsk() {
  const { query, results } = S.ask;
  const head = h("div", { class: "toolbar" },
    h("span", {}, "Best matches for ", h("strong", {}, query)),
    h("button", { class: "link-btn", onclick: () => { S.ask = null; render(); } }, "Clear"),
    h("span", { class: "toolbar-note" }, "This is what your agent would get"));
  if (results === null) return h("div", {}, head, loading());
  if (!results.length) return h("div", {}, head, empty("Nothing found", "No fact that is true now matches this question."));
  return h("div", {}, head, h("div", { class: "list" }, results.map(({ fact }, index) => h("button", {
    class: "row", onclick: () => openFact(fact.id),
  },
    h("div", { class: "row-text" }, h("span", { class: `rank${index === 0 ? " top" : ""}` }, index + 1), fact.statement),
    h("div", { class: "meta" }, h("span", {}, humanize(fact.predicate)), h("span", {}, `since ${date(fact.valid_from)}`), h("span", {}, TRUST[fact.trust][0]))))));
}

// ---- Map --------------------------------------------------------------------------------------

function buildGraph(facts) {
  const nodes = new Map();
  const touch = (id, name, kind) => {
    if (!nodes.has(id)) nodes.set(id, { id, name, kind, x: 0, y: 0, vx: 0, vy: 0, degree: 0, facts: [] });
    return nodes.get(id);
  };
  const edges = [];
  for (const fact of facts) {
    const from = touch(fact.subject_id, fact.subject_name, fact.subject_kind);
    from.facts.push(fact);
    if (fact.object_id) {
      const to = touch(fact.object_id, fact.object_name, fact.object_kind);
      to.facts.push(fact);
      from.degree++;
      to.degree++;
      edges.push({ fact, from, to });
    }
  }
  const list = [...nodes.values()];
  layout(list, edges);
  const kinds = [...new Set(list.map((node) => node.kind))];
  return { nodes: list, edges, kinds, total: facts.length, limited: facts.length >= GRAPH_LIMIT, view: null };
}

// A small force layout, run to the end before drawing, so the map appears settled and does not jiggle.
function layout(nodes, edges) {
  const golden = Math.PI * (3 - Math.sqrt(5));
  nodes.forEach((node, index) => {
    const radius = 26 * Math.sqrt(index + 1);
    node.x = radius * Math.cos(index * golden);
    node.y = radius * Math.sin(index * golden);
  });
  const rounds = nodes.length > 200 ? 140 : 280;
  for (let round = 0; round < rounds; round++) {
    const heat = 1 - round / rounds;
    for (let a = 0; a < nodes.length; a++) {
      for (let b = a + 1; b < nodes.length; b++) {
        const dx = nodes[a].x - nodes[b].x;
        const dy = nodes[a].y - nodes[b].y;
        const square = Math.max(dx * dx + dy * dy, 64);
        const push = 16000 / square;
        nodes[a].vx += dx * push / Math.sqrt(square);
        nodes[a].vy += dy * push / Math.sqrt(square);
        nodes[b].vx -= dx * push / Math.sqrt(square);
        nodes[b].vy -= dy * push / Math.sqrt(square);
      }
    }
    for (const { from, to } of edges) {
      const dx = to.x - from.x;
      const dy = to.y - from.y;
      const distance = Math.max(Math.hypot(dx, dy), 1);
      const pull = (distance - 170) * 0.03;
      from.vx += dx / distance * pull;
      from.vy += dy / distance * pull;
      to.vx -= dx / distance * pull;
      to.vy -= dy / distance * pull;
    }
    for (const node of nodes) {
      node.vx -= node.x * 0.012;
      node.vy -= node.y * 0.012;
      node.x += Math.max(-30, Math.min(30, node.vx)) * heat;
      node.y += Math.max(-30, Math.min(30, node.vy)) * heat;
      node.vx *= 0.55;
      node.vy *= 0.55;
    }
  }
}

const nodeRadius = (node) => 7 + Math.min(node.degree, 12) * 1.25;
const nodeColor = (graph, node) => `var(--node-${(graph.kinds.indexOf(node.kind) % 6) + 1})`;
const shorten = (text, size) => (text.length > size ? `${text.slice(0, size - 1)}…` : text);

function renderMap() {
  const graph = S.graph;
  if (!graph) return loading();
  if (!graph.nodes.length) return empty("Nothing to draw yet", "The map shows the people and things in the facts known now.");

  const svg = s("svg", { class: "map", role: "img", "aria-label": `Map of ${graph.nodes.length} people and things` });
  const world = s("g", {});
  svg.append(world);
  const showAllLabels = graph.nodes.length <= 70;

  const edgeEls = graph.edges.map((edge) => {
    const line = s("line", { class: "edge" });
    const hit = s("line", { class: "edge-hit", onclick: () => openFact(edge.fact.id) }, s("title", {}, edge.fact.statement));
    const label = s("text", { class: "edge-label" }, humanize(edge.fact.predicate));
    label.style.display = "none";
    world.append(line, hit, label);
    return { edge, line, hit, label };
  });
  const nodeEls = graph.nodes.map((node) => {
    const text = s("text", { dy: nodeRadius(node) + 15 }, shorten(node.name, 24));
    if (!showAllLabels && node.degree < 2) text.style.display = "none";
    const group = s("g", { class: "node", tabindex: 0, role: "button", "aria-label": node.name },
      s("title", {}, `${node.name}: ${node.facts.length} fact${node.facts.length === 1 ? "" : "s"}`),
      s("circle", { r: nodeRadius(node), css: { fill: nodeColor(graph, node) } }),
      text);
    group.addEventListener("keydown", (event) => { if (event.key === "Enter") selectNode(node); });
    world.append(group);
    return { node, group, text };
  });

  function place() {
    for (const { edge, line, hit, label } of edgeEls) {
      for (const el of [line, hit]) {
        el.setAttribute("x1", edge.from.x);
        el.setAttribute("y1", edge.from.y);
        el.setAttribute("x2", edge.to.x);
        el.setAttribute("y2", edge.to.y);
      }
      label.setAttribute("x", (edge.from.x + edge.to.x) / 2);
      label.setAttribute("y", (edge.from.y + edge.to.y) / 2 - 4);
    }
    for (const { node, group } of nodeEls) group.setAttribute("transform", `translate(${node.x} ${node.y})`);
  }

  function highlight() {
    const chosen = S.selectedNode;
    const near = new Set();
    if (chosen) {
      near.add(chosen);
      for (const { from, to } of graph.edges) {
        if (from.id === chosen) near.add(to.id);
        if (to.id === chosen) near.add(from.id);
      }
    }
    for (const { node, group, text } of nodeEls) {
      group.classList.toggle("dim", Boolean(chosen) && !near.has(node.id));
      group.classList.toggle("on", node.id === chosen);
      if (!showAllLabels) text.style.display = node.degree >= 2 || near.has(node.id) ? "" : "none";
    }
    for (const { edge, line, label } of edgeEls) {
      const on = chosen && (edge.from.id === chosen || edge.to.id === chosen);
      line.classList.toggle("on", Boolean(on));
      line.classList.toggle("dim", Boolean(chosen) && !on);
      label.style.display = on ? "" : "none";
    }
  }

  function selectNode(node) {
    S.selectedNode = node.id;
    S.drawer = { type: "entity", node };
    highlight();
    renderDrawer();
  }

  const view = graph.view || { x: 0, y: 0, k: 1 };
  const applyView = () => {
    world.setAttribute("transform", `translate(${view.x} ${view.y}) scale(${view.k})`);
    graph.view = view;
  };
  function fit() {
    const box = svg.getBoundingClientRect();
    if (!box.width) return;
    const xs = graph.nodes.map((node) => node.x);
    const ys = graph.nodes.map((node) => node.y);
    const [minX, maxX, minY, maxY] = [Math.min(...xs) - 90, Math.max(...xs) + 90, Math.min(...ys) - 50, Math.max(...ys) + 60];
    view.k = Math.min(1.05, box.width / (maxX - minX), box.height / (maxY - minY));
    view.x = box.width / 2 - ((minX + maxX) / 2) * view.k;
    view.y = box.height / 2 - ((minY + maxY) / 2) * view.k;
    applyView();
  }
  function zoom(factor, cx, cy) {
    const box = svg.getBoundingClientRect();
    const x = cx ?? box.width / 2;
    const y = cy ?? box.height / 2;
    const next = Math.max(0.2, Math.min(4, view.k * factor));
    view.x = x - ((x - view.x) / view.k) * next;
    view.y = y - ((y - view.y) / view.k) * next;
    view.k = next;
    applyView();
  }

  svg.addEventListener("wheel", (event) => {
    event.preventDefault();
    const box = svg.getBoundingClientRect();
    zoom(Math.exp(-event.deltaY * 0.0016), event.clientX - box.left, event.clientY - box.top);
  }, { passive: false });

  let drag = null;
  svg.addEventListener("pointerdown", (event) => {
    const hitNode = nodeEls.find(({ group }) => group.contains(event.target));
    if (event.target.classList?.contains("edge-hit")) return;
    drag = { node: hitNode?.node || null, sx: event.clientX, sy: event.clientY, moved: false, ox: hitNode?.node.x, oy: hitNode?.node.y, vx: view.x, vy: view.y };
    svg.setPointerCapture(event.pointerId);
    if (!hitNode) svg.classList.add("panning");
  });
  svg.addEventListener("pointermove", (event) => {
    if (!drag) return;
    const dx = event.clientX - drag.sx;
    const dy = event.clientY - drag.sy;
    if (Math.hypot(dx, dy) > 3) drag.moved = true;
    if (!drag.moved) return;
    if (drag.node) {
      drag.node.x = drag.ox + dx / view.k;
      drag.node.y = drag.oy + dy / view.k;
      place();
    } else {
      view.x = drag.vx + dx;
      view.y = drag.vy + dy;
      applyView();
    }
  });
  svg.addEventListener("pointerup", () => {
    svg.classList.remove("panning");
    if (drag && !drag.moved) {
      if (drag.node) selectNode(drag.node);
      else if (S.selectedNode) closeSelection();
    }
    drag = null;
  });
  function closeSelection() {
    S.selectedNode = null;
    S.drawer = null;
    highlight();
    renderDrawer();
  }

  place();
  highlight();
  const wrap = h("div", {},
    svg,
    h("div", { class: "map-legend" }, graph.kinds.slice(0, 6).map((kind, index) => h("span", { class: "legend-item" },
      h("span", { class: "legend-swatch", css: { background: `var(--node-${index + 1})` } }), humanize(kind)))),
    h("div", { class: "map-note" }, graph.limited ? `Newest ${GRAPH_LIMIT} facts. Click anything for detail.` : "Click anything for detail. Drag to move, scroll to zoom."),
    h("div", { class: "map-controls" },
      h("button", { "aria-label": "Zoom in", title: "Zoom in", onclick: () => zoom(1.25) }, "+"),
      h("button", { "aria-label": "Zoom out", title: "Zoom out", onclick: () => zoom(0.8) }, "−"),
      h("button", { "aria-label": "Fit to window", title: "Fit to window", onclick: fit }, "⤢")));
  if (graph.view) applyView();
  else requestAnimationFrame(fit);
  return wrap;
}

// ---- Timeline ---------------------------------------------------------------------------------

function renderTimeline() {
  const facts = S.timeline;
  if (!facts) return loading();
  if (!facts.length) return empty("No history yet", "When a fact changes, you will see the old and new versions here.");

  const now = Date.now();
  const lanes = new Map();
  for (const fact of facts) {
    const key = `${fact.subject_id}|${fact.predicate}`;
    if (!lanes.has(key)) lanes.set(key, { subject: fact.subject_name, predicate: fact.predicate, facts: [] });
    lanes.get(key).facts.push(fact);
  }
  const start = Math.min(...facts.map((f) => new Date(f.valid_from).getTime()));
  const span = Math.max(now - start, 86400000);
  const min = start - span * 0.02;
  const width = now - min;
  const at = (time) => ((Math.min(time, now) - min) / width) * 100;
  const order = [...lanes.values()].sort((a, b) => b.facts.length - a.facts.length
    || Math.max(...b.facts.map((f) => new Date(f.valid_from).getTime())) - Math.max(...a.facts.map((f) => new Date(f.valid_from).getTime())));

  const ticks = [0, 0.25, 0.5, 0.75, 1].map((part) => {
    const time = min + width * part;
    return h("span", { css: { left: `${part * 100}%` } }, part === 1 ? "Today" : date(new Date(time).toISOString()));
  });

  return h("div", {},
    h("div", { class: "toolbar" },
      h("div", { class: "timeline-key" },
        h("span", {}, h("span", { class: "key-swatch", css: { background: "var(--accent)" } }), "True now"),
        h("span", {}, h("span", { class: "key-swatch", css: { background: "var(--past-soft)", "box-shadow": "inset 0 0 0 1px var(--line-strong)" } }), "Replaced or ended"),
        h("span", {}, h("span", { class: "key-swatch", css: { background: "var(--red-soft)" } }), "Retracted")),
      h("span", { class: "toolbar-note" }, facts.length >= GRAPH_LIMIT ? `Newest ${GRAPH_LIMIT} facts. What changed comes first.` : "What changed comes first")),
    h("div", { class: "axis" }, ticks),
    order.map((lane) => h("div", { class: "lane" },
      h("div", { class: "lane-label", title: `${lane.subject}: ${humanize(lane.predicate)}` },
        h("span", { class: "lane-subject" }, lane.subject),
        h("span", { class: "lane-predicate" }, humanize(lane.predicate))),
      h("div", { class: "lane-track" }, lane.facts.map((fact) => {
        const status = factStatus(fact);
        const left = at(new Date(fact.valid_from).getTime());
        const end = status.key === "retracted" ? new Date(fact.invalidated_at).getTime() : fact.valid_to ? new Date(fact.valid_to).getTime() : now;
        const kind = status.key === "current" ? "current" : status.key === "retracted" ? "retracted" : "past";
        return h("button", {
          class: `bar ${kind}`, title: `${fact.statement} (${date(fact.valid_from)} to ${fact.valid_to ? date(fact.valid_to) : "now"})`,
          "aria-current": S.drawer?.id === fact.id ? "true" : null,
          css: { left: `${left}%`, width: `calc(${Math.max(at(end) - left, 0)}% - 2px)` },
          onclick: () => openFact(fact.id),
        }, fact.statement);
      })))));
}

// ---- Review -----------------------------------------------------------------------------------

function renderReview() {
  if (S.review === null) return loading();
  if (S.reviewDenied) return empty("Review needs the admin key", "This page is using a key that cannot review. Run memoryos setup to create the admin key.");
  if (!S.review.length) return empty("All clear", "Nothing is waiting. Facts held back from your agent's memory will show up here for you to decide.");
  return h("div", {},
    h("div", { class: "toolbar" }, h("span", { class: "toolbar-note", css: { "margin-left": "0" } },
      "Your agent cannot recall these until you approve them. Check the source: outside content can try to plant false facts.")),
    S.review.map(({ fact, replaces, source }) => h("article", { class: "card" },
      h("div", { class: "card-title" }, fact.statement),
      h("div", { class: "meta" }, h("span", {}, humanize(fact.predicate)), h("span", {}, TRUST[fact.trust][0]), h("span", {}, `added ${ago(fact.recorded_at)}`)),
      replaces && h("div", { class: "quote" }, h("span", { class: "quote-label" }, "Would replace"), replaces.statement),
      source && h("div", { class: "quote" }, h("span", { class: "quote-label" }, `Source: ${source.source}`), source.content),
      h("div", { class: "actions" },
        h("button", { class: "btn primary", onclick: (event) => { event.target.disabled = true; decide(fact.id, "approve"); } }, "Approve"),
        h("button", { class: "btn danger", onclick: (event) => { event.target.disabled = true; decide(fact.id, "reject"); } }, "Reject"),
        h("button", { class: "link-btn", css: { "margin-left": "auto" }, onclick: () => openFact(fact.id) }, "Details")))));
}

function renderNotes() {
  if (S.notes === null) return loading();
  if (!S.notes.length) return empty("No notes", "This agent has not written any working notes.");
  return h("div", {}, S.notes.map((block) => h("article", { class: "card" },
    h("div", { class: "card-title" }, humanize(block.label)),
    h("div", { class: "meta" }, h("span", {}, `version ${block.version}`), h("span", {}, `updated ${ago(block.updated_at)}`),
      h("span", {}, `${block.content.length.toLocaleString()} of ${block.max_chars.toLocaleString()} characters`)),
    h("div", { class: "meter" }, h("span", { css: { width: `${Math.min(100, (block.content.length / block.max_chars) * 100)}%` } })),
    h("div", { class: "note-text" }, block.content))));
}

// ---- Drawer -----------------------------------------------------------------------------------

function renderDrawer() {
  document.getElementById("drawer")?.remove();
  if (!S.drawer || !S.token) return;
  const content = S.drawer.type === "entity" ? renderEntityDrawer(S.drawer.node) : renderFactDrawer(S.drawer);
  document.body.append(h("aside", { class: "drawer", id: "drawer", "aria-label": "Detail" },
    h("div", { class: "drawer-head" },
      h("strong", {}, S.drawer.type === "entity" ? humanize(S.drawer.node.kind) : "Fact"),
      h("span", { class: "spacer" }),
      h("button", { class: "icon-btn", "aria-label": "Close", title: "Close (Esc)", onclick: closeDrawer }, icon(ICONS.close))),
    h("div", { class: "drawer-body" }, content)));
}

function renderEntityDrawer(node) {
  return h("div", {},
    h("div", { class: "drawer-title" }, node.name),
    h("div", { class: "meta" }, `${node.facts.length} fact${node.facts.length === 1 ? "" : "s"} known now`),
    h("div", { class: "section" }, h("h3", {}, "What is known"),
      node.facts.map((fact) => h("button", { class: "mini-row", onclick: () => openFact(fact.id) }, fact.statement))));
}

function renderFactDrawer({ fact, history, sources }) {
  if (!fact) return loading();
  const trust = TRUST[fact.trust];
  const status = factStatus(fact);
  return h("div", {},
    h("div", { class: "drawer-title" }, fact.statement),
    h("div", { class: "meta" }, pill(fact)),
    h("dl", { class: "fields" },
      h("dt", {}, "About"), h("dd", {}, humanize(fact.predicate)),
      h("dt", {}, "True from"), h("dd", {}, date(fact.valid_from)),
      h("dt", {}, "True until"), h("dd", {}, fact.valid_to ? date(fact.valid_to) : status.key === "current" ? "Still true" : "Not set"),
      h("dt", {}, "Learned"), h("dd", {}, `${date(fact.recorded_at)} (${ago(fact.recorded_at)})`),
      fact.invalidated_at && [h("dt", {}, "Retracted"), h("dd", {}, date(fact.invalidated_at))],
      h("dt", {}, "Trust"), h("dd", {}, trust[0], h("div", { class: "help" }, trust[1])),
      h("dt", {}, "Space"), h("dd", {}, fact.scope)),
    history && history.length > 1 && h("div", { class: "section" }, h("h3", {}, "How this changed"),
      h("div", { class: "chain" }, history.map((item) => h("button", {
        class: `chain-item${factStatus(item).key === "current" ? " now" : ""}${item.id === fact.id ? " here" : ""}`,
        onclick: () => openFact(item.id),
      },
        h("div", { class: "chain-text" }, item.statement),
        h("div", { class: "meta" }, `${date(item.valid_from)} to ${item.valid_to ? date(item.valid_to) : "now"}`))))),
    sources && sources.length > 0 && h("div", { class: "section" }, h("h3", {}, "Where it came from"),
      sources.map((episode) => h("div", { class: "quote" },
        h("span", { class: "quote-label" }, `${episode.source}, ${date(episode.occurred_at)}`), episode.content))),
    fact.status === "quarantined" && h("div", { class: "actions" },
      h("button", { class: "btn primary", onclick: () => decide(fact.id, "approve") }, "Approve"),
      h("button", { class: "btn danger", onclick: () => decide(fact.id, "reject") }, "Reject")));
}

// ---- Start ------------------------------------------------------------------------------------

function toggleTheme() {
  const dark = document.documentElement.dataset.theme === "dark"
    || (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.dataset.theme = dark ? "light" : "dark";
  storageSet(localStorage, "memoryos-theme", document.documentElement.dataset.theme);
}

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && S.drawer) { closeDrawer(); return; }
  const typing = /^(INPUT|SELECT|TEXTAREA)$/.test(event.target.tagName);
  if (typing || event.metaKey || event.ctrlKey || event.altKey) return;
  if (event.key === "/") { event.preventDefault(); document.getElementById("ask")?.focus(); }
  const tabs = { 1: "facts", 2: "map", 3: "timeline", 4: "review" };
  if (tabs[event.key] && S.scope) selectTab(tabs[event.key]);
});

// Pasting the new address after `memoryos ui` restarts changes only the fragment, so nothing reloads.
addEventListener("hashchange", () => { if (/t=/.test(location.hash)) location.reload(); });

async function start() {
  const theme = storageGet(localStorage, "memoryos-theme");
  if (theme) document.documentElement.dataset.theme = theme;
  const match = location.hash.match(/t=([^&]+)/);
  if (match) {
    S.token = decodeURIComponent(match[1]);
    storageSet(sessionStorage, "memoryos-token", S.token);
    history.replaceState(null, "", location.pathname);
  } else {
    S.token = storageGet(sessionStorage, "memoryos-token") || "";
  }
  render();
  if (!S.token) return;
  await guarded(async () => {
    await Promise.all([loadScopes(), loadHealth()]);
    S.signature = JSON.stringify(S.scopes);
    render();
    await loadTab();
    render();
  });
  setInterval(() => { if (!document.hidden) refresh(true); }, REFRESH_MS);
  addEventListener("focus", () => refresh(true));
}

start();

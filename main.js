import { basicSetup, EditorView } from "codemirror";
import { Compartment } from "@codemirror/state";
import { python } from "@codemirror/lang-python";
import { vim } from "@replit/codemirror-vim";
import { oneDark } from "@codemirror/theme-one-dark";

const GROUPS = [
  { name: "Basics", range: "01–04", ids: ["01", "02", "03", "04", "04a"] },
  { name: "Multiple heads", range: "05–06", ids: ["05", "06"] },
  { name: "Grouped attention", range: "07", ids: ["07a", "07b"] },
  { name: "Cache & decode", range: "08–11", ids: ["08", "09", "10", "11"] },
  { name: "Serving patterns", range: "12–13", ids: ["12", "13"] },
  { name: "Paged cache", range: "14–16", ids: ["14", "15", "16"] },
  { name: "Numerics", range: "17–19", ids: ["17", "18a", "18b", "19"] },
  { name: "Measurement", range: "20", ids: ["20"] },
];
const PREREQUISITES = {
  "04": ["01", "02", "03"],
  "04a": ["04"],
  "05": ["04"],
  "06": ["05"],
  "07a": ["06"],
  "07b": ["07a"],
  "09": ["08"],
  "10": ["07a"],
  "11": ["08", "10"],
  "12": ["07a"],
  "13": ["07a"],
  "15": ["14"],
  "16": ["14"],
  "18b": ["18a"],
};
const STORAGE_KEY = "inference-workbook-v1";
const STATIC_MODE = import.meta.env.VITE_STATIC_MODE === "1";
const $ = (id) => document.getElementById(id);

function loadStore() {
  try {
    const value = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    const drafts = value.drafts || {};
    const progress = value.progress || {};
    if (drafts["07"] && !drafts["07a"]) drafts["07a"] = drafts["07"];
    if (progress["07"] && !progress["07a"]) progress["07a"] = progress["07"];
    delete drafts["07"];
    delete progress["07"];
    return { drafts, progress, vim: !!value.vim, theme: value.theme === "dark" ? "dark" : "light", last: value.last === "07" ? "07a" : value.last || null };
  } catch {
    return { drafts: {}, progress: {}, vim: false, theme: "light", last: null };
  }
}

const store = loadStore();
let exercises = [];
let currentId = null;
let editor = null;
let loadingEditor = false;
const vimMode = new Compartment();
const editorTheme = new Compartment();
const results = new Map();
const checkerWorkers = new Map();
let checkerRequest = 0;
const pendingChecks = new Map();
let webgpuUnavailable = null;

function runBrowserWorker(kind, payload) {
  let worker = checkerWorkers.get(kind);
  if (!worker) {
    worker = kind === "webgpu"
      ? new Worker(new URL("./browser_webgpu_worker.js", import.meta.url), { type: "module" })
      : new Worker(new URL("./browser_worker.js", import.meta.url), { type: "module" });
    checkerWorkers.set(kind, worker);
  }
  worker.onmessage = ({ data }) => {
    const pending = pendingChecks.get(data.request);
    if (!pending) return;
    pendingChecks.delete(data.request);
    if (data.error) pending.reject(new Error(data.error));
    else pending.resolve(data.result);
  };
  worker.onerror = (event) => {
    for (const [request, pending] of pendingChecks) {
      if (pending.kind !== kind) continue;
      pending.reject(new Error(event.message || `${kind} checker failed.`));
      pendingChecks.delete(request);
    }
  };
  const request = ++checkerRequest;
  return new Promise((resolve, reject) => {
    const timer = kind === "webgpu" ? setTimeout(() => {
      pendingChecks.delete(request);
      worker.terminate();
      checkerWorkers.delete(kind);
      reject(new Error("WebGPU check timed out after 45 seconds."));
    }, 45_000) : null;
    pendingChecks.set(request, {
      kind,
      resolve: (value) => { clearTimeout(timer); resolve(value); },
      reject: (error) => { clearTimeout(timer); reject(error); },
    });
    worker.postMessage({ request, payload });
  });
}

async function browserCheck(item, source) {
  const edits = exercises.flatMap((entry) => {
    const draft = entry.id === item.id ? source : store.drafts[entry.id]?.source;
    return draft ? [{ function: entry.function, source: draft }] : [];
  });
  const payload = { id: item.id, edits };
  let webgpuResult;
  if (!webgpuUnavailable) {
    try {
      webgpuResult = await runBrowserWorker("webgpu", payload);
      if (webgpuResult.passed) return { passed: true, output: `WebGPU: ${webgpuResult.output}` };
    } catch (error) {
      webgpuUnavailable = String(error.message || error);
    }
  }
  if (currentId === item.id) {
    results.set(item.id, { kind: "running", output: "WebGPU could not complete this check. Running CPU PyTorch… The first submission may take about 30 seconds." });
    renderResult();
  }
  const cpuResult = await runBrowserWorker("cpu", payload);
  const gpuOutput = webgpuUnavailable || webgpuResult?.output || "unavailable";
  return { passed: cpuResult.passed, output: `WebGPU: ${gpuOutput}\nCPU: ${cpuResult.output}` };
}

function persist() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(store));
}

function currentExercise() {
  return exercises.find((item) => item.id === currentId);
}

function displayedSource(item) {
  return store.drafts[item.id]?.source ?? item.source;
}

function completion(item) {
  const progress = store.progress[item.id];
  if (!progress) return null;
  if (progress.kind === "manual") return item.checkable ? null : "complete";
  const draft = store.drafts[item.id];
  return !draft || progress.source === draft.source ? "complete" : "stale";
}

function missingPrerequisites(item) {
  // Keep existing work and the open exercise accessible when progress is reset.
  if (item.id === currentId || store.drafts[item.id] || store.progress[item.id]) return [];
  return (PREREQUISITES[item.id] || []).filter((id) => {
    const prerequisite = exercises.find((entry) => entry.id === id);
    return prerequisite && completion(prerequisite) !== "complete";
  });
}

function nextAccessibleIndex(index, direction) {
  const next = index + direction;
  return next >= 0 && next < exercises.length && !missingPrerequisites(exercises[next]).length ? next : -1;
}

function createEditor() {
  editor = new EditorView({
    doc: "",
    extensions: [
      vimMode.of(store.vim ? vim({ status: true }) : []),
      editorTheme.of(store.theme === "dark" ? oneDark : []),
      basicSetup,
      python(),
      EditorView.lineWrapping,
      EditorView.updateListener.of((update) => {
        if (!update.docChanged || loadingEditor || !currentId) return;
        const item = currentExercise();
        const source = update.state.doc.toString();
        if (source === item.source) delete store.drafts[currentId];
        else store.drafts[currentId] = { source, base: item.revision };
        persist();
        renderDraftState();
        renderProgress();
        renderNavigation();
      }),
    ],
    parent: $("editor"),
  });
  editor.dom.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      if (currentExercise()?.checkable) runCheck();
    }
  }, true);
  renderVimButton();
}

function setEditorSource(source) {
  loadingEditor = true;
  editor.dispatch({ changes: { from: 0, to: editor.state.doc.length, insert: source } });
  loadingEditor = false;
}

function renderVimButton() {
  $("vim-toggle").textContent = `Vim ${store.vim ? "on" : "off"}`;
  $("vim-toggle").setAttribute("aria-pressed", String(store.vim));
}

function renderTheme() {
  document.documentElement.dataset.theme = store.theme;
  document.querySelector('meta[name="theme-color"]').content = store.theme === "dark" ? "#181818" : "#ffffff";
  const label = store.theme === "dark" ? "Switch to light mode" : "Switch to dark mode";
  $("theme-toggle").textContent = store.theme === "dark" ? "☼" : "☾";
  $("theme-toggle").setAttribute("aria-label", label);
  $("theme-toggle").title = label;
  $("theme-toggle").setAttribute("aria-pressed", String(store.theme === "dark"));
}

function renderProgress() {
  const completed = exercises.filter((item) => completion(item) === "complete").length;
  const percent = exercises.length ? Math.round((completed / exercises.length) * 100) : 0;
  $("progress-fill").style.width = `${percent}%`;
  $("progress-caption").textContent = `${completed} / ${exercises.length}`;
}

function renderNavigation() {
  const nav = $("exercise-nav");
  nav.replaceChildren();
  for (const group of GROUPS) {
    const items = group.ids.map((id) => exercises.find((item) => item.id === id)).filter(Boolean);
    if (!items.length) continue;
    const section = document.createElement("section");
    section.className = "sidebar-section";
    const heading = document.createElement("h3");
    heading.className = "sidebar-section-title";
    heading.textContent = `${group.range} / ${group.name}`;
    section.append(heading);
    const links = document.createElement("div");
    links.className = "sidebar-section-links";
    for (const item of items) {
      const missing = missingPrerequisites(item);
      const button = document.createElement("button");
      button.type = "button";
      button.className = `exercise-link${item.id === currentId ? " active" : ""}`;
      button.setAttribute("aria-current", item.id === currentId ? "step" : "false");
      button.disabled = missing.length > 0;
      button.title = missing.length ? `Complete ${missing.join(" and ")} first` : `${item.id} · ${item.title}`;
      if (missing.length) button.setAttribute("aria-label", `${item.id} ${item.title}; complete ${missing.join(" and ")} first`);
      const id = document.createElement("span");
      id.className = "exercise-id";
      id.textContent = item.id;
      const name = document.createElement("span");
      name.className = "exercise-name";
      name.textContent = item.title;
      const indicator = document.createElement("span");
      indicator.className = `exercise-indicator${completion(item) === "complete" ? " complete" : ""}`;
      indicator.setAttribute("aria-hidden", "true");
      button.append(id, name, indicator);
      button.addEventListener("click", () => selectExercise(item.id));
      links.append(button);
    }
    section.append(links);
    nav.append(section);
  }
  const index = exercises.findIndex((item) => item.id === currentId);
  if (index >= 0) {
    $("prev-button").disabled = nextAccessibleIndex(index, -1) < 0;
    $("next-button").disabled = nextAccessibleIndex(index, 1) < 0;
  }
}

function renderDraftState() {
  const item = currentExercise();
  if (!item) return;
  const draft = store.drafts[item.id];
  $("reset-button").disabled = !draft;
}

function renderResult() {
  const item = currentExercise();
  const result = results.get(item.id);
  const panel = $("result-panel");
  panel.className = `output${result?.kind ? ` ${result.kind}` : ""}`;
  $("result-state").textContent = result?.kind === "pass" ? "Pass" : result?.kind === "fail" ? "Fail" : result?.kind === "running" ? "Running" : "Ready";
  $("result-output").textContent = result?.output || "Submit your code when you're ready. ⌘/Ctrl + Enter works in the editor.";
}

function selectExercise(id) {
  const item = exercises.find((entry) => entry.id === id);
  if (!item || missingPrerequisites(item).length) return;
  currentId = id;
  store.last = id;
  persist();
  history.replaceState(null, "", `#${id}`);
  const index = exercises.indexOf(item);
  $("exercise-number").textContent = id.toUpperCase();
  $("exercise-title").textContent = item.title;
  $("problem-text").textContent = item.prompt;
  $("solution-toggle").hidden = !item.has_solution;
  $("solution-toggle").textContent = "View solution";
  $("solution-toggle").setAttribute("aria-expanded", "false");
  $("solution-panel").hidden = true;
  $("solution-source").textContent = "";
  $("bottom-position").textContent = `${String(index + 1).padStart(2, "0")} / ${exercises.length}`;
  $("prev-button").disabled = nextAccessibleIndex(index, -1) < 0;
  $("next-button").disabled = nextAccessibleIndex(index, 1) < 0;
  $("run-button").disabled = !item.checkable;
  $("run-button").textContent = "Submit";
  setEditorSource(displayedSource(item));
  renderDraftState();
  renderProgress();
  renderNavigation();
  renderResult();
}

async function runCheck() {
  const item = currentExercise();
  if (!item?.checkable || $("run-button").disabled) return;
  const source = editor.state.doc.toString();
  $("run-button").disabled = true;
  results.set(item.id, { kind: "running", output: STATIC_MODE ? "Trying WebGPU first… First load may take a while." : "Running your code with the local PyTorch checker…" });
  renderResult();
  try {
    let data;
    if (STATIC_MODE) {
      data = await browserCheck(item, source);
    } else {
      const response = await fetch("/api/check", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: item.id, source }),
      });
      data = await response.json();
      if (!response.ok) throw new Error(data.error || "The checker could not run.");
    }
    const passed = !!data.passed;
    results.set(item.id, { kind: passed ? "pass" : "fail", output: data.output || (passed ? "Passed." : "Check failed.") });
    if (passed) store.progress[item.id] = { kind: "verified", source };
    else delete store.progress[item.id];
    persist();
  } catch (error) {
    results.set(item.id, { kind: "fail", output: String(error.message || error) });
  } finally {
    if (currentId === item.id) {
      $("run-button").disabled = false;
      renderResult();
      renderDraftState();
      renderProgress();
      renderNavigation();
    }
  }
}

function installActions() {
  $("clear-progress").addEventListener("click", () => {
    store.progress = {};
    results.clear();
    persist();
    renderProgress();
    renderNavigation();
    renderDraftState();
    renderResult();
  });
  $("vim-toggle").addEventListener("click", () => {
    store.vim = !store.vim;
    persist();
    editor.dispatch({ effects: vimMode.reconfigure(store.vim ? vim({ status: true }) : []) });
    renderVimButton();
    editor.focus();
  });
  $("theme-toggle").addEventListener("click", () => {
    store.theme = store.theme === "dark" ? "light" : "dark";
    persist();
    editor.dispatch({ effects: editorTheme.reconfigure(store.theme === "dark" ? oneDark : []) });
    renderTheme();
  });
  $("reset-button").addEventListener("click", () => {
    const item = currentExercise();
    delete store.drafts[item.id];
    delete store.progress[item.id];
    results.delete(item.id);
    persist();
    selectExercise(item.id);
  });
  $("solution-toggle").addEventListener("click", async () => {
    const item = currentExercise();
    const panel = $("solution-panel");
    if (!panel.hidden) {
      panel.hidden = true;
      $("solution-toggle").textContent = "View solution";
      $("solution-toggle").setAttribute("aria-expanded", "false");
      return;
    }
    try {
      const response = await fetch(STATIC_MODE ? `${import.meta.env.BASE_URL}solutions.json` : `/api/solution?id=${encodeURIComponent(item.id)}`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Could not load the solution.");
      if (currentId !== item.id) return;
      $("solution-source").textContent = STATIC_MODE ? data[item.id] : data.source;
      panel.hidden = false;
      $("solution-toggle").textContent = "Hide solution";
      $("solution-toggle").setAttribute("aria-expanded", "true");
    } catch (error) {
      results.set(item.id, { kind: "fail", output: String(error.message || error) });
      if (currentId === item.id) renderResult();
    }
  });
  $("run-button").addEventListener("click", runCheck);
  $("prev-button").addEventListener("click", () => {
    const index = exercises.findIndex((item) => item.id === currentId);
    const previous = nextAccessibleIndex(index, -1);
    if (previous >= 0) selectExercise(exercises[previous].id);
  });
  $("next-button").addEventListener("click", () => {
    const index = exercises.findIndex((item) => item.id === currentId);
    const next = nextAccessibleIndex(index, 1);
    if (next >= 0) selectExercise(exercises[next].id);
  });
  window.addEventListener("hashchange", () => {
    const id = location.hash.slice(1) === "07" ? "07a" : location.hash.slice(1);
    if (id && id !== currentId) selectExercise(id);
    if (currentId && location.hash !== `#${currentId}`) history.replaceState(null, "", `#${currentId}`);
  });
}

async function main() {
  renderTheme();
  createEditor();
  installActions();
  try {
    const response = await fetch(STATIC_MODE ? `${import.meta.env.BASE_URL}catalog.json` : "/api/exercises");
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Could not load exercises.");
    exercises = data.exercises;
    const requested = location.hash.slice(1) === "07" ? "07a" : location.hash.slice(1);
    const id = [requested, store.last, exercises[0]?.id].find((candidate) => {
      const item = exercises.find((entry) => entry.id === candidate);
      return item && !missingPrerequisites(item).length;
    });
    if (!id) throw new Error("No numbered exercises were found in part1_pytorch.py.");
    selectExercise(id);
  } catch (error) {
    $("exercise-title").textContent = "Could not load exercises";
    $("result-output").textContent = String(error.message || error);
    $("result-panel").classList.add("fail");
  }
}

main();

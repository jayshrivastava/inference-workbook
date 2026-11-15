import { basicSetup, EditorView } from "codemirror";
import { Compartment } from "@codemirror/state";
import { python } from "@codemirror/lang-python";
import { vim } from "@replit/codemirror-vim";
import { oneDark } from "@codemirror/theme-one-dark";

const GROUPS = [
  { name: "Basics", range: "01–04", ids: ["01", "02", "03", "04", "04a"] },
  { name: "Multiple heads", range: "05–06", ids: ["05", "06"] },
  { name: "Grouped attention", range: "07", ids: ["07", "07b"] },
  { name: "Cache & decode", range: "08–11", ids: ["08", "09", "10", "11"] },
  { name: "Serving patterns", range: "12–13", ids: ["12", "13"] },
  { name: "Paged cache", range: "14–16", ids: ["14", "15", "16"] },
  { name: "Numerics", range: "17–19", ids: ["17", "18a", "18b", "19"] },
  { name: "Measurement", range: "20", ids: ["20"] },
];
const STORAGE_KEY = "inference-workbook-v1";
const $ = (id) => document.getElementById(id);

function loadStore() {
  try {
    const value = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    return { drafts: value.drafts || {}, progress: value.progress || {}, vim: !!value.vim, theme: value.theme === "dark" ? "dark" : "light", last: value.last || null };
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

function createEditor() {
  editor = new EditorView({
    doc: "",
    extensions: [
      vimMode.of(store.vim ? vim() : []),
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
      const button = document.createElement("button");
      button.type = "button";
      button.className = `exercise-link${item.id === currentId ? " active" : ""}`;
      button.setAttribute("aria-current", item.id === currentId ? "step" : "false");
      button.title = `${item.id} · ${item.title}`;
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
  if (!item) return;
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
  $("prev-button").disabled = index === 0;
  $("next-button").disabled = index === exercises.length - 1;
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
  results.set(item.id, { kind: "running", output: "Running your code with the local PyTorch checker…" });
  renderResult();
  try {
    const response = await fetch("/api/check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: item.id, source }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "The checker could not run.");
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
    editor.dispatch({ effects: vimMode.reconfigure(store.vim ? vim() : []) });
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
      const response = await fetch(`/api/solution?id=${encodeURIComponent(item.id)}`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Could not load the local solution.");
      if (currentId !== item.id) return;
      $("solution-source").textContent = data.source;
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
    if (index > 0) selectExercise(exercises[index - 1].id);
  });
  $("next-button").addEventListener("click", () => {
    const index = exercises.findIndex((item) => item.id === currentId);
    if (index < exercises.length - 1) selectExercise(exercises[index + 1].id);
  });
  window.addEventListener("hashchange", () => {
    const id = location.hash.slice(1);
    if (id && id !== currentId) selectExercise(id);
  });
}

async function main() {
  renderTheme();
  createEditor();
  installActions();
  try {
    const response = await fetch("/api/exercises");
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Could not load exercises.");
    exercises = data.exercises;
    const requested = location.hash.slice(1);
    const id = [requested, store.last, exercises[0]?.id].find((candidate) => exercises.some((item) => item.id === candidate));
    if (!id) throw new Error("No numbered exercises were found in part1_pytorch.py.");
    selectExercise(id);
  } catch (error) {
    $("exercise-title").textContent = "Could not load exercises";
    $("result-output").textContent = String(error.message || error);
    $("result-panel").classList.add("fail");
  }
}

main();

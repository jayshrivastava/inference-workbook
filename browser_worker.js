const RUNTIME_BASE = "https://mmtftr.github.io/pyodide-pytorch/runtime/";
const MANIFEST_URL = `${RUNTIME_BASE}build-manifest.json`;
const PYODIDE_VERSION = "314.0.2";
const WHEEL_SHA256 = "074e99919ddc271a0e955c90dac57a0aab5569db5bcf3ae0941b75e11b8a5e69";
const base = import.meta.env.BASE_URL;
let ready;
let queue = Promise.resolve();

async function initialize() {
  const manifestResponse = await fetch(MANIFEST_URL);
  if (!manifestResponse.ok) throw new Error("Could not load the PyTorch for Pyodide manifest.");
  const manifest = await manifestResponse.json();
  if (manifest.configuration?.pyodide?.version !== PYODIDE_VERSION || manifest.wheel?.sha256 !== WHEEL_SHA256) {
    throw new Error("The published PyTorch wheel changed; update the pinned browser runtime.");
  }
  const PYODIDE_ROOT = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
  const { loadPyodide } = await import(/* @vite-ignore */ `${PYODIDE_ROOT}pyodide.mjs`);
  const pyodide = await loadPyodide({ indexURL: PYODIDE_ROOT });
  await pyodide.loadPackage(["micropip", "numpy", "typing-extensions", "sympy", "networkx", "jinja2", "fsspec"]);
  pyodide.FS.chdir("/home/pyodide");
  await pyodide.runPythonAsync('import micropip; await micropip.install("filelock==3.32.0")');
  await pyodide.loadPackage(`${RUNTIME_BASE}${encodeURIComponent(manifest.wheel.filename)}`);
  for (const name of ["problems.py", "check_part1.py", "browser_runner.py", "solutions.json"]) {
    const response = await fetch(`${base}${name}`);
    if (!response.ok) throw new Error(`Could not load ${name}.`);
    pyodide.FS.writeFile(`/home/pyodide/${name}`, await response.text());
  }
  await pyodide.runPythonAsync("import browser_runner");
  return pyodide;
}

self.onmessage = ({ data }) => {
  queue = queue.then(async () => {
    try {
      ready ||= initialize().catch((error) => { ready = undefined; throw error; });
      const pyodide = await ready;
      pyodide.globals.set("submission_json", JSON.stringify(data.payload));
      const answer = await pyodide.runPythonAsync("browser_runner.run(submission_json)");
      self.postMessage({ request: data.request, result: JSON.parse(answer) });
    } catch (error) {
      self.postMessage({ request: data.request, error: String(error?.message || error) });
    }
  });
};

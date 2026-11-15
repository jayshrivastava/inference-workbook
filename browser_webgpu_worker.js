const PYODIDE_ROOT = "https://cdn.jsdelivr.net/pyodide/v0.29.4/full/";
const MANIFEST_URL = "https://celsowm.github.io/torch-pyodide/latest.json";
const base = import.meta.env.BASE_URL;
let ready;
let queue = Promise.resolve();

async function initialize() {
  if (!navigator.gpu) throw new Error("WebGPU is not exposed in this browser.");
  if (!await navigator.gpu.requestAdapter()) throw new Error("No WebGPU adapter is available.");
  const response = await fetch(MANIFEST_URL);
  if (!response.ok) throw new Error("Could not load the torch-pyodide manifest.");
  const manifest = await response.json();
  const runtime = await import(/* @vite-ignore */ manifest.runtimeUrl);
  runtime.installTorchRuntime(globalThis);
  const { loadPyodide } = await import(/* @vite-ignore */ `${PYODIDE_ROOT}pyodide.mjs`);
  const pyodide = await loadPyodide({ indexURL: PYODIDE_ROOT });
  await pyodide.loadPackage(["micropip", "numpy"]);
  pyodide.FS.chdir("/home/pyodide");
  pyodide.globals.set("wheel_url", manifest.wheelUrl);
  await pyodide.runPythonAsync("import micropip; await micropip.install(wheel_url)");
  for (const name of ["problems.py", "check_part1.py", "browser_runner.py", "browser_webgpu_runner.py"]) {
    const file = await fetch(`${base}${name}`);
    if (!file.ok) throw new Error(`Could not load ${name}.`);
    pyodide.FS.writeFile(`/home/pyodide/${name}`, await file.text());
  }
  await pyodide.runPythonAsync("import browser_webgpu_runner");
  return pyodide;
}

self.onmessage = ({ data }) => {
  queue = queue.then(async () => {
    try {
      ready ||= initialize().catch((error) => { ready = undefined; throw error; });
      const pyodide = await ready;
      pyodide.globals.set("submission_json", JSON.stringify(data.payload));
      const answer = await pyodide.runPythonAsync("browser_webgpu_runner.run(submission_json)");
      self.postMessage({ request: data.request, result: JSON.parse(answer) });
    } catch (error) {
      self.postMessage({ request: data.request, error: String(error?.message || error) });
    }
  });
};

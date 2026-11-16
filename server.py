"""Local exercise studio. Reads prompts; runs checks on temporary copies."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


HERE = Path(__file__).resolve().parent
SOURCE_FILE = HERE / "part1_pytorch.py"
if not SOURCE_FILE.exists():
    SOURCE_FILE = HERE / "problems.py"
CHECKER_FILE = HERE / "check_part1.py"
DIST = HERE / "dist"
CHECKABLE = {f"{i:02}" for i in range(1, 18) if i != 7} | {"04a", "07a", "07b", "18a", "18b", "19", "20"}
EXERCISE_ID = re.compile(r"^(\d{2}[a-z]?)\.\s*(.*)")
MAX_BODY = 200_000
TITLES = {
    "01": "Scaled QK scores", "02": "Weighted values", "03": "Causal window mask",
    "04": "Single-head attention", "04a": "Predict next token",
    "05": "Multi-head attention", "06": "Multi-query attention",
    "07a": "Grouped-query attention", "07b": "Predict with GQA",
    "08": "Append to KV cache", "09": "Decode one token",
    "10": "Verify draft block", "11": "Commit accepted tokens",
    "12": "Chunked prefill", "13": "Batched GQA decode",
    "14": "Map paged rows", "15": "Read paged KV", "16": "Write paged KV",
    "17": "RoPE", "18a": "Quantize KV cache", "18b": "Dequantize KV cache",
    "19": "KV cache bytes", "20": "Measure latency",
}
HELPER_DETAILS = {
    "01": "qk_scores_one_head(q: [L,D], k: [S,D]) -> [L,S]",
    "02": "weighted_values(weights: [L,S], v: [S,Dv]) -> [L,Dv]",
    "03": "causal_window_mask(past_len: int, q_len: int, window: int | None = None, device: torch.device | str | None = None) -> bool [q_len,past_len+q_len]",
    "04": "causal_attention_one_head(q_new: [L,D], k_all: [S,D], v_all: [S,Dv], past_len: int, window: int | None = None) -> [L,Dv]",
    "07a": "grouped_query_attention(q_new: [L,Hq,D], k_all: [S,Hkv,D], v_all: [S,Hkv,Dv], past_len: int, window: int | None = None) -> [L,Hq,Dv]",
    "08": "append_contiguous_kv(k_cache: [capacity,Hkv,D], v_cache: [capacity,Hkv,Dv], used: int, k_new: [L,Hkv,D], v_new: [L,Hkv,Dv]) -> int (used+L)",
    "14": "paged_cache_rows(page_table: [max_pages], logical_positions: [N], page_size: int) -> [N] physical rows",
}
HELPERS = {
    "04": ("01", "02", "03"), "04a": ("04",),
    "05": ("03", "04"), "06": ("03", "04"), "07a": ("03", "04"),
    "07b": ("07a",), "09": ("03", "07a", "08"),
    "10": ("03", "07a"), "11": ("08",), "12": ("03", "07a"),
    "13": ("03", "07a"), "15": ("14",), "16": ("14",),
}


def catalog() -> list[dict[str, object]]:
    source = SOURCE_FILE.read_text()
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source, filename=str(SOURCE_FILE))
    result = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        prompt = ast.get_docstring(node)
        if not prompt:
            continue
        match = EXERCISE_ID.match(prompt.splitlines()[0])
        if not match:
            continue
        exercise_id, title = match.groups()
        docstring = node.body[0]
        original = "".join(lines[node.lineno - 1:node.end_lineno])
        starter = "".join(lines[node.lineno - 1:docstring.end_lineno]).rstrip() + (
            f'\n    raise NotImplementedError("exercise {exercise_id}")\n'
        )
        if exercise_id in HELPERS:
            prompt = re.sub(r"(?m)^Available helpers?:[^\n]*(?:\n[^\n]+)*", "", prompt).rstrip()
            starter = re.sub(r'(?m)^    Available helpers?:[^\n]*(?:\n    (?!""")[^\n]+)*', "", starter)
            details = "\n".join(f"    {HELPER_DETAILS[helper]}" for helper in HELPERS[exercise_id])
            prompt += "\n\nAvailable functions (inputs and output shapes):\n" + textwrap.dedent(details)
            closing = starter.rfind('    """')
            starter = starter[:closing] + "\n    Available functions (inputs and output shapes):\n" + details + "\n" + starter[closing:]
        notes = [
            textwrap.dedent(stmt.value.value).strip("\n")
            for stmt in node.body[1:]
            if isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ]
        if notes:
            prompt += "\n\nNotes from your file:\n\n" + "\n\n".join(notes)
        executable = [
            stmt for stmt in node.body[1:]
            if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str))
        ]
        is_stub = len(executable) == 1 and isinstance(executable[0], ast.Raise) and (
            isinstance(executable[0].exc, ast.Call)
            and isinstance(executable[0].exc.func, ast.Name)
            and executable[0].exc.func.id == "NotImplementedError"
        )
        result.append({
            "id": exercise_id,
            "title": TITLES.get(exercise_id, title.rstrip(".")),
            "function": node.name,
            "prompt": prompt,
            "source": starter,
            "solution": original,
            "revision": hashlib.sha256(original.encode()).hexdigest(),
            "has_solution": bool(executable) and not is_stub,
            "checkable": exercise_id in CHECKABLE,
        })
    return result


def replace_function(base: str, function_name: str, replacement: str) -> str:
    replacement_tree = ast.parse(replacement, filename="browser_draft.py")
    functions = [node for node in replacement_tree.body if isinstance(node, ast.FunctionDef)]
    if len(replacement_tree.body) != 1 or len(functions) != 1 or functions[0].name != function_name:
        raise ValueError(f"Write one function named {function_name}; keep helpers inside it.")

    tree = ast.parse(base, filename=str(SOURCE_FILE))
    target = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name),
        None,
    )
    if target is None:
        raise ValueError("The exercise function is no longer in the local file.")
    lines = base.splitlines(keepends=True)
    return "".join(lines[:target.lineno - 1]) + replacement.rstrip() + "\n" + "".join(lines[target.end_lineno:])


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIST), **kwargs)

    def json_response(self, status: int, value: object) -> None:
        payload = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/exercises":
            try:
                self.json_response(200, {"exercises": [
                    {key: value for key, value in item.items() if key != "solution"}
                    for item in catalog()
                ]})
            except Exception as error:
                self.json_response(500, {"error": str(error)})
            return
        if parsed.path == "/api/solution":
            exercise_id = parse_qs(parsed.query).get("id", [None])[0]
            item = next((entry for entry in catalog() if entry["id"] == exercise_id), None)
            if item is None or not item["has_solution"]:
                self.json_response(404, {"error": "No local solution for this exercise."})
            else:
                self.json_response(200, {"source": item["solution"]})
            return
        super().do_GET()

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/check":
            self.json_response(404, {"error": "Unknown endpoint"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_BODY:
                raise ValueError("Request is empty or too large.")
            request = json.loads(self.rfile.read(size))
            exercise_id = request.get("id")
            draft = request.get("source")
            exercise = next((item for item in catalog() if item["id"] == exercise_id), None)
            if exercise is None:
                raise ValueError("Unknown exercise.")
            if exercise_id not in CHECKABLE:
                raise ValueError("This exercise does not have a checker yet.")
            if not isinstance(draft, str):
                raise ValueError("Expected Python function source.")
            source = replace_function(SOURCE_FILE.read_text(), str(exercise["function"]), draft)
            with tempfile.TemporaryDirectory(prefix="inference-workbook-") as scratch:
                scratch_path = Path(scratch)
                (scratch_path / "part1_pytorch.py").write_text(source)
                (scratch_path / "check_part1.py").write_bytes(CHECKER_FILE.read_bytes())
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                completed = subprocess.run(
                    [sys.executable, "check_part1.py", exercise_id],
                    cwd=scratch,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=20,
                    check=False,
                )
            # Put interpreter warnings first so the checker's PASS/FAIL line stays last.
            output = (completed.stderr + completed.stdout).strip()
            self.json_response(200, {
                "passed": completed.returncode == 0,
                "output": output[-18_000:],
            })
        except subprocess.TimeoutExpired:
            self.json_response(200, {"passed": False, "output": "Check timed out after 20 seconds."})
        except (SyntaxError, ValueError, json.JSONDecodeError) as error:
            self.json_response(400, {"error": str(error)})
        except Exception as error:
            self.json_response(500, {"error": str(error)})


if __name__ == "__main__":
    if not (DIST / "index.html").exists():
        raise SystemExit("Build the UI first: npm install && npm run build")
    port = int(os.environ.get("STUDIO_PORT", "8765"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Inference Workbook running at http://127.0.0.1:{port}", flush=True)
    server.serve_forever()

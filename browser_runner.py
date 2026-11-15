"""Run a submitted function against the existing checker inside Pyodide."""

from __future__ import annotations

import ast
import contextlib
import importlib
import io
import json
import sys
import traceback
from pathlib import Path


def _merge(base: str, edits: list[dict]) -> str:
    tree = ast.parse(base)
    lines = base.splitlines(keepends=True)
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    changes = []
    for edit in edits:
        name, source = edit["function"], edit["source"]
        replacement = ast.parse(source)
        if (len(replacement.body) != 1
                or not isinstance(replacement.body[0], ast.FunctionDef)
                or replacement.body[0].name != name
                or name not in functions):
            raise ValueError(f"Write one function named {name}; keep helpers inside it.")
        target = functions[name]
        changes.append((target.lineno - 1, target.end_lineno, source.rstrip() + "\n"))
    for start, end, source in sorted(changes, reverse=True):
        lines[start:end] = [source]
    return "".join(lines)


def run(payload_json: str) -> str:
    payload = json.loads(payload_json)
    capture = io.StringIO()
    try:
        base = Path("problems.py").read_text()
        source = _merge(base, payload["edits"])
        Path("part1_pytorch.py").write_text(source)
        sys.modules.pop("part1_pytorch", None)
        sys.modules.pop("check_part1", None)
        importlib.invalidate_caches()
        checker = importlib.import_module("check_part1")
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            getattr(checker, f"check_{payload['id']}")()
        return json.dumps({"passed": True, "output": f"PASS exercise {payload['id']}"})
    except Exception as error:
        detail = "".join(traceback.format_exception_only(type(error), error)).strip()
        return json.dumps({"passed": False, "output": capture.getvalue() + detail})

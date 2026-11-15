"""Add public exercise data and the browser checker to the Pages build."""

import json
import shutil
from pathlib import Path

import server


here = Path(__file__).resolve().parent
server.SOURCE_FILE = here / "problems.py"
dist = here / "dist"
(dist / "catalog.json").write_text(json.dumps({
    "exercises": [
        {key: value for key, value in item.items() if key != "solution"}
        for item in server.catalog()
    ]
}))
for name in ("problems.py", "check_part1.py", "browser_runner.py", "browser_webgpu_runner.py"):
    shutil.copyfile(here / name, dist / name)

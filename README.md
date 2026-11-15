# Inference Workbook

A local website for the Part 1 PyTorch exercises. It reads `problems.py` and `check_part1.py`. If your local, ignored `part1_pytorch.py` exists here, it reads that instead of `problems.py`. The server never writes to either file. Each editor starts with a clean function stub. The problem description includes the prompt and your standalone notes or diagrams. **View solution** reveals an implementation only when one exists in the local file. Code edited in the website is saved as a browser draft; **Reset attempt** restores the clean stub. The theme preference is saved in the browser.

Submitting an exercise runs its browser draft alongside the other functions currently saved in the active Python file. Browser drafts of earlier exercises are not included until you save those functions to your local `part1_pytorch.py`.

## Run

From this folder:

```bash
npm install
npm run build
python3 server.py
```

Open <http://127.0.0.1:8765>. The server listens only on localhost. Use the Python environment that has `torch` installed. Set `STUDIO_PORT` to use another port.

To keep solutions in a local Python file, copy the starter with `cp problems.py part1_pytorch.py` and edit `part1_pytorch.py`. That file is ignored by Git. If it exists, the site uses it instead of `problems.py`.

**Submit** copies the current exercise source and checker into a temporary directory, runs the checker with real PyTorch, and deletes that temporary directory afterward. Every Part 1 exercise has a checker. Passing submissions fill the progress bar.

The editor supports normal and Vim bindings. Browser drafts, progress, the keymap choice, and the theme are stored in this browser's local storage. Your worked `part1_pytorch.py` is ignored by Git; refresh the page after editing it locally to read its latest version.

**Clear progress** in the sidebar removes completion marks for every exercise. It keeps browser code drafts, your Vim preference, and the original Python files.

## Hosting

GitHub Pages cannot run `server.py` or PyTorch. Publishing `dist/` there would show the page but leave exercise loading and Submit broken because both use `/api/...`. A working hosted version needs a Python-capable backend, or a browser-only replacement for the Python API. See [GitHub's Pages documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site).

# OpenCodeReview — run notes (Gradio web UI)

## Reproduce the artifacts

1. Use the system Python 3.12 (the checked-in `.venv` is a partial install
   without gradio): `C:\Program Files\Python312\python.exe` — available as
   `python` on PATH. The full dependency set (gradio 6.19, langgraph,
   langchain*, chromadb, langfuse, …) is already installed there for the
   user profile.
2. No `.env` copying is required: `.env` already exists in this checkout and
   the app does not load it automatically. Note the shell exports
   `PORT=0` globally on this machine; the app now ignores invalid values
   (`_resolve_port()` in `app.py`) and falls back to 7860.
3. No LLM keys are set, so reviews run in degraded mode (reviewer nodes log
   "LLM unavailable — skipping"); the pipeline still completes and the UI
   renders verdict/findings from the aggregator's rule-based path. The
   Smoke Test tab works fully without keys.

## Run the server

```bash
python app.py            # from the repo root; binds 0.0.0.0:7860
```

- Default port **7860** (override with `PORT`, values <1 or >65535 fall
  back to 7860).
- Gradio prints `Running on local URL: http://0.0.0.0:7860`; open
  <http://127.0.0.1:7860/>.
- Logs go to stdout (or redirect to a file when launched detached).
- Stop: `taskkill //PID <pid> //F` (Windows) or close the process.

## Detached launch (Windows, PowerShell)

```powershell
powershell -NoProfile -Command "(Start-Process -FilePath 'python' -ArgumentList 'app.py' -RedirectStandardOutput '<log>' -RedirectStandardError '<log>.err' -WindowStyle Hidden -PassThru).Id"
```

Then confirm: `powershell -NoProfile -Command "Get-Process -Id <pid>"` and
`curl http://127.0.0.1:7860/`.

# Contract Genome Agent — 03 • Runtime Unblock & E2E Pass (GPU, Offline)

**Objective (single, specific):**  
Make the agent **run end‑to‑end offline** with the **GPU LLM** and **pass the full test suite** on a text‑based PDF, resolving the current blockers (models path, prompt placeholders, mocks, optional deps). Each subtask is executed by **one agent at a time** before moving on.

> Scope: We do **not** change OCR/Layout logic here except to avoid import-time failures. The goal is to get `cg run` and all tests green end‑to‑end with the existing text ingestion path.

---

## Acceptance Criteria
- `cg run samples/nda_sample.pdf --report out/report.json --html out/report.html` completes without errors.
- `pytest -q` passes locally **with outbound network blocked** during tests.
- LLM inference uses **GPU** (llama.cpp with `n_gpu_layers>0`) and **no external API** calls.
- Missing model/config produces a **clear actionable error** (but tests run with a mock without requiring the real model).
- Optional **transformers/layout** imports **never** break the core run when disabled.

---

## Deliverables
- Working `.env` and **models directory** recognized at runtime.
- Stable **prompt system** with required placeholder constants + validation.
- `MockLocalLlama` available to tests; real `LocalLlama` used at runtime.
- Optional imports (transformers/torchvision) made **lazy & guarded**.
- Make targets: `doctor`, `e2e`, and improved `check-gpu`.

---

# TASKS (continue numbering)

## Task 17 — Model Assets & Preflight
**Agent:** _Preflight Engineer_  
**Goal:** Ensure model paths exist and produce a clear error if missing.

**Instructions**
1. Create `cg/_preflight.py` with:
   - `def check_models(cfg) -> None:`  
     - Validate `cfg.LLM_MODEL_PATH` exists and file extension is `.gguf`.
     - If missing, raise a `RuntimeError` with actionable guidance (path, expected name, docs section).
2. In `cg/llm/llama_cpp.py`:
   - Before instantiating `Llama`, call `check_models(cfg)`.
   - If an environment variable `CG_ALLOW_MOCK="1"` is set, **skip** the real load and use `MockLocalLlama` (for tests only).
3. Add `make doctor`:
   ```Makefile
   doctor:
	python -c "import torch, os;print('cuda=', torch.cuda.is_available()); from cg.config import Config; from cg._preflight import check_models; check_models(Config()); print('models=ok')"
   ```
4. Update `README.md` with a **Model Prep** section describing where to place the `.gguf` file and how to set `.env`.

**DoD**
- `make doctor` prints `cuda= True` and `models=ok` on a configured machine.
- If the model is missing, it raises a single, readable error with remediation steps.

**Git**
- Commit & push: `feat(preflight): model path checks + doctor target`

---

## Task 18 — Prompt System Stabilization
**Agent:** _Prompt Smith_  
**Goal:** Provide stable prompt templates and required constants referenced by tests.

**Instructions**
1. Implement/finish `cg/llm/prompts.py`:
   - `EMAIL_REQUIRED_PLACEHOLDERS = {"COMPANY_NAME", "COUNTERPARTY_NAME", "CLAUSE_DELTAS"}`
   - `SUGGESTION_PROMPT` and `EMAIL_NEGOTIATION_PROMPT` string templates.
   - `def validate_placeholders(template:str, required:set) -> None` raising `ValueError` if missing.
2. Add unit tests in `tests/test_prompts.py` to assert all placeholders exist and `validate_placeholders` passes.

**DoD**
- Importing `cg.llm.prompts` exposes the constant and functions; tests pass.

**Git**
- Commit & push: `feat(prompts): required placeholders + validation`

---

## Task 19 — Test Mocks & No-Network Guard
**Agent:** _Test Harnesser_  
**Goal:** Unblock tests that expect a `MockLocalLlama` and enforce offline testing.

**Instructions**
1. In `cg/llm/llama_cpp.py`, add:
   ```python
   class MockLocalLlama:
       def __init__(self, *_, **__): pass
       def generate(self, prompt:str, max_tokens:int=256) -> str:
           return "[MOCK] " + (prompt[:120].replace("\n"," ") + "...")
   ```
   - Export it so tests can `from cg.llm.llama_cpp import MockLocalLlama`.
2. Add `tests/conftest.py`:
   - A `block_network` fixture that monkeypatches `socket.socket` to raise on outbound connections.
   - An `autouse=True` fixture to set `os.environ["CG_ALLOW_MOCK"]="1"` so tests use mock by default.
3. Update tests that previously failed import to use the mock.

**DoD**
- `pytest -q` runs fully offline; no import errors; LLM calls resolved by the mock.

**Git**
- Commit & push: `test(harness): MockLocalLlama + network block fixture`

---

## Task 20 — Optional Imports: Lazy & Guarded
**Agent:** _Import Warden_  
**Goal:** Prevent `transformers/torchvision` issues from breaking core runtime when layout is disabled.

**Instructions**
1. In any module importing transformers (e.g., `cg/layout/model.py`):
   - Move heavy imports **inside functions** (`def load_layout_model(...):`).
   - Wrap with `try/except ImportError` and raise a clear, non-fatal `RuntimeError` **only** when the code path is executed (i.e., when `layout=model` is requested).
2. In `pyproject.toml`, create optional extra:
   ```toml
   [project.optional-dependencies]
   layout = ["transformers>=4.41", "accelerate", "Pillow", "opencv-python-headless"]
   ```
3. In `README.md` document:
   - Core install vs. `pip install .[layout]` for layout features.
   - Note that **core run** does not require transformers/torchvision.

**DoD**
- Running `cg run ...` with `LAYOUT_ENABLE=false` works on a machine **without** transformers/torchvision.
- Attempting to enable layout without deps gives a clear instruction instead of a stack trace.

**Git**
- Commit & push: `chore(imports): lazy guarded layout imports + extras`

---

## Task 21 — CLI Wiring & E2E Command
**Agent:** _Runner Conductor_  
**Goal:** Ensure the CLI uses the real GPU LLM at runtime and succeeds end‑to‑end on the sample PDF.

**Instructions**
1. In `cg/graph.py` and `cli.py`:
   - Ensure `run_pipeline(pdf_path, out_json, out_html, use_mock=False)` parameter exists.
   - The CLI should set `use_mock=False` by default; tests can call with `use_mock=True`.
   - On `use_mock=False`, instantiate `LocalLlama` and confirm `n_gpu_layers>0` in logs.
2. Ensure the report files are created and non-empty.
3. Add `make e2e` target:
   ```Makefile
   e2e:
	cg run samples/nda_sample.pdf --report out/report.json --html out/report.html
	@python - <<'PY'
import json; d=json.load(open('out/report.json')); assert d.get('clauses'), 'no clauses'; print('E2E OK')
PY
   ```

**DoD**
- `make e2e` prints `E2E OK`.
- The logs show the GPU LLM path (no network).

**Git**
- Commit & push: `feat(cli): e2e command uses GPU llama.cpp + report checks`

---

## Task 22 — Readme & Troubleshooting
**Agent:** _Doc Wrangler_  
**Goal:** Provide crisp guidance to future contributors/users.

**Instructions**
1. Add a **Troubleshooting** section:
   - Model missing / wrong path
   - CUDA not visible
   - transformers/torchvision conflicts (and solution: `pip install .[layout]` or disable layout)
   - Tests forcing mock via `CG_ALLOW_MOCK=1`
2. Add a quick “Offline Guarantee” note describing the network-blocking tests.

**DoD**
- README updated; newcomers can get to a green E2E within minutes.

**Git**
- Commit & push: `docs: troubleshooting + offline guarantee`

---

## Runbook
```bash
# 1) Verify CUDA and model path
make doctor

# 2) Run tests fully offline (uses mock LLM)
pytest -q

# 3) Real GPU E2E (no mock)
unset CG_ALLOW_MOCK
make e2e
```

---

## Notes
- Keep commits small and one-per-task; always **push after each task**.
- We will handle OCR/Layout robustness and highlighting in the previous increment after runtime is green.

# Contract Genome Agent — 01 • Bootstrap & Hello-World Graph (GPU)

**Objective (single, specific):**  
Stand up a _minimal_ Contract Genome Agent repo with GPU-enabled local LLM, a LangGraph skeleton (ingest → clause_extractor → risk_scorer → suggestion_generator → email_drafter), a CLI entrypoint that runs end‑to‑end on one sample PDF, and basic tests. Each subtask is done by **one agent at a time** before proceeding.

---

## Context & Constraints
- **Privacy-first & offline** by default for SMEs (no external API calls at runtime).
- **GPU REQUIRED** (NVIDIA, CUDA) for local LLM inference.
- **Python** backend using **LangGraph** + **LangChain**.
- **Local models**:  
  - **Layout/structure**: LayoutLMv3 (DocLayNet-tuned) for layout features (later). For bootstrap we use PyMuPDF and deterministic regex to extract the 5 MVP clause types.
  - **LLM**: Mistral‑7B‑Instruct **or** Qwen2‑7B‑Instruct via **llama.cpp** (GGUF) with CUDA/cuBLAS. (Switchable later to vLLM if desired.)
- **MVP clauses**: term, termination, liability cap, SLA, confidentiality.
- **Scoring**: traffic‑light (green / amber / red) vs **golden playbook** in YAML.
- **Cursor workflow**: follow the Tasks below in order; **do not** start the next task until the current task’s DoD is met.  
- **Git**: After each task, **commit + push**. Include short, descriptive commit messages.

---

## Deliverables
- A repo that installs with one command, passes tests, and runs:  
  ```bash
  cg run samples/nda_sample.pdf --report out/report.json --html out/report.html
  ```
- GPU-accelerated local LLM wired into the suggestion_generator and email_drafter nodes.
- A golden_playbook.yaml with example policies & fallback language for the 5 MVP clauses.
- Minimal E2E output: JSON + HTML report (simple, readable).

---

## Acceptance Criteria
- `python -c "import torch; print(torch.cuda.is_available())"` prints `True` on the target machine.
- `cg run ...` executes the LangGraph pipeline and produces non-empty `out/report.json` & `out/report.html`.
- At least **one** unit test per node + one E2E smoke test pass locally.
- No outbound network calls during runtime (model weights loaded from disk).
- Each task ends with a **git push**.

---

## Repository Layout (target)
```
contract-genome-agent/
├─ pyproject.toml
├─ README.md
├─ .gitignore
├─ .env.example
├─ Makefile
├─ cg/                    # package
│  ├─ __init__.py
│  ├─ config.py
│  ├─ graph.py
│  ├─ nodes/
│  │  ├─ ingest.py
│  │  ├─ clause_extractor.py
│  │  ├─ risk_scorer.py
│  │  ├─ suggestion_generator.py
│  │  └─ email_drafter.py
│  ├─ llm/
│  │  ├─ llama_cpp.py
│  │  └─ prompts.py
│  ├─ rules/
│  │  └─ golden_playbook.yaml
│  └─ reports/
│     └─ render.py
├─ samples/
│  └─ nda_sample.pdf              # placeholder (user-provided)
├─ tests/
│  ├─ test_ingest.py
│  ├─ test_clause_extractor.py
│  ├─ test_risk_scorer.py
│  ├─ test_suggestion_generator.py
│  ├─ test_email_drafter.py
│  └─ test_e2e.py
└─ cli.py
```

---

## Setup (one-time)
- **Conda or venv**. Example with venv:
  ```bash
  python -m venv .venv && source .venv/bin/activate
  python -m pip install --upgrade pip
  ```
- **Install CUDA-enabled PyTorch** (adapt the CUDA version as needed):
  ```bash
  pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
  ```
- **Install core deps**:
  ```bash
  pip install langchain langgraph pydantic ruamel.yaml pymupdf python-dotenv jinja2 click
  ```
- **Install llama.cpp with CUDA**:
  ```bash
  CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install --upgrade --force-reinstall llama-cpp-python
  ```
- **Place local models** (no runtime downloads):
  - `models/llm/mistral-7b-instruct.Q5_K_M.gguf` (or Qwen2‑7B instruct GGUF).  
  - (Later) LayoutLMv3 weights in `models/layoutlmv3/`.
- Copy `.env.example` → `.env` and set:
  ```dotenv
  LLM_MODEL_PATH=models/llm/mistral-7b-instruct.Q5_K_M.gguf
  LLM_CTX_SIZE=8192
  LLM_N_GPU_LAYERS=999
  LLM_N_THREADS=8
  ```

---

# TASKS (one agent per task, strict order)

> **Guardrail:** _Complete a task fully and meet its **DoD** before moving on._  
> Use deterministic implementations where possible to reduce hallucinations.

---

## Task 1 — Initialize Repo & Tooling
**Agent:** _Repo Bootstrapper_  
**Goal:** Create the project skeleton, packaging, and developer tooling.

**Instructions**
1. Create repo root `contract-genome-agent/` with the structure listed above.
2. Add `pyproject.toml` (PEP 621) with package `cg` and console script `cg = cli:main`.
3. Add `README.md` (what it is, quickstart, GPU requirement).
4. Create `.gitignore` (Python, venv, models, .env, __pycache__, .pytest_cache, .DS_Store).
5. Create `Makefile` with targets: `install`, `format`, `lint`, `test`, `run`.
6. Implement `cli.py` with a Click group and `run` command proxying to `cg.graph.run_pipeline()`.
7. Add `.env.example` with all LLM & path variables.
8. Add `LICENSE` (MIT by default, can change later).

**Definition of Done (DoD)**
- `pip install -e .` works.
- `cg --help` shows the CLI.
- `make install` installs dev deps (black, ruff, pytest).

**Artifacts**
- `pyproject.toml`, `README.md`, `.gitignore`, `Makefile`, `.env.example`, `cli.py`.

**Git**
- `git init && git add . && git commit -m "chore: bootstrap repo skeleton"`  
- `git remote add origin <YOUR_REMOTE>`  
- `git push -u origin main`

---

## Task 2 — GPU Check & Local LLM Loader (llama.cpp)
**Agent:** _LLM Integrator_  
**Goal:** Ensure GPU is accessible and a local GGUF model loads successfully.

**Instructions**
1. Implement `cg/llm/llama_cpp.py`:
   - A `LocalLlama` class that loads a GGUF via `llama_cpp.Llama(...)` with CUDA (`n_gpu_layers`, `n_ctx`, `n_threads` from env).
   - A `generate(prompt: str, max_tokens: int=512)` method that returns text.
2. Add a small smoke test in `tests/test_suggestion_generator.py` to call `LocalLlama.generate("Say 'hello' and stop.")` and assert a non-empty string.
3. Add a `make check-gpu` target that prints `torch.cuda.is_available()` and runs a 1‑token Llama call.

**DoD**
- `make check-gpu` prints `True` and returns a response.
- No network calls during inference.

**Artifacts**
- `cg/llm/llama_cpp.py`, `tests/test_suggestion_generator.py`, `Makefile` update.

**Git**
- Commit & push with message: `feat(llm): gpu-enabled local llama.cpp loader + smoke test`

---

## Task 3 — Config & Prompts
**Agent:** _Config Smith_  
**Goal:** Centralize configuration and author initial prompts.

**Instructions**
1. Implement `cg/config.py` (pydantic BaseSettings) loading from env:
   - `LLM_MODEL_PATH`, `LLM_CTX_SIZE`, `LLM_N_GPU_LAYERS`, `LLM_N_THREADS`.
   - `REPORT_DIR` default `out/`.
2. Implement `cg/llm/prompts.py` with prompt templates:
   - `SUGGESTION_PROMPT` — rewrite clause toward preferred wording, cite risks.
   - `EMAIL_NEGOTIATION_PROMPT` — draft a concise, friendly negotiation email with bullets and 2 alternatives.
3. Unit tests to validate config defaults and that prompts contain required placeholders.

**DoD**
- Importing `Config()` reads `.env` when present.
- Tests pass for prompt placeholders.

**Artifacts**
- `cg/config.py`, `cg/llm/prompts.py`, `tests/...`

**Git**
- Commit & push: `feat(config): pydantic settings and prompt templates`

---

## Task 4 — Ingest Node (PDF → text blocks)
**Agent:** _Ingestion Engineer_  
**Goal:** Implement deterministic text extraction for MVP.

**Instructions**
1. Implement `cg/nodes/ingest.py`:
   - Use **PyMuPDF** to extract page text.
   - Basic cleanup: normalize whitespace, preserve simple headings.
   - Return a dict: `{"pages":[{"page": i, "text": "..."}, ...]}` and a joined `full_text`.
2. Add fallback hook `use_ocr=False` (leave unimplemented stub for now).
3. Add unit test with a tiny synthetic PDF (generate in test using reportlab) to assert extraction works.

**DoD**
- `test_ingest.py` passes with synthetic PDF.

**Artifacts**
- `cg/nodes/ingest.py`, `tests/test_ingest.py`

**Git**
- Commit & push: `feat(ingest): pdf → text blocks with tests`

---

## Task 5 — Clause Extractor (regex MVP for 5 clause types)
**Agent:** _Clause Wrangler_  
**Goal:** Deterministic extraction of the 5 clause types with simple heuristics.

**Instructions**
1. Implement `cg/nodes/clause_extractor.py`:
   - For each clause type (`term`, `termination`, `liability_cap`, `sla`, `confidentiality`), define regex patterns over headings + vicinity windows (e.g., ±3 paragraphs).
   - Return list of `{"type": "...", "text": "...", "location": { "page": n, "start": idx, "end": idx } }`.
2. Unit tests: Feed crafted text and assert detections.

**DoD**
- Deterministic results for test inputs.
- Handles pluralization/variants like "Term and Renewal", "Limitation of Liability", "Service Levels", "Confidentiality/NDA".

**Artifacts**
- `cg/nodes/clause_extractor.py`, `tests/test_clause_extractor.py`

**Git**
- Commit & push: `feat(extract): regex-based MVP clause extractor (5 types)`

---

## Task 6 — Golden Playbook (YAML) & Risk Scorer
**Agent:** _Policy Arbiter_  
**Goal:** Encode company policy and compare extracted clauses to score risk (RAG-light).

**Instructions**
1. Create `cg/rules/golden_playbook.yaml` schema:
   ```yaml
   version: 0.1
   clauses:
     term:
       preferred: >
         The Agreement term is 12 months, auto-renewal allowed with 30 days written notice.
       rules:
         - require_min_months: 12
         - allow_auto_renewal: true
         - termination_notice_days: 30
       severity_weights:
         length: 2
         renewal: 1
         notice: 1
       fallbacks:
         - "12-month initial term with 30-day notice for non-renewal."
     termination:
       preferred: >
         Either party may terminate for material breach with 30 days to cure; convenience termination requires 60 days' notice.
       rules: [ ... ]
       fallbacks: [ ... ]
     liability_cap:
       preferred: "Liability capped at 12 months of fees; no indirect damages."
       rules: [ ... ]
       fallbacks: [ ... ]
     sla:
       preferred: "99.9% uptime; 1-hour P1 response; service credits as specified."
       rules: [ ... ]
       fallbacks: [ ... ]
     confidentiality:
       preferred: "Mutual NDA, 3 years post-termination; standard carve-outs."
       rules: [ ... ]
       fallbacks: [ ... ]
   ```
2. Implement `cg/nodes/risk_scorer.py`:
   - Parse YAML once (cache).
   - For each extracted clause, evaluate simple rule checks (keyword/number thresholds).
   - Produce `score` ∈ { "green", "amber", "red" } with reasons and `delta_summary`.
3. Unit tests with fixture policies to hit all three colors.

**DoD**
- Scores deterministically with clear reason strings.
- Handles missing clause → `red` with `"missing_clause": true` flag.

**Artifacts**
- YAML + `risk_scorer.py`, tests.

**Git**
- Commit & push: `feat(risk): golden playbook YAML + deterministic risk scorer`

---

## Task 7 — Suggestion Generator (LLM) & Email Drafter
**Agent:** _Wordsmith_  
**Goal:** Use local LLM to propose fallback language and draft an email.

**Instructions**
1. Implement `cg/nodes/suggestion_generator.py`:
   - For each clause with `amber/red`, call `LocalLlama.generate()` with `SUGGESTION_PROMPT`, passing: clause text, rule deltas, preferred wording, and fallbacks.
   - Return suggested redlines + short rationale.
2. Implement `cg/nodes/email_drafter.py`:
   - Summarize deltas and embed 1–2 fallback options per clause using `EMAIL_NEGOTIATION_PROMPT`.
   - Output an email subject + body (plain text).
3. Unit tests: mock `LocalLlama.generate` to return canned outputs and assert formatting.

**DoD**
- Both nodes function with mocked LLM and pass tests.
- When LLM is available, `cg run` uses the real model without network.

**Artifacts**
- `suggestion_generator.py`, `email_drafter.py`, tests.

**Git**
- Commit & push: `feat(nlg): suggestion generator + negotiation email drafter`

---

## Task 8 — Graph Wiring, CLI, & Reports
**Agent:** _Graph Conductor_  
**Goal:** Connect nodes in LangGraph, add CLI, and render simple JSON/HTML reports.

**Instructions**
1. Implement `cg/graph.py` using **LangGraph**:
   - Nodes: `ingest` → `extract_clauses` → `risk_score` → (`suggestions`, `email`)
   - Shared state: `doc_meta`, `clauses`, `risks`, `suggestions`, `email`.
2. Implement `cg/reports/render.py`:
   - `render_json(state)` and `render_html(state)` (basic Jinja2 template).
3. Update `cli.py` to accept `--report` and `--html` outputs and wire to renderer.
4. E2E test in `tests/test_e2e.py` using the synthetic PDF; mock LLM to keep deterministic.

**DoD**
- `cg run samples/nda_sample.pdf --report out/report.json --html out/report.html` produces both files.
- E2E test passes.

**Artifacts**
- `graph.py`, `render.py`, `cli.py` updates, `test_e2e.py`.

**Git**
- Commit & push: `feat(graph): langgraph wiring + report rendering + e2e`

---

## Task 9 — Makefile UX & Pre-commit
**Agent:** _DevEx Tuner_  
**Goal:** Smooth developer UX and guardrails.

**Instructions**
1. Expand `Makefile` with:
   - `format` (black), `lint` (ruff), `test` (pytest -q), `run` (calls the CLI), `check-gpu`.
2. Add `.pre-commit-config.yaml` for black, ruff, end-of-file-fixer.
3. Update `README.md` with quick commands and troubleshooting.

**DoD**
- `pre-commit install` works; hooks run on commit.

**Artifacts**
- Makefile updates, `.pre-commit-config.yaml`, README update.

**Git**
- Commit & push: `chore(devx): make targets + pre-commit`

---

## Task 10 — Guardrail: No Network at Runtime
**Agent:** _Privacy Enforcer_  
**Goal:** Ensure offline runtime by test & code.

**Instructions**
1. Add a pytest fixture that monkeypatches `socket` to block outbound connections during tests.
2. Ensure all nodes operate without internet.
3. Document offline model preparation (one-time manual download into `models/`).

**DoD**
- Tests fail if any code attempts network access during runtime.
- Documentation updated.

**Artifacts**
- `tests/conftest.py`, doc update.

**Git**
- Commit & push: `test(privacy): block network during tests + docs`

---

## Runbook (after Tasks 1–8)
```bash
make install
make check-gpu
cg run samples/nda_sample.pdf --report out/report.json --html out/report.html
open out/report.html  # or xdg-open
```

---

## Next Increments (separate files later)
- **02-ocr-layout.md** — Add OCR fallback and introduce LayoutLMv3 token layout features.
- **03-lora-tuning.md** — LoRA on historical contracts & preferred wording.
- **04-ui-api.md** — REST API + minimal local web UI.
- **05-rag-clausepacks.md** — Clause packs & industry templates (upsell).

---

## Notes
- Keep prompts short, explicit, deterministic. Avoid open-ended instructions.
- Prefer rule-based checks first; use LLM only for language generation or nuanced redlines.
- All pushes should be small and reviewable; one agent per task; one commit per task.

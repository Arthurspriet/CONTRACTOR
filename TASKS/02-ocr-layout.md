# Contract Genome Agent — 02 • OCR Fallback + Layout-Aware Clause Extraction (GPU)

**Objective (single, specific):**  
Add a reliable **OCR+layout** path for scanned PDFs and upgrade clause extraction to use **layout-aware sections** (LayoutLMv3 features when available, otherwise deterministic fallback). Produce **highlighted clause thumbnails** in the HTML report. Each subtask is executed by **one agent at a time** before proceeding.

---

## Context & Constraints
- **Offline-first**: no network at runtime. All models must be present on disk.
- **GPU used**: LLM already on GPU; LayoutLMv3 can run on GPU if weights present.
- **Python** stack with **LangGraph + LangChain** remains unchanged.
- **Backwards compatible**: text-only PDFs still work with the existing path.
- **Deterministic fallback**: If LayoutLMv3 assets are missing, extractor relies on headings/regex heuristics.

---

## Deliverables
- OCR pipeline integrated into graph: `ingest → (text_only | ocr_layout) → clause_extractor → risk_scorer → ...`
- `cg run ...` detects scanned PDFs and switches to OCR automatically.
- Layout-aware clause extraction that anchors on section headings (layout blocks) when available.
- HTML report shows **page images** with **bounding boxes** around extracted clause regions.
- Tests covering scanned PDFs, fallback paths, and no-network guard.

---

## Model Prep (offline)
Place models with the following structure (no downloads at runtime):
```
models/
├─ llm/
│  └─ mistral-7b-instruct.Q5_K_M.gguf           # already used
└─ layoutlmv3/
   ├─ config.json
   ├─ preprocessor_config.json
   ├─ special_tokens_map.json
   ├─ tokenizer.json
   ├─ tokenizer_config.json
   └─ model.safetensors
```
> Use any LayoutLMv3 checkpoint (preferably DocLayNet-tuned) compatible with `AutoProcessor` + `AutoModelForTokenClassification`. Update IDs/labels via the model's `config.id2label`.

Environment variables to add in `.env`:
```
OCR_BACKEND=tesseract         # options: tesseract, none
OCR_DPI=220                   # render DPI for OCR rasterization
LAYOUT_MODEL_DIR=models/layoutlmv3
LAYOUT_ENABLE=true            # if false, skip model and use heuristics
```

---

# TASKS (one agent per task, strict order)

> Continue numbering from the previous increment.

## Task 11 — Dependencies & Install Targets
**Agent:** _Build Steward_  
**Goal:** Add OCR/layout deps and make targets without breaking bootstrap.

**Instructions**
1. In `pyproject.toml` add:
   - `transformers`, `accelerate`, `Pillow`, `opencv-python-headless`, `pdf2image` (if using), but **prefer** PyMuPDF rasterization.
   - `pytesseract` (Python bindings). Note: system `tesseract` binary required (document in README).
2. Update `Makefile`:
   - `install-ocr`: installs new deps.
   - Ensure `make install` includes them or document a separate step.
3. Update `README.md`:
   - Add section for OCR with **Tesseract** install notes (Linux/macOS), Poppler optional if using `pdf2image`.
   - Note: We use **PyMuPDF** to rasterize by default (no Poppler needed).

**DoD**
- `pip install -e .` still works; `pytest -q` still runs.
- `import pytesseract` & `from transformers import AutoModel` succeed.

**Git**
- Commit & push: `chore(deps): add OCR/layout deps + install targets`

---

## Task 12 — Page Rasterization & OCR
**Agent:** _Vision Wrangler_  
**Goal:** Convert PDF pages to images and run OCR to obtain **words with bounding boxes**.

**Instructions**
1. Create `cg/ocr/ocr.py` with:
   - `render_pages(pdf_path, dpi) -> List[PIL.Image]` using **PyMuPDF** (`fitz`) to rasterize.
   - `ocr_page(image) -> {words: [ {text, bbox(x0,y0,x1,y1)} ], width, height}` using **pytesseract** (`image_to_data`).
   - `ocr_document(pdf_path, dpi) -> List[page_dict]`.
2. Add heuristic in `ingest` to detect scanned PDFs:
   - Rule: if `len(full_text.strip()) < MIN_CHARS` or ratio of glyphs/page < threshold → `use_ocr=True`.
   - Store `state["ocr_used"]=True` when triggered.
3. Tests (`tests/test_ocr.py`):
   - Generate a synthetic PDF with an **image of text** (draw text to an image, embed in PDF) so text extraction returns near-empty, but OCR recovers words.
   - Assert non-empty OCR words with bboxes.

**DoD**
- `ocr_document(...)` returns words with pixel coords and page size.
- Heuristic flips `use_ocr=True` for scanned sample.

**Git**
- Commit & push: `feat(ocr): rasterize with PyMuPDF + pytesseract word bboxes`

---

## Task 13 — Layout Features API
**Agent:** _Layout Engineer_  
**Goal:** Build a common **LayoutFeatures** interface used by downstream nodes, with or without a model.

**Instructions**
1. Create `cg/layout/features.py`:
   - Define dataclasses: `Word`, `Block`, `PageLayout`, `LayoutFeatures`.
   - Helpers to convert between pixel coords and **normalized [0,1]** coords.
   - Function `group_words_into_blocks(words) -> List[Block]` using simple XY-cut / line clustering (deterministic heuristic).
2. Create `cg/layout/model.py`:
   - If `LAYOUT_ENABLE=true` and `LAYOUT_MODEL_DIR` exists:
     - Load `AutoProcessor` + `AutoModelForTokenClassification` on **GPU if available**.
     - Build inputs from page image + OCR boxes (as per LayoutLMv3 expectations).
     - Run inference to label tokens/blocks (e.g., `TITLE`, `HEADER`, `PARA`, `LIST`, `TABLE`, etc.).
   - Expose `analyze_pages(images, ocr_pages) -> LayoutFeatures` merging model labels into blocks.
3. Tests (`tests/test_layout.py`):
   - Mock model forward pass to produce deterministic labels.
   - Assert blocks exist, labels propagate, and normalization works.

**DoD**
- `LayoutFeatures` produced for both paths: **heuristic-only** and **model-backed**.
- No network access; model optional.

**Git**
- Commit & push: `feat(layout): features API + optional LayoutLMv3 inference`

---

## Task 14 — Layout-Aware Clause Extraction
**Agent:** _Clause Cartographer_  
**Goal:** Upgrade clause extraction to **anchor on headings/blocks** and compute **region bboxes** per clause.

**Instructions**
1. Update `cg/nodes/clause_extractor.py` to accept `LayoutFeatures`:
   - Detect candidate **section headings** (block label in {TITLE, HEADER, H1/H2-like} or heuristic: bold/ALL CAPS/end with colon).
   - Map sections to text spans + **page/box regions**.
   - For each MVP clause type, search within the most relevant section window (same page and next N blocks) using regex synonyms:
     - term: `\bterm\b|\bduration\b|\brenewal\b`
     - termination: `\btermination\b|\bterminate\b|\bconvenience\b|\bbreach\b`
     - liability_cap: `\blimitation of liability\b|\bliability cap\b`
     - sla: `\bservice level\b|\bsla\b|\buptime\b`
     - confidentiality: `\bconfidential\b|\bnon-disclosure\b|\bnda\b`
   - Return entries include: `type, text, location: {page, bbox_norm:[x0,y0,x1,y1]}, section_title`.
2. Tests (`tests/test_clause_extractor_layout.py`):
   - Fabricate `LayoutFeatures` with a heading and following paragraphs; ensure extractor picks correct region and returns bbox.

**DoD**
- Extractor returns **both text and bbox** per clause, using layout if available, fallback to text offsets otherwise.

**Git**
- Commit & push: `feat(extract): layout-aware sections + clause region bboxes`

---

## Task 15 — Report Thumbnails with Highlights
**Agent:** _Report Artisan_  
**Goal:** Add **page thumbnails** to HTML with rectangles highlighting clause regions.

**Instructions**
1. In `cg/reports/render.py`:
   - During run, save page images to `out/pages/page_{i}.png` (already available from OCR step or re-render).
   - Draw rectangles for each clause region on a copy (use PIL).
   - Embed thumbnails in the HTML next to each clause entry.
2. Add minimal CSS for responsive image grid.
3. Tests (`tests/test_report_highlights.py`):
   - Generate a fake state with one clause having a bbox; run renderer; assert the PNG files exist and HTML references them.

**DoD**
- `out/report.html` shows visible highlighted boxes for extracted clauses.

**Git**
- Commit & push: `feat(report): clause highlight thumbnails in HTML`

---

## Task 16 — Graph Integration & Fallback Robustness
**Agent:** _Graph Orchestrator_  
**Goal:** Wire OCR + layout into the graph and make it robust when assets are missing.

**Instructions**
1. Update `cg/graph.py`:
   - After `ingest`, decide `path = text_only` or `ocr_layout` using the scanned-PDF heuristic.
   - If `LAYOUT_ENABLE` and model present, run model-backed layout; else, heuristic-only grouping.
   - Pass `layout_features` to `clause_extractor`.
2. Update CLI `cg run` options:
   - `--force-ocr/--no-force-ocr`
   - `--layout off|heuristic|model`
3. Extend E2E test to cover:
   - text-only PDF (no OCR path),
   - scanned PDF (OCR path),
   - layout model missing (heuristic path),
   - layout model present (mocked) path.

**DoD**
- All E2E permutations pass locally with **no network**.
- Logged warnings when model missing; execution continues.

**Git**
- Commit & push: `feat(graph): OCR+layout branch + robust fallbacks`

---


## Task 18 — Performance Optimization & Memory Management
**Agent:** _Performance Engineer_
**Goal:** Optimize OCR and layout processing for large documents while maintaining memory efficiency.

**Instructions**
1. Add memory management to OCR pipeline:
   - Process pages in batches (configurable `OCR_BATCH_SIZE`).
   - Clear page images from memory after OCR processing.
   - Add `--max-pages` CLI option to limit processing for testing.
2. Optimize layout model inference:
   - Implement model caching to avoid reloading between pages.
   - Add GPU memory monitoring and fallback to CPU if OOM.
   - Batch multiple pages through model when possible.
3. Add performance metrics:
   - Track OCR processing time per page.
   - Monitor layout model inference time.
   - Log memory usage at key checkpoints.
4. Tests (`tests/test_performance.py`):
   - Test with large PDF (100+ pages) to verify memory stability.
   - Assert processing completes without memory leaks.
   - Verify batch processing works correctly.

**DoD**
- Large documents process without memory issues.
- Performance metrics logged and accessible.
- Configurable batch sizes and limits.

**Git**
- Commit & push: `perf(ocr): memory management + batch processing + metrics`

---

## Task 19 — Validation & Quality Assurance
**Agent:** _Quality Guardian_
**Goal:** Implement comprehensive validation, error handling, and quality metrics for the OCR+layout pipeline.

**Instructions**
1. Add robust error handling throughout the pipeline:
   - OCR failures: graceful degradation with retry logic and fallback to text extraction.
   - Layout model errors: automatic fallback to heuristic grouping with clear logging.
   - Memory pressure: detect and handle OOM conditions gracefully.
   - Invalid bounding boxes: validate and clamp coordinates to page boundaries.
2. Implement quality metrics and validation:
   - OCR confidence scores and word-level quality assessment.
   - Layout block coherence validation (overlapping blocks, reasonable sizes).
   - Clause extraction accuracy metrics (precision/recall against known samples).
   - End-to-end pipeline health checks.
3. Add comprehensive logging and monitoring:
   - Structured logging with different levels (DEBUG, INFO, WARN, ERROR).
   - Performance timing for each pipeline stage.
   - Quality metrics logging for debugging and optimization.
   - Error tracking with context and recovery actions taken.
4. Create validation test suite (`tests/test_validation.py`):
   - Test error conditions: corrupted PDFs, missing models, invalid inputs.
   - Test quality metrics: verify confidence scores, validate bbox coordinates.
   - Test fallback mechanisms: ensure graceful degradation works.
   - Test edge cases: empty pages, single-word documents, extreme aspect ratios.
5. Add CLI validation commands:
   - `cg validate <pdf>`: run full pipeline validation on a document.
   - `cg health-check`: verify all dependencies and models are available.
   - `--validate-only`: run validation without generating reports.

**DoD**
- All error conditions handled gracefully with appropriate fallbacks.
- Quality metrics provide actionable insights for debugging.
- Validation commands help users diagnose issues.
- Comprehensive test coverage for edge cases and error conditions.

**Git**
- Commit & push: `feat(validation): error handling + quality metrics + health checks`

---

## Task 20 — Documentation & User Experience Finalization
**Agent:** _Documentation Specialist_  
**Goal:** Complete comprehensive documentation, create user guides, and ensure production-ready deployment experience.

**Instructions**
1. Create comprehensive documentation in `docs/`:
   - `docs/ocr-setup.md`: Step-by-step OCR installation guide with troubleshooting.
   - `docs/layout-models.md`: LayoutLMv3 model setup, configuration, and performance tuning.
   - `docs/api-reference.md`: Complete API documentation for all OCR/layout functions.
   - `docs/configuration.md`: All environment variables and configuration options explained.
2. Create user-friendly sample workflows:
   - Add `samples/` directory with diverse contract examples (text-only, scanned, mixed).
   - Create `samples/README.md` with example commands and expected outputs.
   - Include sample `.env` file with recommended settings.
3. Enhance CLI user experience:
   - Add `--help` text for all new OCR/layout options.
   - Add progress bars for long-running OCR operations.
   - Implement `--dry-run` mode to preview processing without execution.
   - Add `--verbose` mode with detailed pipeline logging.
4. Create deployment guides:
   - `docs/deployment.md`: Production deployment checklist and best practices.
   - Docker configuration examples for containerized deployment.
   - Performance tuning guide for different hardware configurations.
5. Quality assurance and final testing:
   - Run full test suite across different document types and configurations.
   - Verify all documentation examples work correctly.
   - Test installation process on fresh environment.
   - Benchmark performance and document typical processing times.

**DoD**
- Complete documentation covers all features and setup scenarios.
- New users can successfully set up and run OCR pipeline from documentation alone.
- All CLI options are intuitive and well-documented.
- Production deployment is fully documented with best practices.

**Git**
- Commit & push: `docs(final): comprehensive OCR+layout documentation + UX improvements`

---
## Runbook
```bash
make install-ocr
cg run samples/nda_scan.pdf --report out/report.json --html out/report.html
open out/report.html
```

---

## Notes
- We keep OCR on CPU for portability; **GPU** usage is already satisfied by the LLM.
- If you later prefer GPU OCR, consider PaddleOCR GPU — but pin exact CUDA wheels and remain offline.
- Keep prompts deterministic; use layout to narrow search windows and reduce LLM surface area.

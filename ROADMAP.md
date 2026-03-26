# 🗺️ Project Roadmap & Task Tracker (JUST A SKELETON RIGHT NOW)

I made this with an LLM just to get a structure down and then made a few edits/changes. We can scrap it / change it or just use bullets if prefered.



## 🎯 Project Goal
*[Brief 1-2 sentence description of the final deliverable. E.g., Build a reproducible data pipeline to analyze X and produce a final methodology report.]*

---

## 📅 Milestones
*The high-level phases of the project.*
- [X] **Phase 0: Research and Proposal** (Submit proposal: 2026-03-13)
  - Finish proposal doc
  - Submit on canvas
- [ ] **Phase 1: Infrastructure & Data Ingestion** (Target: 2026-03-27)
  - Set up Git and `uv` environment.
  - Source and download raw datasets.
- [ ] **Phase 2: Cleaning & Exploratory Analysis** (Target: YYYY-MM-DD)
  - Standardize data schemas.
  - Address missing variables and document exclusions.
- [ ] **Phase 3: Core Analysis / Modeling** (Target: YYYY-MM-DD)
  - Execute main analytical scripts.
  - Generate primary visualizations/tables.
- [ ] **Phase 4: Synthesis & Final Deliverables** (Target: YYYY-MM-DD)
  - Draft final report/presentation.
  - Code review and repository cleanup.

---

## 📋 Active Task Tracker
*Move tasks from To Do -> In Progress -> Done as we work. Assign names to avoid duplicated effort.*

### 🔴 To Do
- [ ] **Screenshots:** Get screenshots of data having been loaded for checkpoint (@Name)
- [ ] **Test model:** [Pass a subset of data through a model] (@Name)
- [ ] **Engineering:** Further clean data (@Name)

### 🟡 In Progress
- [ ] **Data:** Write extraction script for raw data (@Chloe)
- [ ] **Model:** [YOLO26 model] (@Ryan and Sabine)
- [ ] **Baseline Model:** [EfficientNet model] (@Mason)
### 🟢 Done
- [x] **Setup:** Initialize Git, `.gitignore`, and `pyproject.toml`
- [x] **Admin:** Form group and finalize project scope
- [X] **Setup:** Finalize `README.md` and verify `uv sync` works on all OS
- [X] **Proposal Draft:** Write proposal document
---

## 🧊 Backlog / The Icebox
*Ideas, extra features, or puzzles we want to solve only if time permits.*

- [ ] *[e.g., Automate the data extraction process]*
- [ ] *[e.g., Incorporate an additional historical dataset]*

---

## 📝 Key Decisions Log
*A quick record of structural choices.*

* **[2026-02-27]:** Decided to use `uv` for dependency management to ensure cross-platform compatibility without WSL.
* **[2026-03-26]:** Decided to use EfficientNet as baseline model. Will be using YOLO26 for main model.
# MCP / AI Tooling Notes (Bonus)

## Goal
This folder documents how we used AI-assisted tooling during the project workflow, what it helped with, and the limitations/guardrails we followed. The intent is transparency and reproducibility (not replacing the analysis).

## What we used AI for (and what we did ourselves)
We used AI as a **coding + documentation assistant**:
- Helped brainstorm Streamlit layout ideas (section organization, UI flow, sidebar controls to add later).
- Helped propose “report-ready” outputs (fit parameter summary table, chi²/RMS reporting).
- Helped spot common data-wrangling pitfalls (dtype coercion, missing column handling, NaNs).
- Helped with wording for notebook markdown (methods explanation, interpretation prompts).

We still **implemented and verified**:
- Data loading/cleaning from OGLE tables (column selection, numeric conversions, NaN handling).
- P–L model definition in log-period space and curve fitting (SciPy `curve_fit`).
- Plot generation and astronomy-specific conventions (magnitude axis inversion, labeling, subsets by type).

## Collaboration breakdown
- **Abdirahman:** initial Streamlit app structure (basic pages/plots), early data ingest and cleaning, initial fit/plot pipeline.
- **Kyle:** refined/updated Streamlit layout and plotting sections, additional comparison plots (type splits), cleanup and consistency passes.
- We both contributed to interpreting outputs and deciding what comparisons to show.

## Reproducibility / setup
Run from the project root:
```bash
streamlit run app.py
ls
ls
cat README.md
git status
git log -1 --name-only

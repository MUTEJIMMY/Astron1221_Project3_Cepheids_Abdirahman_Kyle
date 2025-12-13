# Cepheid Period–Luminosity Streamlit App (ASTRON 1221 Project 3)

This repo contains a Streamlit app that explores the Cepheid period–luminosity (P–L) relation using OGLE Cepheid catalog data.

## What it does
- Loads the Cepheid catalog from `asu.tsv`
- Computes log10(Period) and fits a linear P–L relation using SciPy `curve_fit`
- Visualizes:
  1) Apparent magnitude (Vmag) best-fit
  2) Absolute magnitude best-fit (using an assumed LMC distance modulus)
  3) Classical Cepheids vs other types
  4) FU vs FO vs DM vs other types

## How to run
1. Make sure you are in the repo folder
2. Install dependencies (example):
   - `pip install streamlit pandas numpy matplotlib scipy`
3. Run:
   - `streamlit run app.py`

## Files
- `app.py` — Streamlit application
- `asu.tsv` — OGLE Cepheid catalog table (input data)
- `Project3_Cepheids.ipynb` — notebook used for development / exploration

## Team contributions
- Abdirahman: initial Streamlit prototype, first working end-to-end pipeline (load → fit → plot), and first set of P–L visualizations.
- Kyle: refined Streamlit layout + plotting organization, cleaned up final app flow, and helped validate the fitting outputs.

## Notes / troubleshooting
During development we ran into a breaking change while sorting/filtering the LMC dataset (variable renames + filtering logic). After ~3 hours of debugging, we reverted to a known-good backup from GitHub and rebuilt forward from that stable version.

## AI usage & limitations
AI tools were used as a coding assistant for debugging suggestions and minor refactors, but the scientific choices (model form, fitting approach, and interpretation) were made by the team.
Limitations:
- Fit quality depends on the catalog filtering choices and assumes a simple linear model in log-period space.
- No full uncertainty propagation is performed beyond the covariance from `curve_fit`.
- Results are intended as an instructional exploration rather than a publication-grade calibration.# Astron 1221 Project 3 – Cepheids

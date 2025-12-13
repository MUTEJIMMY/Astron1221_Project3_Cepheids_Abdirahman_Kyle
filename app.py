import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ------------------------------------------------------------
# Streamlit page config
# ------------------------------------------------------------
st.set_page_config(page_title="Cepheid P–L Interactive App", layout="wide")

# ------------------------------------------------------------
# Load Data (cached)
# ------------------------------------------------------------
@st.cache_data
def load_data():
    """
    Load OGLE Cepheid catalog (LMC) from asu.tsv and return a cleaned DataFrame.
    Required columns: Per, Vmag, Imag, Type
    """
    df_raw = pd.read_csv("asu.tsv", sep="\t", comment="#")

    # Convert numeric columns safely (if missing, skip)
    num_cols = ["Per", "Vmag", "Imag", "E(B-V)"]
    for col in num_cols:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

    # Keep only required columns and drop rows missing any required value
    needed = ["Per", "Vmag", "Imag", "Type"]
    for col in needed:
        if col not in df_raw.columns:
            raise ValueError(f"Missing required column '{col}' in asu.tsv")

    clean = df_raw[needed].dropna(subset=needed).copy()
    return clean


clean = load_data()

# ------------------------------------------------------------
# Helper model(s) (keep your names)
# ------------------------------------------------------------
def pl_model(x, a, b):
    """Linear P–L model in log-period space: mag = a*logP + b"""
    return a * x + b


def linear(x, a, b):
    """Same as pl_model; kept for compatibility with your existing style."""
    return a * x + b


# ------------------------------------------------------------
# Helper: compute fit + uncertainty + residual stats
# ------------------------------------------------------------
def fit_pl(logP, mag):
    """
    Fit mag = a*logP + b using curve_fit.
    Returns (a, b, a_err, b_err, resid, rms, chi2_red).
    Uses RMS(resid) as empirical sigma for reduced chi^2 (so chi2_red tends to ~1).
    """
    popt, pcov = curve_fit(pl_model, logP, mag, maxfev=10000)
    a, b = popt
    a_err, b_err = np.sqrt(np.diag(pcov))

    resid = mag - pl_model(logP, a, b)

    # Empirical scatter (ddof=2 for two fitted params)
    rms = np.std(resid, ddof=2)

    # Reduced chi^2 using rms as sigma estimate
    dof = len(resid) - 2
    if dof > 0 and rms > 0:
        chi2 = np.sum((resid / rms) ** 2)
        chi2_red = chi2 / dof
    else:
        chi2_red = np.nan

    return a, b, a_err, b_err, resid, rms, chi2_red


# ------------------------------------------------------------
# Sidebar controls (this is the big “Streamlit upgrade”)
# ------------------------------------------------------------
st.sidebar.title("Controls")

band = st.sidebar.selectbox(
    "Magnitude band",
    options=["Vmag", "Imag"],
    index=0
)

mag_mode = st.sidebar.selectbox(
    "Magnitude mode",
    options=["Apparent", "Absolute (assume μ_LMC)"],
    index=0
)

mu_LMC = st.sidebar.number_input(
    "μ_LMC (distance modulus)",
    min_value=15.0,
    max_value=25.0,
    value=18.5,
    step=0.1
)

# Type filter
all_types = sorted(clean["Type"].astype(str).unique().tolist())
default_types = all_types  # default: include all
selected_types = st.sidebar.multiselect(
    "Cepheid types to include",
    options=all_types,
    default=default_types
)

# Period range slider (log-space)
logP_all = np.log10(clean["Per"].values)
logP_min, logP_max = float(np.nanmin(logP_all)), float(np.nanmax(logP_all))
logP_range = st.sidebar.slider(
    "log10(Period) range",
    min_value=logP_min,
    max_value=logP_max,
    value=(logP_min, logP_max)
)

show_fit = st.sidebar.checkbox("Show best-fit line", value=True)
show_resid = st.sidebar.checkbox("Show residual plot", value=True)
invert_mag_axis = st.sidebar.checkbox("Invert magnitude axis (astronomy convention)", value=True)

st.sidebar.markdown("---")
show_data_preview = st.sidebar.checkbox("Show filtered data preview", value=True)
show_download = st.sidebar.checkbox("Enable CSV download of filtered data", value=True)

# ------------------------------------------------------------
# Apply filters
# ------------------------------------------------------------
df = clean.copy()

df = df[df["Type"].astype(str).isin(selected_types)].copy()

df["logP"] = np.log10(df["Per"].values)
df = df[(df["logP"] >= logP_range[0]) & (df["logP"] <= logP_range[1])].copy()

if len(df) < 3:
    st.error("Not enough data points after filtering. Widen filters or select more types.")
    st.stop()

logP = df["logP"].to_numpy()
mag_app = df[band].to_numpy()

if mag_mode == "Absolute (assume μ_LMC)":
    mag = mag_app - mu_LMC
    y_label = f"{band} (absolute, μ={mu_LMC:.1f})"
else:
    mag = mag_app
    y_label = f"{band} (apparent)"

# ------------------------------------------------------------
# Main title + context
# ------------------------------------------------------------
st.title("Cepheid Period–Luminosity Relation Explorer (LMC)")
st.markdown(
    """
This app lets you interactively explore the Cepheid Period–Luminosity (P–L) relation using OGLE survey data.
Use the sidebar to filter Cepheid types, choose band (V/I), switch apparent vs assumed-absolute magnitudes,
and view best-fit parameters, residuals, and summary statistics.
"""
)

# ------------------------------------------------------------
# Fit and display key stats
# ------------------------------------------------------------
a, b, a_err, b_err, resid, rms, chi2_red = fit_pl(logP, mag)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("N points", f"{len(df)}")
c2.metric("Slope a", f"{a:.4f} ± {a_err:.4f}")
c3.metric("Intercept b", f"{b:.4f} ± {b_err:.4f}")
c4.metric("RMS(resid)", f"{rms:.4f} mag")
c5.metric("Reduced χ²", f"{chi2_red:.3f}" if np.isfinite(chi2_red) else "—")

st.caption(
    "Note: reduced χ² here uses RMS(residuals) as the uncertainty estimate, so it often lands near ~1 by construction."
)

# ------------------------------------------------------------
# Plot: P–L scatter + fit
# ------------------------------------------------------------
xfit = np.linspace(logP.min(), logP.max(), 250)

fig1, ax1 = plt.subplots(figsize=(9, 6))
ax1.scatter(logP, mag, s=18, alpha=0.7)

if show_fit:
    ax1.plot(
        xfit,
        pl_model(xfit, a, b),
        linewidth=2.5,
        label=f"Fit: mag = {a:.2f} logP + {b:.2f}"
    )

ax1.set_xlabel("log10(Period) [days]")
ax1.set_ylabel(y_label)
ax1.set_title("Period–Luminosity Relation (Filtered Selection)")

# In astronomy, magnitude axes are often inverted so brighter appears higher
if invert_mag_axis:
    ax1.invert_yaxis()

ax1.grid(True)

if show_fit:
    ax1.legend()

st.pyplot(fig1)

# ------------------------------------------------------------
# Plot: residuals (optional)
# ------------------------------------------------------------
if show_resid:
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.scatter(logP, resid, s=18, alpha=0.7)
    ax2.axhline(0, color="red")
    ax2.set_xlabel("log10(Period) [days]")
    ax2.set_ylabel("Residual (mag)")
    ax2.set_title("Residuals (data − model)")
    ax2.grid(True)
    # Do NOT invert residual axis (residual sign should be intuitive)
    st.pyplot(fig2)

# ------------------------------------------------------------
# Data preview + download
# ------------------------------------------------------------
if show_data_preview:
    st.subheader("Filtered data preview")
    st.dataframe(df.head(20), use_container_width=True)

if show_download:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered data as CSV",
        data=csv_bytes,
        file_name="cepheids_filtered.csv",
        mime="text/csv"
    )

# ------------------------------------------------------------
# Group comparison section (optional but nice)
# ------------------------------------------------------------
st.markdown("---")
st.header("Quick group comparisons (same filters apply)")

compare_mode = st.selectbox(
    "Choose a comparison",
    options=[
        "None",
        "Classical (FU/FO/DM) vs Other",
        "FU vs FO (fit both)",
    ],
    index=0
)

if compare_mode != "None":
    fig3, ax3 = plt.subplots(figsize=(9, 6))

    # Use the already-filtered df for comparisons
    if compare_mode == "Classical (FU/FO/DM) vs Other":
        classical_labels = ["FU", "FO", "DM"]
        df_classical = df[df["Type"].astype(str).isin(classical_labels)].copy()
        df_other = df[~df["Type"].astype(str).isin(classical_labels)].copy()

        # Plot classical
        ax3.scatter(df_classical["logP"], (df_classical[band].values - mu_LMC) if mag_mode.startswith("Absolute") else df_classical[band].values,
                    s=22, alpha=0.8, label="Classical (FU/FO/DM)")
        # Plot other
        ax3.scatter(df_other["logP"], (df_other[band].values - mu_LMC) if mag_mode.startswith("Absolute") else df_other[band].values,
                    s=22, alpha=0.8, label="Other")

        # Fit + line (only if enough points)
        if len(df_classical) >= 3:
            a_c, b_c, a_c_err, b_c_err, _, _, _ = fit_pl(
                df_classical["logP"].to_numpy(),
                (df_classical[band].to_numpy() - mu_LMC) if mag_mode.startswith("Absolute") else df_classical[band].to_numpy()
            )
            ax3.plot(xfit, pl_model(xfit, a_c, b_c), linewidth=2.5,
                     label=f"Classical fit: {a_c:.2f} logP + {b_c:.2f}")

        if len(df_other) >= 3:
            a_o, b_o, a_o_err, b_o_err, _, _, _ = fit_pl(
                df_other["logP"].to_numpy(),
                (df_other[band].to_numpy() - mu_LMC) if mag_mode.startswith("Absolute") else df_other[band].to_numpy()
            )
            ax3.plot(xfit, pl_model(xfit, a_o, b_o), linewidth=2.5,
                     label=f"Other fit: {a_o:.2f} logP + {b_o:.2f}")

        ax3.set_title("Classical vs Other (group comparison)")

    elif compare_mode == "FU vs FO (fit both)":
        df_fu = df[df["Type"].astype(str) == "FU"].copy()
        df_fo = df[df["Type"].astype(str) == "FO"].copy()

        y_fu = (df_fu[band].values - mu_LMC) if mag_mode.startswith("Absolute") else df_fu[band].values
        y_fo = (df_fo[band].values - mu_LMC) if mag_mode.startswith("Absolute") else df_fo[band].values

        ax3.scatter(df_fu["logP"], y_fu, s=22, alpha=0.85, label="FU")
        ax3.scatter(df_fo["logP"], y_fo, s=22, alpha=0.85, label="FO")

        if len(df_fu) >= 3:
            a_fu, b_fu, _, _, _, _, _ = fit_pl(df_fu["logP"].to_numpy(), y_fu)
            ax3.plot(xfit, pl_model(xfit, a_fu, b_fu), linewidth=2.5,
                     label=f"FU fit: {a_fu:.2f} logP + {b_fu:.2f}")

        if len(df_fo) >= 3:
            a_fo, b_fo, _, _, _, _, _ = fit_pl(df_fo["logP"].to_numpy(), y_fo)
            ax3.plot(xfit, pl_model(xfit, a_fo, b_fo), linewidth=2.5,
                     label=f"FO fit: {a_fo:.2f} logP + {b_fo:.2f}")

        ax3.set_title("FU vs FO (group comparison)")

    ax3.set_xlabel("log10(Period) [days]")
    ax3.set_ylabel(y_label)
    if invert_mag_axis:
        ax3.invert_yaxis()
    ax3.grid(True)
    ax3.legend()
    st.pyplot(fig3)

st.markdown(
    """
**Tip for your presentation/report:**  
Mention that users can dynamically filter types and period range, switch bands, and inspect fit parameters + residuals.
That’s the “interactive Streamlit feature” your instructor was asking for.
"""
)

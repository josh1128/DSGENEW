# dsge_dashboard.py
# -----------------------------------------------------------
# Streamlit app that runs:
#   1) Original model (DSGE.xlsx): IS (DlogGDP), Phillips (Dlog_CPI), Taylor (Nominal rate)
#      - Real rate uses lag 1: RR_{t-1} = i_{t-1} − π_{t-1}
#      - Taylor uses inflation gap (π_t − π*)
#      - Shocks: IS, Phillips, and Taylor (tightening/easing)
#      - Policy shock behavior selector
#      - LaTeX equations shown below charts
#      - Monetary transmission floor on IS real-rate coefficient (optional)
#   2) Simple NK (built-in)
#   3) Macro Dashboard (ported from Dash) — integrated BELOW the DSGE section
#      - Potential Output construction, HP filters, and relevant IS/PC/Taylor columns viewer
# -----------------------------------------------------------

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.filters.hp_filter import hpfilter
import plotly.graph_objects as go
import os

# =========================
# Page setup
# =========================
st.set_page_config(page_title="DSGE IRF + Macro Dashboard", layout="wide")
st.title("DSGE IRF Dashboard — IS, Phillips, Taylor")

st.markdown(
    "- **Units (Original model plots):** Real GDP growth **DlogGDP in %**, Inflation **DlogCPI in %**, "
    "and the **Taylor policy rate in decimals**.\n"
    "- **Taylor** uses the **inflation gap** \\((\\pi_t - \\pi^*)\\)."
)

# =========================
# Helpers
# =========================
def ensure_decimal_rate(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s/100.0 if np.nanmedian(np.abs(s.values)) > 1.0 else s

def fmt_coef(x: float, nd: int = 3) -> str:
    s = f"{x:.{nd}f}"; return f"+{s}" if x >= 0 else s

def build_latex_equation(const_val: float, terms: List[tuple], lhs: str, eps_symbol: str) -> str:
    rhs_terms = "" if not terms else " ".join([f"{fmt_coef(c)}\\,{sym}" for (c, sym) in terms])
    return rf"""\begin{{aligned}} {lhs} &= {const_val:.3f} {rhs_terms} + {eps_symbol} \end{{aligned}}"""

def row_from_params(params_index: pd.Index, values: Dict[str, float]) -> pd.DataFrame:
    cols = list(params_index)
    row = {c: (1.0 if c == "const" else float(values.get(c, 0.0))) for c in cols}
    return pd.DataFrame([row], columns=cols)

def predict_with_params(row_df: pd.DataFrame, beta: pd.Series) -> float:
    x = row_df[beta.index].iloc[0].values.astype(float)
    b = beta.values.astype(float)
    return float(np.dot(x, b))

# =========================
# Simple NK (built-in)
# =========================
@dataclass
class NKParamsSimple:
    sigma: float = 1.00
    kappa: float = 0.10
    phi_pi: float = 1.50
    phi_x: float = 0.125
    rho_i: float = 0.80
    rho_x: float = 0.50
    rho_r: float = 0.80
    rho_u: float = 0.50
    gamma_pi: float = 0.50

class SimpleNK3EqBuiltIn:
    def __init__(self, params: Optional[NKParamsSimple] = None):
        self.p = params or NKParamsSimple()

    def irf(self, shock="demand", T=24, size_pp=1.0, t0=0, rho_override=None):
        p = self.p
        x = np.zeros(T); pi = np.zeros(T); i = np.zeros(T)
        r_nat = np.zeros(T); u = np.zeros(T); e_i = np.zeros(T)

        if shock == "demand":
            r_nat[t0] = size_pp; rho_sh = p.rho_r if rho_override is None else rho_override
        elif shock == "cost":
            u[t0] = size_pp; rho_sh = p.rho_u if rho_override is None else rho_override
        elif shock == "policy":
            e_i[t0] = size_pp; rho_sh = None
        else:
            raise ValueError("shock must be 'demand','cost','policy'")

        for t in range(T):
            if t > t0:
                if shock == "demand": r_nat[t] += (rho_sh or 0.0) * r_nat[t-1]
                elif shock == "cost": u[t] += (rho_sh or 0.0) * u[t-1]

            x_lag = x[t-1] if t>0 else 0.0
            pi_lag = pi[t-1] if t>0 else 0.0
            i_lag = i[t-1] if t>0 else 0.0

            A_x = (1 - p.rho_i) * (p.phi_pi * p.kappa + p.phi_x) - p.kappa
            B_const = (
                p.rho_i * i_lag
                + ((1 - p.rho_i) * p.phi_pi * p.gamma_pi - p.gamma_pi) * pi_lag
                + ((1 - p.rho_i) * p.phi_pi - 1.0) * u[t]
                + e_i[t]
            )
            denom = 1.0 + (A_x / p.sigma)
            num = (p.rho_x * x_lag) - (B_const / p.sigma) + (r_nat[t] / p.sigma)
            x[t] = num / max(denom, 1e-8)
            pi[t] = p.gamma_pi * pi_lag + p.kappa * x[t] + u[t]
            i[t]  = p.rho_i * i_lag + (1 - p.rho_i) * (p.phi_pi * pi[t] + p.phi_x * x[t]) + e_i[t]

        return np.arange(T), x, pi, i

# =========================
# Sidebar (DSGE controls)
# =========================
with st.sidebar:
    st.header("Model selection")
    model_choice = st.selectbox("Choose model version", ["Original (DSGE.xlsx)", "Simple NK (built-in)"], index=0)

    st.header("Simulation settings")
    T = st.slider("Horizon (quarters)", 8, 60, 20, 1)

    neutral_rate_pct = st.number_input(
        "Neutral nominal policy rate — % annual",
        value=2.00, step=0.25, format="%.2f",
        help="Used for NK level plotting."
    )

    if model_choice == "Original (DSGE.xlsx)":
        xlf = st.file_uploader("Upload DSGE.xlsx (IS/PC/Taylor)", type=["xlsx"], key="upload_original",
                               help="If omitted, the app looks for 'DSGE.xlsx' next to this script.")
        fallback = Path(__file__).parent / "DSGE.xlsx"

        rho_sim = st.slider("Policy smoothing ρ (Taylor)", 0.0, 0.95, 0.80, 0.05)

        st.header("Inflation target for Taylor")
        use_sample_mean = st.checkbox("Use sample mean of DlogCPI as target π*", value=False)
        if use_sample_mean:
            target_annual_pct = None
            st.caption("π* will be set to sample mean (quarterly) after data loads.")
        else:
            target_annual_pct = st.slider("π* (annual %)", 0.0, 5.0, 2.0, 0.1)
        st.divider()

        st.header("Shock")
        shock_target = st.selectbox(
            "Apply shock to",
            ["None", "IS (Demand)", "Phillips (Supply)", "Taylor (Policy tightening)", "Taylor (Policy easing)"],
            index=0,
        )
        is_shock_size_pp = st.number_input("IS shock (Δ DlogGDP, pp)", value=0.50, step=0.10, format="%.2f")
        pc_shock_size_pp = st.number_input("Phillips shock (Δ DlogCPI, pp)", value=0.10, step=0.05, format="%.2f")
        policy_shock_bp_abs = st.number_input("Policy shock size (absolute bp)", value=100, step=5, format="%d")
        shock_quarter = st.slider("Shock timing (t)", 1, T-1, 1, 1)
        shock_persist = st.slider("Shock persistence ρ_shock", 0.0, 0.95, 0.0, 0.05)

        st.header("Policy shock behavior")
        policy_mode = st.radio(
            "How should the policy shock enter?",
            ["Add after smoothing (standard)", "Add to target (inside 1−ρ)", "Force local jump (override)"],
            index=0
        )

        st.divider()
        st.header("Variable selection (include/exclude)")

        IS_ALL = ["DlogGDP_L1", "Real_Rate_L1_data", "Dlog FD_Lag1", "Dlog_REER", "Dlog_Energy", "Dlog_NonEnergy"]
        PC_ALL = ["Dlog_CPI_L1", "DlogGDP_L1", "Dlog_Reer_L2", "Dlog_Energy_L1", "Dlog_Non_Energy_L1"]
        TR_ALL = ["Nominal_Rate_L1", "Inflation_Gap", "DlogGDP"]

        with st.expander("IS Curve regressors", expanded=True):
            is_selected = st.multiselect("Use these variables in the IS regression:", IS_ALL, default=IS_ALL, key="is_vars")
        with st.expander("Phillips Curve regressors", expanded=True):
            pc_selected = st.multiselect("Use these variables in the Phillips regression:", PC_ALL, default=PC_ALL, key="pc_vars")
        with st.expander("Taylor Rule regressors", expanded=True):
            tr_selected = st.multiselect("Use these variables in the Taylor (partial adjustment) regression:", TR_ALL, default=TR_ALL, key="tr_vars")

        st.divider()
        mon_pass_floor_pp = st.slider(
            "Min Δg from −100 bp (pp)",
            0.00, 0.50, 0.05, 0.01,
            help="Floor on policy→activity: a 100 bp rate cut must raise quarterly GDP growth by at least this many **percentage points** (shows mainly at t+1 because IS uses RR_{t-1})."
        )

    else:
        st.info("**Mapping:** IS→(σ, ρx, ρr) • Phillips→(κ, γπ, ρu) • Taylor→(φπ, φx, ρi)")
        st.header("Simple NK parameters (pp units)")
        st.subheader("IS")
        sigma = st.slider("σ", 0.2, 5.0, 1.00, 0.05)
        rho_x = st.slider("ρx", 0.0, 0.98, 0.50, 0.02)
        rho_r = st.slider("ρr", 0.0, 0.98, 0.80, 0.02)
        st.subheader("Phillips")
        kappa = st.slider("κ", 0.01, 0.50, 0.10, 0.01)
        gamma_pi = st.slider("γπ", 0.0, 0.95, 0.50, 0.05)
        rho_u = st.slider("ρu", 0.0, 0.98, 0.50, 0.02)
        st.subheader("Taylor")
        phi_pi = st.slider("φπ", 1.0, 3.0, 1.50, 0.05)
        phi_x = st.slider("φx", 0.00, 1.00, 0.125, 0.005)
        rho_i = st.slider("ρi", 0.0, 0.98, 0.80, 0.02)
        st.divider()
        st.header("Shock (IRF mode)")
        shock_type_nk = st.selectbox("Shock type", ["Demand (IS)", "Cost-push (Phillips)", "Policy (Taylor)"], index=0)
        shock_size_pp_nk = st.number_input("Shock size (pp)", value=1.00, step=0.25, format="%.2f")
        shock_quarter_nk = st.slider("Shock timing t", 1, T-1, 1, 1)
        shock_persist_nk = st.slider("Shock persistence ρ_shock", 0.0, 0.98, 0.80, 0.02)
        units_mode = st.radio("NK policy rate units", ["Deviation (pp)", "Level (decimal)"], index=1)

# =========================
# ORIGINAL MODEL (DSGE.xlsx)
# =========================
@st.cache_data(show_spinner=True)
def load_and_prepare_original(file_like_or_path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if file_like_or_path is None:
        raise FileNotFoundError("Upload DSGE.xlsx or place it beside this script.")
    excel_src = file_like_or_path
    is_df = pd.read_excel(excel_src, sheet_name="IS Curve")
    pc_df = pd.read_excel(excel_src, sheet_name="Phillips")
    tr_df = pd.read_excel(excel_src, sheet_name="Taylor")

    for d in (is_df, pc_df, tr_df):
        d["Date"] = pd.to_datetime(d["Date"], format="%Y-%m", errors="raise")

    df = (
        is_df.merge(pc_df, on="Date", how="inner")
             .merge(tr_df, on="Date", how="inner")
             .sort_values("Date")
             .set_index("Date")
    )

    df["Nominal Rate"] = ensure_decimal_rate(df["Nominal Rate"])
    df["DlogGDP_L1"] = df["DlogGDP"].shift(1)
    df["Dlog_CPI_L1"] = df["Dlog_CPI"].shift(1)
    df["Nominal_Rate_L1"] = df["Nominal Rate"].shift(1)
    # ---- RR lag 1 (not lag 2) ----
    df["Real_Rate_L1_data"] = (df["Nominal Rate"] - df["Dlog_CPI"]).shift(1)

    required_cols = [
        "DlogGDP", "DlogGDP_L1", "Dlog_CPI", "Dlog_CPI_L1",
        "Nominal Rate", "Nominal_Rate_L1", "Real_Rate_L1_data",
        "Dlog FD_Lag1", "Dlog_REER", "Dlog_Energy", "Dlog_NonEnergy",
        "Dlog_Reer_L2", "Dlog_Energy_L1", "Dlog_Non_Energy_L1",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing: raise KeyError(f"Missing required columns: {missing}")

    df_est = df.dropna(subset=required_cols).copy()
    if df_est.empty: raise ValueError("No rows remain after dropping NA for required columns. Check your data.")
    return df, df_est

def _clip_and_recentre(beta_raw: pd.Series, means: Dict[str, float], rules: Dict[str, str]) -> pd.Series:
    beta_new = beta_raw.copy()
    for var, rule in rules.items():
        if var in beta_new.index:
            if rule == "nonpos": beta_new[var] = min(float(beta_new[var]), 0.0)
            elif rule == "nonneg": beta_new[var] = max(float(beta_new[var]), 0.0)
    const = float(beta_raw.get("const", 0.0))
    shift = 0.0
    for var, rule in rules.items():
        if var in beta_raw.index:
            shift += (float(beta_raw[var]) - float(beta_new[var])) * float(means.get(var, 0.0))
    beta_new["const"] = const + shift
    return beta_new

def fit_models_original(df_est: pd.DataFrame, pi_star_quarterly: float,
                        is_selected: List[str], pc_selected: List[str], tr_selected: List[str],
                        mon_pass_floor_pp: float):
    # ---------- IS ----------
    if not is_selected:
        raise ValueError("Select at least one regressor for IS (besides constant).")
    X_is = sm.add_constant(df_est[is_selected], has_constant="add")
    y_is = df_est["DlogGDP"]
    model_is = sm.OLS(y_is, X_is).fit()
    X_is_means = {c: float(X_is[c].mean()) for c in X_is.columns if c != "const"}
    beta_is_sim = _clip_and_recentre(model_is.params, X_is_means, rules={"Real_Rate_L1_data": "nonpos"})

    # Pass-through floor: 100bp cut → ≥ mon_pass_floor_pp pp ↑ in growth
    floor_abs = (mon_pass_floor_pp / 100.0) / 0.01  # pp→decimal / 0.01
    if "Real_Rate_L1_data" in beta_is_sim.index:
        current = float(beta_is_sim["Real_Rate_L1_data"])
        desired = -max(abs(current), floor_abs)
        if desired != current:
            mean_rr = float(X_is_means.get("Real_Rate_L1_data", 0.0))
            beta_is_sim["const"] = float(beta_is_sim.get("const", 0.0)) + (current - desired) * mean_rr
            beta_is_sim["Real_Rate_L1_data"] = desired

    # ---------- Phillips ----------
    if not pc_selected:
        raise ValueError("Select at least one regressor for Phillips (besides constant).")
    X_pc = sm.add_constant(df_est[pc_selected], has_constant="add")
    y_pc = df_est["Dlog_CPI"]
    model_pc = sm.OLS(y_pc, X_pc).fit()
    X_pc_means = {c: float(X_pc[c].mean()) for c in X_pc.columns if c != "const"}
    beta_pc_sim = _clip_and_recentre(model_pc.params, X_pc_means, rules={"DlogGDP_L1": "nonneg"})

    # ---------- Taylor (partial adjustment with inflation gap) ----------
    infl_gap_full = df_est["Dlog_CPI"] - pi_star_quarterly
    df_tr = pd.DataFrame(index=df_est.index)
    if "Nominal_Rate_L1" in tr_selected: df_tr["Nominal_Rate_L1"] = df_est["Nominal_Rate_L1"]
    if "Inflation_Gap" in tr_selected:   df_tr["Inflation_Gap"]   = infl_gap_full
    if "DlogGDP" in tr_selected:         df_tr["DlogGDP"]         = df_est["DlogGDP"]
    if df_tr.empty: raise ValueError("Select at least one regressor for Taylor (besides constant).")

    X_tr = sm.add_constant(df_tr, has_constant="add")
    y_tr = df_est["Nominal Rate"]
    model_tr = sm.OLS(y_tr, X_tr).fit()

    b0 = float(model_tr.params.get("const", 0.0))
    rhoh = min(max(float(model_tr.params.get("Nominal_Rate_L1", 0.0)), 0.0), 0.99)
    def safe_div(num, den): return num / den if abs(den) > 1e-8 else np.nan
    alpha_star = safe_div(b0, (1 - rhoh))
    bpi = float(model_tr.params.get("Inflation_Gap", 0.0))
    bg  = float(model_tr.params.get("DlogGDP", 0.0))
    phi_pi_star_raw = safe_div(bpi, (1 - rhoh)) if "Inflation_Gap" in model_tr.params.index else np.nan
    phi_g_star_raw  = safe_div(bg,  (1 - rhoh)) if "DlogGDP"      in model_tr.params.index else np.nan
    phi_pi_star_sim = np.nan if np.isnan(phi_pi_star_raw) else max(0.0, phi_pi_star_raw)
    phi_g_star_sim  = np.nan if np.isnan(phi_g_star_raw)  else max(0.0,  phi_g_star_raw)

    p_ss = float(df_est["Dlog_CPI"].mean())
    g_ss = float(df_est["DlogGDP"].mean())

    return {
        "model_is": model_is, "model_pc": model_pc, "model_tr": model_tr,
        "beta_is_sim": beta_is_sim, "beta_pc_sim": beta_pc_sim,
        "alpha_star": alpha_star, "rho_hat": rhoh,
        "phi_pi_star_sim": phi_pi_star_sim, "phi_g_star_sim": phi_g_star_sim,
        "pi_star_quarterly": float(pi_star_quarterly),
        "p_ss": p_ss, "g_ss": g_ss
    }

def build_shocks_original(T, target, is_size_pp, pc_size_pp, policy_bp_abs, t0, rho):
    is_arr = np.zeros(T); pc_arr = np.zeros(T); pol_arr = np.zeros(T)
    target_norm = (target or "None").strip().lower()
    if target_norm == "is (demand)".lower():
        is_arr[t0] = is_size_pp / 100.0
        for k in range(t0 + 1, T): is_arr[k] = rho * is_arr[k - 1]
    elif target_norm == "phillips (supply)".lower():
        pc_arr[t0] = pc_size_pp / 100.0
        for k in range(t0 + 1, T): pc_arr[k] = rho * pc_arr[k - 1]
    elif target_norm == "taylor (policy tightening)".lower():
        pol_arr[t0] =  (policy_bp_abs / 10000.0)
        for k in range(t0 + 1, T): pol_arr[k] = rho * pol_arr[k - 1]
    elif target_norm == "taylor (policy easing)".lower():
        pol_arr[t0] = -(policy_bp_abs / 10000.0)
        for k in range(t0 + 1, T): pol_arr[k] = rho * pol_arr[k - 1]
    return is_arr, pc_arr, pol_arr

def simulate_original(
    T: int, rho_sim: float, df_est: pd.DataFrame,
    models: Dict[str, sm.regression.linear_model.RegressionResultsWrapper],
    means: Dict[str, float], i_mean_dec: float, real_rate_mean_dec: float,
    pi_star_quarterly: float, is_shock_arr=None, pc_shock_arr=None, policy_shock_arr=None,
    policy_mode: str = "Add after smoothing (standard)", neutral_dec: float = 0.02
):
    """Forward-simulate GDP growth (g), inflation (p), and the nominal rate (i).
       Units: g & p in decimals internally; i in decimals."""
    g = np.zeros(T); p = np.zeros(T); i = np.zeros(T)
    g[0] = float(df_est["DlogGDP"].mean())
    p[0] = float(df_est["Dlog_CPI"].mean())
    i[0] = i_mean_dec

    model_is = models["model_is"]; model_pc = models["model_pc"]
    beta_is_sim = models["beta_is_sim"]; beta_pc_sim = models["beta_pc_sim"]
    phi_pi_star_sim = models.get("phi_pi_star_sim", np.nan)
    phi_g_star_sim  = models.get("phi_g_star_sim",  np.nan)
    p_ss = models["p_ss"]; g_ss = models["g_ss"]

    if is_shock_arr is None: is_shock_arr = np.zeros(T)
    if pc_shock_arr is None: pc_shock_arr = np.zeros(T)
    if policy_shock_arr is None: policy_shock_arr = np.zeros(T)

    # Intercept so long-run equals chosen neutral rate
    alpha_star_sim = neutral_dec - (0.0 if np.isnan(phi_pi_star_sim) else phi_pi_star_sim) * (p_ss - pi_star_quarterly)

    for t in range(1, T):
        # Real rate with 1-quarter lag (decimal)
        rr_lag1 = (i[t - 1] - p[t - 1]) if t >= 1 else real_rate_mean_dec

        # IS (g_t)
        vals_is = {
            "DlogGDP_L1": g[t - 1],
            "Real_Rate_L1_data": rr_lag1,
            "Dlog FD_Lag1": means["Dlog FD_Lag1"],
            "Dlog_REER": means["Dlog_REER"],
            "Dlog_Energy": means["Dlog_Energy"],
            "Dlog_NonEnergy": means["Dlog_NonEnergy"],
        }
        Xis = row_from_params(model_is.params.index, vals_is)
        g[t] = predict_with_params(Xis, beta_is_sim) + is_shock_arr[t]

        # Phillips (p_t)
        vals_pc = {
            "Dlog_CPI_L1": p[t - 1],
            "DlogGDP_L1": g[t - 1],
            "Dlog_Reer_L2": means["Dlog_Reer_L2"],
            "Dlog_Energy_L1": means["Dlog_Energy_L1"],
            "Dlog_Non_Energy_L1": means["Dlog_Non_Energy_L1"],
        }
        Xpc = row_from_params(model_pc.params.index, vals_pc)
        p[t] = predict_with_params(Xpc, beta_pc_sim) + pc_shock_arr[t]

        # Taylor target and partial adjustment (i_t)
        pi_gap_t = p[t] - pi_star_quarterly
        g_dev_t  = g[t] - g_ss
        i_star = (alpha_star_sim
                  + (0.0 if np.isnan(phi_pi_star_sim) else phi_pi_star_sim) * pi_gap_t
                  + (0.0 if np.isnan(phi_g_star_sim)  else phi_g_star_sim)  * g_dev_t)

        eps = policy_shock_arr[t]  # decimal (e.g., 0.01 = 100 bp)
        if policy_mode.startswith("Add after"):
            i_raw = rho_sim * i[t - 1] + (1 - rho_sim) * i_star + eps
        elif policy_mode.startswith("Add to target"):
            i_raw = rho_sim * i[t - 1] + (1 - rho_sim) * (i_star + eps)
        else:  # local-jump override (guard a minimum jump)
            i_raw = rho_sim * i[t - 1] + (1 - rho_sim) * i_star + eps
            if eps > 0: i_raw = max(i_raw, i[t - 1] + abs(eps))
            elif eps < 0: i_raw = min(i_raw, i[t - 1] - abs(eps))
        i[t] = float(i_raw)

    return g, p, i

# =========================
# Run selected model
# =========================
try:
    if model_choice == "Original (DSGE.xlsx)":
        file_source = xlf if 'xlf' in locals() and xlf is not None else (fallback if 'fallback' in locals() else None)
        df_all, df_est = load_and_prepare_original(file_source)

        # Determine π* (quarterly decimal)
        if 'use_sample_mean' in locals() and use_sample_mean:
            pi_star_quarterly = float(df_est["Dlog_CPI"].mean())
            st.info(f"π* set to sample mean of DlogCPI: {pi_star_quarterly:.4f} (quarterly decimal)")
        else:
            annual_pct = target_annual_pct if 'target_annual_pct' in locals() and target_annual_pct is not None else 2.0
            pi_star_quarterly = (annual_pct / 100.0) / 4.0
            st.info(f"π* set to {annual_pct:.2f}% annual ⇒ {pi_star_quarterly:.4f} quarterly (decimal)")

        # Fit with selected regressors (with pass-through floor)
        models_o = fit_models_original(df_est, pi_star_quarterly, is_selected, pc_selected, tr_selected, mon_pass_floor_pp)

        # Warn if policy shock but Real_Rate_L1_data is excluded (kills transmission)
        if isinstance(shock_target, str) and "taylor" in shock_target.lower() and "Real_Rate_L1_data" not in is_selected:
            st.warning("Policy shock selected but **Real_Rate_L1_data** is not in the IS regressors — "
                       "add it back to allow policy to affect GDP.")

        # Anchors & means
        i_mean_dec = float(df_est["Nominal Rate"].mean())
        real_rate_mean_dec = float(df_est["Real_Rate_L1_data"].mean())
        means_o = {
            "Dlog FD_Lag1": float(df_est["Dlog FD_Lag1"].mean()),
            "Dlog_REER": float(df_est["Dlog_REER"].mean()),
            "Dlog_Energy": float(df_est["Dlog_Energy"].mean()),
            "Dlog_NonEnergy": float(df_est["Dlog_NonEnergy"].mean()),
            "Dlog_Reer_L2": float(df_est["Dlog_Reer_L2"].mean()),
            "Dlog_Energy_L1": float(df_est["Dlog_Energy_L1"].mean()),
            "Dlog_Non_Energy_L1": float(df_est["Dlog_Non_Energy_L1"].mean()),
        }

        # Build shocks & simulate
        is_arr, pc_arr, pol_arr = build_shocks_original(
            T, shock_target, is_shock_size_pp, pc_shock_size_pp, policy_shock_bp_abs, shock_quarter, shock_persist
        )
        neutral_dec = neutral_rate_pct / 100.0

        g0, p0, i0 = simulate_original(
            T, rho_sim, df_est, models_o, means_o, i_mean_dec, real_rate_mean_dec, pi_star_quarterly,
            policy_mode=policy_mode, neutral_dec=neutral_dec
        )
        gS, pS, iS = simulate_original(
            T, rho_sim, df_est, models_o, means_o, i_mean_dec, real_rate_mean_dec, pi_star_quarterly,
            is_shock_arr=is_arr, pc_shock_arr=pc_arr, policy_shock_arr=pol_arr,
            policy_mode=policy_mode, neutral_dec=neutral_dec
        )

        # ===== Plot (LEVELS): g & p in %, i in decimal =====
        plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 12, "legend.fontsize": 11})
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        quarters = np.arange(T); vline_kwargs = dict(color="black", linestyle=":", linewidth=1)

        axes[0].plot(quarters, g0*100, label="Baseline", linewidth=2)
        axes[0].plot(quarters, gS*100, label="Shock", linewidth=2)
        axes[0].axvline(shock_quarter, **vline_kwargs)
        axes[0].set_title("Real GDP Growth (DlogGDP, %)"); axes[0].set_ylabel("%")
        axes[0].grid(True, alpha=0.3); axes[0].legend(loc="best")

        axes[1].plot(quarters, p0*100, label="Baseline", linewidth=2)
        axes[1].plot(quarters, pS*100, label="Shock", linewidth=2)
        axes[1].axvline(shock_quarter, **vline_kwargs)
        axes[1].set_title("Inflation (DlogCPI, %)"); axes[1].set_ylabel("%")
        axes[1].grid(True, alpha=0.3); axes[1].legend(loc="best")

        axes[2].plot(quarters, i0, label="Baseline", linewidth=2)
        axes[2].plot(quarters, iS, label="Shock", linewidth=2)
        axes[2].axvline(shock_quarter, **vline_kwargs)
        axes[2].set_title("Nominal Policy Rate (decimal)")
        axes[2].set_xlabel("Quarters ahead"); axes[2].set_ylabel("decimal")
        axes[2].grid(True, alpha=0.3); axes[2].legend(loc="best")

        plt.tight_layout(); st.pyplot(fig)

        # Readout
        if isinstance(shock_target, str) and "taylor" in shock_target.lower():
            delta_i_bp = (iS - i0)[shock_quarter] * 10000.0
            st.info(f"Δ policy rate at t={shock_quarter}: {delta_i_bp:.1f} bp  |  mode: {policy_mode}  |  ρ={rho_sim:.2f}")
            st.caption("Policy affects GDP at **t+1** (IS uses RR_{t-1}).")

        # ===== LaTeX equations (display raw OLS) =====
        st.subheader("Estimated Equations (Original model)")
        m_is = models_o["model_is"]; m_pc = models_o["model_pc"]; m_tr = models_o["model_tr"]
        alpha_star = models_o["alpha_star"]; rho_hat = models_o["rho_hat"]
        phi_pi_star = models_o["phi_pi_star_sim"]; phi_g_star = models_o["phi_g_star_sim"]

        # IS
        is_terms = []
        pretty_map_is = {
            "DlogGDP_L1": r"\Delta \log GDP_{t-1}",
            "Real_Rate_L1_data": r"RR_{t-1}",
            "Dlog FD_Lag1": r"\Delta \log FD_{t-1}",
            "Dlog_REER": r"\Delta \log REER_t",
            "Dlog_Energy": r"\Delta \log Energy_t",
            "Dlog_NonEnergy": r"\Delta \log NonEnergy_t",
        }
        for k, v in m_is.params.items():
            if k == "const": continue
            is_terms.append((float(v), pretty_map_is.get(k, k)))
        st.markdown("**IS Curve (\\(\\Delta \\log GDP_t\\))**")
        st.latex(build_latex_equation(float(m_is.params.get("const", 0.0)), is_terms, r"\Delta \log GDP_t", r"\varepsilon_t"))

        # Phillips
        pc_terms = []
        pretty_map_pc = {
            "Dlog_CPI_L1": r"\Delta \log CPI_{t-1}",
            "DlogGDP_L1": r"\Delta \log GDP_{t-1}",
            "Dlog_Reer_L2": r"\Delta \log REER_{t-2}",
            "Dlog_Energy_L1": r"\Delta \log Energy_{t-1}",
            "Dlog_Non_Energy_L1": r"\Delta \log NonEnergy_{t-1}",
        }
        for k, v in m_pc.params.items():
            if k == "const": continue
            pc_terms.append((float(v), pretty_map_pc.get(k, k)))
        st.markdown("**Phillips Curve (\\(\\Delta \\log CPI_t\\))**")
        st.latex(build_latex_equation(float(m_pc.params.get("const", 0.0)), pc_terms, r"\Delta \log CPI_t", r"u_t"))

        # Taylor rule
        st.markdown("**Taylor Rule (partial adjustment, with inflation gap)**")
        if policy_mode.startswith("Add after"):
            st.latex(r"i_t \;=\; \rho\, i_{t-1} \;+\; (1-\rho)\, i_t^\* \;+\; \varepsilon^{\text{pol}}_t")
        elif policy_mode.startswith("Add to target"):
            st.latex(r"i_t \;=\; \rho\, i_{t-1} \;+\; (1-\rho)\,\big(i_t^\* + \varepsilon^{\text{pol}}_t\big)")
        else:
            st.latex(r"i_t \;=\; \rho\, i_{t-1} \;+\; (1-\rho)\, i_t^\* \;+\; \varepsilon^{\text{pol}}_t \quad (\text{with local-jump override})")

        st.latex(r"i_t^\* \;=\; \alpha^\*_{\text{sim}} \;+\; \phi_{\pi}^\*\,(\pi_t - \pi^\*) \;+\; \phi_{g}^\*\,\big(g_t - \bar g\big)")
        st.caption("Clips enforce: IS real-rate ≤ 0 (with floor), Phillips activity ≥ 0, Taylor φ’s ≥ 0; α* set so long-run equals the neutral rate.")

    else:
        # =========================
        # Simple NK (built-in)
        # =========================
        P = NKParamsSimple(
            sigma=sigma, kappa=kappa, phi_pi=phi_pi, phi_x=phi_x, rho_i=rho_i,
            rho_x=rho_x, rho_r=rho_r, rho_u=rho_u, gamma_pi=gamma_pi
        )
        model = SimpleNK3EqBuiltIn(P)
        label_to_code = {"Demand (IS)": "demand", "Cost-push (Phillips)": "cost", "Policy (Taylor)": "policy"}
        code = label_to_code[shock_type_nk]
        t0 = max(0, min(T-1, shock_quarter_nk - 1))
        rho_for_shock = shock_persist_nk

        st.subheader("Impulse responses")
        h, x0, pi0, i0 = model.irf(code, T, 0.0, t0, rho_for_shock)
        h, xS, piS, iS = model.irf(code, T, shock_size_pp_nk, t0, rho_for_shock)

        # Plot: keep x, π in pp; show rate in **decimal** when in level mode
        neutral_dec = neutral_rate_pct / 100.0
        if units_mode == "Level (decimal)":
            i0_plot = neutral_dec + (i0 / 100.0)     # convert pp to decimal
            iS_plot = neutral_dec + (iS / 100.0)
            i_ylabel = "decimal"
            i_title = "Nominal Policy Rate (level, decimal)"
        else:
            i0_plot = i0
            iS_plot = iS
            i_ylabel = "pp"
            i_title = "Nominal Policy Rate (deviation, pp)"

        plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 12, "legend.fontsize": 11})
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        vline_kwargs = dict(color="black", linestyle=":", linewidth=1)

        axes[0].plot(h, x0, linewidth=2, label="Baseline")
        axes[0].plot(h, xS, linewidth=2, label="Shock")
        axes[0].axvline(t0, **vline_kwargs); axes[0].set_title("Output Gap (x_t, pp)"); axes[0].set_ylabel("pp")
        axes[0].grid(True, alpha=0.3); axes[0].legend(loc="best")

        axes[1].plot(h, pi0, linewidth=2, label="Baseline")
        axes[1].plot(h, piS, linewidth=2, label="Shock")
        axes[1].axvline(t0, **vline_kwargs); axes[1].set_title("Inflation (π_t, pp)"); axes[1].set_ylabel("pp")
        axes[1].grid(True, alpha=0.3); axes[1].legend(loc="best")

        axes[2].plot(h, i0_plot, linewidth=2, label="Baseline")
        axes[2].plot(h, iS_plot, linewidth=2, label="Shock")
        axes[2].axvline(t0, **vline_kwargs)
        axes[2].set_title(i_title)
        axes[2].set_xlabel("Quarters ahead"); axes[2].set_ylabel(i_ylabel)
        axes[2].grid(True, alpha=0.3); axes[2].legend(loc="best")

        plt.tight_layout(); st.pyplot(fig)

        with st.expander("Simple NK equations"):
            st.latex(r"x_t = \rho_x x_{t-1} \;-\; \frac{1}{\sigma}\big( i_t - \pi_{t+1} - r^n_t \big)")
            st.latex(r"\pi_t = \gamma_\pi \pi_{t-1} \;+\; \kappa x_t \;+\; u_t")
            st.latex(r"i_t = \rho_i i_{t-1} \;+\; (1-\rho_i)(\phi_\pi \pi_t + \phi_x x_t) \;+\; \varepsilon^i_t")

except Exception as e:
    st.error(f"Problem loading or running the selected model: {e}")
    st.stop()

# ============================================================
# ===============  MACRO DASHBOARD (BELOW DSGE)  =============
# ============================================================
st.markdown("---")
st.header("Macro Dashboard (below DSGE)")

with st.expander("Data source & settings", expanded=True):
    colA, colB = st.columns([2,1])
    with colA:
        dash_file = st.file_uploader("Upload macro workbook (test.xlsx)", type=["xlsx"], key="macro_xlsx")
        default_path = st.text_input("...or provide a file path", value=r"C:\Users\AC03537\OneDrive - Alberta Central\Desktop\test.xlsx")
    with colB:
        run_build = st.checkbox("(Re)build Potential Output", value=True,
                                help="Recompute potential output, growth, and HP filters.")

if dash_file is None and not os.path.exists(default_path):
    st.warning("Upload `test.xlsx` or provide a valid path to show the dashboard.")
    st.stop()

excel_src = dash_file if dash_file is not None else default_path
sheet_main = 'Potential Output'
sheet_hours = 'Hours'
sheet_is_curve = 'IS Curve'
sheet_phillips = 'Phillips Curve'
sheet_taylor = 'Taylor Rule'

@st.cache_data(show_spinner=True)
def apply_hp_filter(df, column, prefix=None, log_transform=False, exp_transform=False):
    if column not in df.columns:
        return df
    prefix = prefix or column
    series = df[column].replace(0, np.nan).dropna()
    if len(series) < 10:
        return df
    clean_series = np.log(series) if log_transform else series
    cycle, trend = hpfilter(clean_series, lamb=1600)
    if exp_transform:
        trend = np.exp(trend)
    df[f"{prefix}_Trend"] = trend.reindex(df.index)
    df[f"{prefix}_Cycle"] = cycle.reindex(df.index)
    return df

@st.cache_data(show_spinner=True)
def load_macro(excel_src):
    xls = pd.ExcelFile(excel_src)

    # Potential Output block
    data = pd.read_excel(xls, sheet_name=sheet_main, na_values=["NA"])
    cols = ['Date', 'Population', 'Labour Force Participation', 'Labour Productivity', 'NAIRU',
            'Output_Gap_multivariate', 'Output_Gap_Integrated', 'Output_Gap_Internal', 'Real GDP Expenditure']
    for c in cols:
        if c not in data.columns:
            data[c] = np.nan
    data[cols[1:]] = data[cols[1:]].apply(pd.to_numeric, errors='coerce')
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data = data.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    data['Year'] = data['Date'].dt.year

    hours_df = pd.read_excel(xls, sheet_name=sheet_hours, na_values=["NA"])
    hours_df['Date'] = pd.to_datetime(hours_df['Date'], format='%Y-%m', errors='coerce')
    hours_df = hours_df.dropna(subset=['Date'])
    if 'Average Hours Worked' not in hours_df.columns:
        hours_df['Average Hours Worked'] = np.nan
    data = pd.merge_asof(data.sort_values('Date'), hours_df.sort_values('Date'), on='Date', direction='backward')

    # Inputs HP filters
    for col in ['Labour Force Participation', 'Labour Productivity', 'Average Hours Worked', 'Unemployment']:
        if col in data.columns:
            apply_hp_filter(data, col)

    # Potential Output build
    data['LFP_trend_decimal'] = (data['Labour Force Participation_Trend'] if 'Labour Force Participation_Trend' in data.columns
                                 else data['Labour Force Participation']) / 100.0
    data['NAIRU_decimal'] = data['NAIRU'] / 100.0

    # Base year (use 2017 if present; else first obs)
    base_year = 2017
    base_idx = data[data['Date'].dt.year == base_year].index
    base_idx = base_idx[0] if len(base_idx) else data.index[0]

    ahw = data['Average Hours Worked_Trend'] if 'Average Hours Worked_Trend' in data.columns else data['Average Hours Worked']
    lp  = data['Labour Productivity_Trend'] if 'Labour Productivity_Trend' in data.columns else data['Labour Productivity']

    raw_potential_base = (
        data.loc[base_idx, 'Population']
        * data.loc[base_idx, 'LFP_trend_decimal']
        * (1 - data.loc[base_idx, 'NAIRU_decimal'])
        * ahw.loc[base_idx]
        * lp.loc[base_idx]
    )
    real_gdp_base = data.loc[base_idx, 'Real GDP Expenditure']
    scaling_factor = real_gdp_base / raw_potential_base if raw_potential_base != 0 else 1.0

    data['Potential Output'] = (
        data['Population']
        * data['LFP_trend_decimal']
        * (1 - data['NAIRU_decimal'])
        * ahw
        * lp
        * scaling_factor
    )

    apply_hp_filter(data, 'Potential Output', log_transform=True, exp_transform=True)
    data['Potential Output Growth (%)'] = data['Potential Output_Trend'].pct_change() * 100
    data['Output Gap (%)'] = ((data['Real GDP Expenditure'] - data['Potential Output']) / data['Potential Output']) * 100

    # IS Curve
    is_curve_df = pd.read_excel(xls, sheet_name=sheet_is_curve, na_values=["NA"])
    is_curve_df.columns = is_curve_df.columns.str.strip()
    is_curve_df['Date'] = pd.to_datetime(is_curve_df['Date'], errors='coerce')
    is_curve_df = is_curve_df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    is_curve_df['Year'] = is_curve_df['Date'].dt.year
    is_curve_cols = [
        'Real Interest Rate', 'Nominal interest rate', 'Real Effective Exchange Rate',
        'Commodity Price Index', 'Foreign Demand', 'Output Gap', 'Inflation Rate',
        'Commodity Price Index Energy', 'Commodity Price Index Non-Energy', 'Neutral Interest Rate',
        'Expected Future Inflation Rate', "Commodity Price (US deflated)", 'US Imports',
    ]
    for c in is_curve_cols:
        if c not in is_curve_df.columns: is_curve_df[c] = np.nan
    is_curve_df[is_curve_cols] = is_curve_df[is_curve_cols].apply(pd.to_numeric, errors='coerce')
    for col in is_curve_cols:
        apply_hp_filter(is_curve_df, col)

    # Phillips
    phillips_df = pd.read_excel(xls, sheet_name=sheet_phillips, na_values=["NA"])
    phillips_df.columns = phillips_df.columns.str.strip()
    phillips_df['Date'] = pd.to_datetime(phillips_df['Date'], errors='coerce')
    phillips_df = phillips_df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    phillips_df['Year'] = phillips_df['Date'].dt.year
    phillips_cols = ['Inflation rate', 'Real effective exchange rate', 'WTI Crude (USD)', 'Commodity price index', 'Output Gap', 'WTI Crude (CAD)']
    for c in phillips_cols:
        if c not in phillips_df.columns: phillips_df[c] = np.nan
    phillips_df[phillips_cols] = phillips_df[phillips_cols].apply(pd.to_numeric, errors='coerce')
    for col in phillips_cols:
        apply_hp_filter(phillips_df, col)

    # Taylor
    taylor_df = pd.read_excel(xls, sheet_name=sheet_taylor, na_values=["NA"])
    taylor_df.columns = taylor_df.columns.str.strip()
    taylor_df['Date'] = pd.to_datetime(taylor_df['Date'], errors='coerce')
    taylor_df = taylor_df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    taylor_df['Year'] = taylor_df['Date'].dt.year
    taylor_cols = ['Nominal Interest rate', 'Real Interest Rate', 'Inflation rate', 'Target inflation rate', 'Potential GDP']
    for c in taylor_cols:
        if c not in taylor_df.columns: taylor_df[c] = np.nan
    taylor_df[taylor_cols] = taylor_df[taylor_cols].apply(pd.to_numeric, errors='coerce')
    for col in taylor_cols:
        apply_hp_filter(taylor_df, col)

    year_min = int(min(data['Year'].min(), is_curve_df['Year'].min(), phillips_df['Year'].min(), taylor_df['Year'].min()))
    year_max = int(max(data['Year'].max(), is_curve_df['Year'].max(), phillips_df['Year'].max(), taylor_df['Year'].max()))

    return data, is_curve_df, phillips_df, taylor_df, is_curve_cols, phillips_cols, taylor_cols, year_min, year_max

# Load macro workbook
try:
    data, is_curve_df, phillips_df, taylor_df, is_cols, pc_cols, tr_cols, year_min, year_max = load_macro(excel_src)
except Exception as e:
    st.error(f"Failed to load macro workbook: {e}")
    st.stop()

# ------- Controls (re-implementing the Dash panel) -------
left, right = st.columns([1,2])

with left:
    st.subheader("Controls")

    general_options = sorted([
        col for col in data.columns
        if col not in ['Date', 'Year']
        and not col.endswith('_Trend')
        and not col.endswith('_Cycle')
    ])
    selected_general = st.selectbox("General Variable", options=general_options, index=(general_options.index('Potential Output') if 'Potential Output' in general_options else 0))

    show_gdp = st.checkbox("Overlay Real GDP (when Potential Output selected)", value=False)

    selected_is = st.selectbox("IS Curve (relevant)", options=['(none)'] + is_cols, index=0)
    selected_pc = st.selectbox("Phillips Curve (relevant)", options=['(none)'] + pc_cols, index=0)
    selected_tr = st.selectbox("Taylor Rule (relevant)", options=['(none)'] + tr_cols, index=0)

    selected_type = st.radio("Data Type", options=['raw','trend','cycle','raw_trend'], index=0, horizontal=True)
    year_range = st.slider("Year Range", min_value=year_min, max_value=year_max, value=(year_min, year_max), step=1)

with right:
    st.subheader("Macro Chart")
    start_year, end_year = year_range

    # Decide which dataframe & variable to show (priority: Taylor > Phillips > IS > General), matching your Dash logic
    df = data
    selected_var = selected_general
    if selected_tr != '(none)':
        df = taylor_df; selected_var = selected_tr
    elif selected_pc != '(none)':
        df = phillips_df; selected_var = selected_pc
    elif selected_is != '(none)':
        df = is_curve_df; selected_var = selected_is

    filtered_df = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)].copy()

    fig = go.Figure()
    if selected_type == 'trend':
        fig.add_trace(go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df.get(f"{selected_var}_Trend"),
            mode='lines',
            name=f"{selected_var} - Trend",
        ))
    elif selected_type == 'cycle':
        fig.add_trace(go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df.get(f"{selected_var}_Cycle"),
            mode='lines',
            name=f"{selected_var} - Cycle",
        ))
    elif selected_type == 'raw_trend':
        fig.add_trace(go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df.get(selected_var),
            mode='lines',
            name=f"{selected_var} - Raw",
        ))
        fig.add_trace(go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df.get(f"{selected_var}_Trend"),
            mode='lines',
            name=f"{selected_var} - Trend",
        ))
    else:
        fig.add_trace(go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df.get(selected_var),
            mode='lines',
            name=f"{selected_var} - Raw",
        ))

    if show_gdp and selected_general == 'Potential Output':
        overlay_df = data[(data['Year'] >= start_year) & (data['Year'] <= end_year)]
        fig.add_trace(go.Scatter(
            x=overlay_df['Date'],
            y=overlay_df.get('Real GDP Expenditure'),
            mode='lines',
            name='Real GDP',
        ))

    fig.update_layout(
        title={'text': f"{selected_var} ({start_year}–{end_year})", 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="Date",
        yaxis_title=selected_var,
        legend_title="Series",
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified",
        height=520
    )
    st.plotly_chart(fig, use_container_width=True)

st.caption("This section is the Dash dashboard re-implemented in Streamlit so it sits directly beneath the DSGE outputs.")






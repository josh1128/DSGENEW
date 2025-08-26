# dsge_dashboard.py
# -----------------------------------------------------------
# Streamlit app that runs:
#   1) Original model (DSGE.xlsx): IS (DlogGDP), Phillips (Dlog_CPI), Taylor (Nominal rate)
#      - Sidebar toggles to include/exclude regressors in each curve
#      - Taylor uses inflation gap (π_t − π*)
#      - Shocks: IS, Phillips, and Taylor (tightening/easing)
#      - Policy shock behavior selector
#      - LaTeX equations shown below charts
#   2) Simple NK (built-in)
#   + Optimizations: caching, downcasting, memory profiler, float32 arrays, cached shocks
# -----------------------------------------------------------

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
import gc, tracemalloc

plt.rcParams.update({"figure.max_open_warning": 0})

# =========================
# Page setup
# =========================
st.set_page_config(page_title="DSGE IRF Dashboard", layout="wide")
st.title("DSGE IRF Dashboard — IS, Phillips, Taylor")

st.markdown(
    "- **Original**: GDP & CPI in **%** (Dlog × 100); **Nominal rate** in **decimal**.\n"
    "- **Taylor** uses **inflation gap**: \\(\\pi_t - \\pi^*\\).\n"
    "- Use the sidebar to **toggle variables** in each curve."
)

# =========================
# Helpers
# =========================
def ensure_decimal_rate(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s / 100.0 if np.nanmedian(np.abs(s.values)) > 1.0 else s

def fmt_coef(x: float, nd: int = 3) -> str:
    s = f"{x:.{nd}f}"
    return f"+{s}" if x >= 0 else s

def build_latex_equation(const_val: float, terms: List[tuple], lhs: str, eps_symbol: str) -> str:
    rhs_terms = "" if not terms else " ".join([f"{fmt_coef(c)}\\,{sym}" for (c, sym) in terms])
    eq = rf"""
    \begin{{aligned}}
    {lhs} &= {const_val:.3f} {rhs_terms} + {eps_symbol}
    \end{{aligned}}
    """
    return eq

def row_from_params(params_index: pd.Index, values: Dict[str, float]) -> pd.DataFrame:
    cols = list(params_index)
    row = {c: (1.0 if c == "const" else float(values.get(c, 0.0))) for c in cols}
    return pd.DataFrame([row], columns=cols)

def predict_with_params(row_df: pd.DataFrame, beta: pd.Series) -> float:
    x = row_df[beta.index].iloc[0].values.astype(float)
    b = beta.values.astype(float)
    return float(np.dot(x, b))

def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to reduce memory."""
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    for c in df.select_dtypes(include=["int64", "int32"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    return df

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
        x  = np.zeros(T, dtype=np.float32)
        pi = np.zeros(T, dtype=np.float32)
        i  = np.zeros(T, dtype=np.float32)
        r_nat = np.zeros(T, dtype=np.float32)
        u     = np.zeros(T, dtype=np.float32)
        e_i   = np.zeros(T, dtype=np.float32)

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
                if shock == "demand":
                    r_nat[t] += (rho_sh or 0.0) * r_nat[t-1]
                elif shock == "cost":
                    u[t] += (rho_sh or 0.0) * u[t-1]

            x_lag  = x[t-1]  if t > 0 else 0.0
            pi_lag = pi[t-1] if t > 0 else 0.0
            i_lag  = i[t-1]  if t > 0 else 0.0

            A_x = (1 - p.rho_i) * (p.phi_pi * p.kappa + p.phi_x) - p.kappa
            B_const = (
                p.rho_i * i_lag
                + ((1 - p.rho_i) * p.phi_pi * p.gamma_pi - p.gamma_pi) * pi_lag
                + ((1 - p.rho_i) * p.phi_pi - 1.0) * u[t]
                + e_i[t]
            )
            denom = 1.0 + (A_x / p.sigma)
            num = (p.rho_x * x_lag) - (B_const / p.sigma) + (r_nat[t] / p.sigma)
            x[t]  = num / max(denom, 1e-8)
            pi[t] = p.gamma_pi * pi_lag + p.kappa * x[t] + u[t]
            i[t]  = p.rho_i * i_lag + (1 - p.rho_i) * (p.phi_pi * pi[t] + p.phi_x * x[t]) + e_i[t]

        return np.arange(T), x, pi, i

    def simulate_path(self, T, x0, pi0, i0, r_nat=None, u=None, e_i=None):
        p = self.p
        x  = np.zeros(T, dtype=np.float32)
        pi = np.zeros(T, dtype=np.float32)
        i  = np.zeros(T, dtype=np.float32)
        x[0] = float(x0); pi[0] = float(pi0); i[0] = float(i0)

        r_nat = np.zeros(T, dtype=np.float32) if r_nat is None else np.asarray(r_nat, dtype=np.float32)
        u     = np.zeros(T, dtype=np.float32) if u     is None else np.asarray(u,     dtype=np.float32)
        e_i   = np.zeros(T, dtype=np.float32) if e_i   is None else np.asarray(e_i,   dtype=np.float32)

        def _fix_len(arr):
            if len(arr) < T:
                return np.pad(arr, (0, T-len(arr)), constant_values=0.0)
            return arr[:T]
        r_nat = _fix_len(r_nat); u = _fix_len(u); e_i = _fix_len(e_i)

        for t in range(1, T):
            x_lag, pi_lag, i_lag = x[t-1], pi[t-1], i[t-1]

            A_x = (1 - p.rho_i) * (p.phi_pi * p.kappa + p.phi_x) - p.kappa
            B_const = (
                p.rho_i * i_lag
                + ((1 - p.rho_i) * p.phi_pi * p.gamma_pi - p.gamma_pi) * pi_lag
                + ((1 - p.rho_i) * p.phi_pi - 1.0) * u[t]
                + e_i[t]
            )
            denom = 1.0 + (A_x / p.sigma)
            num = (p.rho_x * x_lag) - (B_const / p.sigma) + (r_nat[t] / p.sigma)
            x[t]  = num / max(denom, 1e-8)
            pi[t] = p.gamma_pi * pi_lag + p.kappa * x[t] + u[t]
            i[t]  = p.rho_i * i_lag + (1 - p.rho_i) * (p.phi_pi * pi[t] + p.phi_x * x[t]) + e_i[t]
        return np.arange(T), x, pi, i

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("Model selection")
    model_choice = st.selectbox("Choose model version", ["Original (DSGE.xlsx)", "Simple NK (built-in)"], index=0)

    st.header("Simulation settings")
    T = st.slider("Horizon (quarters)", 8, 60, 20, 1)

    # ---- Memory tools ----
    with st.expander("Memory tools", expanded=False):
        profile_mem = st.checkbox("Profile memory (tracemalloc)", value=False)
        if st.button("Clear caches"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cleared cached data/resources.")

    if model_choice == "Original (DSGE.xlsx)":
        xlf = st.file_uploader("Upload DSGE.xlsx (optional)", type=["xlsx"], key="upload_original",
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

        IS_ALL = ["DlogGDP_L1", "Real_Rate_L2_data", "Dlog FD_Lag1", "Dlog_REER", "Dlog_Energy", "Dlog_NonEnergy"]
        PC_ALL = ["Dlog_CPI_L1", "DlogGDP_L1", "Dlog_Reer_L2", "Dlog_Energy_L1", "Dlog_Non_Energy_L1"]
        TR_ALL = ["Nominal_Rate_L1", "Inflation_Gap", "DlogGDP"]

        with st.expander("IS Curve regressors", expanded=True):
            is_selected = st.multiselect("Use these variables in the IS regression:", IS_ALL, default=IS_ALL, key="is_vars")
        with st.expander("Phillips Curve regressors", expanded=True):
            pc_selected = st.multiselect("Use these variables in the Phillips regression:", PC_ALL, default=PC_ALL, key="pc_vars")
        with st.expander("Taylor Rule regressors", expanded=True):
            tr_selected = st.multiselect("Use these variables in the Taylor (partial adjustment) regression:", TR_ALL, default=TR_ALL, key="tr_vars")

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
        snapback = st.checkbox("Snap-back (no persistence after the shock)", value=True)
        units_mode = st.radio("Policy rate units", ["Deviation (pp)", "Level (% annual)"], index=0)

    neutral_rate_pct = st.number_input(
        "Baseline (neutral) nominal policy rate — % annual",
        value=2.00, step=0.25, format="%.2f",
        help="Long-run policy rate anchor."
    )

# =========================
# ORIGINAL MODEL (DSGE.xlsx)
# =========================
@st.cache_data(ttl=3600, show_spinner=True)
def load_and_prepare_original(file_like_or_path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if file_like_or_path is None:
        raise FileNotFoundError("Upload DSGE.xlsx or place it beside this script.")

    if isinstance(file_like_or_path, (str, Path)):
        p = Path(file_like_or_path)
        if not p.is_absolute(): p = Path.cwd() / p
        if not p.exists(): raise FileNotFoundError(f"Could not find Excel file at: {p}")
        excel_src = p
    else:
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
    df["Real_Rate_L2_data"] = (df["Nominal Rate"] - df["Dlog_CPI"]).shift(2)

    required_cols = [
        "DlogGDP", "DlogGDP_L1", "Dlog_CPI", "Dlog_CPI_L1",
        "Nominal Rate", "Nominal_Rate_L1", "Real_Rate_L2_data",
        "Dlog FD_Lag1", "Dlog_REER", "Dlog_Energy", "Dlog_NonEnergy",
        "Dlog_Reer_L2", "Dlog_Energy_L1", "Dlog_Non_Energy_L1",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df_est = df.dropna(subset=required_cols).copy()
    if df_est.empty:
        raise ValueError("No rows remain after dropping NA for required columns. Check your data.")

    # Memory reduction
    df = downcast_numeric(df)
    df_est = downcast_numeric(df_est)
    return df, df_est

def _clip_and_recentre(beta_raw: pd.Series, means: Dict[str, float], rules: Dict[str, str]) -> pd.Series:
    """Clip selected coefficients per rules and adjust constant so that E[y] is preserved."""
    beta_sim = beta_raw.copy()
    beta_new = beta_raw.copy()
    for var, rule in rules.items():
        if var in beta_sim.index:
            if rule == "nonpos":
                beta_new[var] = min(beta_sim[var], 0.0)
            elif rule == "nonneg":
                beta_new[var] = max(beta_sim[var], 0.0)
    const = beta_sim.get("const", 0.0)
    shift = 0.0
    for var, rule in rules.items():
        if var in beta_sim.index:
            shift += (beta_sim[var] - beta_new[var]) * float(means.get(var, 0.0))
    beta_new["const"] = const + shift
    return beta_new

def fit_models_original(df_est: pd.DataFrame, pi_star_quarterly: float,
                        is_selected: List[str], pc_selected: List[str], tr_selected: List[str]):
    """Fit OLS for IS, Phillips, and Taylor. Clip signs for simulation and keep raw OLS for display."""
    # ---------- IS ----------
    if not is_selected:
        raise ValueError("Select at least one regressor for IS (besides constant).")
    X_is = sm.add_constant(df_est[is_selected], has_constant="add")
    y_is = df_est["DlogGDP"]
    model_is = sm.OLS(y_is, X_is).fit()
    X_is_means = {c: float(X_is[c].mean()) for c in X_is.columns if c != "const"}
    beta_is_sim = _clip_and_recentre(model_is.params, X_is_means, rules={"Real_Rate_L2_data": "nonpos"})

    # ---------- Phillips ----------
    if not pc_selected:
        raise ValueError("Select at least one regressor for Phillips (besides constant).")
    X_pc = sm.add_constant(df_est[pc_selected], has_constant="add")
    y_pc = df_est["Dlog_CPI"]
    model_pc = sm.OLS(y_pc, X_pc).fit()
    X_pc_means = {c: float(X_pc[c].mean()) for c in X_pc.columns if c != "const"}
    beta_pc_sim = _clip_and_recentre(model_pc.params, X_pc_means, rules={"DlogGDP_L1": "nonneg"})

    # ---------- Taylor (with inflation gap) ----------
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
    rhoh = float(model_tr.params.get("Nominal_Rate_L1", 0.0))
    rhoh = min(max(rhoh, 0.0), 0.99)

    def safe_div(num, den): return num / den if abs(den) > 1e-8 else np.nan

    alpha_star = safe_div(b0, (1 - rhoh))
    bpi = float(model_tr.params.get("Inflation_Gap", 0.0))
    bg  = float(model_tr.params.get("DlogGDP", 0.0))
    phi_pi_star_raw = safe_div(bpi, (1 - rhoh)) if "Inflation_Gap" in model_tr.params.index else np.nan
    phi_g_star_raw  = safe_div(bg,  (1 - rhoh)) if "DlogGDP"      in model_tr.params.index else np.nan

    # Simulation clips (theory-consistent)
    phi_pi_star_sim = np.nan if np.isnan(phi_pi_star_raw) else max(0.0, phi_pi_star_raw)
    phi_g_star_sim  = np.nan if np.isnan(phi_g_star_raw)  else max(0.0, phi_g_star_raw)

    # Long-run anchors
    p_ss = float(df_est["Dlog_CPI"].mean())   # \bar{π}
    g_ss = float(df_est["DlogGDP"].mean())    # \bar{g}

    return {
        "model_is": model_is, "model_pc": model_pc, "model_tr": model_tr,
        "beta_is_sim": beta_is_sim, "beta_pc_sim": beta_pc_sim,
        "alpha_star": alpha_star,
        "phi_pi_star": phi_pi_star_raw, "phi_g_star": phi_g_star_raw,         # display
        "phi_pi_star_sim": phi_pi_star_sim, "phi_g_star_sim": phi_g_star_sim, # simulation
        "rho_hat": rhoh, "pi_star_quarterly": float(pi_star_quarterly),
        "p_ss": p_ss, "g_ss": g_ss
    }

@st.cache_data
def build_shocks_original(T, target, is_size_pp, pc_size_pp, policy_bp_abs, t0, rho):
    """Build shock arrays (cached)."""
    is_arr = np.zeros(T, dtype=np.float32); pc_arr = np.zeros(T, dtype=np.float32); pol_arr = np.zeros(T, dtype=np.float32)
    target_norm = (target or "None").strip().lower()

    if target_norm == "is (demand)".lower():
        is_arr[t0] = is_size_pp / 100.0
        for k in range(t0 + 1, T): is_arr[k] = rho * is_arr[k - 1]
    elif target_norm == "phillips (supply)".lower():
        pc_arr[t0] = pc_size_pp / 100.0
        for k in range(t0 + 1, T): pc_arr[k] = rho * pc_arr[k - 1]
    elif target_norm == "taylor (policy tightening)".lower():
        pol_arr[t0] = (policy_bp_abs / 10000.0)
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
    """Forward-simulate GDP growth, inflation, and the rate using estimated OLS with theory-consistent clips."""
    g = np.zeros(T, dtype=np.float32)
    p = np.zeros(T, dtype=np.float32)
    i = np.zeros(T, dtype=np.float32)

    g[0] = float(df_est["DlogGDP"].mean())
    p[0] = float(df_est["Dlog_CPI"].mean())
    i[0] = float(i_mean_dec)

    model_is = models["model_is"]; model_pc = models["model_pc"]
    beta_is_sim = models["beta_is_sim"]; beta_pc_sim = models["beta_pc_sim"]
    phi_pi_star_sim = models.get("phi_pi_star_sim", np.nan)
    phi_g_star_sim  = models.get("phi_g_star_sim",  np.nan)
    p_ss = models["p_ss"]; g_ss = models["g_ss"]

    is_shock_arr = np.zeros(T, dtype=np.float32) if is_shock_arr is None else is_shock_arr.astype(np.float32)
    pc_shock_arr = np.zeros(T, dtype=np.float32) if pc_shock_arr is None else pc_shock_arr.astype(np.float32)
    policy_shock_arr = np.zeros(T, dtype=np.float32) if policy_shock_arr is None else policy_shock_arr.astype(np.float32)

    # Taylor alpha*: anchor long-run to neutral
    alpha_star_sim = float(neutral_dec) - (0.0 if np.isnan(phi_pi_star_sim) else float(phi_pi_star_sim)) * (float(p_ss) - float(pi_star_quarterly))

    for t in range(1, T):
        rr_lag2 = (i[t - 2] - p[t - 2]) if t >= 2 else float(real_rate_mean_dec)

        # IS (use theory-consistent beta)
        vals_is = {
            "DlogGDP_L1": g[t - 1],
            "Real_Rate_L2_data": rr_lag2,
            "Dlog FD_Lag1": float(means["Dlog FD_Lag1"]),
            "Dlog_REER": float(means["Dlog_REER"]),
            "Dlog_Energy": float(means["Dlog_Energy"]),
            "Dlog_NonEnergy": float(means["Dlog_NonEnergy"]),
        }
        Xis = row_from_params(model_is.params.index, vals_is)
        g[t] = predict_with_params(Xis, beta_is_sim) + is_shock_arr[t]

        # Phillips (use theory-consistent beta)
        vals_pc = {
            "Dlog_CPI_L1": p[t - 1],
            "DlogGDP_L1": g[t - 1],
            "Dlog_Reer_L2": float(means["Dlog_Reer_L2"]),
            "Dlog_Energy_L1": float(means["Dlog_Energy_L1"]),
            "Dlog_Non_Energy_L1": float(means["Dlog_Non_Energy_L1"]),
        }
        Xpc = row_from_params(model_pc.params.index, vals_pc)
        p[t] = predict_with_params(Xpc, beta_pc_sim) + pc_shock_arr[t]

        # Taylor target (deviation form + neutral anchor)
        pi_gap_t = p[t] - float(pi_star_quarterly)
        g_dev_t  = g[t] - float(g_ss)
        i_star = (alpha_star_sim
                  + (0.0 if np.isnan(phi_pi_star_sim) else float(phi_pi_star_sim)) * pi_gap_t
                  + (0.0 if np.isnan(phi_g_star_sim)  else float(phi_g_star_sim))  * g_dev_t)

        eps = policy_shock_arr[t]  # decimal (e.g., 0.01 = 100 bp)
        if policy_mode.startswith("Add after"):
            i_raw = rho_sim * i[t - 1] + (1 - rho_sim) * i_star + eps
        elif policy_mode.startswith("Add to target"):
            i_raw = rho_sim * i[t - 1] + (1 - rho_sim) * (i_star + eps)
        else:
            i_raw = rho_sim * i[t - 1] + (1 - rho_sim) * i_star + eps
            if eps > 0:  i_raw = max(i_raw, i[t - 1] + abs(eps))
            elif eps < 0: i_raw = min(i_raw, i[t - 1] - abs(eps))
        i[t] = float(i_raw)

    return g, p, i

# =========================
# Run selected model
# =========================
try:
    if model_choice == "Original (DSGE.xlsx)":
        if profile_mem: tracemalloc.start()

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

        # Fit with selected regressors
        models_o = fit_models_original(df_est, pi_star_quarterly, is_selected, pc_selected, tr_selected)

        # Anchors & means
        i_mean_dec = float(df_est["Nominal Rate"].mean())
        real_rate_mean_dec = float(df_est["Real_Rate_L2_data"].mean())
        means_o = {
            "Dlog FD_Lag1": float(df_est["Dlog FD_Lag1"].mean()),
            "Dlog_REER": float(df_est["Dlog_REER"].mean()),
            "Dlog_Energy": float(df_est["Dlog_Energy"].mean()),
            "Dlog_NonEnergy": float(df_est["Dlog_NonEnergy"].mean()),
            "Dlog_Reer_L2": float(df_est["Dlog_Reer_L2"].mean()),
            "Dlog_Energy_L1": float(df_est["Dlog_Energy_L1"].mean()),
            "Dlog_Non_Energy_L1": float(df_est["Dlog_Non_Energy_L1"].mean()),
        }

        # Baseline simulation
        neutral_dec = neutral_rate_pct / 100.0
        g0, p0, i0 = simulate_original(
            T, rho_sim, df_est, models_o, means_o, i_mean_dec, real_rate_mean_dec, pi_star_quarterly,
            policy_mode=policy_mode, neutral_dec=neutral_dec
        )

        # Only build/compute shock path if user selected a shock
        do_shock = isinstance(shock_target, str) and (shock_target.strip().lower() != "none")
        if do_shock:
            is_arr, pc_arr, pol_arr = build_shocks_original(
                T, shock_target, is_shock_size_pp, pc_shock_size_pp, policy_shock_bp_abs, shock_quarter, shock_persist
            )
            gS, pS, iS = simulate_original(
                T, rho_sim, df_est, models_o, means_o, i_mean_dec, real_rate_mean_dec, pi_star_quarterly,
                is_shock_arr=is_arr, pc_shock_arr=pc_arr, policy_shock_arr=pol_arr,
                policy_mode=policy_mode, neutral_dec=neutral_dec
            )
        else:
            gS, pS, iS = g0, p0, i0

        # Plot IRFs
        plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 12, "legend.fontsize": 11})
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        quarters = np.arange(T, dtype=int)
        vline_kwargs = dict(color="black", linestyle=":", linewidth=1)

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

        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
        gc.collect()

        # Readout & guardrail
        if do_shock and ("taylor" in shock_target.lower()):
            delta_i_bp = (iS - i0)[shock_quarter] * 10000.0
            st.info(f"Δ policy rate at t={shock_quarter}: {delta_i_bp:.1f} bp  |  mode: {policy_mode}  |  ρ={rho_sim:.2f}")
            if "tightening" in shock_target.lower() and delta_i_bp < 0:
                st.warning("Tightening selected but Δi on impact is negative. Check inputs.")

        # ===== LaTeX equations (display raw OLS) =====
        st.subheader("Estimated Equations (Original model)")
        m_is = models_o["model_is"]; m_pc = models_o["model_pc"]; m_tr = models_o["model_tr"]
        alpha_star = models_o["alpha_star"]; phi_pi_star = models_o["phi_pi_star"]; phi_g_star = models_o["phi_g_star"]
        rho_hat = models_o["rho_hat"]

        # IS
        is_terms = []
        pretty_map_is = {
            "DlogGDP_L1": r"\Delta \log GDP_{t-1}",
            "Real_Rate_L2_data": r"RR_{t-2}",
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

        parts = [rf"\rho = {rho_hat:.3f}",
                 rf"\pi^\* = {models_o['pi_star_quarterly']:.4f}",
                 rf"\bar\pi = {models_o['p_ss']:.4f}",
                 rf"\bar g = {models_o['g_ss']:.4f}",
                 rf"i^n = {neutral_dec:.4f}"]
        if not np.isnan(alpha_star): parts.append(rf"\alpha^\* = {alpha_star:.3f}")
        if not np.isnan(phi_pi_star): parts.append(rf"\phi_{{\pi}}^\* = {phi_pi_star:.3f}")
        if not np.isnan(phi_g_star): parts.append(rf"\phi_{{g}}^\* = {phi_g_star:.3f}")

        st.latex(r"i_t^\* \;=\; \alpha^\*_{\text{sim}} \;+\; \phi_{\pi}^\*\,(\pi_t - \pi^\*) \;+\; \phi_{g}^\*\,\big(g_t - \bar g\big)")

        if profile_mem:
            snap = tracemalloc.take_snapshot()
            top = snap.statistics("lineno")[:8]
            st.code("\n".join(str(s) for s in top), language="text")
            tracemalloc.stop()

    else:
        # =========================
        # Simple NK (built-in)
        # =========================
        if profile_mem: tracemalloc.start()

        P = NKParamsSimple(
            sigma=sigma, kappa=kappa, phi_pi=phi_pi, phi_x=phi_x, rho_i=rho_i,
            rho_x=(0.0 if snapback else rho_x), rho_r=rho_r, rho_u=rho_u,
            gamma_pi=(0.0 if snapback else gamma_pi)
        )
        model = SimpleNK3EqBuiltIn(P)
        label_to_code = {"Demand (IS)": "demand", "Cost-push (Phillips)": "cost", "Policy (Taylor)": "policy"}
        code = label_to_code[shock_type_nk]
        t0 = max(0, min(T-1, shock_quarter_nk - 1))
        rho_for_shock = 0.0 if snapback else shock_persist_nk

        st.subheader("Impulse responses (IRF mode)")
        h, x0, pi0, i0 = model.irf(code, T, 0.0, t0, rho_for_shock)
        h, xS, piS, iS = model.irf(code, T, shock_size_pp_nk, t0, rho_for_shock)

        i0_plot, iS_plot = i0.copy(), iS.copy()
        i_ylabel = "pp"
        if units_mode == "Level (% annual)":
            i0_plot = neutral_rate_pct + i0_plot
            iS_plot = neutral_rate_pct + iS_plot
            i_ylabel = "%"

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
        axes[2].set_title("Nominal Policy Rate (i_t)")
        axes[2].set_xlabel("Quarters ahead"); axes[2].set_ylabel(i_ylabel)
        axes[2].grid(True, alpha=0.3); axes[2].legend(loc="best")

        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
        gc.collect()

        with st.expander("Simple NK equations"):
            st.latex(r"x_t = \rho_x x_{t-1} \;-\; \frac{1}{\sigma}\big( i_t - \pi_{t+1} - r^n_t \big)")
            st.latex(r"\pi_t = \gamma_\pi \pi_{t-1} \;+\; \kappa x_t \;+\; u_t")
            st.latex(r"i_t = \rho_i i_{t-1} \;+\; (1-\rho_i)(\phi_\pi \pi_t + \phi_x x_t) \;+\; \varepsilon^i_t")

        if profile_mem:
            snap = tracemalloc.take_snapshot()
            top = snap.statistics("lineno")[:8]
            st.code("\n".join(str(s) for s in top), language="text")
            tracemalloc.stop()

except Exception as e:
    st.error(f"Problem loading or running the selected model: {e}")
    st.stop()





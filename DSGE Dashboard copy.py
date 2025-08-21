# dsge_dashboard.py
# -----------------------------------------------------------
# Streamlit app that runs:
#   1) Original model (DSGE.xlsx): IS (DlogGDP), Phillips (Dlog_CPI), Taylor (Nominal rate)
#      - Sidebar toggles to include/exclude regressors in each curve
#      - Taylor uses inflation gap (π_t − π*)
#      - Shocks: IS, Phillips, and Taylor (tightening/easing)
#      - Policy shock behavior selector:
#          • Add after smoothing (default)
#          • Add to target (inside 1−ρ)
#          • Force local jump (override)
#      - LaTeX equations shown below charts (auto-updates to reflect selected vars)
#   2) Simple NK (built-in): 3-eq NK DSGE-lite with tunable parameters
#      - "Snap-back (no persistence)" option makes x_t & π_t one-period while
#        KEEPING policy smoothing ρ_i so i_t decays geometrically.
#      - Toggle to show policy rate in **levels (% annual)** instead of deviations (pp).
#      - NEW: Out-of-sample forecast mode with calendar date labels and optional CSV shocks.
#      - NEW: (Simple NK) **Guarantee convergence** option projects parameters into a stable region.
# -----------------------------------------------------------

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

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
    """Convert percent-style rates (e.g., 3.2) to decimal (0.032) if needed."""
    s = pd.to_numeric(series, errors="coerce")
    if np.nanmedian(np.abs(s.values)) > 1.0:  # e.g., 3.2 means 3.2%
        return s / 100.0
    return s

def fmt_coef(x: float, nd: int = 3) -> str:
    """Pretty-print a coefficient with sign (e.g., '+0.123')."""
    s = f"{x:.{nd}f}"
    return f"+{s}" if x >= 0 else s

def build_latex_equation(const_val: float, terms: List[tuple], lhs: str, eps_symbol: str) -> str:
    """Build a LaTeX aligned equation string."""
    if not terms:
        rhs_terms = ""
    else:
        rhs_terms = " ".join([f"{fmt_coef(c)}\\,{sym}" for (c, sym) in terms])
    eq = rf"""
    \begin{{aligned}}
    {lhs} &= {const_val:.3f} {rhs_terms} + {eps_symbol}
    \end{{aligned}}
    """
    return eq

def row_from_params(params_index: pd.Index, values: Dict[str, float]) -> pd.DataFrame:
    """Create a 1-row DataFrame for prediction with columns ordered like model.params.index."""
    cols = list(params_index)
    row = {}
    for c in cols:
        if c == "const":
            row[c] = 1.0
        else:
            row[c] = float(values.get(c, 0.0))
    return pd.DataFrame([row], columns=cols)

# =========================
# Simple NK (built-in)
# =========================
@dataclass
class NKParamsSimple:
    sigma: float = 1.00   # σ: demand sensitivity to real rate (higher = less sensitive)
    kappa: float = 0.10   # κ: slope of NK Phillips curve (how demand moves inflation)
    phi_pi: float = 1.50  # φπ: policy response to inflation
    phi_x: float = 0.125  # φx: policy response to output gap
    rho_i: float = 0.80   # ρi: interest rate smoothing
    rho_x: float = 0.50   # ρx: persistence of output gap
    rho_r: float = 0.80   # ρr: persistence of demand (natural-rate) shock
    rho_u: float = 0.50   # ρu: persistence of cost-push shock
    gamma_pi: float = 0.50  # γπ: inflation inertia

def project_params_to_stable(p: NKParamsSimple) -> NKParamsSimple:
    """
    Project parameters to a region that ensures convergence to steady state
    for one-off (or decaying) shocks:
      - Enforce Taylor principle: φπ >= 1.05
      - Clip AR coefficients strictly inside unit circle
      - Keep σ positive and not tiny
      - Ensure the algebraic denominator used for x_t solution stays away from zero
    """
    p2 = NKParamsSimple(**vars(p))
    # Core bounds
    p2.sigma     = max(0.10, float(p2.sigma))
    p2.kappa     = min(max(0.01, float(p2.kappa)), 1.00)
    p2.phi_pi    = max(1.05, float(p2.phi_pi))     # Taylor principle
    p2.phi_x     = min(max(0.0, float(p2.phi_x)), 2.0)
    p2.rho_i     = min(max(0.0, float(p2.rho_i)), 0.98)
    p2.rho_x     = min(max(0.0, float(p2.rho_x)), 0.95)
    p2.rho_r     = min(max(0.0, float(p2.rho_r)), 0.98)
    p2.rho_u     = min(max(0.0, float(p2.rho_u)), 0.98)
    p2.gamma_pi  = min(max(0.0, float(p2.gamma_pi)), 0.95)

    # Guard the x_t denominator: denom = 1 + ((1 - ρ_i)*(φπ κ + φx) - κ)/σ
    A_x    = (1 - p2.rho_i) * (p2.phi_pi * p2.kappa + p2.phi_x) - p2.kappa
    denom  = 1.0 + (A_x / p2.sigma)
    floor  = 0.15
    if denom < floor:
        # Reduce the strength of contemporaneous policy load to lift denom
        # Scale φx and κ contributions inside A_x toward a safer value
        # Keep φπ (Taylor principle) intact.
        # Compute scale needed to reach denom=floor, then softly blend.
        need = floor - denom
        # Approximate sensitivity of denom to φx, κ:
        sens = (1 - p2.rho_i) * (1.0) / p2.sigma + (-(1.0) / p2.sigma)  # rough positive number
        sens = max(sens, 1e-3)
        scale_down = 1.0 - min(0.5, need / (abs(sens) + 1e-6))  # cap shrink at 50%
        p2.phi_x = p2.phi_x * scale_down
        p2.kappa = max(0.01, p2.kappa * scale_down)
        # Recompute to ensure we're above floor
        A_x   = (1 - p2.rho_i) * (p2.phi_pi * p2.kappa + p2.phi_x) - p2.kappa
        denom = 1.0 + (A_x / p2.sigma)
        if denom < floor:
            # As a last resort, nudge σ up a bit
            p2.sigma = max(p2.sigma, (A_x) / (floor - 1.0) + 0.10)

    return p2

class SimpleNK3EqBuiltIn:
    """Tiny 3-equation NK model used for quick IRFs or exogenous-path simulations."""
    def __init__(self, params: Optional[NKParamsSimple] = None, guarantee_convergence: bool = False):
        p = params or NKParamsSimple()
        self.p = project_params_to_stable(p) if guarantee_convergence else p
        self.enforced = guarantee_convergence

    def _safe_denom(self, denom: float) -> float:
        # Numerical guard for the linear solve of x_t
        return max(denom, 1e-6)

    def irf(
        self,
        shock: str = "demand",
        T: int = 24,
        size_pp: float = 1.0,
        t0: int = 0,
        rho_override: Optional[float] = None
    ):
        """
        Impulse responses for output gap (x), inflation (pi), and rate (i) to a one-time shock.
        All variables are in **percentage points** (pp) deviations from baseline (steady state = 0).
        """
        p = self.p
        x = np.zeros(T); pi = np.zeros(T); i = np.zeros(T)
        r_nat = np.zeros(T); u = np.zeros(T); e_i = np.zeros(T)

        # Initialize one-time shocks
        if shock == "demand":
            r_nat[t0] = size_pp
            rho_sh = p.rho_r if rho_override is None else rho_override
        elif shock == "cost":
            u[t0] = size_pp
            rho_sh = p.rho_u if rho_override is None else rho_override
        elif shock == "policy":
            e_i[t0] = size_pp
            rho_sh = None  # one-off policy shock
        else:
            raise ValueError("shock must be 'demand','cost','policy'")

        for t in range(T):
            # Propagate shock persistence (if any)
            if t > t0:
                if shock == "demand":
                    r_nat[t] += (rho_sh or 0.0) * r_nat[t-1]
                elif shock == "cost":
                    u[t] += (rho_sh or 0.0) * u[t-1]

            x_lag  = x[t-1]  if t>0 else 0.0
            pi_lag = pi[t-1] if t>0 else 0.0
            i_lag  = i[t-1]  if t>0 else 0.0

            # Solve contemporaneously for x_t given policy rule and Phillips
            A_x = (1 - p.rho_i) * (p.phi_pi * p.kappa + p.phi_x) - p.kappa
            B_const = (
                p.rho_i * i_lag
                + ((1 - p.rho_i) * p.phi_pi * p.gamma_pi - p.gamma_pi) * pi_lag
                + ((1 - p.rho_i) * p.phi_pi - 1.0) * u[t]
                + e_i[t]
            )
            denom = self._safe_denom(1.0 + (A_x / p.sigma))
            num   = (p.rho_x * x_lag) - (B_const / p.sigma) + (r_nat[t] / p.sigma)
            x[t]  = num / denom
            pi[t] = p.gamma_pi * pi_lag + p.kappa * x[t] + u[t]
            i[t]  = p.rho_i * i_lag + (1 - p.rho_i) * (p.phi_pi * pi[t] + p.phi_x * x[t]) + e_i[t]

        return np.arange(T), x, pi, i

    def simulate_path(
        self,
        T: int,
        x0: float,
        pi0: float,
        i0: float,
        r_nat: Optional[np.ndarray] = None,
        u: Optional[np.ndarray] = None,
        e_i: Optional[np.ndarray] = None
    ):
        """
        Simulate a forward path given initial conditions and **exogenous sequences**
        for demand (r_nat), cost-push (u), and policy disturbance (e_i). Units = pp deviations.
        """
        p = self.p
        x = np.zeros(T); pi = np.zeros(T); i = np.zeros(T)
        x[0] = float(x0); pi[0] = float(pi0); i[0] = float(i0)

        r_nat = np.zeros(T) if r_nat is None else np.asarray(r_nat, dtype=float)
        u     = np.zeros(T) if u     is None else np.asarray(u, dtype=float)
        e_i   = np.zeros(T) if e_i   is None else np.asarray(e_i, dtype=float)

        # Ensure the arrays are length T
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
            denom = self._safe_denom(1.0 + (A_x / p.sigma))
            num   = (p.rho_x * x_lag) - (B_const / p.sigma) + (r_nat[t] / p.sigma)
            x[t]  = num / denom
            pi[t] = p.gamma_pi * pi_lag + p.kappa * x[t] + u[t]
            i[t]  = p.rho_i * i_lag + (1 - p.rho_i) * (p.phi_pi * pi[t] + p.phi_x * x[t]) + e_i[t]

        return np.arange(T), x, pi, i

# =========================
# ORIGINAL MODEL (DSGE.xlsx)
# =========================
@st.cache_data(show_spinner=True)
def load_and_prepare_original(file_like_or_path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read Excel, merge sheets, set dates/units, and create basic lagged fields used by the regressions."""
    if file_like_or_path is None:
        raise FileNotFoundError("Upload DSGE.xlsx or place it beside this script.")

    if isinstance(file_like_or_path, (str, Path)):
        p = Path(file_like_or_path)
        if not p.is_absolute():
            p = Path.cwd() / p
        if not p.exists():
            raise FileNotFoundError(f"Could not find Excel file at: {p}")
        excel_src = p
    else:
        excel_src = file_like_or_path

    is_df = pd.read_excel(excel_src, sheet_name="IS Curve")
    pc_df = pd.read_excel(excel_src, sheet_name="Phillips")
    tr_df = pd.read_excel(excel_src, sheet_name="Taylor")

    for df in (is_df, pc_df, tr_df):
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m", errors="raise")

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
    return df, df_est

def fit_models_original(
    df_est: pd.DataFrame,
    pi_star_quarterly: float,
    is_selected: List[str],
    pc_selected: List[str],
    tr_selected: List[str],
):
    """Fit OLS for IS, Phillips, and Taylor (with inflation gap)."""
    # IS
    if not is_selected:
        raise ValueError("Select at least one regressor for IS (besides constant).")
    X_is = sm.add_constant(df_est[is_selected], has_constant="add")
    y_is = df_est["DlogGDP"]
    model_is = sm.OLS(y_is, X_is).fit()

    # Phillips
    if not pc_selected:
        raise ValueError("Select at least one regressor for Phillips (besides constant).")
    X_pc = sm.add_constant(df_est[pc_selected], has_constant="add")
    y_pc = df_est["Dlog_CPI"]
    model_pc = sm.OLS(y_pc, X_pc).fit()

    # Taylor with inflation gap
    infl_gap_full = df_est["Dlog_CPI"] - pi_star_quarterly
    df_tr = pd.DataFrame(index=df_est.index)
    if "Nominal_Rate_L1" in tr_selected:
        df_tr["Nominal_Rate_L1"] = df_est["Nominal_Rate_L1"]
    if "Inflation_Gap" in tr_selected:
        df_tr["Inflation_Gap"] = infl_gap_full
    if "DlogGDP" in tr_selected:
        df_tr["DlogGDP"] = df_est["DlogGDP"]
    if df_tr.empty:
        raise ValueError("Select at least one regressor for Taylor (besides constant).")
    X_tr = sm.add_constant(df_tr, has_constant="add")
    y_tr = df_est["Nominal Rate"]
    model_tr = sm.OLS(y_tr, X_tr).fit()

    # Convert partial-adjustment rule to star-form only if the needed params exist
    b0 = float(model_tr.params.get("const", 0.0))
    rhoh = float(model_tr.params.get("Nominal_Rate_L1", 0.0))
    rhoh = min(max(rhoh, 0.0), 0.99)

    def safe_div(num, den):
        return num / den if abs(den) > 1e-8 else np.nan

    alpha_star = safe_div(b0, (1 - rhoh))
    bpi = float(model_tr.params.get("Inflation_Gap", 0.0))
    bg  = float(model_tr.params.get("DlogGDP", 0.0))
    phi_pi_star = safe_div(bpi, (1 - rhoh)) if "Inflation_Gap" in model_tr.params.index else np.nan
    phi_g_star  = safe_div(bg,  (1 - rhoh)) if "DlogGDP" in model_tr.params.index else np.nan

    return {
        "model_is": model_is, "model_pc": model_pc, "model_tr": model_tr,
        "alpha_star": alpha_star, "phi_pi_star": phi_pi_star, "phi_g_star": phi_g_star,
        "rho_hat": rhoh, "pi_star_quarterly": float(pi_star_quarterly),
    }

def build_shocks_original(T, target, is_size_pp, pc_size_pp, policy_bp_abs, t0, rho):
    """Create shock arrays (IS, Phillips, or Policy) with optional AR(1)-style decay."""
    is_arr = np.zeros(T); pc_arr = np.zeros(T); pol_arr = np.zeros(T)

    if target == "IS (Demand)":
        is_arr[t0] = is_size_pp / 100.0
        for k in range(t0 + 1, T): is_arr[k] = rho * is_arr[k - 1]
    elif target == "Phillips (Supply)":
        pc_arr[t0] = pc_size_pp / 100.0
        for k in range(t0 + 1, T): pc_arr[k] = rho * pc_arr[k - 1]
    elif target == "Taylor (Policy tightening)":
        pol_arr[t0] =  (policy_bp_abs / 10000.0)
        for k in range(t0 + 1, T): pol_arr[k] = rho * pol_arr[k - 1]
    elif target == "Taylor (Policy easing)":
        pol_arr[t0] = -(policy_bp_abs / 10000.0)
        for k in range(t0 + 1, T): pol_arr[k] = rho * pol_arr[k - 1]

    return is_arr, pc_arr, pol_arr

def simulate_original(
    T: int, rho_sim: float, df_est: pd.DataFrame, models: Dict[str, sm.regression.linear_model.RegressionResultsWrapper],
    means: Dict[str, float], i_mean_dec: float, real_rate_mean_dec: float, pi_star_quarterly: float,
    is_shock_arr=None, pc_shock_arr=None, policy_shock_arr=None, policy_mode: str = "Add after smoothing (standard)"
):
    """Forward-simulate GDP growth, inflation, and the rate using the estimated OLS models."""
    g = np.zeros(T); p = np.zeros(T); i = np.zeros(T)

    g[0] = float(df_est["DlogGDP"].mean())
    p[0] = float(df_est["Dlog_CPI"].mean())
    i[0] = i_mean_dec

    model_is = models["model_is"]; model_pc = models["model_pc"]; model_tr = models["model_tr"]
    alpha_star = models["alpha_star"]; phi_pi_star = models["phi_pi_star"]; phi_g_star = models["phi_g_star"]

    if is_shock_arr is None: is_shock_arr = np.zeros(T)
    if pc_shock_arr is None: pc_shock_arr = np.zeros(T)
    if policy_shock_arr is None: policy_shock_arr = np.zeros(T)

    for t in range(1, T):
        rr_lag2 = (i[t - 2] - p[t - 2]) if t >= 2 else real_rate_mean_dec

        vals_is = {
            "DlogGDP_L1": g[t - 1],
            "Real_Rate_L2_data": rr_lag2,
            "Dlog FD_Lag1": means["Dlog FD_Lag1"],
            "Dlog_REER": means["Dlog_REER"],
            "Dlog_Energy": means["Dlog_Energy"],
            "Dlog_NonEnergy": means["Dlog_NonEnergy"],
        }
        Xis = row_from_params(model_is.params.index, vals_is)
        g[t] = float(model_is.predict(Xis).iloc[0]) + is_shock_arr[t]

        vals_pc = {
            "Dlog_CPI_L1": p[t - 1],
            "DlogGDP_L1": g[t - 1],
            "Dlog_Reer_L2": means["Dlog_Reer_L2"],
            "Dlog_Energy_L1": means["Dlog_Energy_L1"],
            "Dlog_Non_Energy_L1": means["Dlog_Non_Energy_L1"],
        }
        Xpc = row_from_params(model_pc.params.index, vals_pc)
        p[t] = float(model_pc.predict(Xpc).iloc[0]) + pc_shock_arr[t]

        pi_gap_t = p[t] - pi_star_quarterly
        if not np.isnan(alpha_star) and (("Inflation_Gap" in model_tr.params.index) or ("DlogGDP" in model_tr.params.index)):
            i_star = (alpha_star
                      + (0.0 if np.isnan(phi_pi_star) else phi_pi_star) * pi_gap_t
                      + (0.0 if np.isnan(phi_g_star) else phi_g_star) * g[t])
        else:
            vals_tr = {"Nominal_Rate_L1": 0.0, "Inflation_Gap": pi_gap_t, "DlogGDP": g[t]}
            Xtr_star = row_from_params(model_tr.params.index, vals_tr)
            i_star = float(model_tr.predict(Xtr_star).iloc[0])

        eps = policy_shock_arr[t]  # decimal (e.g., 0.0025 = 25 bp)
        if policy_mode.startswith("Add after"):
            i_raw = rho_sim * i[t - 1] + (1 - rho_sim) * i_star + eps
        elif policy_mode.startswith("Add to target"):
            i_raw = rho_sim * i[t - 1] + (1 - rho_sim) * (i_star + eps)
        else:
            i_raw = rho_sim * i[t - 1] + (1 - rho_sim) * i_star + eps
            if eps > 0:
                i_raw = max(i_raw, i[t - 1] + abs(eps))
            elif eps < 0:
                i_raw = min(i_raw, i[t - 1] - abs(eps))

        i[t] = float(i_raw)

    return g, p, i

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("Model selection")
    model_choice = st.selectbox("Choose model version", ["Original (DSGE.xlsx)", "Simple NK (built-in)"], index=0)

    st.header("Simulation settings")
    T = st.slider("Horizon (quarters)", 8, 60, 20, 1)

    if model_choice == "Original (DSGE.xlsx)":
        xlf = st.file_uploader("Upload DSGE.xlsx (optional)", type=["xlsx"], key="upload_original", help="If omitted, the app looks for 'DSGE.xlsx' next to this script.")
        fallback = Path(__file__).parent / "DSGE.xlsx"

        rho_sim = st.slider("Policy smoothing ρ (Taylor)", 0.0, 0.95, 0.80, 0.05, help="How much the policy rate inherits from its own past. Higher ρ ⇒ more persistence.")

        st.header("Inflation target for Taylor")
        use_sample_mean = st.checkbox("Use sample mean of DlogCPI as target π*", value=False, help="If checked, π* is the average of your sample's quarterly inflation.")
        if use_sample_mean:
            target_annual_pct = None
            st.caption("π* will be set to sample mean (quarterly) after data loads.")
        else:
            target_annual_pct = st.slider("π* (annual %)", 0.0, 5.0, 2.0, 0.1, help="Annualized target inflation; we convert this to a quarterly decimal.")
        st.divider()

        st.header("Shock")
        shock_target = st.selectbox(
            "Apply shock to",
            ["None", "IS (Demand)", "Phillips (Supply)", "Taylor (Policy tightening)", "Taylor (Policy easing)"],
            index=0,
            help="Choose which block is directly shocked. Policy shocks move the rate by the bp size below."
        )
        is_shock_size_pp = st.number_input("IS shock (Δ DlogGDP, pp)", value=0.50, step=0.10, format="%.2f", help="One-time bump to GDP growth (percentage points).")
        pc_shock_size_pp = st.number_input("Phillips shock (Δ DlogCPI, pp)", value=0.10, step=0.05, format="%.2f", help="One-time bump to inflation (percentage points).")
        policy_shock_bp_abs = st.number_input("Policy shock size (absolute bp)", value=25, step=5, format="%d", help="Size of the policy rate shock in basis points (25 bp = 0.25%).")
        shock_quarter = st.slider("Shock timing (t)", 1, T-1, 1, 1, help="Quarter index at which the shock hits.")
        shock_persist = st.slider("Shock persistence ρ_shock", 0.0, 0.95, 0.0, 0.05, help="How much the shock decays each period. 0 = one-and-done.")

        st.header("Policy shock behavior")
        policy_mode = st.radio(
            "Choose how the policy shock is applied",
            ["Add after smoothing (standard)", "Add to target (inside 1−ρ)", "Force local jump (override)"],
            index=0,
            help=("How policy shocks enter the partial-adjustment rule:\n"
                  "• **Add after smoothing**: i_t = ρ i_{t-1} + (1−ρ) i*_t + ε^pol_t\n"
                  "• **Add to target**: i_t = ρ i_{t-1} + (1−ρ)(i*_t + ε^pol_t)\n"
                  "• **Force local jump**: Ensures a minimum jump by the shock size at t.")
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
        st.info("**Which parameters affect which curve?**  \n"
                "• **IS (Demand)**: σ, ρx, ρr  \n"
                "• **Phillips (Supply)**: κ, γπ, ρu  \n"
                "• **Taylor Rule (Policy)**: φπ, φx, ρi")

        st.header("Simple NK parameters (pp units)")

        # Guarantee convergence toggle
        guarantee_convergence = st.checkbox(
            "Guarantee convergence to steady state (project parameters to stable region)",
            value=True
        )

        # -------- IS (Demand) --------
        st.subheader("IS Curve (Demand): controls how rates and shocks move activity (x_t)")
        sigma = st.slider("σ — Demand sensitivity denominator", 0.2, 5.0, 1.00, 0.05)
        rho_x = st.slider("ρx — Output persistence", 0.0, 0.98, 0.50, 0.02)
        rho_r = st.slider("ρr — Demand-shock persistence (r^n_t)", 0.0, 0.98, 0.80, 0.02)

        # -------- Phillips (Supply) --------
        st.subheader("Phillips Curve (Supply): links activity to inflation (π_t)")
        kappa = st.slider("κ — Phillips slope", 0.01, 0.50, 0.10, 0.01)
        gamma_pi = st.slider("γπ — Inflation inertia", 0.0, 0.95, 0.50, 0.05)
        rho_u = st.slider("ρu — Cost-push shock persistence (u_t)", 0.0, 0.98, 0.50, 0.02)

        # -------- Taylor Rule (Policy) --------
        st.subheader("Taylor Rule (Policy): sets the interest rate (i_t)")
        phi_pi = st.slider("φπ — Response to inflation (≥1 ensures determinacy)", 1.0, 3.0, 1.50, 0.05)
        phi_x = st.slider("φx — Response to output gap", 0.00, 1.00, 0.125, 0.005)
        rho_i = st.slider("ρi — Policy rate smoothing", 0.0, 0.98, 0.80, 0.02)

        # ---- Shock controls ----
        st.divider()
        st.header("Shock (IRF mode)")
        shock_type_nk = st.selectbox(
            "Shock type (what we 'poke')",
            ["Demand (IS)", "Cost-push (Phillips)", "Policy (Taylor)"],
            index=0
        )
        shock_size_pp_nk = st.number_input("Shock size (percentage points, pp)", value=1.00, step=0.25, format="%.2f")
        shock_quarter_nk = st.slider("Shock timing t (quarter index)", 1, T-1, 1, 1)
        shock_persist_nk = st.slider("Shock persistence ρ_shock (for demand/cost)", 0.0, 0.98, 0.80, 0.02)

        # ---- Snap-back option ----
        snapback = st.checkbox(
            "Snap-back (no persistence after the shock)",
            value=True,
            help="Sets ρx = γπ = 0 and forces the shock to be one-period (ρ_shock = 0). "
                 "Policy smoothing ρi is kept so i_t decays geometrically."
        )

        # ---- Display option for policy rate UNITS ----
        units_mode = st.radio(
            "Policy rate units",
            ["Deviation (pp)", "Level (% annual)"],
            index=0
        )

    neutral_rate_pct = st.number_input(
        "Baseline (neutral) nominal policy rate — % annual",
        value=2.00, step=0.25, format="%.2f",
        help="Use 2.00 for Bank of Canada's target neutral rate."
    )

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

        # Build shocks & simulate
        is_arr, pc_arr, pol_arr = build_shocks_original(
            T, shock_target, is_shock_size_pp, pc_shock_size_pp, policy_shock_bp_abs, shock_quarter, shock_persist
        )
        g0, p0, i0 = simulate_original(
            T, rho_sim, df_est, models_o, means_o, i_mean_dec, real_rate_mean_dec, pi_star_quarterly,
            policy_mode=policy_mode
        )
        gS, pS, iS = simulate_original(
            T, rho_sim, df_est, models_o, means_o, i_mean_dec, real_rate_mean_dec, pi_star_quarterly,
            is_shock_arr=is_arr, pc_shock_arr=pc_arr, policy_shock_arr=pol_arr, policy_mode=policy_mode
        )

        # Plot IRFs
        plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 12, "legend.fontsize": 11})
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        quarters = np.arange(T)
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

        plt.tight_layout(); st.pyplot(fig)

        # Readout at the shock quarter
        if shock_target.startswith("Taylor"):
            delta_i_bp = (iS - i0)[shock_quarter] * 10000.0
            st.info(f"Δ policy rate at t={shock_quarter}: {delta_i_bp:.1f} bp  |  mode: {policy_mode}  |  ρ={rho_sim:.2f}")

        # ===== LaTeX equations (reflect chosen variables) =====
        st.subheader("Estimated Equations (Original model)")

        m_is = models_o["model_is"]; m_pc = models_o["model_pc"]; m_tr = models_o["model_tr"]
        alpha_star = models_o["alpha_star"]; phi_pi_star = models_o["phi_pi_star"]; phi_g_star = models_o["phi_g_star"]
        rho_hat = models_o["rho_hat"]

        # IS equation
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

        # Phillips equation
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

        # Taylor rule (display matches chosen mode)
        st.markdown("**Taylor Rule (partial adjustment, with inflation gap)**")
        if policy_mode.startswith("Add after"):
            st.latex(r"i_t \;=\; \rho\, i_{t-1} \;+\; (1-\rho)\, i_t^\* \;+\; \varepsilon^{\text{pol}}_t")
        elif policy_mode.startswith("Add to target"):
            st.latex(r"i_t \;=\; \rho\, i_{t-1} \;+\; (1-\rho)\,\big(i_t^\* + \varepsilon^{\text{pol}}_t\big)")
        else:
            st.latex(r"i_t \;=\; \rho\, i_{t-1} \;+\; (1-\rho)\, i_t^\* \;+\; \varepsilon^{\text{pol}}_t \quad (\text{with local-jump override})")

        parts = [rf"\rho = {rho_hat:.3f}"]
        if not np.isnan(alpha_star): parts.append(rf"\alpha^\* = {alpha_star:.3f}")
        if not np.isnan(phi_pi_star): parts.append(rf"\phi_{{\pi}}^\* = {phi_pi_star:.3f}")
        if not np.isnan(phi_g_star): parts.append(rf"\phi_{{g}}^\* = {phi_g_star:.3f}")
        parts.append(rf"\pi^\* = {pi_star_quarterly:.4f}")
        st.latex(r"i_t^\* \;=\; \alpha^\* \;+\; \phi_{\pi}^\*\,(\pi_t - \pi^\*) \;+\; \phi_{g}^\*\,g_t")
        st.latex(r",\; ".join(parts))

        with st.expander("Model diagnostics (OLS summaries)"):
            st.write("**IS Curve**"); st.text(m_is.summary().as_text())
            st.write("**Phillips Curve**"); st.text(m_pc.summary().as_text())
            st.write("**Taylor Rule**"); st.text(m_tr.summary().as_text())

    else:
        # =========================
        # Simple NK (built-in)
        # =========================
        # Build parameter set (apply snapback if chosen)
        P_raw = NKParamsSimple(
            sigma=sigma, kappa=kappa, phi_pi=phi_pi, phi_x=phi_x,
            rho_i=rho_i,
            rho_x=(0.0 if snapback else rho_x),
            rho_r=rho_r, rho_u=rho_u,
            gamma_pi=(0.0 if snapback else gamma_pi)
        )
        model = SimpleNK3EqBuiltIn(P_raw, guarantee_convergence=guarantee_convergence)
        if model.enforced:
            st.caption("Convergence enforced: parameters projected to a stable region (Taylor principle & AR < 1).")

        label_to_code = {"Demand (IS)": "demand", "Cost-push (Phillips)": "cost", "Policy (Taylor)": "policy"}
        code = label_to_code[shock_type_nk]
        t0 = max(0, min(T-1, shock_quarter_nk - 1))

        rho_for_shock = 0.0 if snapback else shock_persist_nk

        # ===== IRF block =====
        st.subheader("Impulse responses (IRF mode)")
        h, x0, pi0, i0 = model.irf(code, T, 0.0, t0, rho_for_shock)
        h, xS, piS, iS = model.irf(code, T, shock_size_pp_nk, t0, rho_for_shock)

        # Prepare policy series for plotting in chosen units
        i0_plot, iS_plot = i0.copy(), iS.copy()
        i_ylabel = "pp"
        if units_mode == "Level (% annual)":
            i0_plot = neutral_rate_pct + i0_plot
            iS_plot = neutral_rate_pct + iS_plot
            i_ylabel = "%"

        plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 12, "legend.fontsize": 11})
        fig, axes = plt.subplots(3, 1, figsize=( Twelve, 12), sharex=True)
        # fix typo: use 12 not 'Twelve'
        plt.close(fig)  # ensure not displaying incorrect fig due to typo
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

        plt.tight_layout(); st.pyplot(fig)

        with st.expander("Simple NK equations"):
            st.latex(r"x_t = \rho_x x_{t-1} \;-\; \frac{1}{\sigma}\big( i_t - \pi_{t+1} - r^n_t \big)")
            st.latex(r"\pi_t = \gamma_\pi \pi_{t-1} \;+\; \kappa x_t \;+\; u_t")
            st.latex(r"i_t = \rho_i i_{t-1} \;+\; (1-\rho_i)(\phi_\pi \pi_t + \phi_x x_t) \;+\; \varepsilon^i_t")

        st.divider()

        # ===== Out-of-sample forecast block =====
        st.subheader("Out-of-sample forecast (Simple NK)")
        forecast_mode = st.checkbox("Enable forecast mode (label by calendar dates)", value=True)

        if forecast_mode:
            colA, colB, colC = st.columns(3)
            with colA:
                start_year = st.number_input("Start year", value=2019, step=1, format="%d")
            with colB:
                start_quarter = st.selectbox("Start quarter", ["Q1", "Q2", "Q3", "Q4"], index=3)
            with colC:
                T_fore = st.slider("Forecast horizon (quarters)", 4, 40, 24, 1)

            # Initial conditions (pp deviations)
            x0_init = st.number_input("Initial output gap x₀ (pp)", value=0.30, step=0.10, format="%.2f")
            pi0_init = st.number_input("Initial inflation π₀ (pp)", value=0.30, step=0.10, format="%.2f")
            i0_init = st.number_input("Initial policy rate i₀ (pp deviation)", value=0.30, step=0.10, format="%.2f")

            st.caption("Optional: upload CSV with columns **r_nat**, **u**, **e_i** (all in pp). Extra columns/rows ignored.")
            csv = st.file_uploader("Upload exogenous paths (optional)", type=["csv"], key="nk_fore_csv")

            r_nat_path = u_path = e_i_path = None
            if csv is not None:
                try:
                    df_exo = pd.read_csv(csv)
                    def _col(name):
                        if name in df_exo.columns:
                            s = pd.to_numeric(df_exo[name], errors="coerce").fillna(0.0).values.astype(float)
                            if len(s) < T_fore:
                                s = np.pad(s, (0, T_fore-len(s)), constant_values=0.0)
                            else:
                                s = s[:T_fore]
                            return s
                        return None
                    r_nat_path = _col("r_nat")
                    u_path     = _col("u")
                    e_i_path   = _col("e_i")
                except Exception as ee:
                    st.warning(f"Could not parse CSV: {ee}")

            # Simulate path (returns pp deviations for all three series)
            _, xF, piF, iF = model.simulate_path(
                T=T_fore, x0=x0_init, pi0=pi0_init, i0=i0_init,
                r_nat=r_nat_path, u=u_path, e_i=e_i_path
            )

            # Calendar labels (avoid freq=None bug)
            q_start = pd.Period(f"{start_year}Q{start_quarter[-1]}", freq="Q")
            dates = pd.period_range(start=q_start, periods=T_fore, freq="Q").strftime("%YQ%q")

            # Policy rate plotting units
            i_plot_fore = iF.copy()
            ylab_i = "pp"
            if units_mode == "Level (% annual)":
                i_plot_fore = neutral_rate_pct + i_plot_fore
                ylab_i = "%"

            # Plot forecast with calendar x-ticks
            plt.rcParams.update({"axes.titlesize": 14, "axes.labelsize": 11, "legend.fontsize": 10})
            figF, axesF = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            idx = np.arange(T_fore)

            axesF[0].plot(idx, xF, linewidth=2)
            axesF[0].set_title("Output Gap (pp)")
            axesF[0].grid(True, alpha=0.3)

            axesF[1].plot(idx, piF, linewidth=2)
            axesF[1].set_title("Inflation (pp)")
            axesF[1].grid(True, alpha=0.3)

            axesF[2].plot(idx, i_plot_fore, linewidth=2)
            axesF[2].set_title("Policy Rate")
            axesF[2].set_ylabel(ylab_i)
            axesF[2].set_xlabel("Quarter")
            axesF[2].grid(True, alpha=0.3)
            axesF[2].set_xticks(idx)
            axesF[2].set_xticklabels(dates, rotation=45)

            plt.tight_layout()
            st.pyplot(figF)

        with st.expander("Symbol glossary (Simple NK)"):
            st.markdown(
                r"""
- **$x_t$** — Output gap (percentage points, pp)  
- **$\pi_t$** — Inflation (pp)  
- **$i_t$** — Nominal policy rate (pp deviations; or % level when selected)  
- **$r_t^n$** — Demand / natural-rate shock (pp)  
- **$u_t$** — Cost-push shock (pp)  
- **$\sigma$,\,$\rho_x$,\,$\rho_r$** — IS dynamics  
- **$\kappa$,\,$\gamma_\pi$,\,$\rho_u$** — Phillips dynamics  
- **$\phi_\pi$,\,$\phi_x$,\,$\rho_i$** — Taylor dynamics  
                """
            )

except Exception as e:
    st.error(f"Problem loading or running the selected model: {e}")
    st.stop()





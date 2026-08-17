import io
import re
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ---------------------------
# Constants & helpers
# ---------------------------
_MODEL_TYPES = [
    "Power Law",
    "Linear",
    "Polynomial (2)",
    "Polynomial (3)",
    "Interaction (linear + cross terms)",
    "Exponential",
    "Auto (best fit)",
]

_FONT_FAMILIES = [
    "Arial", "Helvetica", "Times New Roman", "Courier New",
    "Verdana", "Georgia", "Trebuchet MS", "Roboto",
]

_PALETTE = (
    px.colors.qualitative.Plotly
    + px.colors.qualitative.D3
    + px.colors.qualitative.Set3
)


# Map variable names to well-known LaTeX symbols (case-insensitive lookup).
# Includes Greek letter names AND common engineering / CFD descriptive names.
_SYMBOL_MAP = {
    # --- Greek letter names ---
    "rho": r"\rho",
    "mu": r"\mu",
    "epsilon": r"\varepsilon",
    "nu": r"\nu",
    "sigma": r"\sigma",
    "tau": r"\tau",
    "omega": r"\omega",
    "delta": r"\delta",
    "gamma": r"\gamma",
    "alpha": r"\alpha",
    "beta": r"\beta",
    "lambda": r"\lambda",
    "theta": r"\theta",
    "phi": r"\phi",
    "eta": r"\eta",
    "kappa": r"\kappa",
    "zeta": r"\zeta",
    "pi": r"\pi",
    # --- Common engineering descriptive names ---
    "density": r"\rho",
    "viscosity": r"\mu",
    "dynamic viscosity": r"\mu",
    "dynamic_viscosity": r"\mu",
    "kinematic viscosity": r"\nu",
    "kinematic_viscosity": r"\nu",
    "energy dissipation": r"\varepsilon",
    "energy_dissipation": r"\varepsilon",
    "dissipation": r"\varepsilon",
    "edr": r"\varepsilon",
    "volume": "V",
    "vol": "V",
    "stir speed": "N",
    "stir_speed": "N",
    "stirrer speed": "N",
    "stirrer_speed": "N",
    "agitation speed": "N",
    "agitation_speed": "N",
    "rpm": "N",
    "speed": "N",
    "impeller speed": "N",
    "impeller_speed": "N",
    "temperature": "T",
    "temp": "T",
    "pressure": "P",
    "diameter": "D",
    "length": "L",
    "velocity": "U",
    "surface tension": r"\sigma",
    "surface_tension": r"\sigma",
    "shear stress": r"\tau",
    "shear_stress": r"\tau",
    "shear rate": r"\dot{\gamma}",
    "shear_rate": r"\dot{\gamma}",
    "power": "P_w",
    "torque": r"\mathcal{T}",
    "reynolds": r"\mathrm{Re}",
    "reynolds number": r"\mathrm{Re}",
    "reynolds_number": r"\mathrm{Re}",
    "re": r"\mathrm{Re}",
    # --- Compound / CamelCase names ---
    "fillvolume": "V",
    "fill volume": "V",
    "fill_volume": "V",
    "stirspeed": "N",
    "stir speed": "N",
    "stir_speed": "N",
    "impellerspeed": "N",
    "impeller speed": "N",
    "impeller_speed": "N",
    "agitationspeed": "N",
    "agitation speed": "N",
    "agitation_speed": "N",
}


def _render_latex_png(latex_str: str, fontsize: int = 20, dpi: int = 200) -> bytes:
    """Render a LaTeX equation string to a PNG byte buffer using matplotlib."""
    fig, ax = plt.subplots(figsize=(0.1, 0.1))
    ax.axis("off")
    txt = ax.text(
        0.5, 0.5, f"${latex_str}$",
        fontsize=fontsize, ha="center", va="center",
        transform=ax.transAxes,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = txt.get_window_extent(renderer=renderer)
    bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted()).expanded(1.15, 1.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches=bbox_inches,
                transparent=True, pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _fmt(text: str, bold: bool, italic: bool) -> str:
    if bold:
        text = f"<b>{text}</b>"
    if italic:
        text = f"<i>{text}</i>"
    return text


def _latex_equation(result: Dict, dep_var: str = "y") -> str:
    """Build a LaTeX string for the fitted model equation."""
    model = result["model"]
    coeffs = result["coefficients"]
    var_names = result["_var_names"]

    def _vname(n: str) -> str:
        # Try progressively more aggressive normalizations
        def _lookup(s: str) -> Optional[str]:
            k = s.strip().lower()
            for candidate in (
                k,
                k.replace("_", " "),
                k.replace(" ", "_"),
                k.replace("_", ""),
                k.replace(" ", ""),
            ):
                sym = _SYMBOL_MAP.get(candidate)
                if sym is not None:
                    return sym
            return None

        # Extract unit portion from brackets/parens: "Density[kg/m3]" → unit="kg/m3"
        unit_match = re.search(r"[\[\(](.+?)[\]\)]", n)
        unit_str = unit_match.group(1) if unit_match else ""

        # 1) Try the raw name
        sym = _lookup(n)

        # 2) Strip unit portion: "Density[kg/m3]" → "Density"
        if sym is None:
            base = re.sub(r"[\[\(\{].*$", "", n).strip()
            if base != n:
                sym = _lookup(base)

            # 3) Split CamelCase: "FillVolume" → "fill volume", then lookup
            if sym is None:
                split = re.sub(r"([a-z])([A-Z])", r"\1 \2", base).lower()
                sym = _lookup(split)

        if sym is not None:
            if unit_str:
                return sym + r" \; [\mathrm{" + unit_str + "}]"
            return sym

        # Fallback: escape underscores and wrap multi-char names in \mathrm
        safe = n.replace("_", r"\_")
        if len(n) > 1:
            return r"\mathrm{" + safe + "}"
        return safe

    def _coeff(v: float, digits: int = 4) -> str:
        return f"{v:.{digits}g}"

    lhs = _vname(dep_var)

    if model == "Power Law":
        C = coeffs["C"]
        parts = [_coeff(C)]
        for name in var_names:
            exp_val = coeffs[f"exp_{name}"]
            vn = _vname(name)
            if abs(exp_val - 1.0) < 1e-4:
                parts.append(vn)
            elif abs(exp_val) < 1e-6:
                continue
            else:
                parts.append(f"{vn}^{{{_coeff(exp_val)}}}")
        return lhs + " = " + r" \cdot ".join(parts)

    elif model == "Linear":
        terms = [_coeff(coeffs["intercept"])]
        for name in var_names:
            c = coeffs[name]
            sign = "+" if c >= 0 else "-"
            terms.append(f"{sign} {_coeff(abs(c))} \\, {_vname(name)}")
        return lhs + " = " + " ".join(terms)

    elif model.startswith("Polynomial"):
        degree = int(model.split("(")[1].rstrip(")"))
        terms = [_coeff(coeffs["1"])]
        for name in var_names:
            for d in range(1, degree + 1):
                key = f"{name}^{d}" if d > 1 else name
                c = coeffs[key]
                sign = "+" if c >= 0 else "-"
                vn = _vname(name)
                power = f"^{{{d}}}" if d > 1 else ""
                terms.append(f"{sign} {_coeff(abs(c))} \\, {vn}{power}")
        return lhs + " = " + " ".join(terms)

    elif model == "Interaction":
        terms = [_coeff(coeffs["1"])]
        for name in var_names:
            c = coeffs[name]
            sign = "+" if c >= 0 else "-"
            terms.append(f"{sign} {_coeff(abs(c))} \\, {_vname(name)}")
        for i, j in combinations(range(len(var_names)), 2):
            key = f"{var_names[i]}\u00b7{var_names[j]}"
            c = coeffs[key]
            sign = "+" if c >= 0 else "-"
            terms.append(
                f"{sign} {_coeff(abs(c))} \\, {_vname(var_names[i])} \\, {_vname(var_names[j])}"
            )
        return lhs + " = " + " ".join(terms)

    elif model == "Exponential":
        C = coeffs["C"]
        exp_parts = []
        for name in var_names:
            c = coeffs[name]
            sign = "+" if c >= 0 else "-"
            exp_parts.append(f"{sign} {_coeff(abs(c))} \\, {_vname(name)}")
        inner = " ".join(exp_parts).lstrip("+ ")
        return lhs + f" = {_coeff(C)}" + r" \cdot e^{" + inner + "}"

    return result["equation"]


# ---------------------------
# Model fitting functions
# ---------------------------
def _fit_power_law(X: np.ndarray, y: np.ndarray, var_names: List[str]) -> Optional[Dict]:
    """
    Power law: y = C * x1^a1 * x2^a2 * ...
    Fit via linear regression in log space: log(y) = log(C) + a1*log(x1) + ...
    Requires all values > 0.
    """
    if np.any(X <= 0) or np.any(y <= 0):
        return None
    log_X = np.log(X)
    log_y = np.log(y)
    # Add intercept column
    A = np.column_stack([np.ones(len(log_y)), log_X])
    try:
        coeffs, residuals, rank, sv = np.linalg.lstsq(A, log_y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    log_C = coeffs[0]
    exponents = coeffs[1:]
    C = np.exp(log_C)
    y_pred = C * np.prod(X ** exponents, axis=1)
    return _build_result(
        "Power Law", y, y_pred, var_names,
        _power_law_equation(C, exponents, var_names),
        {"C": C, **{f"exp_{n}": e for n, e in zip(var_names, exponents)}},
        raw_coeffs=coeffs,
    )


def _power_law_equation(C: float, exponents: np.ndarray, names: List[str]) -> str:
    parts = [f"{C:.4g}"]
    for name, exp in zip(names, exponents):
        if abs(exp - 1.0) < 1e-4:
            parts.append(f"{name}")
        elif abs(exp) < 1e-6:
            continue
        else:
            parts.append(f"{name}^{exp:.4g}")
    return "y = " + " · ".join(parts)


def _fit_linear(X: np.ndarray, y: np.ndarray, var_names: List[str]) -> Optional[Dict]:
    """Linear: y = a0 + a1*x1 + a2*x2 + ..."""
    A = np.column_stack([np.ones(len(y)), X])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    y_pred = A @ coeffs
    eq = _linear_equation(coeffs, var_names)
    return _build_result("Linear", y, y_pred, var_names, eq,
                         {"intercept": coeffs[0], **{n: c for n, c in zip(var_names, coeffs[1:])}},
                         raw_coeffs=coeffs)


def _linear_equation(coeffs: np.ndarray, names: List[str]) -> str:
    terms = [f"{coeffs[0]:.4g}"]
    for c, n in zip(coeffs[1:], names):
        sign = "+" if c >= 0 else "-"
        terms.append(f"{sign} {abs(c):.4g}·{n}")
    return "y = " + " ".join(terms)


def _fit_polynomial(X: np.ndarray, y: np.ndarray, var_names: List[str], degree: int) -> Optional[Dict]:
    """Polynomial with given degree (each variable independently raised to powers 1..degree)."""
    cols = [np.ones(len(y))]
    col_names = ["1"]
    for vi, name in enumerate(var_names):
        for d in range(1, degree + 1):
            cols.append(X[:, vi] ** d)
            col_names.append(f"{name}^{d}" if d > 1 else name)
    A = np.column_stack(cols)
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    y_pred = A @ coeffs
    eq = _poly_equation(coeffs, col_names)
    return _build_result(f"Polynomial ({degree})", y, y_pred, var_names, eq,
                         {n: c for n, c in zip(col_names, coeffs)},
                         raw_coeffs=coeffs)


def _fit_interaction(X: np.ndarray, y: np.ndarray, var_names: List[str]) -> Optional[Dict]:
    """Linear + all pairwise interaction terms: y = a0 + Σ ai·xi + Σ bij·xi·xj."""
    cols = [np.ones(len(y))]
    col_names = ["1"]
    for vi, name in enumerate(var_names):
        cols.append(X[:, vi])
        col_names.append(name)
    for i, j in combinations(range(len(var_names)), 2):
        cols.append(X[:, i] * X[:, j])
        col_names.append(f"{var_names[i]}·{var_names[j]}")
    A = np.column_stack(cols)
    if A.shape[1] > A.shape[0]:
        return None  # underdetermined
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    y_pred = A @ coeffs
    eq = _poly_equation(coeffs, col_names)
    return _build_result("Interaction", y, y_pred, var_names, eq,
                         {n: c for n, c in zip(col_names, coeffs)},
                         raw_coeffs=coeffs)


def _fit_exponential(X: np.ndarray, y: np.ndarray, var_names: List[str]) -> Optional[Dict]:
    """
    Exponential: y = C * exp(a1*x1 + a2*x2 + ...)
    Fit via log(y) = log(C) + a1*x1 + a2*x2 + ...
    Requires y > 0.
    """
    if np.any(y <= 0):
        return None
    log_y = np.log(y)
    A = np.column_stack([np.ones(len(log_y)), X])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, log_y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    C = np.exp(coeffs[0])
    a_coeffs = coeffs[1:]
    y_pred = C * np.exp(X @ a_coeffs)
    terms = [f"{C:.4g}"]
    exp_parts = []
    for c, n in zip(a_coeffs, var_names):
        sign = "+" if c >= 0 else "-"
        exp_parts.append(f"{sign} {abs(c):.4g}·{n}")
    eq = "y = " + terms[0] + " · exp(" + " ".join(exp_parts).lstrip("+ ") + ")"
    return _build_result("Exponential", y, y_pred, var_names, eq,
                         {"C": C, **{n: c for n, c in zip(var_names, a_coeffs)}},
                         raw_coeffs=coeffs)


def _poly_equation(coeffs: np.ndarray, col_names: List[str]) -> str:
    terms = []
    for c, n in zip(coeffs, col_names):
        if n == "1":
            terms.append(f"{c:.4g}")
        else:
            sign = "+" if c >= 0 else "-"
            terms.append(f"{sign} {abs(c):.4g}·{n}")
    return "y = " + " ".join(terms)


def _build_result(model_name: str, y_actual: np.ndarray, y_pred: np.ndarray,
                  var_names: List[str], equation: str, coefficients: Dict,
                  raw_coeffs: Optional[np.ndarray] = None) -> Dict:
    ss_res = np.sum((y_actual - y_pred) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    n = len(y_actual)
    p = len(var_names)
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2
    rmse = np.sqrt(np.mean((y_actual - y_pred) ** 2))
    mae = np.mean(np.abs(y_actual - y_pred))
    mape = np.mean(np.abs((y_actual - y_pred) / np.where(y_actual == 0, 1e-10, y_actual))) * 100
    return {
        "model": model_name,
        "equation": equation,
        "coefficients": coefficients,
        "_raw_coeffs": raw_coeffs,
        "_var_names": var_names,
        "y_pred": y_pred,
        "R2": r2,
        "R2_adj": r2_adj,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "n_points": n,
        "n_params": p + 1,
    }


def _fit_all_models(X: np.ndarray, y: np.ndarray, var_names: List[str]) -> List[Dict]:
    """Try all model types and return results sorted by R²."""
    results = []
    for fn in [
        lambda: _fit_power_law(X, y, var_names),
        lambda: _fit_linear(X, y, var_names),
        lambda: _fit_polynomial(X, y, var_names, 2),
        lambda: _fit_polynomial(X, y, var_names, 3),
        lambda: _fit_interaction(X, y, var_names),
        lambda: _fit_exponential(X, y, var_names),
    ]:
        try:
            r = fn()
            if r is not None and np.isfinite(r["R2"]):
                results.append(r)
        except Exception:
            pass
    results.sort(key=lambda r: r["R2"], reverse=True)
    return results


def _fit_model(model_type: str, X: np.ndarray, y: np.ndarray, var_names: List[str]) -> Optional[Dict]:
    dispatch = {
        "Power Law": lambda: _fit_power_law(X, y, var_names),
        "Linear": lambda: _fit_linear(X, y, var_names),
        "Polynomial (2)": lambda: _fit_polynomial(X, y, var_names, 2),
        "Polynomial (3)": lambda: _fit_polynomial(X, y, var_names, 3),
        "Interaction (linear + cross terms)": lambda: _fit_interaction(X, y, var_names),
        "Exponential": lambda: _fit_exponential(X, y, var_names),
    }
    fn = dispatch.get(model_type)
    if fn is None:
        return None
    try:
        return fn()
    except Exception:
        return None


def _predict_from_result(result: Dict, X_new: np.ndarray) -> np.ndarray:
    """Generate predictions from a fitted result dict on new X data."""
    model = result["model"]
    coeffs = result["_raw_coeffs"]
    var_names = result["_var_names"]

    if model == "Power Law":
        C = np.exp(coeffs[0])
        exponents = coeffs[1:]
        return C * np.prod(X_new ** exponents, axis=1)

    elif model == "Linear":
        A = np.column_stack([np.ones(len(X_new)), X_new])
        return A @ coeffs

    elif model.startswith("Polynomial"):
        degree = int(model.split("(")[1].rstrip(")"))
        cols = [np.ones(len(X_new))]
        for vi in range(X_new.shape[1]):
            for d in range(1, degree + 1):
                cols.append(X_new[:, vi] ** d)
        A = np.column_stack(cols)
        return A @ coeffs

    elif model == "Interaction":
        cols = [np.ones(len(X_new))]
        for vi in range(X_new.shape[1]):
            cols.append(X_new[:, vi])
        for i, j in combinations(range(X_new.shape[1]), 2):
            cols.append(X_new[:, i] * X_new[:, j])
        A = np.column_stack(cols)
        return A @ coeffs

    elif model == "Exponential":
        C = np.exp(coeffs[0])
        a_coeffs = coeffs[1:]
        return C * np.exp(X_new @ a_coeffs)

    return np.full(len(X_new), np.nan)


# ---------------------------
# State persistence helpers
# ---------------------------
_WIDGET_SUFFIXES = [
    "dep", "indep", "model", "filter_col", "filter_vals",
    "height", "fontfam", "titlefs", "tickfs",
    "titlebold", "titleital",
    "sweep_var", "sweep_min", "sweep_max", "sweep_n",
    "sweep_sets", "sweep_reflines",
    "sweep_leg_pos", "sweep_leg_fs",
    "sweep_xlabel", "sweep_ylabel",
    "sweep_axfs", "sweep_axbold", "sweep_axitalic",
    "sweep_xgrid", "sweep_ygrid",
    "sweep_gridcolor", "sweep_gridwidth",
]


def _save_state(plot_idx: int):
    store = {}
    for suffix in _WIDGET_SUFFIXES:
        wk = f"rom{plot_idx}_{suffix}"
        if wk in st.session_state:
            store[wk] = st.session_state[wk]
    st.session_state[f"_persist_rom{plot_idx}"] = store


def _restore_state(plot_idx: int):
    store = st.session_state.get(f"_persist_rom{plot_idx}", {})
    for wk, val in store.items():
        if wk not in st.session_state:
            st.session_state[wk] = val


# ---------------------------
# Page UI
# ---------------------------
# Fix Streamlit's LaTeX rendering clipping superscripts/exponents
st.markdown(
    """<style>.katex-display { overflow: visible !important; padding: 0.5em 0; }</style>""",
    unsafe_allow_html=True,
)

st.title("Reduced-Order Model Fitting")

if "rom_csv_data" not in st.session_state:
    st.session_state["rom_csv_data"] = None
if "rom_plot_configs" not in st.session_state:
    st.session_state["rom_plot_configs"] = [{}]

# File import
uploaded = st.file_uploader(
    "Upload a CSV file:",
    type=["csv", "tsv", "txt"],
    accept_multiple_files=False,
    key="rom_csv_upload",
)

if uploaded is not None:
    try:
        raw = uploaded.getvalue().decode("utf-8", errors="ignore")
        sio = io.StringIO(raw)
        try:
            data = pd.read_csv(sio, sep=None, engine="python")
        except Exception:
            sio.seek(0)
            data = pd.read_csv(sio, sep=r"\s+", engine="python")
        st.session_state["rom_csv_data"] = data
    except Exception as e:
        st.error(f"Failed to parse file: {e}")

data: Optional[pd.DataFrame] = st.session_state["rom_csv_data"]

if data is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

st.success(f"Loaded **{len(data)}** rows × **{len(data.columns)}** columns")

with st.expander("Preview data", expanded=False):
    st.dataframe(data.head(50), use_container_width=True)

all_cols = list(data.columns)
numeric_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(data[c])]
categorical_cols = [c for c in all_cols if not pd.api.types.is_numeric_dtype(data[c]) or data[c].nunique() <= 20]

if len(numeric_cols) < 2:
    st.warning("Need at least 2 numeric columns (1 dependent + 1 independent).")
    st.stop()

# ---------------------------
# Plot management
# ---------------------------
st.markdown("---")

add_col, remove_col, _ = st.columns([0.2, 0.2, 0.6])
with add_col:
    if st.button("➕ Add fit", use_container_width=True, key="rom_add"):
        st.session_state["rom_plot_configs"].append({})
        st.rerun()
with remove_col:
    if len(st.session_state["rom_plot_configs"]) > 1:
        if st.button("➖ Remove last fit", use_container_width=True, key="rom_remove"):
            st.session_state["rom_plot_configs"].pop()
            st.rerun()

n_plots = len(st.session_state["rom_plot_configs"])

# ---------------------------
# Render each fit
# ---------------------------
for plot_idx in range(n_plots):
    _restore_state(plot_idx)

    st.markdown(f"### Fit {plot_idx + 1}")

    with st.expander(f"Fit {plot_idx + 1} — Configuration", expanded=True):
        # --- Variable selection ---
        v1, v2 = st.columns(2)
        with v1:
            dep_var = st.selectbox(
                "Dependent variable (y):",
                options=numeric_cols,
                index=0,
                key=f"rom{plot_idx}_dep",
            )
        with v2:
            indep_default = [c for c in numeric_cols if c != dep_var][:min(3, len(numeric_cols) - 1)]
            indep_vars = st.multiselect(
                "Independent variables (x):",
                options=[c for c in numeric_cols if c != dep_var],
                default=indep_default,
                key=f"rom{plot_idx}_indep",
            )

        if not indep_vars:
            st.warning("Select at least one independent variable.")
            continue

        # --- Optional data filter ---
        st.markdown("**Data filter (optional)**")
        _fc1, _fc2 = st.columns(2)
        with _fc1:
            filter_col = st.selectbox(
                "Filter by column:",
                options=["(none)"] + categorical_cols,
                index=0,
                key=f"rom{plot_idx}_filter_col",
            )
        filter_vals = None
        if filter_col != "(none)":
            all_vals = sorted(data[filter_col].dropna().unique(), key=str)
            with _fc2:
                filter_vals = st.multiselect(
                    f"Include {filter_col} values:",
                    options=[str(v) for v in all_vals],
                    default=[str(v) for v in all_vals],
                    key=f"rom{plot_idx}_filter_vals",
                )

        # --- Model selection ---
        model_type = st.selectbox(
            "Model structure:",
            options=_MODEL_TYPES,
            index=0,
            key=f"rom{plot_idx}_model",
            help=(
                "**Power Law**: y = C · x₁^a₁ · x₂^a₂ · ... (common for Re, Nu, Sh correlations)\n\n"
                "**Linear**: y = a₀ + a₁x₁ + a₂x₂ + ...\n\n"
                "**Polynomial**: includes x², x³ terms per variable\n\n"
                "**Interaction**: linear + all pairwise cross-products\n\n"
                "**Exponential**: y = C · exp(a₁x₁ + a₂x₂ + ...)\n\n"
                "**Auto**: tries all models and ranks by R²"
            ),
        )

        # --- Plot styling ---
        st.markdown("**Styling**")
        _s1, _s2, _s3 = st.columns(3)
        with _s1:
            plot_height = st.slider("Plot height (px)", 300, 800, 450, 25, key=f"rom{plot_idx}_height")
        with _s2:
            font_family = st.selectbox("Font:", options=_FONT_FAMILIES, index=0,
                                       key=f"rom{plot_idx}_fontfam")
        with _s3:
            tick_font_size = st.slider("Font size", 8, 24, 12, 1, key=f"rom{plot_idx}_tickfs")

        _ff1, _ff2 = st.columns(2)
        with _ff1:
            title_bold = st.checkbox("Title bold", value=True, key=f"rom{plot_idx}_titlebold")
        with _ff2:
            title_italic = st.checkbox("Title italic", value=False, key=f"rom{plot_idx}_titleital")

        # --- Fit Model button ---
        if st.button("🔧 Fit Model", key=f"rom{plot_idx}_fit_go", use_container_width=True):
            # Prepare data
            _fit_df = data.copy()
            if filter_col != "(none)" and filter_vals is not None:
                _fit_df = _fit_df[_fit_df[filter_col].astype(str).isin(filter_vals)]
            _fit_df = _fit_df.dropna(subset=[dep_var] + indep_vars)

            if len(_fit_df) < len(indep_vars) + 2:
                st.warning(f"Not enough data points ({len(_fit_df)}) for {len(indep_vars)} variables. "
                           f"Need at least {len(indep_vars) + 2}.")
            else:
                _y = _fit_df[dep_var].values.astype(float)
                _X = _fit_df[indep_vars].values.astype(float)

                if model_type == "Auto (best fit)":
                    _all_res = _fit_all_models(_X, _y, indep_vars)
                    if not _all_res:
                        st.error("No model could be fitted. Check for zeros/negatives in power-law variables.")
                    else:
                        st.session_state[f"_fit_result_{plot_idx}"] = {
                            "result": _all_res[0],
                            "all_results": _all_res,
                            "fit_data": _fit_df,
                            "dep_var": dep_var,
                            "indep_vars": list(indep_vars),
                            "filter_col": filter_col,
                        }
                else:
                    _res = _fit_model(model_type, _X, _y, indep_vars)
                    if _res is None:
                        st.error(f"Failed to fit {model_type}. "
                                 "Power law and exponential require all-positive values.")
                    else:
                        st.session_state[f"_fit_result_{plot_idx}"] = {
                            "result": _res,
                            "all_results": None,
                            "fit_data": _fit_df,
                            "dep_var": dep_var,
                            "indep_vars": list(indep_vars),
                            "filter_col": filter_col,
                        }
            # Clear stale sweep result when re-fitting
            st.session_state.pop(f"_sweep_result_{plot_idx}", None)

    # --- Check for cached fit result ---
    _fit_cache = st.session_state.get(f"_fit_result_{plot_idx}")
    if _fit_cache is None:
        _save_state(plot_idx)
        st.markdown("---")
        continue

    result = _fit_cache["result"]
    fit_data = _fit_cache["fit_data"]
    dep_var = _fit_cache["dep_var"]
    indep_vars = _fit_cache["indep_vars"]
    filter_col = _fit_cache["filter_col"]

    y = fit_data[dep_var].values.astype(float)
    X = fit_data[indep_vars].values.astype(float)

    # Show comparison table for Auto mode
    if _fit_cache["all_results"] is not None:
        with st.expander(f"Fit {plot_idx + 1} — Model comparison", expanded=True):
            comp_df = pd.DataFrame([
                {"Model": r["model"], "R²": f"{r['R2']:.6f}", "R² adj": f"{r['R2_adj']:.6f}",
                 "RMSE": f"{r['RMSE']:.4g}", "MAE": f"{r['MAE']:.4g}", "MAPE": f"{r['MAPE']:.1f}%"}
                for r in _fit_cache["all_results"]
            ])
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            st.success(f"**Best model: {result['model']}** (R² = {result['R2']:.6f})")

    # --- Display results ---
    with st.expander(f"Fit {plot_idx + 1} — Results", expanded=True):
        st.markdown(f"**Model:** {result['model']}")
        _latex_str = _latex_equation(result, dep_var=dep_var)
        st.latex(_latex_str)

        # Equation PNG download
        if st.button("📥 Download equation as PNG", key=f"btn_eq_png_{plot_idx}",
                     use_container_width=False):
            try:
                _eq_png = _render_latex_png(_latex_str)
                st.session_state[f"_export_eq_png_{plot_idx}"] = _eq_png
            except Exception as _e:
                st.caption(f"Equation PNG export failed: {_e}")
        _cached_eq_png = st.session_state.get(f"_export_eq_png_{plot_idx}")
        if _cached_eq_png is not None:
            st.download_button(
                "⬇ Download PNG", data=_cached_eq_png,
                file_name=f"equation_fit_{plot_idx + 1}.png", mime="image/png",
                key=f"dl_eq_png_{plot_idx}",
            )

        _m1, _m2, _m3, _m4 = st.columns(4)
        _m1.metric("R²", f"{result['R2']:.6f}")
        _m2.metric("R² (adj)", f"{result['R2_adj']:.6f}")
        _m3.metric("RMSE", f"{result['RMSE']:.4g}")
        _m4.metric("MAPE", f"{result['MAPE']:.1f}%")

        st.markdown(f"Fitted on **{result['n_points']}** data points with **{result['n_params']}** parameters")

        # Coefficients table
        coeff_df = pd.DataFrame([
            {"Parameter": k, "Value": f"{v:.6g}"}
            for k, v in result["coefficients"].items()
        ])
        st.dataframe(coeff_df, use_container_width=True, hide_index=True)

    # --- Plots ---
    _base_font = dict(family=font_family, size=tick_font_size)
    y_pred = result["y_pred"]

    # Parity plot
    _title_parity = _fmt(f"Parity Plot — {dep_var} ({result['model']})", title_bold, title_italic)

    fig_parity = go.Figure()

    # Perfect fit line
    _min_val = min(y.min(), y_pred.min())
    _max_val = max(y.max(), y_pred.max())
    _pad = (_max_val - _min_val) * 0.05
    fig_parity.add_trace(go.Scatter(
        x=[_min_val - _pad, _max_val + _pad],
        y=[_min_val - _pad, _max_val + _pad],
        mode="lines", line=dict(color="grey", dash="dash", width=1.5),
        name="Perfect fit", showlegend=True,
    ))

    # ±10% bands
    fig_parity.add_trace(go.Scatter(
        x=[_min_val - _pad, _max_val + _pad],
        y=[(1.1) * (_min_val - _pad), (1.1) * (_max_val + _pad)],
        mode="lines", line=dict(color="rgba(150,150,150,0.4)", dash="dot", width=1),
        name="+10%", showlegend=True,
    ))
    fig_parity.add_trace(go.Scatter(
        x=[_min_val - _pad, _max_val + _pad],
        y=[(0.9) * (_min_val - _pad), (0.9) * (_max_val + _pad)],
        mode="lines", line=dict(color="rgba(150,150,150,0.4)", dash="dot", width=1),
        name="-10%", showlegend=True,
    ))

    # Color by filter column if applicable
    if filter_col != "(none)" and filter_col in fit_data.columns:
        cat_vals = fit_data[filter_col].astype(str).values
        unique_cats = sorted(set(cat_vals))
        for ci, cat in enumerate(unique_cats):
            mask = cat_vals == cat
            fig_parity.add_trace(go.Scatter(
                x=y[mask], y=y_pred[mask],
                mode="markers",
                marker=dict(size=8, color=_PALETTE[ci % len(_PALETTE)]),
                name=cat,
                hovertemplate=f"{cat}<br>Actual: %{{x:.4g}}<br>Predicted: %{{y:.4g}}<extra></extra>",
            ))
    else:
        fig_parity.add_trace(go.Scatter(
            x=y, y=y_pred,
            mode="markers",
            marker=dict(size=8, color=_PALETTE[0]),
            name="Data",
            hovertemplate="Actual: %{x:.4g}<br>Predicted: %{y:.4g}<extra></extra>",
        ))

    fig_parity.update_layout(
        title=dict(text=_title_parity, font=dict(family=font_family, size=tick_font_size + 4)),
        xaxis_title=_fmt(f"Actual {dep_var}", False, False),
        yaxis_title=_fmt(f"Predicted {dep_var}", False, False),
        xaxis=dict(tickfont=_base_font, title_font=_base_font, scaleanchor="y"),
        yaxis=dict(tickfont=_base_font, title_font=_base_font),
        font=_base_font,
        height=plot_height,
        legend=dict(font=_base_font),
        margin=dict(l=60, r=20, t=60, b=60),
    )

    st.plotly_chart(fig_parity, use_container_width=True, key=f"rom_parity_{plot_idx}")

    # Residual plot
    residuals = y - y_pred
    _title_resid = _fmt(f"Residuals — {dep_var}", title_bold, title_italic)

    fig_resid = go.Figure()
    fig_resid.add_hline(y=0, line_color="grey", line_dash="dash", line_width=1)

    if filter_col != "(none)" and filter_col in fit_data.columns:
        cat_vals = fit_data[filter_col].astype(str).values
        unique_cats = sorted(set(cat_vals))
        for ci, cat in enumerate(unique_cats):
            mask = cat_vals == cat
            fig_resid.add_trace(go.Scatter(
                x=y_pred[mask], y=residuals[mask],
                mode="markers",
                marker=dict(size=8, color=_PALETTE[ci % len(_PALETTE)]),
                name=cat,
                hovertemplate=f"{cat}<br>Predicted: %{{x:.4g}}<br>Residual: %{{y:.4g}}<extra></extra>",
            ))
    else:
        fig_resid.add_trace(go.Scatter(
            x=y_pred, y=residuals,
            mode="markers",
            marker=dict(size=8, color=_PALETTE[0]),
            name="Residuals",
            hovertemplate="Predicted: %{x:.4g}<br>Residual: %{y:.4g}<extra></extra>",
        ))

    fig_resid.update_layout(
        title=dict(text=_title_resid, font=dict(family=font_family, size=tick_font_size + 4)),
        xaxis_title=_fmt(f"Predicted {dep_var}", False, False),
        yaxis_title=_fmt("Residual (actual − predicted)", False, False),
        xaxis=dict(tickfont=_base_font, title_font=_base_font),
        yaxis=dict(tickfont=_base_font, title_font=_base_font),
        font=_base_font,
        height=max(300, plot_height - 100),
        legend=dict(font=_base_font),
        margin=dict(l=60, r=20, t=60, b=60),
    )

    st.plotly_chart(fig_resid, use_container_width=True, key=f"rom_resid_{plot_idx}")

    # Download buttons — defer export to button click
    _dl1, _dl2, _ = st.columns([0.2, 0.2, 0.6])
    with _dl1:
        if st.button("📥 Export Parity HTML", key=f"btn_parity_html_{plot_idx}",
                     use_container_width=True):
            _html = fig_parity.to_html(include_plotlyjs="cdn").encode("utf-8")
            st.session_state[f"_export_parity_html_{plot_idx}"] = _html
        _cached_html = st.session_state.get(f"_export_parity_html_{plot_idx}")
        if _cached_html is not None:
            st.download_button(
                "⬇ Download HTML", data=_cached_html,
                file_name=f"parity_plot_{plot_idx + 1}.html", mime="text/html",
                key=f"dl_parity_html_{plot_idx}", use_container_width=True,
            )
    with _dl2:
        if st.button("📥 Export Parity PNG", key=f"btn_parity_png_{plot_idx}",
                     use_container_width=True):
            try:
                _png = fig_parity.to_image(format="png", width=800, height=plot_height, scale=2)
                st.session_state[f"_export_parity_png_{plot_idx}"] = _png
            except Exception as _e:
                st.caption(f"PNG export unavailable: {_e}")
        _cached_png = st.session_state.get(f"_export_parity_png_{plot_idx}")
        if _cached_png is not None:
            st.download_button(
                "⬇ Download PNG", data=_cached_png,
                file_name=f"parity_plot_{plot_idx + 1}.png", mime="image/png",
                key=f"dl_parity_png_{plot_idx}", use_container_width=True,
            )

    # --- Prediction sweep ---
    if len(indep_vars) >= 1 and result.get("_raw_coeffs") is not None:
        with st.expander(f"Fit {plot_idx + 1} — Prediction sweep", expanded=False):
            st.markdown("Sweep one independent variable while holding others fixed. "
                        "Define one or more sets of fixed values.")

            _sw1, _sw2 = st.columns(2)
            with _sw1:
                sweep_var = st.selectbox(
                    "Variable to sweep (X-axis):",
                    options=indep_vars,
                    index=0,
                    key=f"rom{plot_idx}_sweep_var",
                )
            _sweep_col_data = fit_data[sweep_var]
            _sweep_min_default = float(_sweep_col_data.min())
            _sweep_max_default = float(_sweep_col_data.max())
            with _sw2:
                sweep_n = st.slider("Number of points", 20, 500, 100, 10,
                                    key=f"rom{plot_idx}_sweep_n")

            _sr1, _sr2 = st.columns(2)
            with _sr1:
                sweep_min = st.number_input("Sweep min:", value=_sweep_min_default,
                                            format="%g", key=f"rom{plot_idx}_sweep_min")
            with _sr2:
                sweep_max = st.number_input("Sweep max:", value=_sweep_max_default,
                                            format="%g", key=f"rom{plot_idx}_sweep_max")

            other_vars = [v for v in indep_vars if v != sweep_var]

            if other_vars:
                st.markdown("**Fixed variable sets** — define comma-separated values for each row, "
                            "one set per line (e.g. `0.5, 300, 800`)")

                # Build default: one set with median values
                _default_vals = [f"{float(fit_data[v].median()):.4g}" for v in other_vars]
                _default_text = ", ".join(_default_vals)

                sweep_sets_text = st.text_area(
                    f"Fixed values for [{', '.join(other_vars)}] — one set per line:",
                    value=_default_text,
                    height=100,
                    key=f"rom{plot_idx}_sweep_sets",
                    help=f"Each line should have {len(other_vars)} comma-separated values "
                         f"for: {', '.join(other_vars)}",
                )

                # Parse sets
                sweep_sets: List[Tuple[str, List[float]]] = []
                for line in sweep_sets_text.strip().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) != len(other_vars):
                        st.warning(f"Line `{line}` has {len(parts)} values, expected {len(other_vars)}. Skipped.")
                        continue
                    try:
                        vals = [float(p) for p in parts]
                        label = ", ".join(f"{n}={v:.4g}" for n, v in zip(other_vars, vals))
                        sweep_sets.append((label, vals))
                    except ValueError:
                        st.warning(f"Could not parse line `{line}`. Skipped.")
            else:
                # Only one independent variable — no fixed vars needed
                sweep_sets = [("", [])]

            # --- Series color pickers ---
            if sweep_sets:
                st.markdown("**Series colors**")
                _n_sets = len(sweep_sets)
                _color_cols = st.columns(min(_n_sets, 6))
                _sweep_colors: List[str] = []
                for si, (set_label, _) in enumerate(sweep_sets):
                    _clabel = set_label if set_label else f"Series {si + 1}"
                    _default_c = _PALETTE[si % len(_PALETTE)]
                    with _color_cols[si % len(_color_cols)]:
                        _c = st.color_picker(
                            _clabel, value=_default_c,
                            key=f"rom{plot_idx}_sweep_color_{si}",
                        )
                        _sweep_colors.append(_c)

            # --- Horizontal reference lines ---
            st.markdown("**Horizontal reference lines** — one per line: "
                        "`y_value` or `y_value, label`")
            st.text_area(
                "Reference lines:",
                value="",
                height=68,
                key=f"rom{plot_idx}_sweep_reflines",
                help="Add horizontal lines at specific y values. "
                     "Format: y_value or y_value, label (one per line)",
            )

            # --- Legend settings ---
            st.markdown("**Legend**")
            _lg1, _lg2 = st.columns(2)
            _LEGEND_POSITIONS = [
                "Top-right", "Top-left", "Bottom-right", "Bottom-left",
                "Top-center", "Bottom-center", "Outside right",
            ]
            with _lg1:
                sweep_leg_pos = st.selectbox(
                    "Legend position:",
                    options=_LEGEND_POSITIONS,
                    index=0,
                    key=f"rom{plot_idx}_sweep_leg_pos",
                )
            with _lg2:
                sweep_leg_fs = st.slider(
                    "Legend font size", 6, 24, 10, 1,
                    key=f"rom{plot_idx}_sweep_leg_fs",
                )

            # --- Axis labels ---
            st.markdown("**Axis labels** (leave blank for auto)")
            _al1, _al2 = st.columns(2)
            with _al1:
                sweep_xlabel = st.text_input(
                    "X-axis label:", value="",
                    key=f"rom{plot_idx}_sweep_xlabel",
                )
            with _al2:
                sweep_ylabel = st.text_input(
                    "Y-axis label:", value="",
                    key=f"rom{plot_idx}_sweep_ylabel",
                )

            # --- Axis title font formatting ---
            _af1, _af2, _af3 = st.columns(3)
            with _af1:
                sweep_axfs = st.slider(
                    "Axis title font size", 8, 28, 14, 1,
                    key=f"rom{plot_idx}_sweep_axfs",
                )
            with _af2:
                sweep_axbold = st.checkbox(
                    "Axis title bold", value=False,
                    key=f"rom{plot_idx}_sweep_axbold",
                )
            with _af3:
                sweep_axitalic = st.checkbox(
                    "Axis title italic", value=False,
                    key=f"rom{plot_idx}_sweep_axitalic",
                )

            # --- Gridlines ---
            st.markdown("**Gridlines**")
            _gl1, _gl2, _gl3, _gl4 = st.columns(4)
            with _gl1:
                sweep_xgrid = st.checkbox("X gridlines", value=True,
                                          key=f"rom{plot_idx}_sweep_xgrid")
            with _gl2:
                sweep_ygrid = st.checkbox("Y gridlines", value=True,
                                          key=f"rom{plot_idx}_sweep_ygrid")
            with _gl3:
                sweep_gridcolor = st.color_picker("Grid color", value="#E5E5E5",
                                                   key=f"rom{plot_idx}_sweep_gridcolor")
            with _gl4:
                sweep_gridwidth = st.slider("Grid width", 0.5, 3.0, 1.0, 0.25,
                                             key=f"rom{plot_idx}_sweep_gridwidth")

            # --- Generate button ---
            _can_sweep = bool(sweep_sets) and sweep_min < sweep_max
            if not _can_sweep:
                st.info("Configure valid sweep parameters above, then click **Generate**.")

            _sweep_key = f"rom{plot_idx}_sweep_go"
            if st.button("🚀 Generate sweep plot", key=_sweep_key,
                         disabled=not _can_sweep, use_container_width=True):
                # Compute and cache results in session state
                x_sweep = np.linspace(sweep_min, sweep_max, sweep_n)
                _sweep_data: List[Tuple[str, np.ndarray, np.ndarray]] = []
                for set_label, fixed_vals in sweep_sets:
                    X_sw = np.zeros((len(x_sweep), len(indep_vars)))
                    fi = 0
                    for vi, vname in enumerate(indep_vars):
                        if vname == sweep_var:
                            X_sw[:, vi] = x_sweep
                        else:
                            X_sw[:, vi] = fixed_vals[fi]
                            fi += 1
                    y_sw = _predict_from_result(result, X_sw)
                    _sweep_data.append((set_label, x_sweep, y_sw))
                st.session_state[f"_sweep_result_{plot_idx}"] = {
                    "data": _sweep_data,
                    "sweep_var": sweep_var,
                    "dep_var": dep_var,
                    "model": result["model"],
                }

            # --- Display stored sweep results ---
            _stored = st.session_state.get(f"_sweep_result_{plot_idx}")
            if _stored is not None:
                _sv = _stored["sweep_var"]
                _dv = _stored["dep_var"]

                _title_sweep = _fmt(
                    f"{_dv} vs {_sv} ({_stored['model']})",
                    title_bold, title_italic,
                )

                fig_sweep = go.Figure()

                _series_tables: list = []

                for si, (set_label, x_arr, y_arr) in enumerate(_stored["data"]):
                    trace_name = set_label if set_label else _dv
                    color = st.session_state.get(
                        f"rom{plot_idx}_sweep_color_{si}",
                        _PALETTE[si % len(_PALETTE)],
                    )

                    fig_sweep.add_trace(go.Scatter(
                        x=x_arr, y=y_arr,
                        mode="lines",
                        line=dict(color=color, width=2),
                        name=trace_name,
                        hovertemplate=(
                            f"{trace_name}<br>"
                            f"{_sv}=%{{x:.4g}}<br>"
                            f"{_dv}=%{{y:.4g}}<extra></extra>"
                        ),
                    ))

                    # Sample 10 evenly-spaced points for the data table
                    _idx10 = np.linspace(0, len(x_arr) - 1, 10, dtype=int)
                    _rows = [{_sv: f"{x_arr[i]:.4g}", _dv: f"{y_arr[i]:.4g}"} for i in _idx10]
                    _series_tables.append((trace_name, pd.DataFrame(_rows)))

                # Add horizontal reference lines as legend-visible traces
                _rl_text = st.session_state.get(f"rom{plot_idx}_sweep_reflines", "")
                _rl_lines_parsed = []
                for _rl_line in _rl_text.strip().splitlines():
                    _rl_line = _rl_line.strip()
                    if not _rl_line:
                        continue
                    _rl_parts = [p.strip() for p in _rl_line.split(",", 1)]
                    try:
                        _rl_y = float(_rl_parts[0])
                        _rl_label = _rl_parts[1] if len(_rl_parts) > 1 else f"y={_rl_y:.4g}"
                        _rl_lines_parsed.append((_rl_y, _rl_label))
                    except ValueError:
                        pass

                # Distinct colors for reference lines (skip colors already used by series)
                _rl_color_pool = [
                    "#E00000", "#008000", "#0000CD", "#FF8C00",
                    "#8B008B", "#00CED1", "#B8860B", "#4B0082",
                ]
                _n_series = len(_stored["data"])
                for _ri, (_rl_y, _rl_label) in enumerate(_rl_lines_parsed):
                    _rl_color = _rl_color_pool[_ri % len(_rl_color_pool)]
                    # Use a full-width scatter trace so it appears in the legend
                    fig_sweep.add_trace(go.Scatter(
                        x=[sweep_min, sweep_max],
                        y=[_rl_y, _rl_y],
                        mode="lines",
                        line=dict(color=_rl_color, dash="dash", width=1.5),
                        name=_rl_label,
                        showlegend=True,
                        hovertemplate=f"{_rl_label}: %{{y:.4g}}<extra></extra>",
                    ))

                # Build legend position kwargs
                _leg_pos = st.session_state.get(f"rom{plot_idx}_sweep_leg_pos", "Top-right")
                _leg_fs = st.session_state.get(f"rom{plot_idx}_sweep_leg_fs", 10)
                _leg_kwargs: dict = {"font": dict(family=font_family, size=_leg_fs)}
                if _leg_pos == "Top-right":
                    _leg_kwargs.update(x=1, y=1, xanchor="right", yanchor="top")
                elif _leg_pos == "Top-left":
                    _leg_kwargs.update(x=0, y=1, xanchor="left", yanchor="top")
                elif _leg_pos == "Bottom-right":
                    _leg_kwargs.update(x=1, y=0, xanchor="right", yanchor="bottom")
                elif _leg_pos == "Bottom-left":
                    _leg_kwargs.update(x=0, y=0, xanchor="left", yanchor="bottom")
                elif _leg_pos == "Top-center":
                    _leg_kwargs.update(x=0.5, y=1, xanchor="center", yanchor="top")
                elif _leg_pos == "Bottom-center":
                    _leg_kwargs.update(x=0.5, y=0, xanchor="center", yanchor="bottom")
                elif _leg_pos == "Outside right":
                    _leg_kwargs.update(x=1.02, y=1, xanchor="left", yanchor="top")

                # Axis labels (custom or auto)
                _xlabel = st.session_state.get(f"rom{plot_idx}_sweep_xlabel", "") or _sv
                _ylabel = st.session_state.get(f"rom{plot_idx}_sweep_ylabel", "") or _dv

                # Axis title font formatting
                _axfs = st.session_state.get(f"rom{plot_idx}_sweep_axfs", 14)
                _axbold = st.session_state.get(f"rom{plot_idx}_sweep_axbold", False)
                _axitalic = st.session_state.get(f"rom{plot_idx}_sweep_axitalic", False)
                _ax_title_font = dict(family=font_family, size=_axfs)

                # Gridline settings
                _xgrid = st.session_state.get(f"rom{plot_idx}_sweep_xgrid", True)
                _ygrid = st.session_state.get(f"rom{plot_idx}_sweep_ygrid", True)
                _gcolor = st.session_state.get(f"rom{plot_idx}_sweep_gridcolor", "#E5E5E5")
                _gwidth = st.session_state.get(f"rom{plot_idx}_sweep_gridwidth", 1.0)

                fig_sweep.update_layout(
                    title=dict(text=_title_sweep,
                               font=dict(family=font_family, size=tick_font_size + 4)),
                    xaxis_title=_fmt(_xlabel, _axbold, _axitalic),
                    yaxis_title=_fmt(_ylabel, _axbold, _axitalic),
                    xaxis=dict(
                        tickfont=_base_font, title_font=_ax_title_font,
                        showgrid=_xgrid, gridcolor=_gcolor, gridwidth=_gwidth,
                    ),
                    yaxis=dict(
                        tickfont=_base_font, title_font=_ax_title_font,
                        showgrid=_ygrid, gridcolor=_gcolor, gridwidth=_gwidth,
                    ),
                    font=_base_font,
                    height=plot_height,
                    legend=_leg_kwargs,
                    margin=dict(l=60, r=20, t=60, b=60),
                )

                st.plotly_chart(fig_sweep, use_container_width=True,
                                key=f"rom_sweep_{plot_idx}")

                # Data tables per series
                st.markdown("**Sweep data (10 points per series)**")
                for _sname, _sdf in _series_tables:
                    with st.expander(f"📊 {_sname}", expanded=False):
                        st.dataframe(_sdf, use_container_width=True, hide_index=True)

                _sdl1, _sdl2, _ = st.columns([0.2, 0.2, 0.6])
                with _sdl1:
                    if st.button("📥 Export Sweep HTML", key=f"btn_sweep_html_{plot_idx}",
                                 use_container_width=True):
                        _html_s = fig_sweep.to_html(include_plotlyjs="cdn").encode("utf-8")
                        st.session_state[f"_export_sweep_html_{plot_idx}"] = _html_s
                    _cached_shtml = st.session_state.get(f"_export_sweep_html_{plot_idx}")
                    if _cached_shtml is not None:
                        st.download_button(
                            "⬇ Download HTML", data=_cached_shtml,
                            file_name=f"sweep_plot_{plot_idx + 1}.html", mime="text/html",
                            key=f"dl_sweep_html_{plot_idx}", use_container_width=True,
                        )
                with _sdl2:
                    if st.button("📥 Export Sweep PNG", key=f"btn_sweep_png_{plot_idx}",
                                 use_container_width=True):
                        try:
                            _png_s = fig_sweep.to_image(format="png", width=800,
                                                         height=plot_height, scale=2)
                            st.session_state[f"_export_sweep_png_{plot_idx}"] = _png_s
                        except Exception as _e:
                            st.caption(f"PNG export unavailable: {_e}")
                    _cached_spng = st.session_state.get(f"_export_sweep_png_{plot_idx}")
                    if _cached_spng is not None:
                        st.download_button(
                            "⬇ Download PNG", data=_cached_spng,
                            file_name=f"sweep_plot_{plot_idx + 1}.png", mime="image/png",
                            key=f"dl_sweep_png_{plot_idx}", use_container_width=True,
                        )

    _save_state(plot_idx)
    st.markdown("---")

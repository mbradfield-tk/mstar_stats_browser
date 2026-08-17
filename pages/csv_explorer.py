import io
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
import streamlit as st

st.title("CSV Data Explorer")


# ---------------------------
# Trendline helpers
# ---------------------------
_TRENDLINE_TYPES = [
    "None",
    "Linear",
    "Polynomial (2)",
    "Polynomial (3)",
    "Polynomial (4)",
    "Polynomial (5)",
    "Exponential",
    "Logarithmic",
    "Power",
    "Moving Average",
]


def _fit_trendline(
    x: np.ndarray,
    y: np.ndarray,
    kind: str,
    ma_window: int = 5,
) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    """
    Return (x_fit, y_fit, equation_label) or None if fitting fails / not applicable.
    """
    if len(x) < 2:
        return None

    # Sort by x for clean line output
    order = np.argsort(x)
    xs, ys = x[order], y[order]

    # Remove NaN / Inf
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[mask], ys[mask]
    if len(xs) < 2:
        return None

    x_dense = np.linspace(xs.min(), xs.max(), max(200, len(xs)))

    try:
        if kind == "Linear":
            c = np.polyfit(xs, ys, 1)
            label = f"y = {c[0]:.4g}x + {c[1]:.4g}"
            return x_dense, np.polyval(c, x_dense), label

        if kind.startswith("Polynomial"):
            deg = int(kind.split("(")[1].rstrip(")"))
            c = np.polyfit(xs, ys, deg)
            terms = []
            for i, ci in enumerate(c):
                power = deg - i
                if power == 0:
                    terms.append(f"{ci:.4g}")
                elif power == 1:
                    terms.append(f"{ci:.4g}x")
                else:
                    terms.append(f"{ci:.4g}x^{power}")
            label = "y = " + " + ".join(terms)
            return x_dense, np.polyval(c, x_dense), label

        if kind == "Exponential":
            # y = a * exp(b * x)
            if np.any(ys <= 0):
                return None

            def exp_func(xv, a, b):
                return a * np.exp(b * xv)

            log_y = np.log(ys)
            b0, a0 = np.polyfit(xs, log_y, 1)
            popt, _ = curve_fit(exp_func, xs, ys, p0=[np.exp(a0), b0], maxfev=10000)
            label = f"y = {popt[0]:.4g} · e^({popt[1]:.4g}x)"
            return x_dense, exp_func(x_dense, *popt), label

        if kind == "Logarithmic":
            # y = a * ln(x) + b
            if np.any(xs <= 0):
                return None
            log_x = np.log(xs)
            c = np.polyfit(log_x, ys, 1)
            label = f"y = {c[0]:.4g} · ln(x) + {c[1]:.4g}"
            x_pos = x_dense[x_dense > 0]
            return x_pos, np.polyval(c, np.log(x_pos)), label

        if kind == "Power":
            # y = a * x^b
            if np.any(xs <= 0) or np.any(ys <= 0):
                return None
            log_x, log_y = np.log(xs), np.log(ys)
            c = np.polyfit(log_x, log_y, 1)
            a = np.exp(c[1])
            b = c[0]
            label = f"y = {a:.4g} · x^{b:.4g}"
            x_pos = x_dense[x_dense > 0]
            return x_pos, a * x_pos**b, label

        if kind == "Moving Average":
            if len(xs) < ma_window:
                return None
            y_ma = pd.Series(ys).rolling(window=ma_window, center=True, min_periods=1).mean().values
            label = f"MA({ma_window})"
            return xs, y_ma, label

    except Exception:
        return None
    return None


# ---------------------------
# Session state initialization
# ---------------------------
if "csv_data" not in st.session_state:
    st.session_state["csv_data"] = None
if "plot_configs" not in st.session_state:
    st.session_state["plot_configs"] = [{}]  # Start with one empty plot config


# ---------------------------
# File import
# ---------------------------
uploaded = st.file_uploader(
    "Upload a CSV file:",
    type=["csv", "tsv", "txt"],
    accept_multiple_files=False,
    key="csv_upload",
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
        st.session_state["csv_data"] = data
    except Exception as e:
        st.error(f"Failed to parse file: {e}")


# ---------------------------
# Export / Import settings
# ---------------------------
_PLOT_WIDGET_SUFFIXES = [
    "xcol", "ycol", "primary", "secondary", "pmode",
    "pfilt", "sfilt", "type", "msize", "lw", "height",
    "trend", "tw", "ma", "eq", "tdash",
    "refsel", "refagg", "refstyle", "refw", "reflbl", "customref",
    "vgrid", "vgdtick",
    "xlabel", "ylabel",
    "lgshow", "lgpos", "lgfont",
    "axfsize", "xbold", "xitalic", "ybold", "yitalic",
]


def _save_csv_plot_state(plot_idx: int):
    """Save widget values to shadow keys that survive page switches."""
    store = {}
    for suffix in _PLOT_WIDGET_SUFFIXES:
        wk = f"p{plot_idx}_{suffix}"
        if wk in st.session_state:
            store[wk] = st.session_state[wk]
    for key in list(st.session_state.keys()):
        if key.startswith(f"p{plot_idx}_scolor_") or key.startswith(f"p{plot_idx}_sname_"):
            store[key] = st.session_state[key]
    st.session_state[f"_persist_p{plot_idx}"] = store


def _restore_csv_plot_state(plot_idx: int):
    """Restore widget values from shadow keys if they were garbage-collected."""
    store = st.session_state.get(f"_persist_p{plot_idx}", {})
    for wk, val in store.items():
        if wk not in st.session_state:
            st.session_state[wk] = val


def _make_serializable(val):
    """Convert numpy / pandas types to plain Python for JSON."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (list, tuple)):
        return [_make_serializable(v) for v in val]
    return val


with st.expander("\U0001f4be Export / Import Settings", expanded=False):
    exp_col, imp_col = st.columns(2)

    # --- Export ---
    with exp_col:
        st.markdown("**Export current settings**")
        if st.session_state["csv_data"] is not None:
            def _build_export() -> str:
                payload: dict = {}
                # Embed CSV data
                payload["csv_data"] = st.session_state["csv_data"].to_csv(index=False)
                # Number of plots
                n = len(st.session_state.get("plot_configs", [{}]))
                payload["n_plots"] = n
                # Global filters
                gf = st.session_state.get("global_filters", [])
                payload["n_global_filters"] = len(gf)
                # Gather all widget values
                settings: dict = {}
                for fi in range(len(gf)):
                    for suffix in ("col", "range", "vals"):
                        key = f"gf_{fi}_{suffix}"
                        if key in st.session_state:
                            settings[key] = _make_serializable(st.session_state[key])
                for idx in range(n):
                    for suffix in _PLOT_WIDGET_SUFFIXES:
                        key = f"p{idx}_{suffix}"
                        if key in st.session_state:
                            settings[key] = _make_serializable(st.session_state[key])
                    prefix = f"p{idx}_scolor_"
                    for key in list(st.session_state.keys()):
                        if key.startswith(prefix):
                            settings[key] = _make_serializable(st.session_state[key])
                    prefix2 = f"p{idx}_sname_"
                    for key in list(st.session_state.keys()):
                        if key.startswith(prefix2):
                            settings[key] = _make_serializable(st.session_state[key])
                payload["settings"] = settings
                return json.dumps(payload, indent=2)

            st.download_button(
                label="\U0001f4e5 Download settings (.json)",
                data=_build_export(),
                file_name="csv_explorer_settings.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.info("Load a CSV first to export settings.")

    # --- Import ---
    with imp_col:
        st.markdown("**Import saved settings**")
        settings_file = st.file_uploader(
            "Upload settings file:",
            type=["json"],
            key="settings_upload",
        )
        if settings_file is not None:
            if st.button("\U0001f4e4 Apply imported settings", key="apply_settings", use_container_width=True):
                try:
                    payload = json.loads(settings_file.getvalue().decode("utf-8"))
                    # Restore CSV data
                    df = pd.read_csv(io.StringIO(payload["csv_data"]))
                    st.session_state["csv_data"] = df
                    # Restore plot configs
                    n = payload.get("n_plots", 1)
                    st.session_state["plot_configs"] = [{} for _ in range(n)]
                    # Restore global filters
                    n_gf = payload.get("n_global_filters", 0)
                    st.session_state["global_filters"] = [{} for _ in range(n_gf)]
                    # Restore all widget values
                    for key, val in payload.get("settings", {}).items():
                        st.session_state[key] = val
                    st.success("Settings imported successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to import settings: {e}")


data: Optional[pd.DataFrame] = st.session_state["csv_data"]

if data is None:
    st.info("Upload a CSV file to begin exploring.")
    st.stop()

st.success(f"Loaded **{len(data)}** rows × **{len(data.columns)}** columns")

with st.expander("Preview data", expanded=False):
    st.dataframe(data.head(50), use_container_width=True)

# ---------------------------
# Global data filters
# ---------------------------
with st.expander("Filter data", expanded=False):
    st.caption("Apply filters to narrow the dataset before plotting. All plots use the filtered data.")

    if "global_filters" not in st.session_state:
        st.session_state["global_filters"] = []

    _gf_add, _gf_clear, _ = st.columns([0.2, 0.2, 0.6])
    with _gf_add:
        if st.button("➕ Add filter", use_container_width=True, key="gf_add"):
            st.session_state["global_filters"].append({})
            st.rerun()
    with _gf_clear:
        if st.session_state["global_filters"]:
            if st.button("🗑 Clear all filters", use_container_width=True, key="gf_clear"):
                st.session_state["global_filters"] = []
                st.rerun()

    _filter_mask = pd.Series(True, index=data.index)

    for fi, _ in enumerate(st.session_state["global_filters"]):
        st.markdown(f"**Filter {fi + 1}**")
        fc1, fc2 = st.columns([0.3, 0.7])
        with fc1:
            filt_col = st.selectbox(
                "Column:", options=list(data.columns), index=0, key=f"gf_{fi}_col"
            )
        with fc2:
            if pd.api.types.is_numeric_dtype(data[filt_col]):
                col_min = float(data[filt_col].min())
                col_max = float(data[filt_col].max())
                if col_min == col_max:
                    st.info(f"Only one value: {col_min}")
                else:
                    filt_range = st.slider(
                        f"Range for {filt_col}:",
                        min_value=col_min,
                        max_value=col_max,
                        value=(col_min, col_max),
                        key=f"gf_{fi}_range",
                    )
                    _filter_mask &= (data[filt_col] >= filt_range[0]) & (data[filt_col] <= filt_range[1])
            else:
                unique_vals = sorted(data[filt_col].dropna().unique(), key=str)
                filt_vals = st.multiselect(
                    f"Values for {filt_col}:",
                    options=unique_vals,
                    default=unique_vals,
                    key=f"gf_{fi}_vals",
                )
                _filter_mask &= data[filt_col].isin(filt_vals)

    data = data[_filter_mask].reset_index(drop=True)
    if st.session_state["global_filters"]:
        st.caption(f"Filtered: **{len(data)}** rows remaining")

# Identify column types
all_cols = list(data.columns)
numeric_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(data[c])]
categorical_cols = [c for c in all_cols if not pd.api.types.is_numeric_dtype(data[c]) or data[c].nunique() <= 20]

if len(numeric_cols) < 2:
    st.warning("Need at least 2 numeric columns for X and Y axes.")
    st.stop()


# ---------------------------
# Plot management
# ---------------------------
st.markdown("---")

# Add / remove plot buttons
add_col, remove_col, _ = st.columns([0.2, 0.2, 0.6])
with add_col:
    if st.button("➕ Add plot", use_container_width=True):
        st.session_state["plot_configs"].append({})
        st.rerun()
with remove_col:
    if len(st.session_state["plot_configs"]) > 1:
        if st.button("➖ Remove last plot", use_container_width=True):
            st.session_state["plot_configs"].pop()
            st.rerun()

n_plots = len(st.session_state["plot_configs"])

# ---------------------------
# Render each plot
# ---------------------------
for plot_idx in range(n_plots):
    _restore_csv_plot_state(plot_idx)

    st.markdown(f"### Plot {plot_idx + 1}")

    with st.expander(f"Plot {plot_idx + 1} — Configuration", expanded=True):
        # --- Axis selection ---
        ax_c1, ax_c2 = st.columns(2)
        with ax_c1:
            x_col = st.selectbox(
                "X-axis column:",
                options=numeric_cols,
                index=0,
                key=f"p{plot_idx}_xcol",
            )
        with ax_c2:
            y_default = 1 if len(numeric_cols) > 1 else 0
            y_col = st.selectbox(
                "Y-axis column:",
                options=numeric_cols,
                index=y_default,
                key=f"p{plot_idx}_ycol",
            )

        # --- Custom axis labels ---
        lbl_c1, lbl_c2 = st.columns(2)
        with lbl_c1:
            x_label = st.text_input("X-axis label:", value="", key=f"p{plot_idx}_xlabel",
                                    placeholder=x_col, help="Leave blank to use column name.")
        with lbl_c2:
            y_label = st.text_input("Y-axis label:", value="", key=f"p{plot_idx}_ylabel",
                                    placeholder=y_col, help="Leave blank to use column name.")
        x_axis_label = x_label if x_label else x_col
        y_axis_label = y_label if y_label else y_col

        # --- Axis label formatting ---
        fmt_c1, fmt_c2, fmt_c3, fmt_c4, fmt_c5 = st.columns(5)
        with fmt_c1:
            axis_label_size = st.slider("Axis label size", 8, 28, 14, 1, key=f"p{plot_idx}_axfsize")
        with fmt_c2:
            x_bold = st.checkbox("X bold", value=False, key=f"p{plot_idx}_xbold")
        with fmt_c3:
            x_italic = st.checkbox("X italic", value=False, key=f"p{plot_idx}_xitalic")
        with fmt_c4:
            y_bold = st.checkbox("Y bold", value=False, key=f"p{plot_idx}_ybold")
        with fmt_c5:
            y_italic = st.checkbox("Y italic", value=False, key=f"p{plot_idx}_yitalic")

        def _fmt_label(text: str, bold: bool, italic: bool) -> str:
            if bold:
                text = f"<b>{text}</b>"
            if italic:
                text = f"<i>{text}</i>"
            return text

        x_axis_label_fmt = _fmt_label(x_axis_label, x_bold, x_italic)
        y_axis_label_fmt = _fmt_label(y_axis_label, y_bold, y_italic)

        # --- Categorical grouping ---
        cat_c1, cat_c2 = st.columns(2)
        with cat_c1:
            primary_cat = st.selectbox(
                "Primary category:",
                options=["(none)"] + categorical_cols,
                index=0,
                key=f"p{plot_idx}_primary",
                help="Each unique value creates a separate subplot or marker style.",
            )
        with cat_c2:
            secondary_cat = st.selectbox(
                "Secondary category (series within each subplot):",
                options=["(none)"] + categorical_cols,
                index=0,
                key=f"p{plot_idx}_secondary",
                help="Each unique value creates a separate colored series.",
            )

        # --- Primary display mode ---
        primary_display = "Separate subplots"
        if primary_cat != "(none)":
            primary_display = st.radio(
                "Primary category display:",
                options=["Separate subplots", "Same plot (different markers)"],
                index=0,
                horizontal=True,
                key=f"p{plot_idx}_pmode",
                help="Separate subplots: one subplot per value. Same plot: all values on one plot distinguished by marker shape.",
            )

        # --- Category filters ---
        primary_filter = None
        secondary_filter = None
        if primary_cat != "(none)" or secondary_cat != "(none)":
            filt_c1, filt_c2 = st.columns(2)
            if primary_cat != "(none)":
                _all_primary = sorted(data[primary_cat].dropna().unique(), key=str)
                with filt_c1:
                    primary_filter = st.multiselect(
                        f"Filter {primary_cat} values:",
                        options=_all_primary,
                        default=_all_primary,
                        key=f"p{plot_idx}_pfilt",
                    )
            if secondary_cat != "(none)":
                _all_secondary = sorted(data[secondary_cat].dropna().unique(), key=str)
                with filt_c2:
                    secondary_filter = st.multiselect(
                        f"Filter {secondary_cat} values:",
                        options=_all_secondary,
                        default=_all_secondary,
                        key=f"p{plot_idx}_sfilt",
                    )

        # --- Series colours ---
        _color_defaults = (
            px.colors.qualitative.Plotly
            + px.colors.qualitative.D3
            + px.colors.qualitative.Set3
        )
        _has_sec = secondary_cat != "(none)"
        _has_pri = primary_cat != "(none)"
        _combined = _has_pri and primary_display == "Same plot (different markers)"

        if _has_sec:
            _cseries = sorted(
                (secondary_filter if secondary_filter is not None
                 else list(data[secondary_cat].dropna().unique())),
                key=str,
            )
        elif _combined:
            _cseries = sorted(
                (primary_filter if primary_filter is not None
                 else list(data[primary_cat].dropna().unique())),
                key=str,
            )
        else:
            _cseries = [y_col]

        _cseries = [str(v) for v in _cseries]

        if _cseries:
            st.markdown("**Series colours & names**")
            _n_ccols = min(len(_cseries), 4)
            for _ci, _cname in enumerate(_cseries):
                if _ci % _n_ccols == 0:
                    _ccol_list = st.columns(_n_ccols)
                with _ccol_list[_ci % _n_ccols]:
                    st.color_picker(
                        _cname,
                        value=_color_defaults[_ci % len(_color_defaults)],
                        key=f"p{plot_idx}_scolor_{_ci}",
                    )
                    st.text_input(
                        "Legend name:",
                        value="",
                        key=f"p{plot_idx}_sname_{_ci}",
                        placeholder=_cname,
                    )

        # --- Plot style ---
        style_c1, style_c2, style_c3 = st.columns(3)
        with style_c1:
            plot_type = st.selectbox(
                "Plot type:",
                options=["Scatter", "Line", "Scatter + Line"],
                index=0,
                key=f"p{plot_idx}_type",
            )
        with style_c2:
            marker_size = st.slider("Marker size", 1, 20, 6, 1, key=f"p{plot_idx}_msize")
        with style_c3:
            line_width = st.slider("Line width", 0.5, 5.0, 1.5, 0.1, key=f"p{plot_idx}_lw")

        height_per = st.slider("Height per subplot (px)", 200, 800, 400, 25, key=f"p{plot_idx}_height")

        # --- Legend ---
        st.markdown("**Legend**")
        _lg1, _lg2, _lg3 = st.columns(3)
        with _lg1:
            show_legend_opt = st.checkbox("Show legend", value=True, key=f"p{plot_idx}_lgshow")
        with _lg2:
            legend_position = st.selectbox(
                "Legend position:",
                options=["Right", "Left", "Top", "Bottom", "Top-right", "Top-left", "Bottom-right", "Bottom-left"],
                index=0,
                key=f"p{plot_idx}_lgpos",
                disabled=not show_legend_opt,
            )
        with _lg3:
            legend_font_size = st.slider("Legend font size", 6, 24, 12, 1, key=f"p{plot_idx}_lgfont",
                                         disabled=not show_legend_opt)

        # --- Vertical gridlines ---
        st.markdown("**Vertical gridlines**")
        _vg1, _vg2 = st.columns(2)
        with _vg1:
            show_vgrid = st.checkbox("Show vertical gridlines", value=False, key=f"p{plot_idx}_vgrid")
        with _vg2:
            vgrid_dtick = st.number_input(
                "Gridline spacing (0 = auto):",
                min_value=0.0,
                value=0.0,
                step=0.1,
                format="%g",
                key=f"p{plot_idx}_vgdtick",
                disabled=not show_vgrid,
            )

        # --- Trendline options ---
        st.markdown("**Trendline**")
        trend_c1, trend_c2, trend_c3 = st.columns(3)
        with trend_c1:
            trend_type = st.selectbox(
                "Trendline type:",
                options=_TRENDLINE_TYPES,
                index=0,
                key=f"p{plot_idx}_trend",
            )
        with trend_c2:
            trend_width = st.slider("Trendline width", 0.5, 5.0, 2.0, 0.1, key=f"p{plot_idx}_tw")
        with trend_c3:
            if trend_type == "Moving Average":
                ma_window = st.number_input("Window size", 2, 200, 5, 1, key=f"p{plot_idx}_ma")
            else:
                ma_window = 5
            show_equation = st.checkbox("Show equation", value=True, key=f"p{plot_idx}_eq")

        trend_dash = st.selectbox(
            "Trendline style:",
            options=["dash", "dot", "dashdot", "solid"],
            index=0,
            key=f"p{plot_idx}_tdash",
        )

        # --- Reference lines ---
        use_primary = primary_cat != "(none)"
        use_secondary = secondary_cat != "(none)"
        combine_primary = use_primary and primary_display == "Same plot (different markers)"

        _ref_line_candidates = []
        if use_primary and not combine_primary:
            for _pv in (primary_filter if primary_filter else sorted(data[primary_cat].dropna().unique(), key=str)):
                _pv_df = data[data[primary_cat] == _pv]
                if not _pv_df[y_col].dropna().empty:
                    _ref_line_candidates.append(str(_pv))
        elif use_secondary:
            for _sv in (secondary_filter if secondary_filter else sorted(data[secondary_cat].dropna().unique(), key=str)):
                _sv_df = data[data[secondary_cat] == _sv]
                if not _sv_df[y_col].dropna().empty:
                    _ref_line_candidates.append(str(_sv))

        ref_series_selected: List[str] = []
        ref_line_style = "dash"
        ref_line_width = 1.5
        ref_show_label = True
        ref_agg = "Mean"
        custom_ref_lines: List[Tuple[float, str]] = []

        st.markdown("**Reference lines**")
        if _ref_line_candidates:
            ref_series_selected = st.multiselect(
                "Draw horizontal reference lines from categories:",
                options=_ref_line_candidates,
                default=[],
                key=f"p{plot_idx}_refsel",
                help="Select category values to draw as horizontal reference lines.",
            )
            if ref_series_selected:
                _rc1, _rc2, _rc3, _rc4 = st.columns(4)
                with _rc1:
                    ref_agg = st.selectbox(
                        "Aggregation:",
                        ["Mean", "Median", "Min", "Max", "First"],
                        index=0,
                        key=f"p{plot_idx}_refagg",
                        help="How to compute the reference value for multi-point series.",
                    )
                with _rc2:
                    ref_line_style = st.selectbox(
                        "Ref line style:",
                        ["dash", "dot", "dashdot", "solid"],
                        index=0,
                        key=f"p{plot_idx}_refstyle",
                    )
                with _rc3:
                    ref_line_width = st.slider("Ref line width", 0.5, 4.0, 1.5, 0.1, key=f"p{plot_idx}_refw")
                with _rc4:
                    ref_show_label = st.checkbox("Show ref label", value=True, key=f"p{plot_idx}_reflbl")
        else:
            st.caption("Select a primary or secondary category to enable category-based reference lines.")

        # Manual custom reference lines
        _custom_ref_str = st.text_input(
            "Custom reference lines (comma-separated y-values, e.g. '10.5, 25, 100'):",
            value="",
            key=f"p{plot_idx}_customref",
        )
        if _custom_ref_str.strip():
            for _part in _custom_ref_str.split(","):
                _part = _part.strip()
                if not _part:
                    continue
                try:
                    custom_ref_lines.append((float(_part), _part))
                except ValueError:
                    pass
        if custom_ref_lines and not ref_series_selected:
            _rc_c1, _rc_c2, _rc_c3 = st.columns(3)
            with _rc_c1:
                ref_line_style = st.selectbox(
                    "Ref line style:",
                    ["dash", "dot", "dashdot", "solid"],
                    index=0,
                    key=f"p{plot_idx}_refstyle",
                )
            with _rc_c2:
                ref_line_width = st.slider("Ref line width", 0.5, 4.0, 1.5, 0.1, key=f"p{plot_idx}_refw")
            with _rc_c3:
                ref_show_label = st.checkbox("Show ref label", value=True, key=f"p{plot_idx}_reflbl")

    # --- Build the plot ---
    # Marker symbols for combined mode
    _MARKER_SYMBOLS = [
        "circle", "square", "diamond", "cross", "x",
        "triangle-up", "triangle-down", "triangle-left", "triangle-right",
        "pentagon", "hexagon", "star", "hexagram", "star-triangle-up",
        "hourglass", "bowtie",
    ]

    # Determine subplot groups
    if use_primary:
        all_primary_vals = sorted(data[primary_cat].dropna().unique(), key=str)
        primary_vals = [v for v in all_primary_vals if primary_filter is None or v in primary_filter]
        if not primary_vals:
            st.warning("No primary category values selected — nothing to plot.")
            continue
    else:
        primary_vals = [None]

    if combine_primary:
        # All on one subplot
        n_subplots = 1
        subplot_titles = None
    else:
        n_subplots = len(primary_vals)
        subplot_titles = tuple(str(v) if v is not None else "" for v in primary_vals) if use_primary else None

    fig = make_subplots(
        rows=n_subplots, cols=1,
        shared_xaxes=True,
        vertical_spacing=min(0.08, 0.9 / max(n_subplots - 1, 1)),
        subplot_titles=subplot_titles,
    )

    # Color palette
    base_colors = (
        px.colors.qualitative.Plotly
        + px.colors.qualitative.D3
        + px.colors.qualitative.Set3
    )

    # Plot mode
    if plot_type == "Scatter":
        mode = "markers"
    elif plot_type == "Line":
        mode = "lines"
    else:
        mode = "lines+markers"

    # Build series-to-colour mapping from colour pickers
    _series_color_map: Dict[str, str] = {}
    _series_name_map: Dict[str, str] = {}
    if use_secondary:
        _all_color_svals = sorted(
            (v for v in data[secondary_cat].dropna().unique()
             if secondary_filter is None or v in secondary_filter),
            key=str,
        )
        for _ci, _sv in enumerate(_all_color_svals):
            _series_color_map[str(_sv)] = st.session_state.get(
                f"p{plot_idx}_scolor_{_ci}",
                base_colors[_ci % len(base_colors)],
            )
            _custom_name = st.session_state.get(f"p{plot_idx}_sname_{_ci}", "")
            if _custom_name:
                _series_name_map[str(_sv)] = _custom_name
    elif combine_primary:
        for _ci, _pv in enumerate(primary_vals):
            _series_color_map[str(_pv)] = st.session_state.get(
                f"p{plot_idx}_scolor_{_ci}",
                base_colors[_ci % len(base_colors)],
            )
            _custom_name = st.session_state.get(f"p{plot_idx}_sname_{_ci}", "")
            if _custom_name:
                _series_name_map[str(_pv)] = _custom_name
    else:
        _custom_name = st.session_state.get(f"p{plot_idx}_sname_0", "")
        if _custom_name:
            _series_name_map[y_col] = _custom_name

    # Track which legend entries have been shown
    legend_shown = set()

    equations_text: List[str] = []

    for p_idx, pval in enumerate(primary_vals):
        row_idx = 1 if combine_primary else (p_idx + 1)

        # Filter for this primary group
        if use_primary:
            sub_df = data[data[primary_cat] == pval].copy()
        else:
            sub_df = data.copy()

        marker_symbol = _MARKER_SYMBOLS[p_idx % len(_MARKER_SYMBOLS)] if combine_primary else "circle"

        # Determine series groups
        if use_secondary:
            all_sec_vals = sorted(sub_df[secondary_cat].dropna().unique(), key=str)
            secondary_vals = [v for v in all_sec_vals if secondary_filter is None or v in secondary_filter]
        else:
            secondary_vals = [None]

        for s_idx, sval in enumerate(secondary_vals):
            if use_secondary:
                color = _series_color_map.get(str(sval), base_colors[s_idx % len(base_colors)])
            elif combine_primary:
                color = _series_color_map.get(str(pval), base_colors[p_idx % len(base_colors)])
            else:
                color = st.session_state.get(f"p{plot_idx}_scolor_0", base_colors[0])

            if use_secondary:
                series_df = sub_df[sub_df[secondary_cat] == sval]
                base_name = str(sval)
            else:
                series_df = sub_df
                base_name = y_col

            # Build series name depending on mode
            if combine_primary:
                series_name = f"{pval} / {base_name}" if use_secondary else str(pval)
                legend_group = series_name
            else:
                series_name = base_name
                legend_group = base_name

            # Apply custom legend name if set
            if use_secondary:
                display_name = _series_name_map.get(str(sval), series_name)
            elif combine_primary:
                display_name = _series_name_map.get(str(pval), series_name)
            else:
                display_name = _series_name_map.get(base_name, series_name)

            xv = series_df[x_col].values
            yv = series_df[y_col].values

            # Only show legend entry once per unique legend group
            show_legend = legend_group not in legend_shown
            legend_shown.add(legend_group)

            fig.add_trace(
                go.Scatter(
                    x=xv,
                    y=yv,
                    mode=mode,
                    name=display_name,
                    legendgroup=legend_group,
                    showlegend=show_legend,
                    marker=dict(size=marker_size, color=color, symbol=marker_symbol),
                    line=dict(width=line_width, color=color),
                    hovertemplate=(
                        f"{display_name}<br>"
                        f"{x_col}=%{{x}}<br>"
                        f"{y_col}=%{{y}}<extra></extra>"
                    ),
                ),
                row=row_idx, col=1,
            )

            # --- Trendline ---
            if trend_type != "None":
                xn = pd.to_numeric(pd.Series(xv), errors="coerce").values
                yn = pd.to_numeric(pd.Series(yv), errors="coerce").values
                result = _fit_trendline(xn, yn, trend_type, ma_window)
                if result is not None:
                    xf, yf, eq_label = result

                    trend_label = f"{display_name} — {eq_label}"

                    fig.add_trace(
                        go.Scatter(
                            x=xf,
                            y=yf,
                            mode="lines",
                            name=trend_label if show_equation else f"{display_name} trend",
                            legendgroup=f"{legend_group}_trend",
                            showlegend=show_legend,
                            line=dict(
                                width=trend_width,
                                color=color,
                                dash=trend_dash,
                            ),
                            hovertemplate=f"{eq_label}<extra></extra>",
                        ),
                        row=row_idx, col=1,
                    )

                    if show_equation:
                        ctx = f"{str(pval) + ' / ' if use_primary and pval is not None else ''}{base_name}"
                        equations_text.append(f"**{ctx}**: `{eq_label}`")

        if not combine_primary:
            fig.update_yaxes(title_text=y_axis_label_fmt, title_font_size=axis_label_size, row=row_idx, col=1)

    if combine_primary:
        fig.update_yaxes(title_text=y_axis_label_fmt, title_font_size=axis_label_size, row=1, col=1)

    # --- Draw reference lines ---
    _ref_colors_palette = [
        "#AA0000", "#0072B2", "#E69F00", "#009E73", "#CC79A7",
        "#56B4E9", "#D55E00", "#000000",
    ]
    _all_ref_lines: List[Tuple[float, str]] = []

    if ref_series_selected:
        _agg_funcs = {"Mean": "mean", "Median": "median", "Min": "min", "Max": "max", "First": "first"}
        _agg_fn = _agg_funcs.get(ref_agg, "mean")
        for _ref_name in ref_series_selected:
            if use_primary and not combine_primary:
                _ref_df = data[data[primary_cat].astype(str) == _ref_name]
            elif use_secondary:
                _ref_df = data[data[secondary_cat].astype(str) == _ref_name]
            else:
                continue
            _ref_y = _ref_df[y_col].dropna()
            if _ref_y.empty:
                continue
            if _agg_fn == "first":
                _ref_val = float(_ref_y.iloc[0])
            else:
                _ref_val = float(getattr(_ref_y, _agg_fn)())
            _label = f"{_ref_name}: {_ref_val:.4g}" if ref_agg != "First" or len(_ref_y) == 1 else f"{_ref_name}: {_ref_val:.4g}"
            _all_ref_lines.append((_ref_val, _ref_name))

    _all_ref_lines.extend(custom_ref_lines)

    if _all_ref_lines:
        for _ri, (_ref_val, _ref_name) in enumerate(_all_ref_lines):
            _ref_color = _ref_colors_palette[_ri % len(_ref_colors_palette)]
            for _si in range(1, n_subplots + 1):
                fig.add_hline(
                    y=_ref_val,
                    line_color=_ref_color,
                    line_width=ref_line_width,
                    line_dash=ref_line_style,
                    opacity=0.8,
                    row=_si, col=1,
                )
                if ref_show_label:
                    _xd_ref = f"x{_si} domain" if _si > 1 else "x domain"
                    _yr_ref = f"y{_si}" if _si > 1 else "y"
                    fig.add_annotation(
                        x=0.02, xref=_xd_ref,
                        y=_ref_val, yref=_yr_ref,
                        text=f"{_ref_name}: {_ref_val:.4g}",
                        showarrow=False,
                        xanchor="left",
                        yanchor="bottom",
                        font=dict(size=10, color=_ref_color),
                        yshift=4,
                        bgcolor="rgba(255,255,255,0.7)",
                        bordercolor="rgba(0,0,0,0.1)",
                        borderwidth=1,
                    )

    fig.update_xaxes(title_text=x_axis_label_fmt, title_font_size=axis_label_size,
                      title_standoff=15, row=n_subplots, col=1)

    # Apply vertical gridline settings
    _vgrid_kwargs = dict(showgrid=show_vgrid)
    if show_vgrid and vgrid_dtick > 0:
        _vgrid_kwargs["dtick"] = vgrid_dtick
    for _r in range(1, n_subplots + 1):
        fig.update_xaxes(**_vgrid_kwargs, row=_r, col=1)

    total_height = max(300, n_subplots * height_per)

    _legend_pos_map = {
        "Right":        dict(x=1.02, y=1.0, xanchor="left",  yanchor="top"),
        "Left":         dict(x=-0.15, y=1.0, xanchor="left",  yanchor="top"),
        "Top":          dict(x=0.5,  y=1.05, xanchor="center", yanchor="bottom", orientation="h"),
        "Bottom":       dict(x=0.5,  y=-0.15, xanchor="center", yanchor="top", orientation="h"),
        "Top-right":    dict(x=1.0,  y=1.0, xanchor="right", yanchor="top"),
        "Top-left":     dict(x=0.0,  y=1.0, xanchor="left",  yanchor="top"),
        "Bottom-right": dict(x=1.0,  y=0.0, xanchor="right", yanchor="bottom"),
        "Bottom-left":  dict(x=0.0,  y=0.0, xanchor="left",  yanchor="bottom"),
    }
    _legend_kwargs = _legend_pos_map.get(legend_position, _legend_pos_map["Right"])
    _legend_kwargs.update(
        bgcolor="rgba(255,255,255,0.7)",
        bordercolor="rgba(0,0,0,0.1)",
        borderwidth=1,
        font=dict(size=legend_font_size),
    )

    # Adjust margins to prevent legend clipping in static PNG export
    _margin = dict(l=60, r=20, t=40, b=80)
    if show_legend_opt:
        if legend_position in ("Right",):
            _margin["r"] = 250
        elif legend_position in ("Left",):
            _margin["l"] = 250
        elif legend_position in ("Bottom",):
            _margin["b"] = 100
        elif legend_position in ("Top",):
            _margin["t"] = 100

    fig.update_layout(
        height=total_height,
        showlegend=show_legend_opt,
        legend=_legend_kwargs,
        margin=_margin,
    )

    st.plotly_chart(fig, use_container_width=True, theme="streamlit", key=f"plot_{plot_idx}")

    # --- Download buttons ---
    _dl1, _dl2, _ = st.columns([0.2, 0.2, 0.6])
    with _dl1:
        _html_fig = go.Figure(fig)
        _html_fig.update_layout(width=1000)
        _html_bytes = _html_fig.to_html(include_plotlyjs="cdn").encode("utf-8")
        st.download_button(
            "📥 Download HTML",
            data=_html_bytes,
            file_name=f"plot_{plot_idx + 1}.html",
            mime="text/html",
            key=f"dl_html_{plot_idx}",
            use_container_width=True,
        )
    with _dl2:
        try:
            _png_width = 1000 if legend_position in ("Right", "Left") else 900
            _png_bytes = fig.to_image(format="png", width=_png_width, height=total_height, scale=2)
            st.download_button(
                "📥 Download PNG",
                data=_png_bytes,
                file_name=f"plot_{plot_idx + 1}.png",
                mime="image/png",
                key=f"dl_png_{plot_idx}",
                use_container_width=True,
            )
        except Exception as _e:
            st.caption(f"PNG export unavailable: {_e}")

    # Show equations below the plot
    if equations_text:
        with st.expander("Trendline equations", expanded=False):
            for eq in equations_text:
                st.markdown(eq)

    _save_csv_plot_state(plot_idx)

    st.markdown("---")

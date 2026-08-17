import io
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------------------------
# Constants & helpers
# ---------------------------
_FONT_FAMILIES = [
    "Arial", "Helvetica", "Times New Roman", "Courier New",
    "Verdana", "Georgia", "Trebuchet MS", "Palatino Linotype",
    "Garamond", "Comic Sans MS", "Open Sans", "Roboto",
]

_AGG_MAP = {
    "Mean": "mean", "Median": "median", "Min": "min",
    "Max": "max", "Sum": "sum", "First": "first",
}


def _fmt(text: str, bold: bool, italic: bool) -> str:
    if bold:
        text = f"<b>{text}</b>"
    if italic:
        text = f"<i>{text}</i>"
    return text


@st.cache_data(show_spinner=False)
def _get_column_types(columns: tuple, dtypes: tuple, nuniques: tuple):
    numeric = [c for c, dt, _ in zip(columns, dtypes, nuniques) if pd.api.types.is_numeric_dtype(pd.Series(dtype=dt))]
    categorical = [c for c, dt, nu in zip(columns, dtypes, nuniques)
                   if not pd.api.types.is_numeric_dtype(pd.Series(dtype=dt)) or nu <= 20]
    return numeric, categorical


def _compute_agg_and_pct(df, cat_col, selected_conditions, selected_vars, agg_fn):
    cat_str = df[cat_col].astype(str)
    mask = cat_str.isin(selected_conditions)
    agg_data = df.loc[mask].groupby(cat_str[mask])[list(selected_vars)].agg(agg_fn)
    return agg_data


# Widget keys that need to persist across page switches
_WIDGET_SUFFIXES = [
    "cat", "agg", "conds", "base", "normmode", "refval", "vars", "type",
    "yvar", "xgrp", "xgrp_vals",
    "height", "showvals", "lgfont", "fill", "opacity",
    "fontfam", "titlefs", "axlabelfs", "tickfs",
    "titlebold", "titleital", "axbold", "axital", "title",
]


def _save_plot_state(plot_idx: int):
    """Save widget values to shadow keys that survive page switches."""
    store = {}
    for suffix in _WIDGET_SUFFIXES:
        wk = f"cp{plot_idx}_{suffix}"
        if wk in st.session_state:
            store[wk] = st.session_state[wk]
    # Also save colour picker keys
    for key in list(st.session_state.keys()):
        if key.startswith(f"cp{plot_idx}_color_"):
            store[key] = st.session_state[key]
    st.session_state[f"_persist_cp{plot_idx}"] = store


def _restore_plot_state(plot_idx: int):
    """Restore widget values from shadow keys if they were garbage-collected."""
    store = st.session_state.get(f"_persist_cp{plot_idx}", {})
    for wk, val in store.items():
        if wk not in st.session_state:
            st.session_state[wk] = val


st.title("Comparative Plots")

# ---------------------------
# Session state initialization
# ---------------------------
if "comp_csv_data" not in st.session_state:
    st.session_state["comp_csv_data"] = None
if "comp_plot_configs" not in st.session_state:
    st.session_state["comp_plot_configs"] = [{}]

# ---------------------------
# File import
# ---------------------------
uploaded = st.file_uploader(
    "Upload a CSV file:",
    type=["csv", "tsv", "txt"],
    accept_multiple_files=False,
    key="comp_csv_upload",
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
        st.session_state["comp_csv_data"] = data
    except Exception as e:
        st.error(f"Failed to parse file: {e}")

data: Optional[pd.DataFrame] = st.session_state["comp_csv_data"]

if data is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

st.success(f"Loaded **{len(data)}** rows × **{len(data.columns)}** columns")

# ---------------------------
# Row filtering
# ---------------------------
with st.expander("Filter rows", expanded=False):
    filter_col = st.selectbox(
        "Column to filter by:",
        options=["(no filter)"] + list(data.columns),
        index=0,
        key="comp_filter_col",
    )
    if filter_col != "(no filter)":
        unique_vals = sorted(data[filter_col].dropna().unique(), key=str)
        unique_vals_str = [str(v) for v in unique_vals]
        selected_filter_vals = st.multiselect(
            f"Keep rows where **{filter_col}** is:",
            options=unique_vals_str,
            default=unique_vals_str,
            key="comp_filter_vals",
        )
        if selected_filter_vals:
            data = data[data[filter_col].astype(str).isin(selected_filter_vals)]
        else:
            st.warning("No filter values selected — showing all rows.")
    st.caption(f"**{len(data)}** rows after filtering")

# ---------------------------
# Exclude specific rows
# ---------------------------
with st.expander("Exclude rows", expanded=False):
    st.caption("Uncheck rows in the **Include** column to exclude them from analysis.")
    _display = data.copy()
    _display.insert(0, "Include", True)
    _display.index = range(len(_display))
    edited = st.data_editor(
        _display,
        use_container_width=True,
        hide_index=True,
        key="comp_row_exclude",
        column_config={"Include": st.column_config.CheckboxColumn("Include", default=True)},
        disabled=[c for c in _display.columns if c != "Include"],
    )
    _keep_mask = edited["Include"].astype(bool)
    if not _keep_mask.all():
        n_excluded = (~_keep_mask).sum()
        data = data.iloc[_keep_mask.values].reset_index(drop=True)
        st.caption(f"**{n_excluded}** row(s) excluded — **{len(data)}** rows remaining")
    else:
        st.caption(f"**{len(data)}** rows (none excluded)")

with st.expander("Preview data", expanded=False):
    st.dataframe(data.head(50), use_container_width=True)

# Identify column types (cached)
_col_tuple = tuple(data.columns)
_dtype_tuple = tuple(str(data[c].dtype) for c in data.columns)
_nunique_tuple = tuple(data[c].nunique() for c in data.columns)
numeric_cols, categorical_cols = _get_column_types(_col_tuple, _dtype_tuple, _nunique_tuple)

if not numeric_cols:
    st.warning("Need at least one numeric column to compare.")
    st.stop()
if not categorical_cols:
    st.warning("Need at least one categorical column for grouping.")
    st.stop()

# Color palette
_palette = (
    px.colors.qualitative.Plotly
    + px.colors.qualitative.D3
    + px.colors.qualitative.Set3
)


# ---------------------------
# Plot management
# ---------------------------
st.markdown("---")

add_col, remove_col, _ = st.columns([0.2, 0.2, 0.6])
with add_col:
    if st.button("➕ Add plot", use_container_width=True, key="comp_add"):
        st.session_state["comp_plot_configs"].append({})
        st.rerun()
with remove_col:
    if len(st.session_state["comp_plot_configs"]) > 1:
        if st.button("➖ Remove last plot", use_container_width=True, key="comp_remove"):
            st.session_state["comp_plot_configs"].pop()
            st.rerun()

n_plots = len(st.session_state["comp_plot_configs"])

# ---------------------------
# Generate button
# ---------------------------
if st.button("🔄 Generate / Update Plots", use_container_width=True, key="comp_generate_btn", type="primary"):
    st.session_state["comp_plots_ready"] = True

if not st.session_state.get("comp_plots_ready", False):
    st.info("Configure your plots below, then click **Generate / Update Plots**.")

# ---------------------------
# Collect plot configurations
# ---------------------------
_plot_cfgs = []  # store resolved config per plot for deferred rendering

for plot_idx in range(n_plots):
    # Restore any widget state that was lost during page switch
    _restore_plot_state(plot_idx)

    st.markdown(f"### Plot {plot_idx + 1}")

    with st.expander(f"Plot {plot_idx + 1} — Configuration", expanded=True):
        # --- Data selection ---
        cfg_c1, cfg_c2 = st.columns(2)
        with cfg_c1:
            _cat_default = categorical_cols.index("Condition") if "Condition" in categorical_cols else 0
            cat_col = st.selectbox(
                "Grouping column (conditions):",
                options=categorical_cols,
                index=_cat_default,
                key=f"cp{plot_idx}_cat",
            )
        with cfg_c2:
            agg_method = st.selectbox(
                "Aggregation:",
                options=["Mean", "Median", "Min", "Max", "Sum", "First"],
                index=0,
                key=f"cp{plot_idx}_agg",
            )

        condition_values = sorted(data[cat_col].dropna().unique(), key=str)
        condition_values_str = [str(v) for v in condition_values]

        if len(condition_values) < 2:
            st.warning("Need at least 2 unique values in the grouping column.")
            continue

        selected_conditions = st.multiselect(
            "Conditions to compare:",
            options=condition_values_str,
            default=condition_values_str,
            key=f"cp{plot_idx}_conds",
        )

        if len(selected_conditions) < 2:
            st.warning("Select at least 2 conditions.")
            continue

        norm_mode = st.radio(
            "Normalization:",
            options=["Base condition", "Reference value"],
            index=0,
            horizontal=True,
            key=f"cp{plot_idx}_normmode",
        )

        if norm_mode == "Base condition":
            base_condition = st.selectbox(
                "Base condition (reference = 100%):",
                options=selected_conditions,
                index=0,
                key=f"cp{plot_idx}_base",
            )
            ref_value = None
        else:
            base_condition = None
            ref_value = st.number_input(
                "Reference value (= 100%):",
                value=1.0,
                format="%g",
                key=f"cp{plot_idx}_refval",
            )
            if ref_value == 0:
                st.warning("Reference value cannot be zero.")
                continue

        plot_type = st.selectbox(
            "Plot type:",
            options=["Spider / Radar", "Grouped Bar", "Both", "Categorical Bar"],
            index=0,
            key=f"cp{plot_idx}_type",
        )

        if plot_type == "Categorical Bar":
            y_var = st.selectbox(
                "Numeric variable (y-axis):",
                options=numeric_cols,
                index=0,
                key=f"cp{plot_idx}_yvar",
            )
            _avail_xgrp = [c for c in categorical_cols if c != cat_col]
            if not _avail_xgrp:
                st.warning("Need a second categorical column for x-axis grouping.")
                continue
            x_group_col = st.selectbox(
                "X-axis grouping column:",
                options=_avail_xgrp,
                index=0,
                key=f"cp{plot_idx}_xgrp",
            )
            _xg_vals = sorted(data[x_group_col].dropna().unique(), key=str)
            _xg_vals_str = [str(v) for v in _xg_vals]
            selected_x_groups = st.multiselect(
                "X-axis groups to show:",
                options=_xg_vals_str,
                default=_xg_vals_str,
                key=f"cp{plot_idx}_xgrp_vals",
            )
            if not selected_x_groups:
                st.warning("Select at least one x-axis group.")
                continue
            selected_vars = [y_var]
        else:
            selected_vars = st.multiselect(
                "Numeric variables to compare:",
                options=numeric_cols,
                default=numeric_cols[:min(3, len(numeric_cols))],
                key=f"cp{plot_idx}_vars",
            )
            if not selected_vars:
                st.warning("Select at least one numeric variable.")
                continue

        # --- Plot styling ---
        st.markdown("**Styling**")
        _s1, _s2, _s3 = st.columns(3)
        with _s1:
            plot_height = st.slider("Plot height (px)", 300, 1000, 500, 25, key=f"cp{plot_idx}_height")
        with _s2:
            show_values = st.checkbox("Show values on plot", value=True, key=f"cp{plot_idx}_showvals")
        with _s3:
            legend_font = st.slider("Legend font size", 8, 24, 12, 1, key=f"cp{plot_idx}_lgfont")

        _r1, _r2 = st.columns(2)
        with _r1:
            fill_radar = st.checkbox("Fill radar areas", value=True, key=f"cp{plot_idx}_fill")
        with _r2:
            radar_opacity = st.slider("Radar fill opacity", 0.0, 1.0, 0.3, 0.05,
                                      key=f"cp{plot_idx}_opacity", disabled=not fill_radar)

        # --- Font customization ---
        st.markdown("**Fonts & formatting**")
        _f1, _f2, _f3, _f4 = st.columns(4)
        with _f1:
            font_family = st.selectbox("Font family:", options=_FONT_FAMILIES, index=0,
                                       key=f"cp{plot_idx}_fontfam")
        with _f2:
            title_font_size = st.slider("Title font size", 10, 32, 16, 1, key=f"cp{plot_idx}_titlefs")
        with _f3:
            axis_label_font_size = st.slider("Axis label font size", 8, 28, 14, 1, key=f"cp{plot_idx}_axlabelfs")
        with _f4:
            tick_font_size = st.slider("Tick font size", 8, 24, 12, 1, key=f"cp{plot_idx}_tickfs")

        _ff1, _ff2, _ff3, _ff4 = st.columns(4)
        with _ff1:
            title_bold = st.checkbox("Title bold", value=True, key=f"cp{plot_idx}_titlebold")
        with _ff2:
            title_italic = st.checkbox("Title italic", value=False, key=f"cp{plot_idx}_titleital")
        with _ff3:
            axis_bold = st.checkbox("Axis labels bold", value=False, key=f"cp{plot_idx}_axbold")
        with _ff4:
            axis_italic = st.checkbox("Axis labels italic", value=False, key=f"cp{plot_idx}_axital")

        plot_title = st.text_input("Plot title:", value="", key=f"cp{plot_idx}_title",
                                   placeholder="Leave blank for auto title")

        # --- Condition colours ---
        st.markdown("**Condition colours**")
        _n_ccols = min(len(selected_conditions), 4)
        for _ci, _cname in enumerate(selected_conditions):
            if _ci % _n_ccols == 0:
                _cc_cols = st.columns(_n_ccols)
            with _cc_cols[_ci % _n_ccols]:
                st.color_picker(
                    _cname,
                    value=_palette[_ci % len(_palette)],
                    key=f"cp{plot_idx}_color_{_ci}",
                )

    # Save widget state before rendering gate
    _save_plot_state(plot_idx)

    if not st.session_state.get("comp_plots_ready", False):
        continue

    # --- Categorical Bar: compute & render, then skip to next plot ---
    if plot_type == "Categorical Bar":
        agg_fn = _AGG_MAP[agg_method]
        cat_str = data[cat_col].astype(str)
        x_str = data[x_group_col].astype(str)
        mask = cat_str.isin(selected_conditions) & x_str.isin(selected_x_groups)
        grouped = data.loc[mask].groupby([cat_str[mask], x_str[mask]])[y_var].agg(agg_fn)
        pivot = grouped.unstack(level=1)
        pivot = pivot.reindex(index=selected_conditions, columns=selected_x_groups)

        if ref_value is not None:
            base_vals = pd.Series(ref_value, index=selected_x_groups)
            _norm_label = f"{ref_value:g}"
        else:
            if base_condition not in pivot.index or pivot.loc[base_condition].isna().all():
                st.error(f"Base condition '{base_condition}' has no data after aggregation.")
                _save_plot_state(plot_idx)
                st.markdown("---")
                continue
            base_vals = pivot.loc[base_condition]
            _norm_label = base_condition
            zero_groups = [g for g in selected_x_groups if pd.isna(base_vals.get(g)) or base_vals.get(g) == 0]
            if zero_groups:
                st.warning(f"Base has zero/missing values for groups: {', '.join(zero_groups)} — excluded.")
                selected_x_groups = [g for g in selected_x_groups if g not in zero_groups]
                if not selected_x_groups:
                    _save_plot_state(plot_idx)
                    st.markdown("---")
                    continue
                pivot = pivot[selected_x_groups]
                base_vals = pivot.loc[base_condition]

        pct_pivot = (pivot / base_vals) * 100

        with st.expander(f"Plot {plot_idx + 1} — Aggregated data", expanded=False):
            st.markdown("**Aggregated values**")
            st.dataframe(pivot.reset_index(), use_container_width=True)
            st.markdown("**Relative to base (%)**")
            st.dataframe(pct_pivot.round(2).reset_index(), use_container_width=True)

        _condition_colors = {
            cond: st.session_state.get(f"cp{plot_idx}_color_{ci}", _palette[ci % len(_palette)])
            for ci, cond in enumerate(selected_conditions)
        }
        _base_font = dict(family=font_family, size=tick_font_size)

        if plot_title.strip():
            _title_text = plot_title.strip()
        else:
            _title_text = f"{y_var} — % relative to {_norm_label}"
        _title_formatted = _fmt(_title_text, title_bold, title_italic)

        fig_cat = go.Figure()
        for ci, cond in enumerate(selected_conditions):
            if cond not in pct_pivot.index:
                continue
            values = [pct_pivot.loc[cond].get(g, float('nan')) for g in selected_x_groups]
            color = _condition_colors.get(cond, _palette[ci % len(_palette)])
            text_vals = [f"{v:.1f}%" if not pd.isna(v) else "" for v in values] if show_values else None
            fig_cat.add_trace(go.Bar(
                x=selected_x_groups,
                y=values,
                name=cond,
                marker_color=color,
                text=text_vals,
                textposition="outside" if show_values else "none",
                textfont=dict(family=font_family, size=tick_font_size),
                hovertemplate=f"{cond}<br>%{{x}}: %{{y:.1f}}%<extra></extra>",
            ))

        _x_title = _fmt(x_group_col, axis_bold, axis_italic)
        _y_title = _fmt(f"% of {y_var} relative to base", axis_bold, axis_italic)

        fig_cat.update_layout(
            title=dict(text=_title_formatted, font=dict(family=font_family, size=title_font_size)),
            barmode="group",
            xaxis_title=_x_title,
            yaxis_title=_y_title,
            yaxis=dict(
                ticksuffix="%",
                title_font=dict(family=font_family, size=axis_label_font_size),
                tickfont=dict(family=font_family, size=tick_font_size),
            ),
            xaxis=dict(
                tickfont=dict(family=font_family, size=tick_font_size),
                title_font=dict(family=font_family, size=axis_label_font_size),
            ),
            font=_base_font,
            height=plot_height,
            legend=dict(font=dict(family=font_family, size=legend_font)),
            margin=dict(l=60, r=20, t=60, b=80),
        )

        fig_cat.add_hline(
            y=100, line_dash="dash", line_color="grey", line_width=1.5,
            annotation_text=f"100% ({_norm_label})", annotation_position="top right",
            annotation_font=dict(family=font_family, size=tick_font_size),
        )

        st.plotly_chart(fig_cat, use_container_width=True, key=f"comp_catbar_{plot_idx}")

        _dl1, _dl2, _ = st.columns([0.2, 0.2, 0.6])
        with _dl1:
            _html = fig_cat.to_html(include_plotlyjs="cdn").encode("utf-8")
            st.download_button(
                "📥 Download HTML", data=_html,
                file_name=f"cat_bar_plot_{plot_idx + 1}.html", mime="text/html",
                key=f"dl_catbar_html_{plot_idx}", use_container_width=True,
            )
        with _dl2:
            try:
                _png = fig_cat.to_image(format="png", width=900, height=plot_height, scale=2)
                st.download_button(
                    "📥 Download PNG", data=_png,
                    file_name=f"cat_bar_plot_{plot_idx + 1}.png", mime="image/png",
                    key=f"dl_catbar_png_{plot_idx}", use_container_width=True,
                )
            except Exception as _e:
                st.caption(f"PNG export unavailable: {_e}")

        st.markdown("---")
        continue

    # --- Compute data (cached) ---
    agg_fn = _AGG_MAP[agg_method]
    agg_data = _compute_agg_and_pct(
        data, cat_col, tuple(selected_conditions), tuple(selected_vars), agg_fn
    )

    if ref_value is not None:
        base_row = pd.Series(ref_value, index=selected_vars)
        _norm_label = f"{ref_value:g}"
    else:
        if base_condition not in agg_data.index:
            st.error(f"Base condition '{base_condition}' has no data after aggregation.")
            continue
        base_row = agg_data.loc[base_condition]
        _norm_label = base_condition

        zero_vars = [v for v in selected_vars if base_row[v] == 0]
        if zero_vars:
            st.warning(f"Base has zero values for: {', '.join(zero_vars)} — excluded.")
            selected_vars = [v for v in selected_vars if v not in zero_vars]
            if not selected_vars:
                continue
            base_row = agg_data.loc[base_condition, selected_vars]
            agg_data = agg_data[selected_vars]

    pct_data = (agg_data / base_row) * 100

    with st.expander(f"Plot {plot_idx + 1} — Aggregated data", expanded=False):
        st.markdown("**Aggregated values**")
        st.dataframe(agg_data.loc[selected_conditions].reset_index(), use_container_width=True)
        st.markdown("**Relative to base (%)**")
        st.dataframe(pct_data.loc[selected_conditions].round(2).reset_index(), use_container_width=True)

    # Build colour map
    _condition_colors = {
        cond: st.session_state.get(f"cp{plot_idx}_color_{ci}", _palette[ci % len(_palette)])
        for ci, cond in enumerate(selected_conditions)
    }

    _show_spider = plot_type in ("Spider / Radar", "Both")
    _show_bar = plot_type in ("Grouped Bar", "Both")

    _base_font = dict(family=font_family, size=tick_font_size)

    # Auto title
    if plot_title.strip():
        _title_text = plot_title.strip()
    else:
        _title_text = f"{'Spider' if _show_spider and not _show_bar else 'Bar' if _show_bar and not _show_spider else 'Comparative'} — % relative to {_norm_label}"

    _title_formatted = _fmt(_title_text, title_bold, title_italic)

    # ----- Spider / Radar plot -----
    if _show_spider:
        fig_spider = go.Figure()

        for ci, cond in enumerate(selected_conditions):
            values = pct_data.loc[cond, selected_vars].values.tolist()
            values_closed = values + [values[0]]
            vars_closed = selected_vars + [selected_vars[0]]
            color = _condition_colors.get(cond, _palette[ci % len(_palette)])

            fig_spider.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=vars_closed,
                name=cond,
                line=dict(color=color, width=2),
                fill="toself" if fill_radar else "none",
                opacity=radar_opacity if fill_radar else 1.0,
                hovertemplate=f"{cond}<br>%{{theta}}: %{{r:.1f}}%<extra></extra>",
            ))

        fig_spider.update_layout(
            title=dict(text=_title_formatted, font=dict(family=font_family, size=title_font_size)),
            polar=dict(
                radialaxis=dict(
                    visible=True, showticklabels=True, ticksuffix="%",
                    tickfont=dict(family=font_family, size=tick_font_size),
                ),
                angularaxis=dict(
                    tickfont=dict(family=font_family, size=axis_label_font_size),
                ),
            ),
            font=_base_font,
            height=plot_height,
            legend=dict(font=dict(family=font_family, size=legend_font)),
            margin=dict(l=80, r=80, t=60, b=40),
        )

        st.plotly_chart(fig_spider, use_container_width=True, key=f"comp_spider_{plot_idx}")

        _dl1, _dl2, _ = st.columns([0.2, 0.2, 0.6])
        with _dl1:
            _html = fig_spider.to_html(include_plotlyjs="cdn").encode("utf-8")
            st.download_button(
                "📥 Download HTML", data=_html,
                file_name=f"spider_plot_{plot_idx + 1}.html", mime="text/html",
                key=f"dl_spider_html_{plot_idx}", use_container_width=True,
            )
        with _dl2:
            try:
                _png = fig_spider.to_image(format="png", width=900, height=plot_height, scale=2)
                st.download_button(
                    "📥 Download PNG", data=_png,
                    file_name=f"spider_plot_{plot_idx + 1}.png", mime="image/png",
                    key=f"dl_spider_png_{plot_idx}", use_container_width=True,
                )
            except Exception as _e:
                st.caption(f"PNG export unavailable: {_e}")

    # ----- Grouped Bar plot -----
    if _show_bar:
        fig_bar = go.Figure()

        for ci, cond in enumerate(selected_conditions):
            values = pct_data.loc[cond, selected_vars].values.tolist()
            color = _condition_colors.get(cond, _palette[ci % len(_palette)])
            text_vals = [f"{v:.1f}%" for v in values] if show_values else None

            fig_bar.add_trace(go.Bar(
                x=selected_vars,
                y=values,
                name=cond,
                marker_color=color,
                text=text_vals,
                textposition="outside" if show_values else "none",
                textfont=dict(family=font_family, size=tick_font_size),
                hovertemplate=f"{cond}<br>%{{x}}: %{{y:.1f}}%<extra></extra>",
            ))

        _y_title = _fmt("% relative to base", axis_bold, axis_italic)

        fig_bar.update_layout(
            title=dict(text=_title_formatted, font=dict(family=font_family, size=title_font_size)),
            barmode="group",
            yaxis_title=_y_title,
            yaxis=dict(
                ticksuffix="%",
                title_font=dict(family=font_family, size=axis_label_font_size),
                tickfont=dict(family=font_family, size=tick_font_size),
            ),
            xaxis=dict(
                tickfont=dict(family=font_family, size=tick_font_size),
                title_font=dict(family=font_family, size=axis_label_font_size),
            ),
            font=_base_font,
            height=plot_height,
            legend=dict(font=dict(family=font_family, size=legend_font)),
            margin=dict(l=60, r=20, t=60, b=80),
        )

        fig_bar.add_hline(
            y=100, line_dash="dash", line_color="grey", line_width=1.5,
            annotation_text=f"100% ({_norm_label})", annotation_position="top right",
            annotation_font=dict(family=font_family, size=tick_font_size),
        )

        st.plotly_chart(fig_bar, use_container_width=True, key=f"comp_bar_{plot_idx}")

        _dl1, _dl2, _ = st.columns([0.2, 0.2, 0.6])
        with _dl1:
            _html = fig_bar.to_html(include_plotlyjs="cdn").encode("utf-8")
            st.download_button(
                "📥 Download HTML", data=_html,
                file_name=f"bar_plot_{plot_idx + 1}.html", mime="text/html",
                key=f"dl_bar_html_{plot_idx}", use_container_width=True,
            )
        with _dl2:
            try:
                _png = fig_bar.to_image(format="png", width=900, height=plot_height, scale=2)
                st.download_button(
                    "📥 Download PNG", data=_png,
                    file_name=f"bar_plot_{plot_idx + 1}.png", mime="image/png",
                    key=f"dl_bar_png_{plot_idx}", use_container_width=True,
                )
            except Exception as _e:
                st.caption(f"PNG export unavailable: {_e}")

    st.markdown("---")

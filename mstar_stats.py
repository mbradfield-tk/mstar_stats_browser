import io
import os
import re
import subprocess
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import streamlit as st


# ---------------------------
# Helpers
# ---------------------------
@st.cache_data(show_spinner=False)
def _read_bytes_as_df(file_bytes: bytes) -> pd.DataFrame:
    """
    Parse an uploaded text file (bytes) into a DataFrame.
    - Auto-detects delimiter (tabs, spaces, etc.) using engine='python' + sep=None.
    - Falls back to a generic whitespace regex if needed.
    - Assumes the first column is time.
    - Coerces all columns to numeric where possible and sorts by time.
    """
    text = file_bytes.decode("utf-8", errors="ignore")
    sio = io.StringIO(text)
    try:
        df = pd.read_csv(sio, sep=None, engine="python")
    except Exception:
        sio.seek(0)
        df = pd.read_csv(sio, sep=r"\s+", engine="python")

    if df.shape[1] < 2:
        raise ValueError("Parsed file has fewer than 2 columns (time + variables required).")

    time_col = df.columns[0]
    if time_col != "Time":
        df = df.rename(columns={time_col: "Time"})

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
    return df


def _collect_columns(dfs: Dict[str, pd.DataFrame], match_mode: str = "Intersection") -> List[str]:
    col_sets = [set([c for c in df.columns if c != "Time"]) for df in dfs.values()]
    if not col_sets:
        return []
    return sorted(set.intersection(*col_sets)) if match_mode == "Intersection" else sorted(set.union(*col_sets))


def _maybe_subsample(df: pd.DataFrame, step: int) -> pd.DataFrame:
    return df if step <= 1 else df.iloc[::step, :].reset_index(drop=True)


def _parse_float_list(text: str) -> List[float]:
    if not text or not text.strip():
        return []
    tokens = re.split(r"[,\s;]+", text.strip())
    vals = []
    for t in tokens:
        if not t:
            continue
        try:
            vals.append(float(t))
        except ValueError:
            pass
    return vals


def _parse_labels_list(text: str) -> List[str]:
    """
    Parse a label list separated by commas/semicolons/spaces, preserving tokens that contain spaces via quotes.
    Practical rule: split on commas/semicolons OR whitespace, then trim.
    """
    if not text or not text.strip():
        return []
    # Allow quoted tokens with spaces: "Peak velocity"
    # Simple approach: split by comma/semicolon first, then split leftover on whitespace.
    raw = re.split(r"[;,]+", text.strip())
    out = []
    for chunk in raw:
        chunk = chunk.strip()
        if not chunk:
            continue
        # Further split on whitespace only if it looks like multiple labels without quotes
        if '"' in chunk or "'" in chunk:
            # remove quotes
            cleaned = chunk.replace('"', "").replace("'", "").strip()
            if cleaned:
                out.append(cleaned)
        else:
            parts = re.split(r"\s+", chunk)
            out.extend([p for p in parts if p])
    return out


def _legend_position(pos: str):
    mapping = {
        "best":          (1.02, 1.0, "left", "top"),
        "upper right":   (1.0, 1.0, "right", "top"),
        "upper left":    (0.0, 1.0, "left", "top"),
        "lower left":    (0.0, 0.0, "left", "bottom"),
        "lower right":   (1.0, 0.0, "right", "bottom"),
        "right":         (1.02, 1.0, "left", "top"),
        "center left":   (0.0, 0.5, "left", "middle"),
        "center right":  (1.0, 0.5, "right", "middle"),
        "lower center":  (0.5, 0.0, "center", "bottom"),
        "upper center":  (0.5, 1.0, "center", "top"),
    }
    return mapping.get(pos, mapping["best"])


def _style_to_dash(style: str) -> str:
    return {"solid": "solid", "dashed": "dash", "dashdot": "dashdot", "dotted": "dot"}.get(style, "dash")


def _fmt_with_fallback(pattern: str, **kwargs) -> str:
    """Safely format a label pattern; fallback to 'x=..' or 'y=..' if formatting fails."""
    try:
        return pattern.format(**kwargs)
    except Exception:
        if "x" in kwargs:
            return f"x={kwargs['x']}"
        if "y" in kwargs:
            return f"y={kwargs['y']}"
        return ""


# ---------------------------
# UI: Uploader and options
# ---------------------------
st.title("M-Star Stats Explorer")

input_mode = st.radio(
    "Data source:",
    options=["Upload files", "Folder with subfolders"],
    index=0,
    horizontal=True,
    help=(
        "Upload files: manually upload one or more files.\n"
        "Folder with subfolders: browse to a parent folder, pick a filename, "
        "and load that file from every subfolder (each subfolder = one data series)."
    ),
)

files = None
_folder_dfs: Dict[str, pd.DataFrame] = {}

if "mstar_dfs" not in st.session_state:
    st.session_state["mstar_dfs"] = {}

if input_mode == "Upload files":
    files = st.file_uploader(
        "Upload one or more .txt files (first column = time; other columns = variables):",
        type=["txt", "tsv", "csv"],
        accept_multiple_files=True,
    )

    # Display the parent folder of uploaded files using macOS Spotlight + file-size matching
    if files:
        _matched_parent = None
        try:
            _candidate_dirs_per_file = []
            for f in files:
                result = subprocess.run(
                    ["mdfind", "-name", f.name],
                    capture_output=True, text=True, timeout=5,
                )
                dirs_for_this_file = set()
                for line in result.stdout.strip().splitlines():
                    if line.endswith("/" + f.name) or line.endswith(os.sep + f.name):
                        dirs_for_this_file.add(os.path.dirname(line))
                _candidate_dirs_per_file.append((f, dirs_for_this_file))

            if _candidate_dirs_per_file:
                _common_dirs = _candidate_dirs_per_file[0][1].copy()
                for _, ds in _candidate_dirs_per_file[1:]:
                    _common_dirs = _common_dirs & ds

                if len(_common_dirs) > 1:
                    _verified_dirs = set()
                    for cand_dir in _common_dirs:
                        all_match = True
                        for f, _ in _candidate_dirs_per_file:
                            disk_path = os.path.join(cand_dir, f.name)
                            if os.path.isfile(disk_path):
                                if os.path.getsize(disk_path) != f.size:
                                    all_match = False
                                    break
                            else:
                                all_match = False
                                break
                        if all_match:
                            _verified_dirs.add(cand_dir)
                    if _verified_dirs:
                        _common_dirs = _verified_dirs

                if _common_dirs:
                    _best = max(_common_dirs, key=len)
                    _matched_parent = os.path.basename(_best)
        except Exception:
            pass
        if _matched_parent:
            st.caption(f"📂 {_matched_parent}")

else:
    # --- Folder with subfolders mode ---
    st.markdown("**Browse for a parent folder** containing subfolders (each subfolder = one run/series).")

    if "browse_src_path" not in st.session_state:
        st.session_state["browse_src_path"] = os.path.expanduser("~")

    _src_current = st.session_state["browse_src_path"]
    st.markdown(f"📂 Browsing: **{_src_current}**")

    try:
        _src_subdirs = sorted(
            d for d in os.listdir(_src_current)
            if os.path.isdir(os.path.join(_src_current, d)) and not d.startswith(".")
        )
    except PermissionError:
        _src_subdirs = []
        st.warning("Permission denied for this directory.")

    _src_c1, _src_c2 = st.columns([0.3, 0.7])
    with _src_c1:
        if st.button("⬆ Parent", use_container_width=True, key="src_parent_btn"):
            _parent = os.path.dirname(_src_current)
            if _parent and _parent != _src_current:
                st.session_state["browse_src_path"] = _parent
                st.rerun()
    with _src_c2:
        _src_chosen = st.selectbox(
            "Navigate into subfolder:",
            options=["(stay here)"] + _src_subdirs,
            index=0,
            key="src_subdir_select",
        )
        if _src_chosen != "(stay here)":
            st.session_state["browse_src_path"] = os.path.join(_src_current, _src_chosen)
            st.rerun()

    if st.button("✅ Use this folder", type="primary", use_container_width=True, key="src_use_btn"):
        st.session_state["src_folder_selected"] = _src_current

    _src_manual = st.text_input(
        "Or enter path manually:",
        value="",
        placeholder="/path/to/parent/folder",
        key="src_manual_path",
    )
    if _src_manual and _src_manual.strip():
        st.session_state["src_folder_selected"] = _src_manual.strip()

    _src_folder = st.session_state.get("src_folder_selected", "")

    if _src_folder and os.path.isdir(_src_folder):
        st.info(f"Selected: `{_src_folder}`")

        # Discover subfolders
        try:
            _run_dirs = sorted(
                d for d in os.listdir(_src_folder)
                if os.path.isdir(os.path.join(_src_folder, d)) and not d.startswith(".")
            )
        except PermissionError:
            _run_dirs = []

        if not _run_dirs:
            st.warning("No subfolders found in the selected folder.")
        else:
            # Collect filenames from each subfolder and find the intersection
            _file_sets = []
            for rd in _run_dirs:
                rd_path = os.path.join(_src_folder, rd)
                rd_files = set(
                    f for f in os.listdir(rd_path)
                    if os.path.isfile(os.path.join(rd_path, f)) and not f.startswith(".")
                )
                _file_sets.append(rd_files)

            _common_files = sorted(set.intersection(*_file_sets)) if _file_sets else []

            if not _common_files:
                st.warning("No common filenames found across all subfolders.")
            else:
                st.caption(f"Found {len(_run_dirs)} subfolder(s) with {len(_common_files)} common file(s).")
                _chosen_file = st.selectbox(
                    "Select file to analyze:",
                    options=_common_files,
                    index=0,
                    key="src_file_select",
                )

                if _chosen_file:
                    # Load the chosen file from every subfolder, keyed by subfolder name
                    _parse_errors_folder: List[Tuple[str, str]] = []
                    for rd in _run_dirs:
                        fpath = os.path.join(_src_folder, rd, _chosen_file)
                        try:
                            with open(fpath, "rb") as fh:
                                df = _read_bytes_as_df(fh.read())
                            # Use subfolder name (minus Stats_ prefix) as the series key
                            display_key = rd.replace("Stats_", "").replace("stats_", "") or rd
                            _folder_dfs[display_key] = df
                        except Exception as e:
                            _parse_errors_folder.append((rd, str(e)))

                    if _parse_errors_folder:
                        st.error("Some subfolders failed to parse:")
                        for rname, msg in _parse_errors_folder:
                            st.write(f"• **{rname}** — {msg}")
    elif _src_folder:
        st.warning("Folder not found. Please check the path.")

with st.expander("Plot & parsing options", expanded=True):
    left, mid, right = st.columns([1.1, 1.1, 1.0])
    with left:
        match_mode = st.radio(
            "Match variables by:",
            options=["Intersection", "Union"],
            index=0,
            help=(
                "Intersection: only variables present across all files.\n"
                "Union: variables present in any file (missing series are skipped per subplot)."
            ),
        )
        zero_time = st.checkbox("Zero time per file (subtract each file's minimum time)", value=False)
    with mid:
        subsample = st.number_input("Subsample factor (plot every Nth point)", min_value=1, max_value=10_000, value=1, step=1)
        sharex = st.checkbox("Share X axis across subplots", value=True)
    with right:
        height_per_subplot = st.slider("Height per subplot (inches)", 1.0, 10.0, 5.0, 0.1)
        legend_loc = st.selectbox(
            "Legend location",
            options=[
                "best", "upper right", "upper left", "lower left", "lower right",
                "right", "center left", "center right", "lower center", "upper center"
            ],
            index=0,
        )
    # Global data line width
    data_line_width = st.slider("Data line width", 0.5, 6.0, 1.6, 0.1)

with st.expander("Legend options", expanded=True):
    legend_mode = st.radio(
        "Legend behavior",
        options=["Per-file (toggle across subplots)", "Per-trace (independent)"],
        index=0,
        help=(
            "Per-file: one legend entry per file; clicking toggles that file in ALL subplots.\n"
            "Per-trace: a legend entry for every subplot-series; clicking toggles only that series."
        ),
    )

with st.expander("Series averages", expanded=False):
    show_avg = st.checkbox("Show average value per series", value=False)
    avg_c1, avg_c2, avg_c3 = st.columns([1.2, 0.9, 0.9])
    with avg_c1:
        avg_start_time = st.number_input(
            "Calculate average from time (s):",
            min_value=0.0,
            value=5.0,
            step=0.1,
            format="%.2f",
            help="Only data points at or after this time are included in the average.",
        )
    with avg_c2:
        avg_line_style = st.selectbox(
            "Average line style",
            ["solid", "dashed", "dashdot", "dotted"],
            index=1,
            key="avg_line_style",
        )
        avg_line_width = st.slider("Average line width", 0.5, 4.0, 1.5, 0.1, key="avg_line_width")
    with avg_c3:
        avg_line_opacity = st.slider("Average line opacity", 0.1, 1.0, 0.7, 0.05, key="avg_line_opacity")
        show_avg_annotation = st.checkbox("Show value as annotation", value=True, key="avg_annotation")
    show_avg_in_legend = st.checkbox("Show average lines in legend", value=True, key="avg_legend")

with st.expander("Reference lines (vertical = Time, horizontal = y-value)", expanded=False):
    # --- Vertical lines
    vcol_left, vcol_mid, vcol_right = st.columns([1.2, 0.9, 0.9])
    with vcol_left:
        vline_input = st.text_input(
            "Vertical lines at Time (x):",
            value="",
            placeholder="e.g., 0.5, 1, 2.25",
            help="Enter times (x-values) separated by commas, spaces, or semicolons.",
        )
    with vcol_mid:
        vline_color = st.color_picker("V‑line color", "#AA0000")
        vline_style = st.selectbox("V‑line style", ["solid", "dashed", "dashdot", "dotted"], index=1)
    with vcol_right:
        vline_width = st.slider("V‑line width", 0.5, 4.0, 1.5, 0.1)
        vline_alpha = st.slider("V‑line opacity", 0.1, 1.0, 0.8, 0.05)

    # NEW — Vertical line labels
    st.markdown("**Vertical line labels**")
    lv1, lv2, lv3, lv4 = st.columns([1.2, 0.9, 0.8, 0.8])
    with lv1:
        vline_labels_text = st.text_input(
            "Labels (optional, one per x; leave blank to use pattern):",
            value="",
            placeholder='e.g., "start", "event A", "peak"',
            help="Number of labels should match the number of x-values if provided.",
        )
    with lv2:
        vlabel_pattern = st.text_input("Label pattern", value="t = {x:.3g}")
    with lv3:
        vlabel_position = st.selectbox("Position", ["top", "bottom"], index=0)
    with lv4:
        vlabel_font = st.number_input("Font size", 6, 32, 11, 1)
    vlabel_offset = st.slider("Vertical label offset (pixels, +moves outward)", -40, 80, 8, 1)

    st.markdown("---")

    # --- Horizontal lines: Set 1 (existing)
    hcol_left, hcol_mid, hcol_right = st.columns([1.2, 0.9, 0.9])
    with hcol_left:
        hline_input = st.text_input(
            "Horizontal lines (y-values) — Set 1:",
            value="",
            placeholder="e.g., 0, 100, 250",
            help="Enter y-values separated by commas, spaces, or semicolons. Applied to all subplots.",
            key="hset_0_vals",
        )
    with hcol_mid:
        hline_color = st.color_picker("H‑line color — Set 1", "#0072B2", key="hset_0_color")
        hline_style = st.selectbox("H‑line style — Set 1", ["solid", "dashed", "dashdot", "dotted"], index=2, key="hset_0_style")
    with hcol_right:
        hline_width = st.slider("H‑line width — Set 1", 0.5, 4.0, 1.5, 0.1, key="hset_0_width")
        hline_alpha = st.slider("H‑line opacity — Set 1", 0.1, 1.0, 0.8, 0.05, key="hset_0_alpha")

    # NEW — Labels for Horizontal Set 1
    lh1, lh2, lh3, lh4 = st.columns([1.2, 0.8, 0.8, 0.8])
    with lh1:
        hset0_label_pattern = st.text_input("Label pattern — Set 1", value="y = {y:.3g}")
    with lh2:
        hset0_label_position = st.selectbox("Label position — Set 1", ["left", "right"], index=0)
    with lh3:
        hset0_label_font = st.number_input("Font size — Set 1", 6, 32, 11, 1)
    with lh4:
        hset0_label_offset = st.slider("Label offset (px) — Set 1", -60, 120, 6, 1)

    # Additional horizontal line sets
    st.markdown("**Additional horizontal line sets**")
    add_sets = st.number_input("Number of extra sets", min_value=0, max_value=5, value=0, step=1)
    default_h_colors = ["#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]

    extra_h_sets = []
    for idx in range(add_sets):
        st.markdown(f"**Set {idx+2}**")
        c1, c2, c3 = st.columns([1.2, 0.9, 0.9])
        with c1:
            vals = st.text_input(
                "Horizontal lines (y-values):",
                value="",
                placeholder="e.g., 0.01 0.02 0.05",
                key=f"hset_{idx+1}_vals",
            )
        with c2:
            color = st.color_picker(
                "Color",
                default_h_colors[idx % len(default_h_colors)],
                key=f"hset_{idx+1}_color",
            )
            style = st.selectbox(
                "Style",
                ["solid", "dashed", "dashdot", "dotted"],
                index=(idx % 4),
                key=f"hset_{idx+1}_style",
            )
        with c3:
            width = st.slider("Width", 0.5, 4.0, 1.5, 0.1, key=f"hset_{idx+1}_width")
            alpha = st.slider("Opacity", 0.1, 1.0, 0.8, 0.05, key=f"hset_{idx+1}_alpha")

        # NEW — Labeling for each extra set
        l1, l2, l3, l4 = st.columns([1.2, 0.8, 0.8, 0.8])
        with l1:
            label_pattern = st.text_input("Label pattern", value="y = {y:.3g}", key=f"hset_{idx+1}_label_pattern")
        with l2:
            label_position = st.selectbox("Label position", ["left", "right"], index=0, key=f"hset_{idx+1}_label_pos")
        with l3:
            label_font = st.number_input("Font size", 6, 32, 11, 1, key=f"hset_{idx+1}_font")
        with l4:
            label_offset = st.slider("Label offset (px)", -60, 120, 6, 1, key=f"hset_{idx+1}_offset")

        extra_h_sets.append(
            dict(
                vals=vals, color=color, style=style, width=width, alpha=alpha,
                label=f"Y threshold S{idx+2}",
                label_pattern=label_pattern, label_position=label_position,
                label_font=label_font, label_offset=label_offset
            )
        )

    show_line_labels = st.checkbox(
        "Show line labels in legend (labels appear once, on the top subplot)",
        value=True
    )


# ---------------------------
# Parse uploaded files / merge folder data
# ---------------------------
dfs: Dict[str, pd.DataFrame] = {}

if input_mode == "Upload files" and files:
    parse_errors: List[Tuple[str, str]] = []
    for f in files:
        try:
            df = _read_bytes_as_df(f.getvalue())
            dfs[f.name] = df
        except Exception as e:
            parse_errors.append((f.name, str(e)))

    if parse_errors:
        st.error("One or more files failed to parse:")
        for fname, msg in parse_errors:
            st.write(f"• **{fname}** — {msg}")
    if dfs:
        st.session_state["mstar_dfs"] = dfs
elif input_mode == "Folder with subfolders" and _folder_dfs:
    dfs = _folder_dfs
    st.session_state["mstar_dfs"] = dfs

    if dfs:
        with st.expander("Preview first rows from each file", expanded=False):
            for name, df in dfs.items():
                st.markdown(f"**{name}**")
                st.dataframe(df.head(10), use_container_width=True)

# ---------------------------
# Variable selection & plotting
# ---------------------------
# Restore from session state if no fresh data
if not dfs and st.session_state.get("mstar_dfs"):
    dfs = st.session_state["mstar_dfs"]

if dfs:
    all_vars = _collect_columns(dfs, match_mode=match_mode)
    if not all_vars:
        st.warning("No variable columns found (besides 'Time'). Check your files/headers.")
    else:
        # Variable selection via sidebar checkboxes
        with st.sidebar:
            st.header("Variables")
            default_vars = all_vars[: min(4, len(all_vars))]

            _sel_all, _sel_none = st.columns(2)
            if _sel_all.button("Select all", use_container_width=True):
                for var in all_vars:
                    st.session_state[f"var_cb_{var}"] = True
            if _sel_none.button("Clear all", use_container_width=True):
                for var in all_vars:
                    st.session_state[f"var_cb_{var}"] = False

            chosen_vars = []
            for var in all_vars:
                # If this variable already has a stored state, keep it; otherwise default
                if f"var_cb_{var}" in st.session_state:
                    default_val = st.session_state[f"var_cb_{var}"]
                else:
                    default_val = var in default_vars
                if st.checkbox(var, value=default_val, key=f"var_cb_{var}"):
                    chosen_vars.append(var)

        if chosen_vars:
            # Prepare data per file
            prepared: Dict[str, pd.DataFrame] = {}
            for name, df in dfs.items():
                dfp = df.copy()
                if zero_time:
                    t0 = dfp["Time"].min()
                    dfp["Time"] = dfp["Time"] - t0
                if subsample > 1:
                    dfp = _maybe_subsample(dfp, subsample)
                prepared[name] = dfp

            # Colors per file (consistent across subplots)
            base_colors = (
                px.colors.qualitative.Plotly
                + px.colors.qualitative.D3
                + px.colors.qualitative.Set3
                + px.colors.qualitative.T10
            )
            file_names = list(prepared.keys())
            color_map = {fname: base_colors[i % len(base_colors)] for i, fname in enumerate(file_names)}

            # --- Series (data-set) display labels ---
            with st.expander("Series labels (rename data sets)", expanded=False):
                st.caption("Override the file-name labels shown in legends and hover tooltips.")
                label_map: Dict[str, str] = {}
                for j, fname in enumerate(file_names):
                    display_label = st.text_input(
                        f"Label for file {j+1} ({fname})",
                        value=fname,
                        key=f"series_label_{j}",
                    )
                    label_map[fname] = display_label

            # --- Subplot headings & axis labels ---
            with st.expander("Subplot headings & axis labels", expanded=False):
                # Default X-axis label (used to pre-fill each subplot's X label)
                default_x_label = st.text_input(
                    "Default X-axis label",
                    value="Time",
                    help="Used to pre-fill each subplot's X label; you can still override per subplot below.",
                    key="axis_default_x",
                )

                # Note to user when sharex is on
                if sharex:
                    st.caption("ℹ️ X axes are shared: Plotly shows the X-axis title on the bottom subplot by default, "
                            "but you can still set them per subplot here.")

                # Build per-subplot heading and label inputs
                subplot_headings, axis_x_labels, axis_y_labels = [], [], []
                for i, var in enumerate(chosen_vars, start=1):
                    c0, c1, c2 = st.columns(3)
                    heading = c0.text_input(
                        f"Heading for subplot {i}",
                        value=var,
                        key=f"heading_{var}",
                        help="Custom title displayed above this subplot.",
                    )
                    ylab = c1.text_input(
                        f"Y label for subplot {i}",
                        value=var,
                        key=f"ylab_{var}"
                    )
                    xlab = c2.text_input(
                        f"X label for subplot {i}",
                        value=default_x_label,
                        key=f"xlab_{var}"
                    )
                    subplot_headings.append(heading)
                    axis_y_labels.append(ylab)
                    axis_x_labels.append(xlab)


            # Create subplots
            n_subplots = len(chosen_vars)
            max_spacing = 1.0 / max(n_subplots - 1, 1)
            v_spacing = min(0.06, max_spacing * 0.9)
            height_px = int(max(250, n_subplots * height_per_subplot * 96))  # ~96 px/inch
            fig = make_subplots(
                rows=n_subplots, cols=1,
                shared_xaxes=sharex,
                vertical_spacing=v_spacing,
                subplot_titles=tuple(subplot_headings),
            )

            # Legend config depending on mode
            per_file = legend_mode.startswith("Per-file")
            legend_groupclick = "togglegroup" if per_file else "toggleitem"

            # Add data traces
            for i, var in enumerate(chosen_vars, start=1):
                any_series = False
                for name, dfp in prepared.items():
                    if var not in dfp.columns:
                        continue

                    display_name = label_map.get(name, name)

                    if per_file:
                        fig.add_trace(
                            go.Scatter(
                                x=dfp["Time"].values,
                                y=dfp[var].values,
                                mode="lines",
                                name=display_name,
                                line=dict(color=color_map[name], width=data_line_width),
                                showlegend=(i == 1),
                                legendgroup=display_name,
                                hovertemplate=f"{display_name}<br>Time=%{{x}}<br>{var}=%{{y}}<extra></extra>",
                            ),
                            row=i, col=1
                        )
                    else:
                        entry_name = f"{display_name} — {var}"
                        fig.add_trace(
                            go.Scatter(
                                x=dfp["Time"].values,
                                y=dfp[var].values,
                                mode="lines",
                                name=entry_name,
                                line=dict(color=color_map[name], width=data_line_width),
                                showlegend=True,
                                hovertemplate=f"{display_name}<br>Var: {var}<br>Time=%{{x}}<br>Value=%{{y}}<extra></extra>",
                            ),
                            row=i, col=1
                        )
                    any_series = True

                # fig.update_yaxes(title_text=var, row=i, col=1)

                fig.update_yaxes(title_text=axis_y_labels[i-1], row=i, col=1)
                # fig.update_xaxes(title_text=axis_x_labels[i-1], row=i, col=1)

                if not any_series:
                    fig.add_annotation(
                        text="No data in any file for this variable",
                        xref=f"x{i}" if i > 1 else "x",
                        yref=f"y{i}" if i > 1 else "y",
                        x=0.5, y=0.5, showarrow=False, font=dict(color="red"),
                        row=i, col=1
                    )

            # ------------- Series averages -------------
            if show_avg:
                avg_dash = _style_to_dash(avg_line_style)
                avg_table_rows: List[Dict] = []
                for i, var in enumerate(chosen_vars, start=1):
                    for name, dfp in prepared.items():
                        if var not in dfp.columns:
                            continue
                        display_name = label_map.get(name, name)
                        # Filter data at or after start time
                        mask = dfp["Time"] >= avg_start_time
                        filtered = dfp.loc[mask, var].dropna()
                        if filtered.empty:
                            continue
                        avg_val = filtered.mean()
                        t_max = dfp["Time"].max()

                        avg_table_rows.append({
                            "Series": display_name,
                            "Variable": var,
                            "Average": avg_val,
                            "From Time": avg_start_time,
                            "N points": len(filtered),
                        })

                        # Draw average line from start_time to end of data
                        avg_trace_name = f"{display_name} avg={avg_val:.4g}"
                        if per_file:
                            fig.add_trace(
                                go.Scatter(
                                    x=[avg_start_time, t_max],
                                    y=[avg_val, avg_val],
                                    mode="lines",
                                    name=avg_trace_name,
                                    line=dict(
                                        color="black",
                                        width=avg_line_width,
                                        dash=avg_dash,
                                    ),
                                    opacity=avg_line_opacity,
                                    showlegend=(show_avg_in_legend and i == 1),
                                    legendgroup=f"{display_name}_avg",
                                    hovertemplate=f"{display_name} avg: {avg_val:.4g}<extra></extra>",
                                ),
                                row=i, col=1,
                            )
                        else:
                            fig.add_trace(
                                go.Scatter(
                                    x=[avg_start_time, t_max],
                                    y=[avg_val, avg_val],
                                    mode="lines",
                                    name=avg_trace_name,
                                    line=dict(
                                        color="black",
                                        width=avg_line_width,
                                        dash=avg_dash,
                                    ),
                                    opacity=avg_line_opacity,
                                    showlegend=show_avg_in_legend,
                                    hovertemplate=f"{display_name} avg ({var}): {avg_val:.4g}<extra></extra>",
                                ),
                                row=i, col=1,
                            )

                        # Annotation with the average value
                        if show_avg_annotation:
                            xdomain_ref = f"x{i} domain" if i > 1 else "x domain"
                            yref = f"y{i}" if i > 1 else "y"
                            fig.add_annotation(
                                x=1.0, xref=xdomain_ref,
                                y=avg_val, yref=yref,
                                text=f"{display_name}: {avg_val:.4g}",
                                showarrow=False,
                                xanchor="left",
                                font=dict(size=10, color="black"),
                                xshift=4,
                                bgcolor="rgba(255,255,255,0.7)",
                                bordercolor="rgba(0,0,0,0.1)",
                                borderwidth=1,
                            )

                # Display averages table
                if avg_table_rows:
                    avg_df = pd.DataFrame(avg_table_rows)
                    with st.expander("Average values table", expanded=True):
                        st.dataframe(
                            avg_df.style.format({"Average": "{:.4g}", "From Time": "{:.2f}"}),
                            use_container_width=True,
                        )

            # ------------- Reference lines (shapes) -------------
            vlines = _parse_float_list(vline_input)
            hlines = _parse_float_list(hline_input)
            v_dash = _style_to_dash(vline_style)
            h_dash = _style_to_dash(hline_style)

            # Build horizontal line sets (Set 1 + additional sets)
            hline_sets = []
            if hlines:
                hline_sets.append(dict(
                    values=hlines, color=hline_color, dash=h_dash, width=hline_width, alpha=hline_alpha,
                    label="Y threshold",
                    label_pattern=hset0_label_pattern,
                    label_position=hset0_label_position,
                    label_font=hset0_label_font,
                    label_offset=hset0_label_offset,
                ))
            for s in extra_h_sets:
                vals = _parse_float_list(s["vals"])
                if vals:
                    hline_sets.append(
                        dict(
                            values=vals,
                            color=s["color"],
                            dash=_style_to_dash(s["style"]),
                            width=s["width"],
                            alpha=s["alpha"],
                            label=s["label"],
                            label_pattern=s["label_pattern"],
                            label_position=s["label_position"],
                            label_font=s["label_font"],
                            label_offset=s["label_offset"],
                        )
                    )

            # Add shapes (lines) across all subplots
            for i in range(1, n_subplots + 1):
                # Vertical lines
                for x in vlines:
                    fig.add_vline(
                        x=x, line_color=vline_color, line_width=vline_width, line_dash=v_dash,
                        opacity=vline_alpha, row=i, col=1,
                    )
                # Horizontal line sets
                for hset in hline_sets:
                    for y in hset["values"]:
                        fig.add_hline(
                            y=y,
                            line_color=hset["color"],
                            line_width=hset["width"],
                            line_dash=hset["dash"],
                            opacity=hset["alpha"],
                            row=i, col=1,
                        )

            # ------------- NEW: Inline labels for reference lines -------------
            # Vertical labels: build label list (either explicit list or from pattern)
            vline_labels_list = _parse_labels_list(vline_labels_text)
            # If provided count mismatches, fall back to pattern for all lines
            use_explicit_vlabels = (len(vline_labels_list) == len(vlines))

            for i in range(1, n_subplots + 1):
                # Axis refs for this row
                xref = f"x{i}" if i > 1 else "x"
                ydomain_ref = f"y{i} domain" if i > 1 else "y domain"  # normalized [0,1] for this subplot

                # Vertical line labels
                if vlines:
                    for idx, x in enumerate(vlines):
                        label_text = (
                            vline_labels_list[idx]
                            if use_explicit_vlabels
                            else _fmt_with_fallback(vlabel_pattern, x=x)
                        )
                        # Place at top/bottom of the subplot using y-domain coordinates (stable under zoom)
                        y_in_domain = 1.0 if vlabel_position == "top" else 0.0
                        # Offset in pixels away from the plot area
                        yshift = vlabel_offset if vlabel_position == "top" else -vlabel_offset
                        fig.add_annotation(
                            x=x, xref=xref,
                            y=y_in_domain, yref=ydomain_ref,
                            text=label_text,
                            showarrow=False,
                            font=dict(size=vlabel_font, color=vline_color),
                            align="center",
                            yshift=yshift,
                            bgcolor="rgba(255,255,255,0.6)",
                            bordercolor="rgba(0,0,0,0.1)", borderwidth=1,
                        )

                # Horizontal line labels for each set
                for hset in hline_sets:
                    xdomain_ref = f"x{i} domain" if i > 1 else "x domain"
                    yref = f"y{i}" if i > 1 else "y"

                    at_left = (hset["label_position"] == "left")
                    x_in_domain = 0.0 if at_left else 1.0
                    xshift = (hset["label_offset"] if at_left else -hset["label_offset"])

                    for y in hset["values"]:
                        label_text = _fmt_with_fallback(hset["label_pattern"], y=y)
                        fig.add_annotation(
                            x=x_in_domain, xref=xdomain_ref,
                            y=y, yref=yref,
                            text=label_text,
                            showarrow=False,
                            xanchor=("left" if at_left else "right"),
                            font=dict(size=hset["label_font"], color=hset["color"]),
                            xshift=xshift,
                            bgcolor="rgba(255,255,255,0.6)",
                            bordercolor="rgba(0,0,0,0.1)", borderwidth=1,
                        )

            # Optional legend entries for ref lines (appear once on top subplot)
            if show_line_labels:
                if vlines:
                    fig.add_trace(
                        go.Scatter(
                            x=[None], y=[None], mode="lines",
                            name="Time marker",
                            line=dict(color=vline_color, width=vline_width, dash=v_dash),
                            showlegend=True, legendgroup="refs", visible="legendonly",
                        ),
                        row=1, col=1
                    )
                for hset in hline_sets:
                    fig.add_trace(
                        go.Scatter(
                            x=[None], y=[None], mode="lines",
                            name=hset["label"],
                            line=dict(color=hset["color"], width=hset["width"], dash=hset["dash"]),
                            showlegend=True, legendgroup="refs", visible="legendonly",
                        ),
                        row=1, col=1
                    )

            # Layout: legend position & behavior
            lx, ly, xa, ya = _legend_position(legend_loc)
            fig.update_layout(
                height=height_px,
                legend=dict(
                    x=lx, y=ly, xanchor=xa, yanchor=ya,
                    bgcolor="rgba(255,255,255,0.6)",
                    bordercolor="rgba(0,0,0,0.1)",
                    borderwidth=1,
                    groupclick=("togglegroup" if legend_mode.startswith("Per-file") else "toggleitem"),
                ),
                margin=dict(l=60, r=20, t=50, b=40),
            )
            # fig.update_xaxes(title_text="Time", row=n_subplots, col=1)
            fig.update_xaxes(title_text=axis_x_labels[i-1], row=i, col=1)

            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
        else:
            st.info("Select one or more variables to plot.")
else:
    if input_mode == "Upload files":
        st.info("Upload one or more `.txt` files to begin.")
    else:
        st.info("Select a parent folder and choose a file to analyze.")


# ---------------------------
# Batch File Rename — Append Suffix
# ---------------------------
st.markdown("---")
st.subheader("Batch File Rename — Append Suffix")

with st.expander("Rename files in subfolders", expanded=False):
    st.caption("Select a parent folder. Each subfolder's name (minus `Stats_`) will be appended to every file inside it.")

    # --- Browsable folder picker ---
    if "browse_path" not in st.session_state:
        st.session_state["browse_path"] = os.path.expanduser("~")

    browse_current = st.session_state["browse_path"]

    st.markdown(f"📂 Browsing: **{browse_current}**")

    # List sub-directories of the current path
    try:
        _subdirs = sorted(
            d for d in os.listdir(browse_current)
            if os.path.isdir(os.path.join(browse_current, d)) and not d.startswith(".")
        )
    except PermissionError:
        _subdirs = []
        st.warning("Permission denied for this directory.")

    nav_col1, nav_col2 = st.columns([0.3, 0.7])
    with nav_col1:
        if st.button("⬆ Parent folder", use_container_width=True):
            parent = os.path.dirname(browse_current)
            if parent and parent != browse_current:
                st.session_state["browse_path"] = parent
                st.rerun()
    with nav_col2:
        chosen_subdir = st.selectbox(
            "Navigate into subfolder:",
            options=["(stay here)"] + _subdirs,
            index=0,
            key="browse_subdir_select",
        )
        if chosen_subdir != "(stay here)":
            st.session_state["browse_path"] = os.path.join(browse_current, chosen_subdir)
            st.rerun()

    if st.button("✅ Use this folder", type="primary", use_container_width=True):
        st.session_state["rename_folder_selected"] = browse_current

    # Also allow manual override
    manual_path = st.text_input(
        "Or enter path manually:",
        value="",
        placeholder="/path/to/parent/folder",
    )
    if manual_path and manual_path.strip():
        st.session_state["rename_folder_selected"] = manual_path.strip()

    rename_folder = st.session_state.get("rename_folder_selected", "")
    if rename_folder:
        st.info(f"Selected parent folder: `{rename_folder}`")

    # --- Build preview across all subfolders ---
    if rename_folder and os.path.isdir(rename_folder):
        try:
            subfolder_names = sorted(
                d for d in os.listdir(rename_folder)
                if os.path.isdir(os.path.join(rename_folder, d)) and not d.startswith(".")
            )
        except PermissionError:
            subfolder_names = []
            st.warning("Permission denied.")

        if not subfolder_names:
            st.info("No subfolders found in the selected folder.")
        else:
            # Build a combined preview table
            preview_rows = []
            for sub in subfolder_names:
                suffix_base = sub.replace("Stats_", "").replace("stats_", "")
                suffix = f"_{suffix_base}" if suffix_base else ""
                sub_path = os.path.join(rename_folder, sub)
                sub_files = sorted(
                    f for f in os.listdir(sub_path)
                    if os.path.isfile(os.path.join(sub_path, f)) and not f.startswith(".")
                )
                for fname in sub_files:
                    base, ext = os.path.splitext(fname)
                    new_name = f"{base}{suffix}{ext}"
                    preview_rows.append({
                        "Subfolder": sub,
                        "Suffix": suffix,
                        "Original": fname,
                        "Renamed": new_name,
                    })

            if preview_rows:
                st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, height=300)

                if st.button("Apply rename to all subfolders", type="primary"):
                    total_renamed = 0
                    errors = []
                    for sub in subfolder_names:
                        suffix_base = sub.replace("Stats_", "").replace("stats_", "")
                        suffix = f"_{suffix_base}" if suffix_base else ""
                        if not suffix:
                            continue
                        sub_path = os.path.join(rename_folder, sub)
                        sub_files = sorted(
                            f for f in os.listdir(sub_path)
                            if os.path.isfile(os.path.join(sub_path, f)) and not f.startswith(".")
                        )
                        for fname in sub_files:
                            base, ext = os.path.splitext(fname)
                            new_name = f"{base}{suffix}{ext}"
                            src = os.path.join(sub_path, fname)
                            dst = os.path.join(sub_path, new_name)
                            if os.path.exists(dst):
                                errors.append(f"Skipped {sub}/{fname}: target already exists.")
                                continue
                            try:
                                os.rename(src, dst)
                                total_renamed += 1
                            except OSError as e:
                                errors.append(f"Failed {sub}/{fname}: {e}")

                    if total_renamed:
                        st.success(f"Renamed {total_renamed} file(s) across {len(subfolder_names)} subfolder(s).")
                    if errors:
                        for err in errors:
                            st.warning(err)
            else:
                st.info("No files found in any subfolder.")
    elif rename_folder:
        st.warning("Folder not found. Please check the path.")
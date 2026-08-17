import streamlit as st

st.set_page_config(
    page_title="M-Star Tools",
    layout="wide",
)

pg = st.navigation([
    st.Page("mstar_stats.py", title="M-Star Stats Explorer", icon="📊"),
    st.Page("pages/csv_explorer.py", title="CSV Data Explorer", icon="📈"),
    st.Page("pages/comparative_plots.py", title="Comparative Plots", icon="🕸️"),
    st.Page("pages/fit_ROM.py", title="ROM Fitting", icon="🔧"),
])

pg.run()

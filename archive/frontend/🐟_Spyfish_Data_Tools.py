import streamlit as st

st.set_page_config(
    page_title="Spyfish Aotearoa Toolkit",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🐟 Spyfish Aotearoa Toolkit")
st.caption(
    "A collection of tools for rangers and scientists working with Spyfish Aotearoa data."
)

st.divider()
st.subheader("Quick links")

cols = st.columns(3)
with cols[0]:
    st.page_link(
        "pages/📥_Export_Biigle_Annotations.py",
        label="Export BIIGLE Annotations",
        icon="📥",
    )
    st.caption(
        "Export Biigle annotations, review max count, and investigate size results."
    )
with cols[1]:
    st.page_link(
        "pages/📺_View_Deployment_Videos.py",
        label="View Deployment Videos",
        icon="📺",
    )
    st.caption("View videos from the deployments.")
with cols[2]:
    st.markdown("TBD")


st.divider()
st.markdown(
    "For more info about Spyfish Aotearoa check here: https://spyfish.notion.site/overview  \n"
    "For any issues please write to Kalindi or add your issues here: https://github.com/wildlifeai/Spyfish-Aotearoa-toolkit/issues"
)

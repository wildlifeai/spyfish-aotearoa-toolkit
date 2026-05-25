import streamlit as st

st.set_page_config(
    page_title="Spyfish Data Tools",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🐟 Spyfish Aotearoa Data Tools")
st.caption(
    "A collection of tools for rangers and scientists working with Spyfish Aotearoa data."
)

st.divider()
st.subheader("Quick links")

cols = st.columns(2, border=True)
with cols[0]:
    st.page_link(
        "pages/⚙️_Deployment_Management.py",
        label="Deployment Management",
        icon="⚙️",
    )
    st.caption("Manage deployment workflows and send videos to analysis pipelines.")

st.text(" ")

with cols[1]:
    st.page_link(
        "pages/🔍_Error_Review.py",
        label="Error Review",
        icon="🔍",
    )
    st.caption("Review validation errors and data quality issues.")

cols = st.columns(2, border=True)
with cols[0]:
    st.page_link(
        "pages/📺_View_Deployment_Videos.py",
        label="View Deployment Videos",
        icon="📺",
    )
    st.caption("View videos from the deployments.")

st.divider()
st.markdown(
    "For more info about Spyfish Aotearoa check here: https://spyfish.notion.site/overview  \n"
    "For any issues please write to Kalindi or add your issues here: https://github.com/wildlifeai/Spyfish-Aotearoa-toolkit/issues"
)

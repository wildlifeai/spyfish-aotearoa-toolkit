import hmac
import streamlit as st
from botocore.exceptions import ClientError

from spyfish.config import config
from spyfish.storage.s3_handler import S3Handler

# --- Helper to generate a presigned URL ---
@st.cache_resource
def get_s3_handler():
    return S3Handler()

def get_presigned_url(key: str, expires_in: int = 3600) -> str | None:
    handler = get_s3_handler()
    try:
        return handler.generate_presigned_url(key, expiration=expires_in)
    except Exception as e:
        st.error(f"AWS Error: {e}")
        return None

def check_password():
    """Returns True if the user entered the correct password."""
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if not st.session_state.password_correct:
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            app_password = st.secrets.get("APP_PASSWORD")

            if app_password is not None and hmac.compare_digest(password, app_password):
                st.session_state.password_correct = True
                st.rerun()
            else:
                st.error("❌ Incorrect password")
        return False
    return True

# --- MAIN APP ---
if not check_password():
    st.stop()
else:
    st.write("You are logged in! 🎉")

    # --- Streamlit UI ---
    st.title("Deployment Video player")

    # --- Checkbox: Use direct S3 path OR DropID ---
    use_direct_path = st.checkbox("Provide full S3 path instead of DropID")
    if use_direct_path:
        s3_key = st.text_input("Enter full S3 object path (key)")
    else:
        drop_id = st.text_input("Provide DropID").strip()
        # Legacy Spyfish prefix pattern for videos
        s3_key = f"media/{drop_id[:16]}/{drop_id[:27]}/{drop_id}" if drop_id else ""
        if drop_id and not s3_key.endswith(".mp4"):
            s3_key += ".mp4"

    if st.button("Generate URL and play video"):
        if not s3_key:
            st.error("Please provide an S3 path or a DropID.")
        else:

            ps_url = get_presigned_url(s3_key)

            if ps_url:
                st.subheader("Video preview.")
                st.write("Does the path look ok? (In the future this will check automatically.)")
                st.code(s3_key, language="text")
                st.write("The video box will show even when there are issues, so check above/try again later, or raise an issue.")
                st.video(ps_url)
                st.caption("Generated Presigned URL:")
                st.code(ps_url, language="text")
            else:
                st.error(f"Video not found at path: {s3_key}")

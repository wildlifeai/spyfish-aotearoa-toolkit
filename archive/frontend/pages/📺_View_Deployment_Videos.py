import hmac

import streamlit as st
from botocore.exceptions import ClientError

from sftk.common import S3_BUCKET
from sftk.s3_handler import S3Handler


# --- Helper to generate a presigned URL ---
@st.cache_resource
def _get_s3_client():
    """Get a cached S3 client."""
    s3_handler = S3Handler()
    return s3_handler.s3


st.write("This is Work in Progress, please share any issues.")


# TODO check if the file exists, add DropID validation etcetc
# --- Helper to generate a presigned URL ---
def get_presigned_url(key: str, expires_in: int = 3600) -> str | None:
    s3 = _get_s3_client()
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return None  # Object not found
        raise  # Re-raise other S3 client errors
    return s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=expires_in,
    )


# --- Simple password gate ---
def check_password():
    """Returns True if the user entered the correct password."""

    if "password_correct" not in st.session_state:
        # First run, ask for the password.
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
        drop_id = st.text_input("Provide DropID")
        s3_key = f"media/{drop_id[:16]}/{drop_id[:27]}/{drop_id}"
        if drop_id:
            if not s3_key.endswith(".mp4"):
                s3_key += ".mp4"
            else:
                s3_key = ""

    if st.button("Generate URL and play video"):
        if not s3_key:
            st.error("Please provide an S3 path or a DropID.")
        else:

            ps_url = get_presigned_url(s3_key)

            if ps_url:
                st.subheader("Video preview.")
                st.write(
                    "Does the path look ok? (In the future this will check automatically.)"
                )
                st.code(s3_key, language="text")
                st.write(
                    "The video box will show even when there are issues, so check above/try again later, or raise an issue."
                )
                st.video(ps_url)
            else:
                st.error(f"Video not found at path: {s3_key}")

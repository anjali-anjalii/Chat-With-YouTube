# app.py

import streamlit as st
from bot import process_video, get_qa_chain
import traceback

st.set_page_config(page_title="YouTube Video Chatbot", layout="wide")

st.title("🎥 YouTube Transcript Chatbot")
st.markdown("""
Paste a YouTube video URL to analyze its transcript and chat with the content.  
Great for lectures, interviews, and podcasts.
""")

# --------- Session State Initialization ----------
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "video_url" not in st.session_state:
    st.session_state.video_url = ""
if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""


# ---------- Video Input Form ----------
with st.form("video_form"):
    video_url = st.text_input("Enter a YouTube video URL:", placeholder="https://www.youtube.com/watch?v=...")
    submitted = st.form_submit_button("Process Video")

if submitted:
    if not video_url.strip():
        st.warning("Please enter a valid URL.")
    else:
        try:
            with st.spinner("Fetching transcript and creating vector index..."):
                vectorstore = process_video(video_url)
                retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
                qa_chain = get_qa_chain(retriever)

                # Store in session
                st.session_state.qa_chain = qa_chain
                st.session_state.vectorstore = vectorstore
                st.session_state.chat_history = []
                st.session_state.video_url = video_url
                st.success("✅ Video processed! You can now ask questions.")
        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(traceback.format_exc())


# ---------- Handle Chat Input ----------
def handle_user_input():
    user_input = st.session_state.chat_input.strip()
    if not user_input:
        return

    # Avoid repeating the same question
    if st.session_state.chat_history and user_input == st.session_state.chat_history[-1][0]:
        return

    try:
        response = st.session_state.qa_chain.invoke({"question": user_input})
        st.session_state.chat_history.append((user_input, response))
    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(traceback.format_exc())

    # Clear input field
    st.session_state.chat_input = ""


# ---------- Chat Interface ----------
if st.session_state.qa_chain:
    st.subheader("💬 Chat about the video")

    # Show existing chat history
    with st.container():
        for i, (q, a) in enumerate(st.session_state.chat_history):
            st.markdown(f"<div style='margin-bottom: 10px;'><b>🧑 You:</b> {q}</div>", unsafe_allow_html=True)
            st.markdown(
                f"""<div style='background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin-bottom: 20px; color: #000;'>
                <b>🤖 Bot:</b> {a}</div>""",
                unsafe_allow_html=True
            )

    # Input box with callback
    st.text_input(
        "Ask a question:",
        key="chat_input",
        placeholder="e.g., What is the video about?",
        on_change=handle_user_input
    )

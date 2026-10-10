import os
import streamlit as st
import requests
import time
import uuid
import logfire
from dotenv import load_dotenv

env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(env_path)

try:
    token = os.getenv("LOGFIRE_TOKEN")
    if not token:
        print("No LogFire token found. Please set the LOGFIRE_TOKEN environment variable.")
    logfire.configure(token=token)
    LOGFIRE_STATUS = "Connected and Tracing"
except Exception as e:
    LOGFIRE_STATUS = f"Error: {e}"
    print(f"Error connecting to LogFire: {e}")

# page config

st.set_page_config(
    page_title="Enterprise Agentic RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

AI_AVATAR="https://cdn-icons-png.flaticon.com/512/3135/3135715.png"
USER_AVATAR="https://cdn-icons-png.flaticon.com/512/149/149071.png"

## Session management
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    logfire.info(f"New Session ID: {st.session_state.session_id}")

if "messages" not in st.session_state:
    st.session_state.messages = []

## Side bar
with st.sidebar:
    st.title("Agent OS")
    st.markdown("---")
    st.success(f"logfire status: {LOGFIRE_STATUS}")
    st.info(f"Memory ID: {st.session_state.session_id[:8]}")

    if st.button("Clear History and Memory", width="stretch", type="primary"):
        logfire.warn(f"Memory wipe triggered for session {st.session_state.session_id}")
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()


## Main Chat
st.title("Enterprise Agentic Assistant")

## Display history

for message in st.session_state.messages:
    avatar = AI_AVATAR if message["role"] == "assistant" else USER_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.write(message["content"])

def stream_words(text):
    for w in text.split(" "):
        yield w + " "
        time.sleep(0.01)

if prompt := st.chat_input("Ask about your Document"):
    with logfire.span("User chat interaction", user_query=prompt, session_id=st.session_state.session_id):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=AI_AVATAR):
            data = None
            with st.status("Agent is thinking", expanded=True) as status:
                try:
                    with logfire.span("Calling RAG Backend"):
                        base_url = os.getenv("BACKEND_URL", "http://localhost:8000")
                        payload = {"query": prompt, "thread_id": st.session_state.session_id}
                        response = requests.post(f"{base_url}/query", json=payload, timeout=60)
                        response.raise_for_status()
                        data = response.json()

                    for step in data.get("plan", []):
                        st.write(f"⚙️ {step}")

                    status.update(label="✅ Answer Synthesized", state="complete", expanded=False)
                except Exception as e:
                    logfire.error(f"UI-Backend Connection Failed: {e}")
                    status.update(label="❌ Connection Failed", state="error")
                    st.error(f"Backend error: {e}")
                    st.stop()

            # Sources: outside st.status, one flat expander (no nesting)
            sources = data.get("documents", [])
            if sources:
                with st.expander(f"📄 Retrieved context ({len(sources)} chunks)"):
                    for i, src in enumerate(sources, 1):
                        st.markdown(f"**Chunk {i}**")
                        st.text(src.removeprefix("CONTENT: ")[:1500])
                        st.divider()

            full_answer = data.get("answer") or "No response."
            st.write_stream(stream_words(full_answer))

            st.session_state.messages.append({"role": "assistant", "content": full_answer})
            logfire.info("Chat cycle completed successfully.")
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


## Chat Input
if prompt := st.chat_input("Ask about your Document"):
    ## Start trace: User interaction
    with logfire.span("User chat interaction", user_query=prompt, session_id=st.session_state.session_id, start=True):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        ## Assistant Response
        with st.chat_message("assistant", avatar=AI_AVATAR):
            with st.status("Agent is thinking", expanded=True) as status:
                try:
                    with logfire.span("Calling RAG Backend"):
                        base_url = os.getenv("BACKEND_URL", "http://localhost:8000")
                        url = f"{base_url}/query"
                        payload = {"query": prompt, "thread_id": st.session_state.session_id}
                        response = requests.post(url, json=payload, timeout=60)
                        data = response.json()

                    steps = data.get("thought_process",  [])
                    for step in steps:
                        st.write(f" {step}")

                    status.update(label="Answer Synthesized", state="complete")

                    # Show Reasoning Steps from Backend
                    steps = data.get("thought_process", [])
                    for step in steps:
                        st.write(f"⚙️ {step}")
                    
                    status.update(label="✅ Answer Synthesized", state="complete", expanded=False)
                    
                    # --- SHOW SOURCES (NESTED EXPANDABLES) ---
                    sources = data.get("sources", [])
                    if sources:
                        with st.expander("📄 View Retrieved Context (Sources)"):
                            for i, source in enumerate(sources):
                                # Create a preview title for each chunk
                                preview = source[:100].replace("\n", " ") + "..."
                                with st.expander(f"Chunk {i+1}: {preview}"):
                                    st.info(source)
                except Exception as e:
                    logfire.error(f"❌ UI-Backend Connection Failed: {e}")
                    status.update(label="❌ Connection Failed", state="error")
                    st.error("Backend Offline.")
                    st.stop()

            # Final Answer Streaming
            answer_placeholder = st.empty()
            full_answer = data.get("answer", "No response.")
            
            curr_text = ""
            for char in full_answer:
                curr_text += char
                answer_placeholder.markdown(curr_text + "▌")
                time.sleep(0.005)
            
            answer_placeholder.markdown(full_answer)
            st.session_state.messages.append({"role": "assistant", "content": full_answer})
            logfire.info("✅ Chat cycle completed successfully.")
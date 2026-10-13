import logfire
from app.agents.state import AgentState
from app.config import settings
from langchain_groq import ChatGroq
from app.gateways.client import extract_cache_status, portkey_client

def generate_node(state: AgentState):
    """
    Synthesizes a response using both 
    Documentation Context and Conversation History
    """
    query = state["current_query"]

    history_str = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "User" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    user_msg = state["messages"][-1]["content"] if state["messages"] else ""

    if query == "CONVERSATIONAL":
        logfire.info("Generating conversational response using memory")
        prompt = f"""
        You are a friendly and helpful Enterprise AI Assistant.
        Answer the user's latest message using the conversation history below.

        CONVERSATION HISTORY:
        {history_str}

        LATEST MESSAGE:
        "{user_msg}"
        """

    else:
        logfire.info("Generating technical RAG response")
        max_context_chars = 25000
        full_context = ""

        for doc in state["documents"]:
            if len(full_context) + len(doc) < max_context_chars:
                full_context += doc + "\n\n"
            else:
                logfire.warning("Context truncated to fit Froq TPM limits")
                break

        prompt = f"""
        You are a Senior Technical Architect.
        Answer the question using the TECHNICAL CONTEXT provided.
        
        TECHNICAL CONTEXT:
        {full_context}
        
        CONVERSATION HISTORY:
        {history_str}

        USER QUESTION:
        "{user_msg}"
        """

    with logfire.span("LLM Synthesis"):
        try:
            response = portkey_client.chat.completions.create(
                messages=[{"role":"user", "content":prompt}],
                temperature=0.1,
            )
            content = response.choices[0].message.content
            cache_status = extract_cache_status(response)
            is_cache_hit = cache_status == "HIT"

            if is_cache_hit:
                logfire.info("🔥Gateway Cache hit - response served from portkey")
                plan_update = state["plan"] + ["Cache: Hit"]
                status = "Cache hit - instant response"
            else:
                logfire.info("🔥Gateway Cache miss - response generated from LLM")
                plan_update = state["plan"]
                status = "Cache miss - generating response"

            return {
                "final_answer": content,
                "status":status,
                "plan": plan_update,
                "messages": [{"role": "assistant", "content": content}]
            }

        except Exception as e:
            logfire.error(f"LLM synthesis failed: {str(e)}")
            raise
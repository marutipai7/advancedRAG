from app.agents.state import AgentState
from app.config import settings
from app.gateways.client import get_langchain_llm
import logfire

llm = get_langchain_llm(feature="planner")

def planner_node(state: AgentState):
    """
    The planner determines if a seach is needed based on the entire conversation
    """
    history = ""
    for message in state["messages"][:-1]:
        role = "User" if message["role"] == "User" else "Assistant"
        history += f"{role}: {message['content']}\n"

    user_message = state["messages"][-1]["content"] if state["messages"] else ""

    prompt = f"""
    You are an intelligent Assistant Planner.
    Analyze the convesation history and the latest user message.

    CONVERSATION HISTORY:
    {history}

    LATEST MESSAGE:
    "{user_message}"

    Task:
    1. If the latest message is greeting (Hi, Hello) or 
    a question that can be answered using ONLY the 
    conversation history above (e.g, "what is my name), respond with 'CONVERSATIONAL'.

    2. If it is a technical question about Kubernetes, 
    Intel, or Networking that requires a fresh documentation, 
    output a refined search query

    Output ONLY 'CONVERSATIONAL' or the search query
    """

    with logfire.span("Planner Decision"):
        decision = llm.invoke(prompt).content.strip()
        logfire.info(f"Planner decision: {decision}")

    if decision == "CONVERSATIONAL":
        return {
            "current_query": "CONVERSATIONAL",
            "status": "Handling conversationally (using memory)....",
            "plan" : ["Intent: Conversational/Memory", "Retrieval: Skipped"]
        }

    return{
        "current_query": decision,
        "status": f"Technical research needed. Searching for: {decision}",
        "plan" : ["Intent: Technical", f"Search Term: {decision}"]
    }
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.agents.state import AgentState
from app.agents.nodes.retriever import retriever_node
from app.agents.nodes.planner import planner_node
from app.agents.nodes.responder import generate_node


## Initialize State Graph
workflow = StateGraph(AgentState)

## Define nodes
workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("responder", generate_node)

## Define adges and Routing
def route_planner(state: AgentState):
    if state["current_query"] == "CONVERSATIONAL":
        return "responder"
    else:
        return "retriever"

workflow.set_entry_point("planner")

## conditional Edge: Planner > Router > Retriever or Responder
workflow.add_conditional_edges(
    "planner",
    route_planner,
    {
        "retriever": "retriever",
        "responder": "responder",
    }
)

workflow.add_edge("retriever", "responder")
workflow.add_edge("responder", END)

## Memory
checkpointer = MemorySaver()


## compile the graph with memory

rag_agent = workflow.compile(checkpointer=checkpointer)


import logfire
import os
from dotenv import load_dotenv

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
from fastapi import FastAPI, Response
from app.agents.graph import rag_agent
from app.gaurdrails.rails import initialize_rails, gaurd

from pydantic import BaseModel
from typing import Optional


app = FastAPI(title="Enterprise Agentic RAG API")

class QueryRequest(BaseModel):
    query: str
    thread_id: Optional[str] = "default_user"

@app.on_event("startup")
async def startup_event():
    initialize_rails()


@app.get("/")
def home():
    return {"message": "Enterprise Agentic RAG API is Live"}

@app.get("/graph")
def get_graph_image():
    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        logfire.error(f"Error generating graph: {str(e)}")
        return {"error": str(e)}


@app.post("/query")
async def query_endpoint(request: QueryRequest):
    query = request.query
    thread_id = request.thread_id

    initial_state = {
        "messages": [{"role": "User", "content": query}],
        "current_query": query,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph",
    }

    # config for Memory (thread_id)
    config = {"configurable": {"thread_id": thread_id}}

    try:
        rail_fired, rail_response = await gaurd(query)

        if rail_fired:
            logfire.info(f"🛡️ Guardrails fired | query='{query[:80]}', thread ={thread_id}")
            return {
                "question": query,
                "answer": rail_response,
                "plan": ["intent : Guardrails fired", "Retrieval: Skipped"],
                "status": "Blocked by Guardrails",
                "documents": [],
            }

        final_output = rag_agent.invoke(initial_state, config)

        return {
            "question": query,
            "answer": final_output.get("final_answer"),
            "plan": final_output.get("plan"),
            "status": final_output.get("status"),
            "documents": final_output.get("documents", []),
        }

    except Exception as e:
        logfire.error(f"Error processing query: {str(e)}")
        return {
            "question": query,
            "answer": "I apologize, an error occurred while processing your query. Please try again later.",
            "plan": ["Error"],
            "status": "Error",
            "documents": [],
        }

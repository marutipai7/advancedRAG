import time
import copy
import json
import os
import requests
import logfire

API_URL = "http://localhost:8000/query"
RESPONSE_TRUNCATE = 300
DELAY_BETWEEN_CALLS = 10
REQUEST_TIMEOUT = 120


def detect_tool(thought_process: list) -> str:
    joined = " ".join(thought_process)
    if "gaurdrails fired" in joined:
        return "gaurdrails"
    if "intent: technical" in joined or "search term:" in joined or "context retrieved" in joined:
        return "retrieve_documents"
    if "conversation" in joined or "memory" in joined:
        return "direct_answer"
    return "unknown"

def run_pipeline(golden_dataset: dict, progress_callback=None) -> dict:
    dataset = copy.deepcopy(golden_dataset)
    samples = dataset["rag_samples"]
    n = len(samples)

    with logfire.span(" Eval Phase 1 - Live Pipeline", total_samples=n):
        for i, sample in enumerate(samples):
            question = sample["question"]

            if progress_callback:
                progress_callback(i, n, question, "calling")
                
            with logfire.span(
                f" Live query {i+1}/{n}",
                question=question[:80],
                domain=sample.get("domain", ""),
            ):
                try:
                    resp = requests.post(
                        API_URL,
                        json={"query": question, "thread_id": f"eval_run_{i}"},
                        timeout=REQUEST_TIMEOUT,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    raw_answer = data.get("answer", "")
                    thought_process = data.get("plan", [])
                    sources = data.get("documents", [])

                    sample["actual_answer"] = raw_answer[:RESPONSE_TRUNCATE]
                    sample["actual_contexts"] = sources[:5]
                    sample["actual_tools_called"] = [detect_tool(thought_process)]

                    logfire.info(
                        "Response Captured",
                        tool=sample["actual_tools_called"][0],
                        response_chars = len(raw_answer),
                        context_chunks = len(sources),
                    )

                except requests.exceptions.RequestException as e:
                    logfire.error(f"Cannot reach FastAPI: {str(e)}")
                    sample["actual_answer"] = "Error calling API"
                    sample["actual_contexts"] = sample.get("relevant_contexts", [])
                    sample["actual_tools_called"] = ["Unknown"]

                except Exception as e:
                    logfire.error(f"query failed: {str(e)}")
                    sample["actual_answer"] = "Error processing response"
                    sample["actual_contexts"] = sample.get("relevant_contexts", [])
                    sample["actual_tools_called"] = ["Unknown"]

            if progress_callback:
                progress_callback(i, n, question, "done", sample["actual_answer"])

            if i < n - 1:
                time.sleep(DELAY_BETWEEN_CALLS)

    return dataset

def save_results(dataset: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)



def load_golden_dataset() -> dict:
    golden_path = os.path.join(
        os.path.dirname(__file__),
        "golden_dataset.json"
    )

    with open(golden_path, "r", encoding="utf-8") as f:
        return json.load(f)




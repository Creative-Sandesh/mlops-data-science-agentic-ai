from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.config import settings
from app.rag.ingestion import chunk_text
from app.rag.vector_store import RAGVectorStore
from app.llm_client import LLMService
from app.tools import get_current_weather

app = FastAPI(title="Task 1 AI Assistant API", version="1.0.0")

# Initialize persistent components
vector_db = RAGVectorStore()
llm_service = LLMService()

class IngestRequest(BaseModel):
    document_text: str

class QueryRequest(BaseModel):
    user_query: str
    use_rag: bool = True

@app.get("/health")
def health_check():
    return {"status": "healthy", "provider": settings.LLM_PROVIDER}

@app.post("/ingest")
def ingest_document(payload: IngestRequest):
    """Chunks incoming document and uploads vectors to Vector DB."""
    try:
        chunks = chunk_text(payload.document_text)
        vector_db.add_documents(chunks)
        return {"status": "success", "chunks_processed": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
def chat_endpoint(payload: QueryRequest):
    """Main query handler utilizing RAG, Tools, and Structured Output."""
    retrieved_context = []
    
    if payload.use_rag:
        retrieved_context = vector_db.retrieve(payload.user_query, top_k=3)
    
    response = llm_service.generate_with_rag(
        prompt=payload.user_query, 
        retrieved_context=retrieved_context
    )
    return {"response": response, "retrieved_context": retrieved_context}

# ---------------------------------------------------------------------------------------------
# W16 agentic feature (W17: exposed over HTTP; prompt/config come from configs/versions.yaml)
# ---------------------------------------------------------------------------------------------
class AgentRequest(BaseModel):
    question: str
    version: str = "v3"  # which prompt/config version from configs/versions.yaml to use


_agents: dict = {}


@app.post("/agent")
def agent_endpoint(payload: AgentRequest):
    """Answer with the tool-using agent loop. Returns the answer plus the full step-by-step trace."""
    from agent.llm import ChatLLM
    from agent.loop import run_agent
    from agent.tools import ToolBox
    from src.run_queries import load_version

    try:
        if payload.version not in _agents:
            cfg = load_version(payload.version)
            _agents[payload.version] = (cfg, ToolBox(top_k=cfg["top_k"], chunk_size=cfg["chunk_size"],
                                                  error_hints=cfg.get("tool_error_hints", False)),
                                        ChatLLM(model=cfg["model"], temperature=cfg["temperature"]))
        cfg, toolbox, llm = _agents[payload.version]
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown prompt version '{payload.version}'")
    trace = run_agent(payload.question, cfg["prompt_text"], toolbox, llm, cfg["max_iterations"],
                      cfg["lenient_json"], prompt_version=payload.version)
    return {"answer": trace["final_answer"], "trace": trace}

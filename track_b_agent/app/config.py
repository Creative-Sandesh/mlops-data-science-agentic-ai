from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Every field can be overridden by an environment variable or a .env file.

    W17 changes vs. W15: the default LLM endpoint is now a local Ollama server (OpenAI-compatible
    API) so experiments run offline and free; RAG / agent knobs (TOP_K, CHUNK_*, AGENT_MAX_ITERATIONS)
    were promoted to settings because they are the things MLflow tracks per experiment.
    """
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    LLM_PROVIDER: str = "ollama"
    LLM_API_BASE: str = "http://localhost:11434/v1"
    LLM_API_KEY: str = "ollama"  # Ollama ignores it; set a real key for OpenAI/Gemini/vLLM endpoints
    LLM_MODEL_NAME: str = "gemma:latest"

    # model tuning parameters
    TEMPERATURE: float = 0.2
    TOP_P: float = 0.9
    MAX_TOKENS: int = 512

    VECTOR_DB_URL: str = "http://localhost:6333"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

    # retrieval knobs (tracked in MLflow)
    TOP_K: int = 3
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # agent loop (W16)
    AGENT_MAX_ITERATIONS: int = 6

    SYSTEM_PROMPT: str = (
        "You are an expert AI assistant. Answer user question precisely using retrieved contexts or "
        "available tools. Always format JSON output accurately. If the answer is not found in the "
        "retrieved contexts, respond with 'I don't know'."
    )


settings = Settings()

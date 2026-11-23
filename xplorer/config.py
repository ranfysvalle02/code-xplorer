"""Central configuration, loaded from the environment with local-first defaults.

Nothing here is required to be set: the defaults target a local Ollama instance
and the bundled `docker compose` MongoDB Atlas container. Voyage and Azure are
optional and only activate when their keys are present.
"""
import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Config:
    # MongoDB
    mdb_uri: str = ""
    db_name: str = "code_search_db"
    collection_name: str = "code_snippets"

    # Ollama
    ollama_host: str = "http://localhost:11434"
    embedding_model: str = "ollama/qwen3-embedding:4b"
    llm_model: str = "qwen3.6:27b"
    llm_fast_model: str = "qwen3:14b"
    num_ctx_cap: int = 16384

    # Agent
    agent_max_iterations: int = 3

    # Voyage (optional)
    voyage_api_key: str = ""

    # Azure OpenAI (optional)
    azure_endpoint: str = ""
    azure_api_key: str = ""
    azure_deployment: str = "gpt-4o"
    azure_api_version: str = "2025-04-01-preview"

    # Flask
    flask_host: str = "127.0.0.1"
    flask_port: int = 5001

    @property
    def azure_enabled(self) -> bool:
        return bool(self.azure_endpoint and self.azure_api_key and self.azure_deployment)

    @property
    def voyage_enabled(self) -> bool:
        return bool(self.voyage_api_key)


def load_config() -> Config:
    """Build a Config from environment variables (call after load_dotenv())."""
    return Config(
        mdb_uri=os.getenv("MDB_URI", "mongodb://localhost:27018/?directConnection=true"),
        db_name=os.getenv("DB_NAME", "code_search_db"),
        collection_name=os.getenv("COLLECTION_NAME", "code_snippets"),
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "ollama/qwen3-embedding:4b"),
        llm_model=os.getenv("LLM_MODEL", "qwen3.6:27b"),
        llm_fast_model=os.getenv("LLM_FAST_MODEL", "qwen3:14b"),
        num_ctx_cap=_int("OLLAMA_NUM_CTX_CAP", 16384),
        agent_max_iterations=_int("AGENT_MAX_ITERATIONS", 3),
        voyage_api_key=os.getenv("VOYAGE_API_KEY", "") or "",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "") or "",
        azure_api_key=os.getenv("AZURE_OPENAI_API_KEY", "") or "",
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o") or "gpt-4o",
        azure_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
        flask_host=os.getenv("FLASK_HOST", "127.0.0.1"),
        flask_port=_int("FLASK_PORT", 5001),
    )

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    frontend_origin: str = "http://localhost:3000"
    model_checkpoint: str = "ml/artifacts/best_model.pt"
    class_mapping: str = "ml/artifacts/class_to_idx.json"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    llm_provider: str = "ollama"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    max_upload_mb: int = 10
    low_confidence_threshold: float = 0.55
    feedback_path: str = "data/feedback.jsonl"

settings = Settings()

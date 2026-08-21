from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Sentilytics"
    API_V1_STR: str = "/v1"
    
    # Security
    JWT_SECRET_KEY: str = "your-secret-key" # In production, set via env
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/sentiment_db"
    
    # Rate Limiting
    RATE_LIMIT_IP: str = "100/minute"
    RATE_LIMIT_USER: str = "1000/minute"
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # LLM & NLP Engine Configuration
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-1.5-flash"
    OLLAMA_BASE_URL: Optional[str] = None
    OLLAMA_MODEL: str = "llama3"
    LLM_PROVIDER: str = "auto"
    LLM_TIMEOUT_SECONDS: float = 10.0
    ENABLE_LLM_TIER: bool = True

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

settings = Settings()

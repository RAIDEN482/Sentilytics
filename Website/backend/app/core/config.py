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

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

settings = Settings()

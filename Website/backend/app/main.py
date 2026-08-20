from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from app.api.v1.router import api_router
from app.core.config import settings
import json
import os

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs"
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/v1/health", tags=["health"])
async def health_check():
    return {"status": "ok"}

@app.on_event("startup")
async def startup_event():
    # Only dump openapi if we are running locally / generating it
    # We dump it to root directory to be easily accessed by frontend tools
    openapi_schema = app.openapi()
    with open("openapi.json", "w") as f:
        json.dump(openapi_schema, f, indent=2)

    # Initialize Database & Seed Default User
    from app.core.database import engine, async_session_maker
    from app.models.base import Base
    from app import models # Crucial: registers all tables with Base.metadata without shadowing global app
    from app.models.user import User
    from app.core.security import get_password_hash
    from sqlalchemy.future import select
    import uuid
    import logging



    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing database tables: {e}")

    try:
        DEFAULT_USER_ID = uuid.UUID("4a3b2c1d-5e6f-7a8b-9c0d-1e2f3a4b5c6d")
        async with async_session_maker() as session:
            result = await session.execute(select(User).filter_by(email="test@test.com"))
            user = result.scalar_one_or_none()
            if not user:
                user = User(
                    id=DEFAULT_USER_ID,
                    email="test@test.com",
                    hashed_password=get_password_hash("test"),
                    tier="free",
                    monthly_analysis_limit=1000
                )
                session.add(user)
                await session.commit()
                logger.info("Default test user test@test.com seeded.")
            else:
                logger.info("Default test user already exists.")
    except Exception as e:
        logger.error(f"Error seeding test user: {e}")



import socket
from collections.abc import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from app.core.config import settings

def is_postgres_available(host: str = "localhost", port: int = 5432) -> bool:
    try:
        s = socket.create_connection((host, port), timeout=1)
        s.close()
        return True
    except OSError:
        return False

# Detect the database URL and fallback to SQLite if postgres is unreachable
db_url = settings.DATABASE_URL
if db_url.startswith("postgresql"):
    host = "localhost"
    port = 5432
    try:
        if "@" in db_url:
            authority = db_url.split("@")[1].split("/")[0]
            if ":" in authority:
                host, port_str = authority.split(":")
                port = int(port_str)
            else:
                host = authority
    except Exception:
        pass
    
    if not is_postgres_available(host, port):
        print(f"WARNING: PostgreSQL at {host}:{port} is unreachable. Falling back to SQLite.")
        db_url = "sqlite+aiosqlite:///./sentiment_db.db"

if "sqlite" in db_url:
    engine = create_async_engine(
        db_url,
        echo=False,
        future=True,
    )
else:
    engine = create_async_engine(
        db_url,
        echo=False,
        future=True,
        pool_size=5,
        max_overflow=10,
    )

async_session_maker = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session


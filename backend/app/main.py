from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import logger
from app.database.session import Base, engine, check_db_connection
from app.database.redis import init_redis, close_redis, check_redis_connection
from app.api.v1.api import api_router
from app.schemas.health import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.PROJECT_NAME} backend...")

    # Initialize Redis connection pool
    await init_redis()

    # Attempt automatic schema creation if PostgreSQL is available
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables initialized successfully.")
    except Exception as e:
        logger.warning(f"Could not automatically create DB tables on startup (DB may still be starting): {e}")

    yield

    logger.info("Shutting down backend...")
    await close_redis()


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# Set up CORS
if settings.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# Root Health Check as required by Phase 0 deliverable (GET /health)
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def root_health_check():
    """
    Deliverable GET /health endpoint returning:
    {
        "status": "running",
        "database": "connected",
        "redis": "connected"
    }
    """
    db_status = await check_db_connection()
    redis_status = await check_redis_connection()

    return HealthResponse(
        status="running",
        database=db_status,
        redis=redis_status
    )


# Include API v1 routes
app.include_router(api_router, prefix=settings.API_V1_STR)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

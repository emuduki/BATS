from fastapi import APIRouter
from app.schemas.health import HealthResponse
from app.database.session import check_db_connection
from app.database.redis import check_redis_connection

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def get_health():
    """
    Health check endpoint returning system operational status,
    PostgreSQL connection state, and Redis cache connection state.
    """
    db_status = await check_db_connection()
    redis_status = await check_redis_connection()

    return HealthResponse(
        status="running",
        database=db_status,
        redis=redis_status
    )

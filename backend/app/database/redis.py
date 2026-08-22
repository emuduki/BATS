from typing import Optional
import redis.asyncio as aioredis
from app.core.config import settings
from app.core.logging import logger

redis_client: Optional[aioredis.Redis] = None


async def init_redis() -> None:
    global redis_client
    try:
        redis_client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )
        logger.info("Redis client initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize Redis client: {e}")


async def close_redis() -> None:
    global redis_client
    if redis_client:
        await redis_client.close()
        logger.info("Redis connection closed.")


async def get_redis() -> Optional[aioredis.Redis]:
    return redis_client


async def check_redis_connection() -> str:
    """Dynamically checks if Redis server is reachable."""
    try:
        if redis_client is None:
            # Attempt one-off ping via temporary connection if client not initialized
            temp_client = aioredis.from_url(settings.REDIS_URL, socket_timeout=2)
            pong = await temp_client.ping()
            await temp_client.close()
            return "connected" if pong else "disconnected"

        pong = await redis_client.ping()
        return "connected" if pong else "disconnected"
    except Exception as e:
        logger.warning(f"Redis connection check failed: {e}")
        return "disconnected"

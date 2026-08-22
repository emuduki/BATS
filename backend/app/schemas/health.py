from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall FastAPI server status", example="running")
    database: str = Field(..., description="PostgreSQL connection status", example="connected")
    redis: str = Field(..., description="Redis cache connection status", example="connected")

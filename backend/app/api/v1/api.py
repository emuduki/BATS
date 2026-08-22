from fastapi import APIRouter
from app.api.v1 import health, signals, market, simulator

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(signals.router)
api_router.include_router(market.router)
api_router.include_router(simulator.router)

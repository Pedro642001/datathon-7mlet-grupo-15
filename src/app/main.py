from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from src.app.controllers.recommend import router as recommend_router
from src.app.services.model_service_wrapper import ModelService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('Startup: initializing ModelService')
    # instantiate singleton to load model/scaler/encoders
    ModelService.get_instance()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title='Offer Recommender API',
        version='v1',
        docs_url='/docs',
        redoc_url='/redoc',
        lifespan=lifespan,
    )

    # Add CORS (allow local testing)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(recommend_router)

    return app

app = create_app()

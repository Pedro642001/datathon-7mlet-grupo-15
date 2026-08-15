from fastapi import APIRouter, Depends, HTTPException
from typing import Dict
import logging

from src.app.schemas.schemas import FeaturesPayload, RecommendResponse
from src.app.services.model_service_wrapper import ModelService

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get('/', tags=['root'])
async def root():
    return {'status': 'ok', 'message': 'Offer Recommender API', 'version': 'v1'}

@router.post('/api/v1/recommend', response_model=RecommendResponse, tags=['recommend'])
async def recommend(payload: FeaturesPayload, svc: ModelService = Depends(ModelService.get_instance)):
    # Ensure service ready
    if not svc.ready:
        raise HTTPException(status_code=503, detail='Service initializing')
    try:
        rec, confidence = svc.recommend(payload.features)
        return RecommendResponse(recommended=rec, confidence=confidence, feature_names_expected=svc.feature_names)
    except Exception as e:
        logger.exception('Recommendation error: %s', e)
        raise HTTPException(status_code=500, detail='Internal recommendation error')

@router.get('/health', tags=['health'])
async def health(svc: ModelService = Depends(ModelService.get_instance)):
    return {'status': 'ready' if svc.ready else 'initializing'}

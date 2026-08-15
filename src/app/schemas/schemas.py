from pydantic import BaseModel, Field
from typing import Dict, List, Union

class FeaturesPayload(BaseModel):
    # Accept numeric or raw categorical strings per feature
    features: Dict[str, Union[str, float]] = Field(..., description='Feature name -> numeric value or raw category string')

class RecommendResponse(BaseModel):
    recommended: int
    confidence: float
    feature_names_expected: List[str]

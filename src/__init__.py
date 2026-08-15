"""src package for Offer Recommender project.

Structure:
- data_preparation.py: ETL pipeline (creates data/processed/{encoders.json,feature_names.json,scaler.pkl,train_clean.csv,test_clean.csv})
- recommender.py: Baseline and ThompsonSampler (contextual) + OfferRecommender wrapper
- train.py: training pipeline + MLflow logging, saves models/*.json
- utils.py: compatibility wrappers delegating to ModelService
- app/: modular FastAPI application (main, controllers, schemas, services/model_service_wrapper.py)

Use src.app.main:app for the FastAPI entrypoint.
"""

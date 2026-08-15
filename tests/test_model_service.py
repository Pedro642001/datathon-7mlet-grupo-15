from src.app.services.model_service_wrapper import ModelService


def test_model_service_recommend():
    svc = ModelService.get_instance()
    assert svc.ready, "ModelService should be ready"
    sample = {
        "age": 35,
        "job": "admin.",
        "marital": "married",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "mon",
        "campaign": 1,
        "pdays": 999,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp.var.rate": 1.1,
        "cons.price.idx": 93.994,
        "cons.conf.idx": -36.4,
        "euribor3m": 4.857,
        "nr.employed": 5191.0,
    }

    rec, conf = svc.recommend(sample)
    assert isinstance(rec, int)
    assert isinstance(conf, float)
    assert 0.0 <= conf <= 1.0

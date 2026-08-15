from src.utils import load_model_and_scaler, prepare_features


def test_utils_wrappers():
    model, scaler, feature_names = load_model_and_scaler()
    assert model is not None
    assert scaler is not None
    assert isinstance(feature_names, list)
    # prepare features uses the loaded service
    row = {feature_names[0]: 1.0}
    df = prepare_features(row)
    assert df.shape[1] == len(feature_names)

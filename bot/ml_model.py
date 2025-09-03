import joblib
import xgboost as xgb

def load_model(path="model/xgb_model.joblib"):
    return joblib.load(path)

def predict(model, X):
    dtest = xgb.DMatrix(X)
    return model.predict(dtest)

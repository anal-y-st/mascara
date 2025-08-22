import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def evaluate_numpy(y_true, y_pred):
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return dict(r2=r2, mae=mae, rmse=rmse)

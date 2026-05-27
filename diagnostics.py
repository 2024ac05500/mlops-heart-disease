import joblib
import numpy as np
import pandas as pd
import traceback

M = joblib.load('models/model.joblib')
print('MODEL TYPE:', type(M))
print('HAS predict:', hasattr(M, 'predict'))
print('HAS predict_proba:', hasattr(M, 'predict_proba'))
print('HAS decision_function:', hasattr(M, 'decision_function'))

arr = np.array([63,1,1,145,233,1,2,150,0,2.3,3,0,6]).reshape(1, -1)
try:
    print('PRED (np):', M.predict(arr))
except Exception:
    print('PRED (np) FAILED')
    traceback.print_exc()

try:
    df = pd.DataFrame(arr)
    print('PRED (df):', M.predict(df))
except Exception:
    print('PRED (df) FAILED')
    traceback.print_exc()

try:
    if hasattr(M, 'predict_proba'):
        print('PROBA:', M.predict_proba(arr))
except Exception:
    traceback.print_exc()

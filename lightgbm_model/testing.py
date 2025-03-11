import pandas as pd

import joblib

model = joblib.load(r'C:\Users\Administrator\work\credit_card_elig\core\credit_card_approval\lightgbm_model\lightgbm_credit_card_approval.pkl')
print('Model loaded successfully!')

X_test = pd.read_csv(r'')
y_test = pd.read_csv(r'')

if isinstance(y_test,pd.DataFrame):
    y_test = y_test.squeeze()
    
y_pred_proba = model.predict(X_test)

print('Prediction done!')
y_pred = (y_pred_proba >= 0.5).astype(int)
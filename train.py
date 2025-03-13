import xgboost as xgb
from sklearn.metrics import roc_auc_score,accuracy_score,balanced_accuracy_score,classification_report

from datetime import datetime
import joblib

from xgboost_model.optuna_optimization import X_train_smote,X_test_smote,y_train,y_test

timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

params = {
    'learning_rate': 0.1040275387678505, 
    'n_estimators': 339,
    'max_depth': 4,
    'min_child_weight': 7, 
    'gamma': 4.09616589780161,
    'subsample': 0.8628345990575292, 
    'colsample_bytree': 0.537160980152967, 
    'reg_lambda': 0.7535887845167186, 
    'reg_alpha': 0.006751151129803729,
    'scale_pos_weight': 52.63966167633958
    }


model = xgb.XGBClassifier(**params)
model.fit(X_train_smote,y_train)
y_pred_proba = model.predict(X_test_smote)
y_pred = (y_pred_proba >= 0.5).astype(int)

# print(full_data['Family Status'].value_counts())
print(model.score(X_test_smote,y_test))
# print(xgb_model.get_params())

auc = roc_auc_score(y_test,y_pred_proba)
accuracy = accuracy_score(y_test,y_pred)

balanced_acc = balanced_accuracy_score(y_test, y_pred)
print(f"Balanced Accuracy: {balanced_acc:.4f}")

class_report = classification_report(y_test, y_pred)

print(f"XGBoost AUC-ROC: {auc:.4f}")
print(f"XGBoost Accuracy: {accuracy:.4f}")
print("XGBoost Classification Report:\n", classification_report(y_test, y_pred))

logfile = r'C:\Users\Administrator\work\credit_card_elig\core\credit_card_approval\metrics\XGBoost_performance.log'
log_entry = f"""
Timestamp: {timestamp}
Model: XGBoost
Best Hyperparameters: {params}
Best AUC score on training: {auc:.6f}

Test Performance:
AUC-ROC Score: {auc:.6f}
Accuracy: {accuracy:.6f}
Classification Report:
{class_report}
"""

with open(logfile,'a') as f:
    f.write(log_entry)
    
joblib.dump(model,'xgboost_credit_card_approval.pkl')
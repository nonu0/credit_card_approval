from datetime import datetime
from core.credit_card_approval.helper_functions.preprocessing import full_data
from core.credit_card_approval.helper_functions.train_test_split import custom_train_test_split

from collections import Counter

import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score,classification_report, roc_auc_score,confusion_matrix,precision_recall_curve



X = full_data.drop(columns=['Is High Risk'])
y = full_data['Is High Risk']

X_train,X_test,y_train,y_test = custom_train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

train_data = lgb.Dataset(X_train,label=y_train)
test_data = lgb.Dataset(X_test,y_test)


params = {
    'objective':'binary',
    'metric':'auc',
    'boosting_type':'gbdt',
    'learning_rate': 0.06652923282198506,
    'num_leaves': 130, 
    'max_depth': 10,
    'min_child_samples': 14,
    'min_split_gain': 0.18969158967598235,
    'colsample_bytree': 0.9535532123159376, 
    'subsample': 0.27456705520848657, 
    'reg_alpha': 0.7404069484338288, 
    'reg_lambda': 0.8874890838479117, 
    'scale_pos_weight': 40.73474668908436,
    'random_state':42,
    }


model = lgb.train(params=params,train_set=train_data,valid_sets=test_data,num_boost_round=100)

y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)
 # Compute precision-recall curve
# precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

# # # Find best threshold to maximize precision + recall
# best_threshold = thresholds[np.argmax(precision + recall)]

# # # Apply new threshold
# y_pred = (y_pred_proba > best_threshold).astype(int)
# print(best_threshold)
timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
auc_score = roc_auc_score(y_test, y_pred_proba)
conf_matrix = confusion_matrix(y_test, y_pred)
acc_score = accuracy_score(y_test, y_pred)
auc_score = auc_score
class_report = classification_report(y_test, y_pred)

# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("Accuracy Score:", accuracy_score(y_test, y_pred))
# print("ROC-AUC Score:", auc_score)
# print("Classification Report:\n", classification_report(y_test, y_pred))

logfile = r'C:\Users\Administrator\work\credit_card_elig\core\credit_card_approval\metrics\lightGBM_performance.log'
log_entry = f"""
Timestamp: {timestamp}
Model: LightGBM
Best Hyperparameters: {params}
Best AUC score on training: {auc_score:.6f}

Test Performance:
AUC-ROC Score: {auc_score:.6f}
Accuracy: {acc_score:.6f}
Classification Report:
{class_report}
"""
# print(log_entry)
with open(logfile,'a') as f:
    f.write(log_entry)
    
print(f'Model performance logged to {logfile} successfully!')
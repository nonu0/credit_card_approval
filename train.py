from preprocessing import full_data
from train_test_split import custom_train_test_split

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
    'learning_rate':0.08930127645584529,
    'num_leaves':64,
    'max_depth':11,
    'min_child_samples':46,
    'min_split_gain':0.2510859852252205,
    'colsample_bytree':0.9131142068386219,
    'subsample': 0.3793325893079907,
    'reg_alpha': 0.376758550961793, 
    'reg_lambda': 0.1202772045279857,
    'scale_pos_weight': 15.40489260342057,
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


print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))
print("Classification Report:\n", classification_report(y_test, y_pred))
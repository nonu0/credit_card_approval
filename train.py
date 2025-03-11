from helper_functions.preprocessing import full_data
from helper_functions.train_test_split import custom_train_test_split

from collections import Counter

import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score,classification_report,balanced_accuracy_score,cohen_kappa_score,log_loss
from imblearn.over_sampling import SMOTE


full_data.drop(columns=['ID'],inplace=True)

label_encode_cols = ['FLAG_OWN_CAR', 'Job', 'Education', 'FLAG_OWN_REALTY']
onehot_encode_cols = ['Gender', 'Housing', 'Occupation', 'Family Status']

X = full_data.drop(columns=['Is High Risk'])
y = full_data['Is High Risk']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

label_encoders = {}
for col in label_encode_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.fit_transform(X_test[col])
    label_encoders[col] = le
    
X_train = pd.get_dummies(X_train,columns=onehot_encode_cols,drop_first=True)
X_test = pd.get_dummies(X_test,columns=onehot_encode_cols,drop_first=True)
X_test = X_test.reindex(columns=X_train.columns,fill_value=0)
# print(X_train.columns)
# print(X_test)

xgb_model = xgb.XGBClassifier(
    learning_rate=0.089,
    max_depth=11,
    n_estimators=100,
    subsample=0.38,
    colsample_bytree=0.91,
    reg_alpha=0.37,
    reg_lambda=0.12,
    scale_pos_weight=15.40,
    # use_label_encoder=False,
    eval_metric="auc"
)

smote = SMOTE(random_state=42)
X_train_res,y_train_res = smote.fit_resample(X_train,y_train)

xgb_model.fit(X_train,y_train)
# xgb_model.fit(X_train_res,y_train_res)
y_pred_proba = xgb_model.predict_proba(X_test)[:,1]
# print(y_pred_proba[:50])
y_pred = (y_pred_proba >= 0.5).astype(int)
# print(xgb_model.score(X_test,y_test))
# print(xgb_model.get_params())

# auc = roc_auc_score(y_test,y_pred_proba)
accuracy = accuracy_score(y_test,y_pred)

# balanced_acc = balanced_accuracy_score(y_test, y_pred)
# print(f"Balanced Accuracy: {balanced_acc:.4f}")


# print(f"XGBoost AUC-ROC: {auc:.4f}")
# print(f"XGBoost Accuracy: {accuracy:.4f}")
# print("XGBoost Classification Report:\n", classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
import numpy as np

feat_imp = xgb_model.feature_importances_
feature_names = X_train.columns
sorted_idx = np.argsort(feat_imp)
print(full_data['Occupation'].value_counts())

plt.figure(figsize=(10,6))
plt.barh(feature_names[sorted_idx],feat_imp[sorted_idx],color='blue')
plt.xlabel('Feature importance')
plt.ylabel('Feature names')
# plt.show()

# kappa = cohen_kappa_score(y_test, y_pred)
# print(f"Cohen’s Kappa Score: {kappa:.4f}")

# logloss = log_loss(y_test, y_pred_proba)
# print(f"Log Loss: {logloss:.4f}")

# # print('original',(X_train))


# # print("Resampled class distribution:", (x_train_smote))

from helper_functions.preprocessing import full_data

import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score,roc_auc_score,balanced_accuracy_score

education_order = ['Lower secondary','Secondary ','Incomplete higher' , 'Higher education','Academic degree']
family_status_order = ['Separated', 'Single ', 'Widow','Civil marriage', 'Married']

label_encoded_cols = ['FLAG_OWN_CAR','FLAG_OWN_REALTY']
onehot_encoded_cols = ['Gender','Job','Occupation','Housing']

full_data.drop(columns=['ID','FLAG_MOBIL'],inplace=True)

X = full_data.drop(columns=['Is High Risk'])
y = full_data['Is High Risk']



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

ordinal_encoder = OrdinalEncoder(categories=[education_order,family_status_order])
X_train[['Education','Family Status']] = ordinal_encoder.fit_transform(X_train[['Education','Family Status']])

label_encoders = {}
for col in label_encoded_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    label_encoders[col] = le
    # print(label_encoders)
    
X_train_smote = pd.get_dummies(X_train,columns=onehot_encoded_cols,drop_first=True)
X_test_smote = pd.get_dummies(X_test,columns=onehot_encoded_cols,drop_first=True)
X_test_smote.reindex(columns=X_train_smote.columns,fill_value=0)
    
def objective(trial):
    params = {
         "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 20.0),  # Handle imbalance
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train_smote,y_train)
    y_pred_proba = model.predict(X_test_smote)
    auc = roc_auc_score(y_test,y_pred_proba)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    # print(y_pred_proba)
    return auc

study = optuna.create_study(direction='maximize')
study.optimize(objective,n_trials=50)

best_params = study.best_params
auc_score = study.best_value
print(best_params)
print(auc_score)
# Xtrain = le.fit_transform(X_train[[label_encoded_cols]])
# print(X_train)
# X_train[label_encoded_cols] = X_train[label_encoded_cols].apply(le.fit_transform)

# print(ordinal_encoder.categories_)
# print(full_data)

# Print the mapping of categories to numerical values
# for col, categories in zip(['Education', 'Family Status'], ordinal_encoder.categories_):
#     print(f"Mapping for {col}:")
#     for i, category in enumerate(categories):
#         print(f"  {category} → {i}")
#     print()

# for col,categories in zip(['Education','Family Status'],ordinal_encoder.categories_):
#     print(f'Mapping for {col}')
#     for i,category in enumerate(categories):
#         print(i)
#         print(category)

# ed_map = {}

# for i,edu_type in enumerate(education_order):
#     ed_map[edu_type] = i
#     print(ed_map)

# print(X_train)


# import matplotlib.pyplot as plt
# import numpy as np

# feat_imp = xgb_model.feature_importances_
# feature_names = X_train.columns
# sorted_idx = np.argsort(feat_imp)

# plt.figure(figsize=(10,6))
# plt.barh(feature_names[sorted_idx],feat_imp[sorted_idx],color='blue')
# plt.xlabel('Feature importance')
# plt.ylabel('Feature names')
# plt.show()

# kappa = cohen_kappa_score(y_test, y_pred)
# print(f"Cohen’s Kappa Score: {kappa:.4f}")

# logloss = log_loss(y_test, y_pred_proba)
# print(f"Log Loss: {logloss:.4f}")

# # print('original',(X_train))


# # print("Resampled class distribution:", (x_train_smote))

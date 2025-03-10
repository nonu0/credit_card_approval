import optuna
import lightgbm as lgb
from sklearn.metrics import roc_auc_score,accuracy_score

from datetime import datetime
from collections import Counter

from preprocessing import full_data
from train import train_data,test_data,X_train,X_test,y_train,y_test

class_counts = Counter(y_train)




def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.3),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'subsample': trial.suggest_float('subsample', 0.1, 1.0),  
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 100.0),  
        'random_state': 42
    }

    model = lgb.train(
        params,
        lgb.Dataset(X_train, label=y_train),
        valid_sets=[lgb.Dataset(X_test, label=y_test)],
        num_boost_round=1000,
        # early_stopping_rounds=30,
        # verbose_eval=False
    )

    y_pred_proba = model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred_proba)

    return auc

# Run Optuna optimization
study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=50)
best_params = study.best_params
Auc_score = study.best_value
accuracy = accuracy_score(y_test,y_pred)
# Get best hyperparameters
# print("Best params:", best_params)
# print("Best AUC score:", study.best_value)


# log_filename = 'lightGBM_performance.log'
# timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# log_entry = f"""
# TimeStamp: {timestamp}
# Model: LightGBM
# Best HyperParameters: {best_params}
# Best AUC Score on training: {auc}

# Test Performance:
# AUC-ROC Score:{Auc_score:.6f}
# Accuracy: {accuracy:.6f}
# Classification Report:
# {classificarion_report(y_test,y_pred)}
# """
# # print(log_entry)
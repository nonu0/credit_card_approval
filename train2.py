
from train_test_split import custom_train_test_split
from sklearn.metrics import precision_recall_curve
from collections import Counter
import lightgbm as lgb
import numpy as np

from preprocessing import full_data

X = full_data.drop(columns=['Is High Risk'])
y = full_data['Is High Risk']
# print(y)

X_train,X_test,y_train,y_test = custom_train_test_split(X=X,y=y,test_size=0.2,random_state=42,stratify=y)

class_counter = Counter()
# print(X_train)
# print(len(X_train))
# print(y_train)
# print(len(y_test))
# print(y_test)
# print(len(y_train))

# # Split dataset using stratified sampling
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# # Compute class weights
# class_counts = Counter(y_train)
# scale_pos_weight = class_counts[0] / class_counts[1]

# # Cap the class weight to avoid overfitting
# adjusted_scale_pos_weight = min(scale_pos_weight, 30)

# # Train LightGBM with adjusted class weights
# model = lgb.LGBMClassifier(scale_pos_weight=adjusted_scale_pos_weight, random_state=42)
# model.fit(X_train, y_train)

# # Get prediction probabilities
# y_pred_proba = model.predict_proba(X_test)[:, 1]

# # Compute precision-recall curve
# precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

# # Find best threshold to maximize precision + recall
# best_threshold = thresholds[np.argmax(precision + recall)]

# # Apply new threshold
# y_pred = (y_pred_proba > best_threshold).astype(int)

# cm = confusion_matrix(y_test,y_pred)


# print('Confusion matrix:\n',cm)
# accuracy = accuracy_score(y_test,y_pred)
# print('Accuracy score:\n',accuracy)
# roc_auc = roc_auc_score(y_test,y_pred_proba)
# print('Roc-Auc Score:\n',roc_auc)

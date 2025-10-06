import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

df = pd.read_csv("creditcard.csv")

print("Dataset shape:", df.shape)
print("Fraud cases:", df['Class'].sum())
print("Non-Fraud cases:", len(df) - df['Class'].sum())

scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df[['Amount']])
df['scaled_time'] = scaler.fit_transform(df[['Time']])
df = df.drop(['Amount', 'Time'], axis=1)

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

over = SMOTE(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.5)
resample_pipeline = Pipeline([('over', over), ('under', under)])

X_res, y_res = resample_pipeline.fit_resample(X_train, y_train)
print("Resampled shape:", np.bincount(y_res))

log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
log_reg.fit(X_res, y_res)
y_pred_lr = log_reg.predict(X_test)

print("\nLogistic Regression")
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
print("ROC AUC:", roc_auc_score(y_test, log_reg.predict_proba(X_test)[:,1]))

rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight="balanced")
rf.fit(X_res, y_res)
y_pred_rf = rf.predict(X_test)

print("\nRandom Forest")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("ROC AUC:", roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]))

xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb.fit(X_res, y_res)
y_pred_xgb = xgb.predict(X_test)

print("\nXGBoost")
print(confusion_matrix(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))
print("ROC AUC:", roc_auc_score(y_test, xgb.predict_proba(X_test)[:,1]))

plt.figure(figsize=(8,6))
for name, model in {"LogReg": log_reg, "RandomForest": rf, "XGBoost": xgb}.items():
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, model.predict_proba(X_test)[:,1]):.3f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
for name, model in {"LogReg": log_reg, "RandomForest": rf, "XGBoost": xgb}.items():
    prec, rec, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:,1])
    plt.plot(rec, prec, label=name)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves")
plt.legend()
plt.show()

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_res)
plt.figure(figsize=(7,6))
sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1], hue=y_res, palette="coolwarm", alpha=0.5)
plt.title("PCA: Fraud vs Non-Fraud")
plt.show()

tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
tsne_result = tsne.fit_transform(X_res[:5000])
plt.figure(figsize=(7,6))
sns.scatterplot(x=tsne_result[:,0], y=tsne_result[:,1], hue=y_res[:5000], palette="coolwarm", alpha=0.5)
plt.title("t-SNE: Fraud vs Non-Fraud")
plt.show()

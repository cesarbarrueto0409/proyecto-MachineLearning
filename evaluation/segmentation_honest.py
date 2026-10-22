"""Honest re-evaluation of the segmentation project.

Same cleaning as the notebook, then:
  * K-Means (k = 6) on standardized variables, with silhouette scores for k = 2..8.
  * KNN that predicts a big wine buyer (wine spend above the median) WITHOUT wine spend
    or the cluster label among its inputs, scored with 5-fold stratified cross-validation.

Run from the repository root:  python evaluation/segmentation_honest.py
Requires pandas, numpy and scikit-learn. Writes only aggregated statistics (no customer rows)
to evaluation/summary.json.
"""
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

df = pd.read_csv('data-raw/marketing_campaign.csv', sep='\t')  # CSV del repo proyecto-MachineLearning
print('raw rows', len(df))
df = df.drop(columns=['ID', 'Dt_Customer'])
df = df.dropna(subset=['Income'])
df = df[df['Income'] < 100000]
df = df[df['Year_Birth'] > 1940]
df['Edades'] = 2024 - df['Year_Birth']
df = df.drop(columns=['Year_Birth'])
df['N_hijos'] = df['Kidhome'] + df['Teenhome']
df = df.drop(columns=['Kidhome', 'Teenhome'])
df['Educacion'] = pd.Categorical(df['Education']).codes
df = df[~df['Marital_Status'].isin(['Alone', 'Absurd', 'YOLO'])].copy()
df['Estado_Civil'] = pd.Categorical(df['Marital_Status']).codes
df = df.drop(columns=['Education', 'Marital_Status'])
df = df[df['MntMeatProducts'] <= 1500].copy()
print('clean rows', len(df))

feats = ['Income','Recency','MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds','NumDealsPurchases','NumWebPurchases','NumCatalogPurchases','NumStorePurchases','NumWebVisitsMonth','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','AcceptedCmp1','AcceptedCmp2','Complain','Z_CostContact','Z_Revenue','Response','Edades','N_hijos','Educacion','Estado_Civil']
X_raw = df[feats].to_numpy(dtype=float)

out = {}
# Notebook-style K-Means (no scaling), k=6
km_raw = KMeans(n_clusters=6, n_init=10, random_state=42).fit(X_raw)
# Scaled K-Means, k=6 (recommended)
Xs = StandardScaler().fit_transform(X_raw)
km = KMeans(n_clusters=6, n_init=10, random_state=42).fit(Xs)
labels = km.labels_
print('silhouette scaled k=6', round(silhouette_score(Xs, labels), 3))
print('silhouette raw k=6', round(silhouette_score(X_raw, km_raw.labels_), 3))
sil_by_k = {}
for k in range(2, 9):
    lab = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(Xs)
    sil_by_k[k] = round(float(silhouette_score(Xs, lab)), 3)
print('silhouette by k (scaled):', sil_by_k)

df['cluster'] = labels
wine_idx = int(df.groupby('cluster')['MntWines'].mean().idxmax())
print('cluster with highest mean wine:', wine_idx)
summary = df.groupby('cluster')[['MntWines','MntMeatProducts','MntFruits','MntFishProducts','MntSweetProducts','MntGoldProds','Income','Edades','N_hijos']].mean().round(1)
sizes = (df['cluster'].value_counts(normalize=True).sort_index()*100).round(1)
print(summary.to_string())
print('share %', sizes.to_dict())

# Honest KNN, task B: predict high wine buyer from NON-wine features
high_wine = (df['MntWines'] > df['MntWines'].median()).astype(int)
drop_b = ['MntWines','cluster']
Xb = df.drop(columns=drop_b).to_numpy(dtype=float)
Xb = StandardScaler().fit_transform(Xb)
yb = high_wine.to_numpy()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
acc_by_k = {}
for k in [1,3,5,7,9,11,15,21]:
    acc_by_k[k] = round(float(cross_val_score(KNeighborsClassifier(n_neighbors=k), Xb, yb, cv=skf).mean()), 3)
print('task B cv accuracy by k', acc_by_k, 'baseline', round(max(yb.mean(), 1-yb.mean()),3))

# Task A (reproduce notebook target): cluster-wine membership from non-leaking features
ya = (df['cluster'] == wine_idx).astype(int).to_numpy()
drop_a = ['MntWines','cluster']
Xa = StandardScaler().fit_transform(df.drop(columns=drop_a).to_numpy(dtype=float))
acc_a = {}
for k in [1,3,5,7,9,11,15,21]:
    acc_a[k] = round(float(cross_val_score(KNeighborsClassifier(n_neighbors=k), Xa, ya, cv=skf).mean()), 3)
print('task A cv accuracy by k', acc_a, 'baseline', round(max(ya.mean(), 1-ya.mean()),3))

best_k_b = max(acc_by_k, key=acc_by_k.get)
Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=0.3, random_state=11, stratify=yb)
m = KNeighborsClassifier(n_neighbors=best_k_b).fit(Xtr, ytr)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yte, m.predict(Xte)).tolist()
print('holdout cm (task B, best k)', best_k_b, cm, 'acc', round(m.score(Xte, yte),3))

# aggregated stats (no rows)
stats = {
 'n_rows': int(len(df)),
 'features_kmeans': feats,
 'silhouette_scaled_k6': round(float(silhouette_score(Xs, labels)),3),
 'silhouette_by_k': sil_by_k,
 'clusters': [
   {'id': int(c), 'share_pct': float(sizes[c]), 'is_wine_cluster': bool(c==wine_idx),
    'means': {col: float(summary.loc[c,col]) for col in summary.columns}}
   for c in sorted(df['cluster'].unique())
 ],
 'knn_task_B_high_wine_buyer': {'cv_accuracy_by_k': acc_by_k, 'baseline_majority': round(float(max(yb.mean(),1-yb.mean())),3), 'best_k': int(best_k_b), 'holdout_confusion_matrix': cm, 'holdout_accuracy': round(float(m.score(Xte,yte)),3)},
 'knn_task_A_reproduces_cluster': {'cv_accuracy_by_k': acc_a, 'baseline_majority': round(float(max(ya.mean(),1-ya.mean())),3)},
}
json.dump(stats, open('evaluation/summary.json','w'), ensure_ascii=False, indent=2)
print('saved')

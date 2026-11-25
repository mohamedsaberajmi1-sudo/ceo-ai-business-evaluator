"""Build notebooks/ml_report.ipynb (temporary helper, not part of the app)."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
nb = nbf.v4.new_notebook()
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# CEO AI — Rapport ML du projet

**Cours :** Machine Learning  
**Projet :** évaluation d'idées de business et analyse de projets existants  
**Auteur :** étudiant(e) — à compléter  
**Date :** 2026

Ce notebook présente :

1. le jeu de données **synthétique documenté** (6 000 idées) ;
2. l'analyse exploratoire (EDA) ;
3. la comparaison des classifieurs et régresseurs ;
4. les métriques de test, la matrice de confusion, la courbe ROC ;
5. l'importance des variables et SHAP ;
6. la segmentation KMeans ;
7. les conclusions et limites.

> Important : le jeu de données fourni est **synthétique**. Il est généré à partir
> d'hypothèses sectorielles documentées dans `configs/markets.yaml` (marges,
> concurrence, intensité capitalistique, croissance, risque réglementaire...).
> Il ne s'agit pas de données d'entreprises réelles. Le code accepte aussi un vrai
> dataset CSV/XLSX placé dans `data/` avec le même schéma.
""")

code("""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path.cwd()
if not (ROOT / "models").exists():
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (10, 5)
print("Racine du projet :", ROOT)
""")

md("""
## 1. Chargement et contrôle du dataset

Le schéma cible contient 21 variables explicatives et 3 cibles :

- `success` : classification binaire ;
- `roi_percent` : régression ;
- `breakeven_months` : régression.
""")

code("""
df = pd.read_csv(ROOT / "data" / "synthetic_businesses.csv")
print(f"Dimensions : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
print(f"Taux de succès : {df['success'].mean():.1%}")
display(df.head(3))
display(df.dtypes.to_frame("dtype"))
display(df.describe(include="all").T.head(30))
""")

md("""
### 1.1 Valeurs manquantes

Quelques valeurs manquantes ont été injectées volontairement (budget, taille
d'équipe, expérience du fondateur, marketing) pour rendre l'imputation du
pipeline réellement utile.
""")

code("""
missing = df.isna().sum().sort_values(ascending=False)
missing = missing[missing > 0]
display(missing.to_frame("valeurs_manquantes"))
""")

md("## 2. Analyse exploratoire (EDA)")

code("""
sector_success = (
    df.groupby("sector")["success"]
    .agg(["mean", "count"])
    .sort_values("mean", ascending=False)
)
fig, ax = plt.subplots(figsize=(11, 5))
sns.barplot(x=sector_success.index, y=sector_success["mean"], ax=ax, color="#0B5FFF")
ax.set_title("Taux de succès par secteur (dataset synthétique)")
ax.set_ylabel("Taux de succès")
ax.set_xlabel("Secteur")
ax.set_ylim(0, 1)
ax.tick_params(axis="x", rotation=35)
for i, value in enumerate(sector_success["mean"]):
    ax.text(i, value + 0.015, f"{value:.0%}", ha="center", fontsize=8)
plt.tight_layout()
plt.show()
display(sector_success.round(3))
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.histplot(df["roi_percent"].clip(-100, 400), bins=40, ax=axes[0], color="#0B5FFF")
axes[0].set_title("Distribution du ROI (%) — tronquée à 400%")
sns.histplot(df["breakeven_months"], bins=35, ax=axes[1], color="#2F855A")
axes[1].set_title("Distribution du seuil de rentabilité (mois)")
plt.tight_layout()
plt.show()
""")

code("""
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ("idea_id",)]
corr = df[numeric_cols].corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(13, 9))
sns.heatmap(corr, cmap="RdBu_r", center=0, annot=False, ax=ax)
ax.set_title("Matrice de corrélation des variables numériques")
plt.tight_layout()
plt.show()

print("Corrélations avec success :")
display(corr["success"].drop("success").sort_values(key=abs, ascending=False).head(12).to_frame("corrélation"))
""")

md("""
## 3. Comparaison des modèles

Le pipeline scikit-learn enchaîne :

1. `SimpleImputer` (médiane / mode) ;
2. `StandardScaler` pour les variables numériques ;
3. `OneHotEncoder(handle_unknown="ignore")` pour les variables catégorielles ;
4. le modèle (régression logistique, Random Forest, Gradient Boosting, XGBoost).

La validation croisée est stratifiée pour la classification et standard pour la
régression. Le meilleur modèle de base est ensuite optimisé avec
`RandomizedSearchCV`.
""")

code("""
metrics = json.loads((ROOT / "models" / "metadata.json").read_text(encoding="utf-8"))
comparison = pd.read_csv(ROOT / "models" / "model_comparison.csv")

cls = comparison[comparison["kind"] == "classification"].copy()
cols = ["model", "accuracy_mean", "precision_mean", "recall_mean", "f1_mean", "roc_auc_mean", "roc_auc_std", "fit_seconds"]
display(cls[cols].sort_values("roc_auc_mean", ascending=False).round(4))
print("Meilleur classifieur :", metrics["classification"]["best_model"])
print("Métriques de test :", {k: round(v, 4) for k, v in metrics["classification"]["test_metrics"].items() if isinstance(v, (int, float))})
""")

code("""
reg = comparison[comparison["kind"] == "regression"].copy()
for target in ("roi_percent", "breakeven_months"):
    print(f"\\n### Cible : {target}")
    subset = reg[reg["target"] == target]
    display(subset[["model", "mae_mean", "rmse_mean", "r2_mean", "fit_seconds"]].sort_values("mae_mean").round(4))
    print("Métriques de test :", {k: round(v, 4) for k, v in metrics["regression"][target]["metrics"].items() if isinstance(v, (int, float))})
""")

md("""
## 4. Évaluation sur le jeu de test

### 4.1 Matrice de confusion et courbe ROC
""")

code("""
cm = pd.read_csv(ROOT / "models" / "confusion_matrix.csv", index_col=0)
fig, ax = plt.subplots(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
ax.set_title("Matrice de confusion — classifieur final")
ax.set_xlabel("Prédit")
ax.set_ylabel("Réel")
plt.tight_layout()
plt.show()
display(cm)
""")

code("""
roc = pd.read_csv(ROOT / "models" / "roc_curve.csv")
auc = metrics["classification"]["test_metrics"]["roc_auc"]
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(roc["fpr"], roc["tpr"], color="#0B5FFF", linewidth=2.2, label=f"AUC = {auc:.3f}")
ax.plot([0, 1], [0, 1], "--", color="#9CA3AF", label="Hasard")
ax.set_xlabel("Taux de faux positifs")
ax.set_ylabel("Taux de vrais positifs")
ax.set_title("Courbe ROC — jeu de test")
ax.legend()
plt.tight_layout()
plt.show()
""")

md("### 4.2 Régression : prédit vs réel")

code("""
for suffix, title in (("roi", "ROI (%)"), ("breakeven", "Seuil de rentabilité (mois)")):
    pred = pd.read_csv(ROOT / "models" / f"test_predictions_{suffix}.csv")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(pred["y_true"], pred["y_pred"], alpha=0.32, s=14, color="#0B5FFF")
    low = min(pred["y_true"].min(), pred["y_pred"].min())
    high = max(pred["y_true"].max(), pred["y_pred"].max())
    ax.plot([low, high], [low, high], "--", color="#D64545", label="Idéal")
    ax.set_xlabel("Valeur réelle")
    ax.set_ylabel("Prédiction")
    ax.set_title(f"Prédit vs réel — {title}")
    ax.legend()
    plt.tight_layout()
    plt.show()
""")

md("""
## 5. Interprétabilité

L'importance par permutation mesure la baisse de performance quand une variable
est mélangée. SHAP fournit en plus une explication locale : le chatbot peut donc
dire quels facteurs poussent le score vers le haut ou vers le bas.
""")

code("""
importance = pd.read_csv(ROOT / "models" / "feature_importance_classification.csv")
top = importance.head(14).sort_values("importance_mean")
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.barh(top["feature"], top["importance_mean"], color="#0B5FFF")
ax.set_title("Importance par permutation — classifieur")
ax.set_xlabel("Baisse moyenne de ROC-AUC")
plt.tight_layout()
plt.show()
display(importance.head(12).round(4))
""")

code("""
shap_path = ROOT / "models" / "shap_importance.csv"
if shap_path.exists():
    shap_df = pd.read_csv(shap_path).head(14).sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(shap_df["display"], shap_df["mean_abs_shap"], color="#2F855A")
    ax.set_title("Importance globale SHAP")
    ax.set_xlabel("|SHAP| moyen")
    plt.tight_layout()
    plt.show()
    display(shap_df.round(4))
else:
    print("SHAP non disponible : le rapport utilise l'importance par permutation.")
""")

md("## 6. Segmentation KMeans des idées")

code("""
kmeans = metrics["clustering"]
print("Nombre de segments :", kmeans["n_clusters"])
profiles = pd.DataFrame(kmeans["profiles"])
display(profiles.round(3))

fig, ax = plt.subplots(figsize=(10, 4.8))
plot_df = profiles.sort_values("roi_percent", ascending=False)
sns.barplot(data=plot_df, x="segment_name", y="roi_percent", ax=ax, color="#0B5FFF", errorbar=None)
ax.set_title("ROI moyen par segment KMeans")
ax.set_xlabel("")
ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.show()
""")

md("""
## 7. Conclusions

### Ce qui fonctionne

- Le pipeline est **reproductible** et sauvegardé avec `joblib`.
- La classification atteint un **ROC-AUC proche de 0,79** sur le test, avec un
  rappel et un F1 équilibrés. La régression ROI atteint un **R² proche de 0,81** ;
  le seuil de rentabilité un **R² proche de 0,75**.
- Les variables les plus utiles sont cohérentes avec la théorie business :
  différenciation, taille/croissance du marché, intensité concurrentielle,
  préparation de l'équipe et modèle de revenus.
- SHAP permet des explications locales, utilisées directement dans le chatbot.

### Limites

- Le dataset est **synthétique** : les relations apprises reflètent les hypothèses
  du générateur, pas nécessairement l'économie réelle.
- Le ROI et le seuil de rentabilité sont des estimations ; ils ne remplacent pas
  une étude de marché ou un conseil financier.
- Les données réelles, si elles sont fournies, doivent respecter le même schéma
  et peuvent nécessiter un recalibrage.
- Le succès est un concept bruité : même un bon modèle reste un outil d'aide à la
  décision, avec une incertitude explicite.

### Pistes d'amélioration

1. Remplacer le dataset synthétique par des données réelles (Kaggle, incubateurs).
2. Ajouter une calibration probabiliste (`CalibratedClassifierCV`, courbe de fiabilité).
3. Tester LightGBM/CatBoost et des modèles temporels plus avancés (Prophet, ARIMA).
4. Ajouter une validation temporelle pour les prévisions.
5. Mettre en place un suivi de dérive (data drift) en production.
""")

code("""
print("Résumé final")
print("-" * 60)
print(f"Classifieur : {metrics['classification']['best_model']}")
print(f"ROC-AUC test : {metrics['classification']['test_metrics']['roc_auc']:.3f}")
print(f"F1 test : {metrics['classification']['test_metrics']['f1']:.3f}")
for target in ("roi_percent", "breakeven_months"):
    m = metrics["regression"][target]["metrics"]
    print(f"{target} : MAE={m['mae']:.2f} | R2={m['r2']:.3f}")
print("Dataset : synthétique,", metrics["dataset"]["rows"], "lignes")
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}
output = ROOT / "notebooks" / "ml_report.ipynb"
nbf.write(nb, output)
print("Notebook written to", output)

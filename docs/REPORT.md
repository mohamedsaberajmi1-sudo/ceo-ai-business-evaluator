# Rapport — CEO AI : agent intelligent d'évaluation d'idées et de projets business

**Méthode :** Machine Learning  
**Établissement :** Institut Supérieur de l'Informatique (ISI) — Mahdia  
**Étudiant :** Mohamed Saber Ajmi  
**Année universitaire :** 2026 / 2027  

**Nom du professeur :** _________________________________________________

---

## 2. Introduction et problématique

La validation d'une idée de projet business est une décision complexe et incertaine. CEO AI est un agent intelligent qui évalue une idée de business ou diagnostique un projet existant.

**Problématique :** comment estimer la probabilité de succès et le ROI d'une idée avant le lancement ?

**Utilisateurs :** entrepreneurs, porteurs de projets, consultants, étudiants.

## 3. Objectifs et cahier des charges

Objectif général : construire un pipeline ML complet et un agent conversationnel.

- **Mode A — Idée :** conversation guidée, score de viabilité 0–100 (rubrique 12 critères + probabilité ML), probabilité de succès, scénarios financiers, SWOT, risques, plan d'action, what-if, export PDF.
- **Mode B — Projet existant :** import CSV/Excel, KPI, benchmarks, diagnostics, prévisions, actions, export PDF.

## 4. Données

- **Source :** jeu de données **synthétique**, généré (`src/data_generation.py`, seed 42), **6 000 lignes**.
- **Taux de succès :** 58,97 % ; ROI moyen : 49,42 % ; seuil de rentabilité moyen : 37,17 mois.
- **Schéma :** 21 variables explicatives, 3 cibles (`success`, `roi_percent`, `breakeven_months`), colonnes descriptives ignorées.

## 5. Méthodologie ML

- **Prétraitement :** ColumnTransformer (imputation médiane/mode, StandardScaler, OneHotEncoder).
- **Split :** entraînement 80 % / test 20 %, `random_state=42`, stratification.
- **Validation croisée :** K-fold stratifié 3 divisions (classification), K-fold 3 (régression).
- **Tuning :** `RandomizedSearchCV`.
- **Modèles testés :** classification (régression logistique, RF, Gradient Boosting, XGBoost) ; régression (Ridge, RF, Gradient Boosting, XGBoost).

## 6. Résultats

- **Classifieur final :** « Tuned Logistic Regression (baseline) » — C = 0,03 ; ROC-AUC test = 0,7900.
  - Accuracy = 0,7208 ; Precision = 0,7891 ; Recall = 0,7189 ; F1 = 0,7524 ; ROC-AUC = 0,7900.
  - Matrice de confusion : [[356, 136], [199, 509]].
- **Régression :** « Tuned XGBoost Regressor » — ROI : MAE = 54,62, RMSE = 85,46, R² = 0,811 ; Break-even : MAE = 7,42, RMSE = 10,30, R² = 0,749.

## 7. Architecture du système

Modules : `app.py` (Streamlit), `src/agent.py` (états), `src/scoring.py` (moteur), `src/nlg.py` (NLG), `src/storage.py` (SQLite), `src/report.py` (PDF).

## 8. Interface et fonctionnalités

Streamlit, 3 langues (EN/FR/AR), preset Tunisie, what-if, comparaison, upload CSV/Excel, export PDF.

## 9. Tests et validation

`python -m pytest -q` → **21 tests passed**. Scénario réel : idée SaaS évaluée → score global 87,4/100, probabilité de succès 95,2 %, ROI estimé -77,7 %.

## 10. Limites et perspectives

Données synthétiques, ROI estimé non financier, besoin de données réelles, améliorations futures.

## 11. Conclusion

## 12. Annexes

- Installation : `pip install -r requirements.txt`
- Entraînement : `python -m src.train`
- Application : `streamlit run app.py`
- Tests : `python -m pytest -q`

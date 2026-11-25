# CEO AI — Évaluation d'idées et analyse de projets

**Projet de cours Machine Learning** — chatbot multilingue (anglais, français, arabe tunisien) qui évalue une idée de business ou diagnostique une entreprise existante à partir de ses indicateurs mensuels.

> ⚠️ **Données synthétiques.** Le jeu de données fourni (`data/synthetic_businesses.csv`, 6 000 lignes) est **généré** à partir d'hypothèses sectorielles documentées dans `configs/markets.yaml`. Il ne représente pas des entreprises réelles. Le pipeline accepte aussi un vrai dataset CSV/XLSX de même schéma.

> ⚠️ **Avertissement.** CEO AI est un outil d'aide à la décision. Ses scores, prévisions et recommandations ne constituent pas un conseil financier, juridique ou commercial.

---

## Captures d'écran

> **Placeholder.** Les captures de l'interface seront ajoutées dans `docs/images/` lors de la publication. Écrans recommandés : page d'accueil du chat, rapport d'évaluation d'idée, analyse d'entreprise et comparaison de deux évaluations.

| Écran | Fichier suggéré |
|---|---|
| Accueil et conversation | `docs/images/ui_accueil.png` |
| Rapport d'évaluation d'idée | `docs/images/ui_rapport_idee.png` |
| Analyse d'entreprise | `docs/images/ui_analyse_entreprise.png` |
| Historique et comparaison | `docs/images/ui_comparaison.png` |

---

## 1. Fonctionnalités

### Mode A — Évaluer une idée
- Conversation guidée : secteur, marché cible, budget, prix, clients, différenciation, concurrence, localisation, etc.
- Score de viabilité 0–100 (rubrique transparente 12 critères + probabilité ML).
- Probabilité de succès, niveau de confiance et explication des facteurs positifs/négatifs (SHAP / importance des variables).
- Scénarios financiers pessimiste / réaliste / optimiste (revenu, profit, ROI, seuil de rentabilité), en TND ou USD selon le preset.
- Graphique radar, SWOT, risques, plan d'action priorisé, simulateur *what-if*.
- Export PDF du rapport complet.

### Mode B — Analyser une entreprise existante
- Import CSV/Excel d'un historique mensuel (revenu, coût, clients, marketing…) ou saisie manuelle d'un snapshot.
- Calcul des KPI : marge nette, croissance, churn, CAC, LTV, ratio LTV/CAC, panier moyen, ratio marketing.
- Détection de problèmes, comparaison aux benchmarks sectoriels, actions priorisées avec impact mensuel estimé.
- Prévision du revenu sur 6 mois (Holt-Winters / ARIMA, repli linéaire) avec intervalle de confiance.
- Export PDF.

### Transverse
- Interface Streamlit avec historique de chat, upload, graphiques Plotly et rechargement d'évaluations.
- Sauvegarde SQLite des rapports, rechargement et comparaison de 2 à 3 idées.
- LLM optionnel (Ollama / OpenAI / Anthropic) via `.env`; **repli hors ligne entièrement fonctionnel** à base de règles et de templates.
- Trois langues : `en`, `fr`, `ar` (arabe standard/tunisien en écriture arabe ; le parseur accepte aussi le tunisien en lettres latines).
- Pied de page localisé, identique à l'écran et dans le PDF :
  - EN : `CEO AI is AI and can make mistakes.`
  - FR : `CEO AI est une IA et peut faire des erreurs.`
  - AR : `CEO AI ذكاء اصطناعي وقد يرتكب أخطاء.`

---

## 2. Installation et exécution (Windows)

Prérequis : Python 3.10+ et `pip`.

```powershell
cd C:\chat-bot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Entraîner les modèles (génère aussi le dataset s'il est absent) :

```powershell
python -m src.train
```

Lancer l'application :

```powershell
streamlit run app.py
```

Ouvrir ensuite <http://localhost:8501>.

### Commandes utiles

```powershell
python -m src.train --quick                  # entraînement rapide (2 folds, moins d'itérations)
python -m src.train --dataset data/mon_dataset.csv
python -m pytest -q                          # tests
python -m nbconvert --execute --inplace notebooks/ml_report.ipynb
```

> `pytest` est épinglé à `<9` : pytest 9 provoque un crash interne sur l'environnement Python 3.10 utilisé pour ce projet.

---

## 3. Structure du projet

```text
C:\chat-bot
├── app.py                     # interface Streamlit
├── requirements.txt
├── README.md
├── .env.example               # configuration LLM optionnelle
├── configs/
│   ├── app.yaml               # nom du bot, langues, chemins, poids, seuils
│   └── markets.yaml           # presets Tunisie / générique, secteurs, benchmarks
├── data/
│   ├── synthetic_businesses.csv   # dataset synthétique (6 000 lignes)
│   ├── synthetic_metadata.json
│   ├── sample_business.csv        # historique mensuel d'exemple (mode B)
│   └── evaluations.db             # SQLite (créé à l'usage)
├── models/                    # artefacts joblib + métriques + SHAP
├── notebooks/
│   └── ml_report.ipynb        # rapport ML exécuté (EDA, modèles, métriques)
├── src/
│   ├── config.py              # chargement YAML, footer, presets
│   ├── data_generation.py     # génération du dataset synthétique documenté
│   ├── preprocessing.py       # schéma, pipelines scikit-learn
│   ├── train.py               # CV, tuning, comparaison, SHAP, KMeans
│   ├── scoring.py             # rubrique, ML, finances, what-if
│   ├── forecasting.py         # Holt-Winters / ARIMA + repli
│   ├── business_analysis.py   # KPI, diagnostics, benchmarks, actions
│   ├── agent.py               # machine à états conversationnelle multilingue
│   ├── nlg.py                 # réponses hors ligne + LLM optionnel
│   ├── report.py              # export PDF (reportlab, arabe façonné)
│   ├── storage.py             # persistance SQLite
│   ├── i18n.py                # traductions UI / rapports
│   └── utils.py               # helpers
└── tests/                     # 21 tests pytest
```

---

## 4. Données et schéma

### 4.1 Dataset synthétique

- **6 000 lignes**, générées avec `seed=42`.
- **Taux de succès : 58,97 %** (classe positive).
- Les relations sont construites à partir des hypothèses de `configs/markets.yaml` : marges brutes, concurrence, intensité capitalistique, croissance, risque réglementaire, scalabilité, etc.
- Des valeurs manquantes sont injectées volontairement (budget, taille d'équipe, expérience du fondateur, marketing) afin que l'imputation du pipeline soit réellement utile.

### 4.2 Colonnes

**Variables explicatives (21) :**

| Type | Colonnes |
|---|---|
| Catégorielles | `sector`, `target_market`, `revenue_model`, `funding_stage`, `country_preset`, `city_tier` |
| Numériques | `budget`, `team_size`, `price`, `competitor_intensity`, `differentiation`, `market_size_score`, `market_growth_score`, `audience_fit`, `operational_complexity`, `regulatory_risk`, `timing_score`, `marketing_spend`, `expected_customers`, `founder_experience_years`, `team_readiness` |

**Cibles :**
- `success` — classification binaire ;
- `roi_percent` — régression ;
- `breakeven_months` — régression.

Des colonnes descriptives (`idea_id`, `country`, `city`) peuvent être présentes : elles sont ignorées par les modèles.

### 4.3 Remplacer par un vrai dataset

Placez un fichier CSV ou Excel dans `data/` avec les mêmes colonnes, puis :

```powershell
python -m src.train --dataset data/real_businesses.csv
```

Les artefacts sont réécrits dans `models/`. Le flag `synthetic` de `models/metadata.json` passe à `false`.

---

## 5. Pipeline Machine Learning

### 5.1 Prétraitement

Pipeline scikit-learn unique, sauvegardé avec le modèle :

1. imputation des numériques par la **médiane** ;
2. imputation des catégorielles par la **mode** (`most_frequent`) ;
3. **standardisation** (`StandardScaler`) des numériques ;
4. **one-hot encoding** (`OneHotEncoder(handle_unknown="ignore")`) des catégorielles.

Séparation **80 % entraînement / 20 % test**, `random_state=42`, stratification pour la classification.

### 5.2 Classification — succès

Modèles comparés en validation croisée stratifiée 3 folds :

| Modèle | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|
| Régression logistique (baseline) | 0,715 | 0,746 | **0,790** |
| Random Forest | 0,713 | 0,773 | 0,776 |
| Gradient Boosting | 0,715 | 0,768 | 0,775 |
| XGBoost | 0,713 | 0,766 | 0,776 |

Le meilleur baseline (régression logistique) est **optimisé par `RandomizedSearchCV`** (12 candidats, 3 folds, métrique ROC-AUC).

**Métriques sur le jeu de test (1 200 lignes) :**

| Métrique | Valeur |
|---|---:|
| Accuracy | **72,08 %** |
| Precision | **78,91 %** |
| Recall | **71,89 %** |
| F1-score | **75,24 %** |
| ROC-AUC | **79,00 %** |

Matrice de confusion : 356 vrais négatifs, 136 faux positifs, 199 faux négatifs, 509 vrais positifs.

### 5.3 Régression — ROI et seuil de rentabilité

Modèles comparés : Ridge, Random Forest, Gradient Boosting, XGBoost. Le meilleur baseline XGBoost est optimisé par `RandomizedSearchCV` (9 candidats, 3 folds, MAE).

| Cible | Modèle retenu | MAE test | RMSE test | R² test |
|---|---|---:|---:|---:|
| `roi_percent` | XGBoost optimisé | **54,62** | **85,46** | **0,811** |
| `breakeven_months` | XGBoost optimisé | **7,42** | **10,30** | **0,749** |

Les prédictions sont bornées à des plages métier raisonnables (ROI −100 % à 700 %, rentabilité 1 à 60 mois).

### 5.4 Interprétabilité

- **Permutation importance** sur le jeu de test pour la classification et les deux régressions.
- **SHAP** : `TreeExplainer` ou `LinearExplainer`, importance globale agrégée par variable d'origine (`models/shap_importance.csv`, `models/shap_background.joblib`).
- Les conducteurs positifs/négatifs du rapport sont dérivés de ces contributions, avec repli sur la rubrique si SHAP est indisponible.
- **KMeans** (5 segments) sur les variables de marché ; le segment prédit est affiché dans le rapport.

### 5.5 Artefacts produits dans `models/`

```text
classifier.joblib
regression_roi.joblib
regression_breakeven.joblib
kmeans.joblib
metadata.json                 # configuration complète, hyperparamètres, schéma
metrics.json                  # métriques finales + comparaison
model_comparison.csv
feature_importance_classification.csv
feature_importance_roi.csv
feature_importance_breakeven.csv
shap_importance.csv
shap_background.joblib
roc_curve.csv
confusion_matrix.csv
test_predictions_*.csv
```

---

## 6. Mode B — Analyse d'une entreprise existante

Le fichier attendu est un historique mensuel. Exemple fourni : `data/sample_business.csv`.

Colonnes minimales : `date`, `revenue`, `cost` (ou `costs`). Colonnes optionnelles : `customers`, `marketing_spend` / `marketing`, `orders`.

Traitement :

1. contrôle qualité et complétion des mois manquants ;
2. calcul des KPI et des tendances ;
3. comparaison aux benchmarks du secteur ;
4. détection de problèmes (marge faible, churn élevé, CAC excessif, croissance négative, etc.) ;
5. prévision du revenu à 6 mois ;
6. actions priorisées avec impact mensuel estimé ;
7. rapport PDF.

La saisie manuelle crée un snapshot synthétique de 12 mois à partir du revenu, du coût, des clients, du marketing et de la croissance fournis.

---

## 7. Configuration

### 7.1 Nom du bot

Le nom est centralisé dans `configs/app.yaml` :

```yaml
bot_name: "CEO AI"
```

Il est réutilisé dans l'interface, l'avatar, le PDF, les titres et le pied de page. Le changer dans ce seul fichier suffit.

### 7.2 Marchés et secteurs

`configs/markets.yaml` contient :

- les presets `tunisia` (TND) et `generic` (USD) ;
- les villes, facteurs de coût, budgets/prix/clients typiques ;
- les hypothèses sectorielles et les benchmarks ;
- les poids de génération du dataset.

Ajouter un preset ou un secteur ne nécessite aucune modification Python.

### 7.3 LLM optionnel

Copiez `.env.example` vers `.env` et renseignez le fournisseur :

```dotenv
LLM_PROVIDER=none          # none | ollama | openai | anthropic
LLM_MODEL=
OLLAMA_BASE_URL=http://localhost:11434
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

Sans clé ni serveur, l'application reste **100 % fonctionnelle** grâce au générateur de réponses hors ligne. Aucun secret n'est codé en dur.

---

## 8. Tests

```powershell
python -m pytest -q
```

Couverture :

- pipeline et schéma du dataset ;
- scoring, plages de valeurs, localisation, what-if, presets ;
- analyse d'entreprise et prévisions ;
- machine conversationnelle ;
- persistance SQLite ;
- export PDF (contenu et pied de page) ;
- présence exacte des trois traductions du pied de page, sans lien.

---

## 9. Notebook ML

`notebooks/ml_report.ipynb` présente le rapport complet :

- chargement et contrôle du dataset ;
- valeurs manquantes et EDA ;
- comparaison des modèles ;
- métriques, matrice de confusion, courbe ROC ;
- importance des variables et SHAP ;
- segmentation KMeans ;
- limites et conclusion.

Pour le régénérer après un nouvel entraînement :

```powershell
python notebooks/_build_notebook.py
python -m nbconvert --execute --inplace notebooks/ml_report.ipynb --ExecutePreprocessor.timeout=300
```

---

## 10. Limites

- Le dataset est **synthétique** : les métriques mesurent la capacité du pipeline à retrouver les relations simulées, pas la performance sur des entreprises réelles.
- Les probabilités et prévisions sont des estimations, pas des garanties.
- Le mode B dépend de la qualité et de la longueur de l'historique fourni ; un historique court réduit la fiabilité des prévisions.
- Le LLM est optionnel ; le mode hors ligne utilise des règles, donc ses formulations restent génériques.
- L'outil ne remplace ni une étude de marché, ni un expert-comptable, ni un conseil juridique.

---

## 11. Résumé des commandes

```powershell
# installation
pip install -r requirements.txt

# données + entraînement complet
python -m src.train

# application
streamlit run app.py

# tests
python -m pytest -q

# notebook
python -m nbconvert --execute --inplace notebooks/ml_report.ipynb
```

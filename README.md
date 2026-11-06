# ML Churn Prediction — Projet MLOps

**Industrialisation d'un modèle de prédiction de l'attrition des clients télécom**  
Projet académique de Master 2 Data Science & Intelligence Artificielle  
**Auteur :** Gibril KAMADEU  
**Version documentée :** `v1.0.0` (le README est ajouté après la création initiale de ce tag)

## Présentation

Ce projet met en œuvre une chaîne MLOps de bout en bout : génération et exploration de données, entraînement de modèles de classification, exposition d'une API de prédiction, tests, conteneurisation, intégration continue et observabilité.

Il s'agit d'un **prototype pédagogique local**, et non d'un service prêt à être exposé en production.

## Architecture

```text
Dataset CSV → Exploration / entraînement → models/model.pkl
                                          ↓
Client HTTP → API FastAPI (/predict) → Prédiction de churn
                 │                 │
                 │ métriques       └── logs → Logstash → Elasticsearch → Kibana
                 └── Prometheus → Grafana

GitHub → Pull Request → GitHub Actions (Pytest + build Docker)
```

## Données et modèles

- **Données :** 2 000 observations synthétiques dans `data/raw/churn.csv`.
- **Cible :** `churn` (classification binaire).
- **Variables explicatives :** `age`, `anciennete`, `revenu`, `nb_reclamations`, `montant_mensuel`, `retards_paiement`.
- **Évaluation :** séparation stratifiée apprentissage/test (80 % / 20 %, `random_state=42`).
- **Modèles comparés :** régression logistique, arbre de décision et Random Forest.

Résultats obtenus lors de l'entraînement de référence :

| Modèle | Accuracy | Précision | Rappel | F1-score |
|---|---:|---:|---:|---:|
| Régression logistique | 62,50 % | 58,21 % | 45,35 % | 50,98 % |
| Arbre de décision | **63,25 %** | **58,28 %** | **51,16 %** | **54,49 %** |
| Random Forest | 60,50 % | 55,15 % | 43,60 % | 48,70 % |

**Point important :** l'arbre de décision obtient les meilleurs scores sur ce découpage de test. Le script `src/train.py` sauvegarde néanmoins **Random Forest** dans `models/model.pkl`, conformément au scénario pédagogique d'industrialisation. Il ne sélectionne pas automatiquement le meilleur modèle. Ces scores sur données synthétiques et un seul découpage ne démontrent pas une performance en production.

## Technologies

| Domaine | Outils |
|---|---|
| Data Science | Python, pandas, scikit-learn, Jupyter, joblib |
| API | FastAPI, Uvicorn |
| Tests | Pytest, Selenium, Apache JMeter |
| Conteneurisation | Docker, Docker Compose |
| CI | GitHub Actions |
| Métriques | Prometheus, Grafana |
| Journaux | Elasticsearch, Logstash, Kibana |
| Versionnement | Git, GitHub, branches et Pull Requests |

## Démarrage rapide

### Prérequis

Git, Python **3.11**, Docker avec le plugin Docker Compose et suffisamment de mémoire pour exécuter six services, notamment la pile ELK. Environnement de référence : Ubuntu sous WSL2.

```bash
git clone https://github.com/kagibru/ml-churn-project.git
cd ml-churn-project
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Le dataset et le modèle sont présents dans le dépôt. Pour les régénérer ou réentraîner :

```bash
python src/generate_data.py
python src/train.py
```

**Attention :** ces commandes peuvent modifier les données ou écraser `models/model.pkl`. Elles ne sont pas nécessaires pour un simple démarrage si les fichiers existent déjà.

Démarrer les services :

```bash
docker compose config
docker compose up -d --build
docker compose ps
```

Arrêter les services :

```bash
docker compose down
```

### Accès aux interfaces

| Service | Adresse |
|---|---|
| API | http://localhost:8000 |
| Documentation Swagger | http://localhost:8000/docs |
| Santé de l'API | http://localhost:8000/health |
| Métriques API | http://localhost:8000/metrics |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3001 |
| Kibana | http://localhost:5601 |
| Elasticsearch | http://localhost:9200 |

### Exemple de prédiction

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"age":35,"anciennete":24,"revenu":350000,"nb_reclamations":2,"montant_mensuel":15000,"retards_paiement":1}'
```

La réponse contient notamment la classe prédite (`CHURN` ou `NO_CHURN`), une probabilité, la version du modèle et la durée de traitement. La prédiction exacte dépend du modèle sauvegardé.

## Tests et intégration continue

Tests API :

```bash
python -m pytest tests/test_api.py -v
```

Tests Selenium (nécessitent un navigateur et un environnement adaptés) :

```bash
python -m pytest tests/test_selenium.py -v
```

Le workflow `.github/workflows/ci.yml` exécute les **tests Pytest de l'API** et la **construction de l'image Docker** lors des événements configurés sur `main` et `develop`. La réussite de la CI ne signifie pas que les tests Selenium, JMeter ou toute la pile d'observabilité sont exécutés automatiquement.

### Tests de charge JMeter

Plan : `performance/jmeter/churn_load_test.jmx` ; résultats : `performance/jmeter/results/`.

| Utilisateurs virtuels | Requêtes | Temps moyen | P95 | Débit | Erreurs |
|---:|---:|---:|---:|---:|---:|
| 100 | 1 000 | 2 725 ms | 3 783 ms | 26,4 req/s | 0 |
| 500 | 5 000 | 17 711 ms | 19 906 ms | 25,6 req/s | 0 |
| 1 000 | 10 000 | 36 463 ms | 39 932 ms | 25,5 req/s | 0 |

**Lecture :** le débit plafonne autour de 25–26 requêtes/s tandis que la latence augmente fortement sous charge. Ces mesures concernent l'environnement testé et ne constituent pas un engagement de performance.

## Structure du dépôt

```text
.github/workflows/ci.yml          # Intégration continue
data/raw/churn.csv                # Données synthétiques
notebooks/exploration.ipynb       # Analyse exploratoire
src/generate_data.py             # Génération des données
src/train.py                     # Entraînement et comparaison
models/model.pkl                 # Modèle sauvegardé
api/main.py                      # API FastAPI
tests/                           # Tests Pytest et Selenium
performance/jmeter/              # Plan et résultats de charge
monitoring/prometheus/           # Collecte des métriques
monitoring/grafana/              # Tableau de bord
monitoring/logstash/             # Pipeline de journaux
docs/                            # Documentation et captures de validation
Dockerfile                       # Image de l'API
docker-compose.yml               # Orchestration des six services
requirements.txt                 # Dépendances Python
```

## Documentation et livrables

- [Architecture](docs/architecture.md)
- [Documentation technique](docs/documentation_technique.md)
- [Guide d'installation et de déploiement](docs/guide_installation.md)
- [Rapport d'industrialisation](docs/rapport_industrialisation.md)
- [Présentation finale (PDF)](docs/Presentation_Finale_ML_Churn_Gibril_KAMADEU.pdf)
- [Captures de validation](docs/validation/)

Le dépôt regroupe les **19 livrables du projet**, comprenant les données, le notebook, le code, le modèle, l'API, les tests, la conteneurisation, la CI, l'observabilité et les documents de restitution.

## Limites et précautions

- Les données sont **synthétiques** : aucune généralisation métier ne peut être affirmée sans données réelles et validation complémentaire.
- La sélection d'un modèle devrait idéalement utiliser une validation croisée ou un jeu de validation indépendant, en réservant le test final à une évaluation non biaisée.
- La configuration Docker est prévue pour un **usage local de démonstration**. Les identifiants par défaut et la configuration de sécurité simplifiée doivent être remplacés avant tout déploiement réel.
- Le pipeline actuel réalise une **intégration continue**, pas un déploiement automatique en production.

## Versionnement

Flux de travail utilisé : `feature/final-deliverables` → `develop` → `main`, avec Pull Requests et vérifications GitHub Actions. Le tag `v1.0.0` identifie le premier état de livraison ; l'ajout ultérieur de ce README nécessitera un nouveau commit et, si l'on veut qu'il figure dans une Release figée, un nouveau tag (par exemple `v1.0.1`).

**Dépôt :** https://github.com/kagibru/ml-churn-project

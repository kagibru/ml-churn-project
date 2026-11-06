# Documentation technique — ML Churn Prediction

**Projet :** Industrialisation MLOps d'un modèle de prédiction du départ des abonnés télécoms  
**Version documentaire :** 1.0 — 6 novembre 2026  
**Référentiel :** dépôt `ml-churn-project`  
**Document complémentaire :** [`architecture.md`](architecture.md)

## 1. Objet et périmètre

Le projet fournit une API REST capable d'estimer le risque de résiliation (*churn*) d'un client à partir de six caractéristiques. Il met en pratique un cycle MLOps comprenant la génération et l'exploration des données, l'entraînement, l'exposition d'un modèle, les tests, la conteneurisation, l'intégration continue, la journalisation et la supervision.

Le système est un **prototype pédagogique exécuté localement**, et non une plateforme de production. Les diagrammes détaillés des flux de prédiction, de journalisation et de supervision figurent dans `docs/architecture.md`.

## 2. Organisation du dépôt

| Chemin | Responsabilité |
|---|---|
| `data/raw/churn.csv` | Dataset de travail |
| `src/generate_data.py` | Génération des données synthétiques |
| `notebooks/exploration.ipynb` | Analyse exploratoire et graphiques |
| `src/train.py` | Entraînement et évaluation de trois classifieurs |
| `models/model.pkl` | Modèle Random Forest sérialisé avec Joblib |
| `api/main.py` | API FastAPI, journalisation et instrumentation |
| `tests/test_api.py` | Tests automatisés de l'API |
| `tests/test_selenium.py` | Tests fonctionnels Selenium |
| `performance/jmeter/churn_load_test.jmx` | Scénario de test de charge |
| `performance/jmeter/results/` | Résultats des campagnes JMeter |
| `Dockerfile` | Construction de l'image de l'API |
| `docker-compose.yml` | Orchestration des six services |
| `.github/workflows/ci.yml` | Pipeline d'intégration continue |
| `monitoring/prometheus/prometheus.yml` | Configuration de la collecte de métriques |
| `monitoring/logstash/pipeline/` | Pipeline de réception et transfert des logs |
| `monitoring/grafana/dashboard.json` | Export du tableau de bord |
| `docs/architecture.md` | Architecture et flux techniques |
| `docs/validation/` | Captures de vérification |

Certains fichiers sont des artefacts générés : leur présence doit être contrôlée après clonage du dépôt.

## 3. Données et préparation

Le fichier `data/raw/churn.csv` contient les observations utilisées pour entraîner les modèles. La cible binaire est `churn` : `1` désigne le départ du client et `0` son maintien. Les six variables explicatives sont :

| Variable | Type attendu dans l'API | Signification |
|---|---|---|
| `age` | Entier | Âge du client |
| `anciennete` | Entier | Ancienneté du client |
| `revenu` | Nombre décimal | Revenu déclaré ou simulé |
| `nb_reclamations` | Entier | Nombre de réclamations |
| `montant_mensuel` | Nombre décimal | Montant mensuel facturé |
| `retards_paiement` | Entier | Nombre de retards de paiement |

Les unités exactes des montants et de l'ancienneté doivent être confirmées à partir du générateur de données avant tout usage métier. Le notebook permet de vérifier dimensions, valeurs manquantes, distributions et corrélations. L'utilisation de données synthétiques limite l'extrapolation des performances à des clients réels.

## 4. Entraînement et évaluation

Le script `src/train.py` lit `data/raw/churn.csv`, sépare la cible des variables explicatives et crée un jeu d'entraînement de **80 %** et un jeu de test de **20 %** avec `random_state=42` et stratification selon `churn`.

Trois modèles sont entraînés :

1. **Régression logistique** : pipeline `StandardScaler` puis `LogisticRegression(max_iter=1000)`.
2. **Arbre de décision** : `DecisionTreeClassifier(max_depth=5, random_state=42)`.
3. **Random Forest** : `RandomForestClassifier(n_estimators=200, random_state=42)`.

Pour chacun, le script calcule **accuracy**, **precision**, **recall** et **F1-score** sur le jeu de test. Les valeurs sont affichées en console ; elles ne sont pas enregistrées dans un registre d'expériences.

**Choix du modèle livré :** le script sauvegarde explicitement le Random Forest dans `models/model.pkl` avec `joblib.dump`, conformément à l'objectif pédagogique. **Ce choix n'est pas le résultat d'une sélection automatique du meilleur modèle selon les métriques.**

Commande depuis la racine du dépôt :

```bash
python src/train.py
```

Le chargement de l'artefact Joblib suppose un environnement logiciel compatible. Ne jamais désérialiser un fichier `.pkl` provenant d'une source non fiable.

## 5. API de prédiction

### 5.1. Initialisation

L'application `api/main.py` initialise FastAPI en version applicative `1.0.0` et charge `models/model.pkl` au démarrage. **Le fichier modèle doit donc exister avant de lancer l'API.**

### 5.2. Routes

| Méthode | Route | Fonction |
|---|---|---|
| `GET` | `/` | Informations de base et statut |
| `GET` | `/health` | Vérification simple de disponibilité (`UP`) |
| `GET` | `/model/info` | Nom du modèle et version |
| `POST` | `/predict` | Prédiction de churn et probabilité |
| `GET` | `/metrics` | Métriques exposées par l'instrumentation Prometheus |
| `GET` | `/docs` | Documentation interactive générée par FastAPI |

### 5.3. Contrat de requête

```json
{
  "age": 35,
  "anciennete": 5,
  "revenu": 450000,
  "nb_reclamations": 2,
  "montant_mensuel": 25000,
  "retards_paiement": 2
}
```

Les six champs sont obligatoires. Pydantic valide leurs types, mais le schéma actuel **n'impose pas de bornes métier** (par exemple, âge positif ou nombre de retards non négatif).

### 5.4. Réponse

```json
{
  "prediction": "CHURN",
  "probability": 0.67,
  "model_version": "1.0.0",
  "duration_ms": 18.51
}
```

Cet exemple est **illustratif**, et non la sortie garantie pour la requête ci-dessus. La classe vaut `CHURN` si la prédiction est `1`, sinon `NO_CHURN`. Le champ `probability` correspond à la probabilité estimée pour la classe `1` par `predict_proba`; `duration_ms` mesure le temps écoulé dans le traitement de la fonction, et non la latence réseau de bout en bout.

**Attention à l'ordre des variables :** l'API construit un tableau dans l'ordre `age`, `anciennete`, `revenu`, `nb_reclamations`, `montant_mensuel`, `retards_paiement`. Cet ordre doit correspondre exactement à celui utilisé pendant l'entraînement. Une évolution recommandée consiste à fournir au modèle un DataFrame avec noms de colonnes explicites et à vérifier cet ordre automatiquement.

### 5.5. Exemple d'appel

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"age":35,"anciennete":5,"revenu":450000,"nb_reclamations":2,"montant_mensuel":25000,"retards_paiement":2}'
```

## 6. Conteneurisation et orchestration

Le `Dockerfile` construit une image de l'application Python. Le fichier `docker-compose.yml` déclare les services suivants :

| Service | Image / version | Accès depuis la machine hôte | Rôle |
|---|---|---|---|
| `api` | `ml-churn-api:1.0` (build local) | `http://localhost:8000` | Inférence REST |
| `elasticsearch` | `8.17.3` | `http://localhost:9200` | Indexation des événements |
| `logstash` | `8.17.3` | Aucun port publié | Réception et transformation des logs |
| `kibana` | `8.17.3` | `http://localhost:5601` | Exploration des logs |
| `prometheus` | `v3.5.0` | `http://localhost:9090` | Collecte des métriques |
| `grafana` | `11.6.0` | `http://localhost:3001` | Tableaux de bord |

Les ports Elasticsearch, Kibana, Prometheus et Grafana sont liés à `127.0.0.1` sur l'hôte. Le port 8000 de l'API est publié sans restriction d'interface dans la configuration fournie : il faut donc limiter l'accès au réseau local de test si nécessaire.

Les volumes Docker `esdata`, `prometheus_data` et `grafana_data` assurent la persistance de leurs données respectives. Les configurations Prometheus et Logstash sont montées en lecture seule. L'API reçoit `LOGSTASH_HOST=logstash` et `LOGSTASH_PORT=5000`.

Elasticsearch dispose d'un *healthcheck* et Logstash/Kibana attendent son état sain. Les autres `depends_on` contrôlent principalement l'ordre de démarrage, sans constituer des vérifications complètes de disponibilité applicative.

Commandes :

```bash
docker compose up -d --build
docker compose ps
docker compose logs -f api
# Arrêt sans suppression des volumes :
docker compose down
```

Ne pas utiliser `docker compose down -v` si l'on souhaite conserver les données Elasticsearch, Prometheus et Grafana.

## 7. Intégration continue et gestion du code

Le workflow `.github/workflows/ci.yml` se déclenche sur les **push** et **pull requests** ciblant `main` ou `develop`.

Le job `test` s'exécute sous `ubuntu-latest`, configure Python **3.11**, installe `requirements.txt`, puis lance :

```bash
python -m pytest tests/test_api.py -v
```

Le job `docker-build` dépend de la réussite de `test`. Il configure Docker Buildx et construit l'image `ml-churn-api:1.0` avec `docker/build-push-action@v6`.

**Limites du pipeline actuel :** `push: false` signifie que l'image n'est pas publiée dans un registre. Aucun déploiement automatique n'est défini. Les tests Selenium et JMeter ne sont pas inclus dans ce workflow.

## 8. Stratégie de tests et validation

| Niveau | Outil | Objectif | Exécution |
|---|---|---|---|
| Tests API | Pytest | Vérifier les routes et réponses | Local et CI |
| Tests fonctionnels | Selenium | Vérifier le parcours automatisé prévu | Local |
| Tests de charge | Apache JMeter | Mesurer latence, débit et erreurs | Local |
| Tests d'intégration | Docker Compose, curl | Vérifier les services et les échanges | Local |
| Observabilité | Kibana, Prometheus, Grafana | Contrôler traces et métriques | Local |

Commandes de tests :

```bash
python -m pytest tests/test_api.py -v
python -m pytest tests/test_selenium.py -v
```

Les captures de validation sont conservées dans `docs/validation/`. Elles constituent des **preuves ponctuelles**, pas un dispositif de surveillance continue.

### 8.1. Campagnes JMeter

Les résultats rapportés pendant la validation locale sont les suivants :

| Utilisateurs simulés | Requêtes | Temps moyen | P95 | Débit | Erreurs |
|---:|---:|---:|---:|---:|---:|
| 100 | 1 000 | 2 725 ms | 3 783 ms | 26,4 req/s | 0 |
| 500 | 5 000 | 17 711 ms | 19 906 ms | 25,6 req/s | 0 |
| 1 000 | 10 000 | 36 463 ms | 39 932 ms | 25,5 req/s | 0 |

**Interprétation :** le débit observé reste proche de 25–26 requêtes/s lorsque la charge augmente, tandis que la latence croît fortement. Cela suggère une saturation de capacité dans **cet environnement de test**, sans permettre d'isoler à lui seul le composant limitant. L'absence d'erreurs ne signifie pas que le service respecte un objectif de temps de réponse en production. Les résultats détaillés sont conservés dans `performance/jmeter/results/`.

Ces mesures proviennent des campagnes de validation du projet ; elles doivent être réexécutées pour toute comparaison après modification de l'infrastructure ou du modèle.

## 9. Journalisation centralisée : ELK

L'API configure un `LogstashHandler` Python qui prépare des événements JSON comprenant le niveau, le message, le nom du logger, la version du modèle et l'identifiant applicatif. Elle envoie les événements par connexion TCP à `logstash:5000` avec un délai de connexion de **0,5 seconde**. Si Logstash est indisponible ou si la transmission échoue avec une erreur réseau/valeur, le gestionnaire ignore cette erreur pour laisser le traitement de prédiction continuer.

Les événements sont indexés dans Elasticsearch et consultables depuis Kibana, notamment au moyen du motif `ml-churn-logs-*` utilisé pendant la validation.

**Points de vigilance :** le gestionnaire ouvre une connexion pour chaque événement, ce qui peut ajouter un coût par requête. Les événements ne sont pas mis en file durablement ; une indisponibilité de Logstash peut entraîner leur perte. Le journal d'erreur courant ne joint pas explicitement la pile d'exception dans l'appel `logger.error("Prediction error")`.

## 10. Supervision : Prometheus et Grafana

`prometheus_fastapi_instrumentator` instrumente l'application et expose `/metrics`. Prometheus collecte périodiquement les métriques définies dans `monitoring/prometheus/prometheus.yml`. Grafana affiche les séries collectées et peut être initialisé à partir du dashboard exporté.

Les indicateurs présentés comprennent les volumes de requêtes, les réponses réussies, les erreurs et les temps de réponse. Les métriques utilisées dans l'architecture incluent `http_requests_total` et les compteurs/sommes des durées HTTP. La disponibilité de la cible Prometheus peut être contrôlée avec une requête `up{job="ml-churn-api"}` ; la valeur `1` indique une cible joignable lors du dernier *scrape*.

**Distinction importante :** les durées issues des métriques HTTP peuvent différer du `duration_ms` calculé à l'intérieur de la route `/predict`, car les périmètres de mesure ne sont pas identiques.

## 11. Sécurité, reproductibilité et limites

Le projet fonctionne comme un environnement pédagogique. Avant un usage réel, les travaux prioritaires sont :

- **Sécurisation :** activer l'authentification et le chiffrement d'Elasticsearch, remplacer `admin/admin` dans Grafana, protéger l'API (authentification, TLS, restrictions réseau).
- **Validation des entrées :** imposer des plages admissibles et vérifier la cohérence métier des six variables.
- **Reproductibilité ML :** enregistrer les métriques, les versions de données, les paramètres, les dépendances et les artefacts d'entraînement.
- **Sélection de modèle :** formaliser un critère de choix et comparer les performances sur données indépendantes ; le Random Forest actuel est imposé par le scénario pédagogique.
- **Déploiement :** ajouter la publication d'image, le scan de vulnérabilités, les environnements de livraison et les mécanismes de retour arrière.
- **Résilience :** renforcer la disponibilité des services, mettre en file les événements de logs et définir des limites de ressources.
- **Supervision ML :** suivre la dérive des données, la dérive des prédictions et les performances réelles lorsque les étiquettes de churn deviennent disponibles.
- **Charge :** approfondir l'analyse du débit plafonnant observé par JMeter et définir des objectifs de service mesurables.

Le projet ne comprend pas, à ce stade, de registre de modèles, de réentraînement automatisé, de surveillance de dérive ou de déploiement continu démontré.

## 12. Vérifications rapides après installation

Depuis la racine du dépôt, une fois l'environnement préparé :

```bash
# Si l'artefact modèle doit être régénéré :
python src/train.py

# Lancement des services :
docker compose up -d --build

# État des conteneurs :
docker compose ps

# Disponibilité de l'API :
curl -fsS http://localhost:8000/health

# Informations du modèle :
curl -fsS http://localhost:8000/model/info

# Endpoint des métriques :
curl -fsS http://localhost:8000/metrics | head
```

Pour les validations complètes, consulter les captures `docs/validation/`, les résultats JMeter et le futur guide d'installation.

## 13. Conclusion

Le projet relie un modèle de classification à une API REST conteneurisée et à une chaîne de contrôle opérationnel. Les tests automatisés, la construction Docker dans GitHub Actions, la centralisation ELK et les métriques Prometheus/Grafana démontrent plusieurs pratiques fondamentales de MLOps. Le périmètre actuel correspond à une **industrialisation partielle et vérifiable en laboratoire** : la mise en production nécessiterait notamment un déploiement automatisé, un renforcement de la sécurité et un suivi de la qualité du modèle dans le temps.

---

**Sources internes :** `src/train.py`, `api/main.py`, `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`, `docs/architecture.md` et éléments de validation du projet.

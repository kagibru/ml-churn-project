# Guide d’installation et de mise en service — ML Churn Prediction

**Projet :** Industrialisation MLOps d’un modèle de prédiction du churn télécom  
**Version :** 1.0 — 6 novembre 2026  
**Public cible :** enseignant, évaluateur ou développeur souhaitant reproduire le prototype  
**Environnement de référence :** Ubuntu sous WSL2, Docker et Python 3.11

## 1. À propos de ce guide

Ce guide accompagne la prise en main du projet, depuis le clonage du dépôt jusqu’à la vérification de l’API et des outils de supervision. L’objectif est de permettre à une personne qui n’a pas participé au développement de lancer le système et de comprendre rapidement le rôle de chaque service.

Le projet est un **prototype pédagogique local**. Il ne doit pas être exposé tel quel sur Internet : plusieurs réglages, notamment les identifiants Grafana et la sécurité Elasticsearch, ont été simplifiés pour les travaux pratiques.

## 2. Prérequis

- Git, pour récupérer le code source ;
- Docker Engine avec le plugin Docker Compose (`docker compose`) ou Docker Desktop avec intégration WSL2 ;
- Python **3.11**, `venv` et `pip`, pour exécuter les scripts hors conteneur ;
- Un terminal Ubuntu/WSL2 et, facultativement, VS Code ;
- Un navigateur et `curl` pour consulter et tester les interfaces.

La pile comporte six conteneurs, dont Elasticsearch, Logstash et Kibana. Prévoir plusieurs gigaoctets de mémoire disponibles et suffisamment d’espace disque. Les besoins exacts dépendent de la machine et de la charge.

Vérifier les outils :

```bash
 git --version
 python3.11 --version
 docker --version
 docker compose version
 docker info
```

Si `docker info` échoue, vérifier que Docker est démarré et accessible depuis la distribution WSL. Ne pas poursuivre avant d’avoir corrigé ce point.

## 3. Récupérer le projet

Dans Ubuntu/WSL :

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/kagibru/ml-churn-project.git
cd ml-churn-project
```

**Attention aux branches :** les derniers livrables ont été préparés sur `feature/final-deliverables`. Tant qu’ils ne sont pas fusionnés, le contenu de `main` peut être incomplet. Pour reproduire la version finale, utiliser la branche dans laquelle les livrables auront été fusionnés ; à défaut, après publication de la branche :

```bash
git fetch origin
git switch feature/final-deliverables
```

Vérifier les éléments nécessaires :

```bash
ls Dockerfile docker-compose.yml requirements.txt
ls api/main.py src/train.py
ls monitoring/prometheus/prometheus.yml
ls monitoring/logstash/pipeline/
```

## 4. Préparer l’environnement Python

Depuis la racine du dépôt :

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Le préfixe `(.venv)` dans le terminal confirme que l’environnement virtuel est activé. Il isole les dépendances du projet de celles du système.

## 5. Vérifier les données et le modèle

Le service API charge le modèle depuis `models/model.pkl` **au démarrage**. Il faut donc disposer de ce fichier avant de construire ou de lancer le conteneur.

```bash
ls -lh data/raw/churn.csv models/model.pkl
```

Si le dataset n’est pas présent, le générer avec :

```bash
python src/generate_data.py
```

Si le modèle n’est pas présent, ou si l’on souhaite reproduire son entraînement :

```bash
mkdir -p models
python src/train.py
```

Le script utilise une séparation apprentissage/test de **80 % / 20 %**, stratifiée, et compare trois algorithmes : régression logistique, arbre de décision et Random Forest. Il enregistre **Random Forest** dans `models/model.pkl` pour la suite du TP ; il ne sélectionne pas automatiquement le meilleur algorithme. Les scores affichés sont calculés sur le jeu de test.

> Les commandes de génération et d’entraînement supposent une exécution depuis la **racine du dépôt**, car les chemins des scripts sont relatifs à ce dossier.

## 6. Démarrer les services Docker

Depuis la racine du dépôt :

```bash
docker compose config
docker compose up -d --build
```

La première commande vérifie la syntaxe de la configuration. La seconde construit l’image de l’API et démarre les six services en arrière-plan. Le premier démarrage peut être long, car les images doivent être téléchargées et Elasticsearch initialisé.

Consulter l’état :

```bash
docker compose ps
```

Si un service ne démarre pas, afficher ses journaux :

```bash
docker compose logs --tail=100 api
docker compose logs --tail=100 elasticsearch
docker compose logs --tail=100 logstash
```

L’API dépend de Logstash dans la configuration Compose, et Logstash attend qu’Elasticsearch soit déclaré sain. Cela ne garantit pas que toutes les interfaces soient immédiatement disponibles ; patienter quelques instants et réessayer si nécessaire.

## 7. Accéder aux interfaces

| Composant | Adresse locale | Utilité |
|---|---|---|
| API FastAPI | `http://localhost:8000` | Service de prédiction |
| Swagger UI | `http://localhost:8000/docs` | Documentation interactive de l’API |
| Contrôle de santé | `http://localhost:8000/health` | Vérification de l’API |
| Métriques | `http://localhost:8000/metrics` | Mesures exposées à Prometheus |
| Elasticsearch | `http://localhost:9200` | Indexation des journaux |
| Kibana | `http://localhost:5601` | Recherche et exploration des logs |
| Prometheus | `http://localhost:9090` | Collecte et requêtes de métriques |
| Grafana | `http://localhost:3001` | Tableaux de bord |

Dans `docker-compose.yml`, Elasticsearch, Kibana, Prometheus et Grafana sont liés à `127.0.0.1` sur l’hôte. L’API, en revanche, publie le port `8000` sans restriction explicite d’interface : il convient de ne pas l’exposer à un réseau non fiable.

## 8. Tester l’API

Vérifier la disponibilité :

```bash
curl -s http://localhost:8000/health
```

Réponse attendue :

```json
{"status":"UP"}
```

Obtenir les informations sur le modèle :

```bash
curl -s http://localhost:8000/model/info
```

Effectuer une prédiction :

```bash
curl -s -X POST 'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
    "age": 35,
    "anciennete": 5,
    "revenu": 450000,
    "nb_reclamations": 2,
    "montant_mensuel": 25000,
    "retards_paiement": 2
  }'
```

La réponse contient les champs `prediction` (`CHURN` ou `NO_CHURN`), `probability`, `model_version` et `duration_ms`. **La probabilité et la classe dépendent du modèle réellement entraîné : ne pas attendre une valeur numérique fixe.**

## 9. Exécuter les tests automatisés

Avec `.venv` activé et les dépendances installées :

```bash
python -m pytest tests/test_api.py -v
```

Les tests Selenium peuvent ensuite être exécutés :

```bash
python -m pytest tests/test_selenium.py -v
```

Ces derniers nécessitent un navigateur et un pilote compatibles avec la configuration du test. Consulter le message d’erreur si ces dépendances ne sont pas disponibles. Le workflow GitHub Actions exécute les **tests API**, puis construit l’image Docker ; il n’exécute pas actuellement Selenium.

## 10. Vérifier la journalisation ELK

Après avoir effectué quelques prédictions :

```bash
docker compose logs --tail=50 api
curl -s 'http://localhost:9200/_cat/indices?v'
```

Ouvrir Kibana à `http://localhost:5601`. Dans **Discover**, utiliser la vue de données `ML Churn Logs` si elle est déjà configurée, ou créer une vue correspondant au motif `ml-churn-logs-*`. Ajuster la période d’affichage pour retrouver les événements récents.

L’API transmet ses événements à Logstash via TCP sur le port interne `5000`. En cas d’indisponibilité de Logstash, l’API ignore l’erreur de transmission et poursuit le traitement de la prédiction ; les journaux centralisés peuvent donc être incomplets pendant une panne.

## 11. Vérifier Prometheus et Grafana

Ouvrir `http://localhost:9090` et exécuter la requête PromQL :

```promql
up{job="ml-churn-api"}
```

La valeur `1` indique que Prometheus a réussi sa dernière collecte pour cette cible. Si aucun résultat n’apparaît, vérifier le nom du job dans `monitoring/prometheus/prometheus.yml` et la disponibilité de `/metrics`.

Ouvrir ensuite `http://localhost:3001`. La configuration actuelle prévoit les identifiants pédagogiques `admin` / `admin` ; Grafana peut demander de changer le mot de passe au premier accès. Ajouter Prometheus comme source de données avec l’URL interne `http://prometheus:9090`, puis importer, si nécessaire, le tableau de bord fourni dans `monitoring/grafana/dashboard.json`.

**Point de reproductibilité :** la configuration Compose conserve les données Grafana dans un volume, mais ne provisionne pas automatiquement la source de données ni le dashboard à partir du fichier JSON. Une installation neuve peut donc nécessiter cette configuration manuelle.

## 12. Tests de charge JMeter (facultatif)

Le scénario JMeter est conservé dans :

`performance/jmeter/churn_load_test.jmx`

Il peut être ouvert dans Apache JMeter pour inspecter les groupes d’utilisateurs, les requêtes et les assertions. Avant de lancer une charge importante, vérifier l’URL et le port de l’API dans le plan de test. Les résultats de référence se trouvent dans `performance/jmeter/results/` ; ils permettent de comparer les campagnes sans les relancer systématiquement.

Les campagnes de charge peuvent consommer beaucoup de ressources. Il est préférable de les effectuer après avoir vérifié le fonctionnement nominal des six services.

## 13. Arrêter et redémarrer le système

Arrêter les conteneurs en conservant les volumes :

```bash
docker compose down
```

Redémarrer :

```bash
docker compose up -d
```

**Ne pas utiliser `docker compose down -v` sans intention explicite de supprimer les données persistantes** : cette option efface les volumes Elasticsearch, Prometheus et Grafana associés à la pile.

## 14. Dépannage rapide

| Symptôme | Vérification ou action |
|---|---|
| `docker: command not found` | Installer Docker et vérifier l’intégration WSL2. |
| Impossible de joindre Docker | Vérifier `docker info` et le démarrage du moteur. |
| Port déjà utilisé | Identifier le processus occupant `8000`, `3001`, `5601`, `9090` ou `9200` ; adapter les ports si nécessaire. |
| API en échec au démarrage | Vérifier la présence de `models/model.pkl`, puis `docker compose logs api`. |
| `ModuleNotFoundError` en local | Activer `.venv` et réinstaller `requirements.txt`. |
| Elasticsearch indisponible | Contrôler la mémoire disponible et `docker compose logs elasticsearch`. |
| Kibana vide | Générer des prédictions, vérifier Logstash et la plage temporelle de Discover. |
| Prometheus affiche `up=0` | Vérifier `/metrics`, le réseau Compose et la cible configurée. |
| Dashboard Grafana absent | Configurer la source Prometheus et importer le JSON fourni. |
| Modifications du code non visibles | Reconstruire l’API : `docker compose up -d --build api`. |

## 15. Sécurité et limites de cette installation

La configuration a été conçue pour une démonstration locale. Elasticsearch est démarré avec `xpack.security.enabled=false`, et Grafana possède des identifiants initiaux connus. Les échanges entre services ne sont pas configurés ici avec TLS. L’API n’implémente pas de mécanisme d’authentification dans le code fourni.

Avant toute utilisation réelle, il faudrait au minimum protéger les secrets, restreindre les ports publiés, activer l’authentification et le chiffrement, définir une politique de conservation des logs et sécuriser le processus de déploiement. Le pipeline GitHub Actions actuel **teste et construit** l’image ; il ne publie pas l’image et ne déploie pas automatiquement l’application.

## 16. Vérification finale

L’installation peut être considérée comme fonctionnelle lorsque les six conteneurs attendus sont démarrés, que `/health` retourne `UP`, qu’une requête `/predict` produit une réponse JSON valide, que les journaux de prédiction sont consultables dans Elasticsearch/Kibana et que Prometheus collecte les métriques de l’API. Grafana doit ensuite pouvoir afficher les indicateurs à partir de sa source Prometheus.

Pour approfondir l’architecture et les choix de conception, consulter [`architecture.md`](architecture.md) et [`documentation_technique.md`](documentation_technique.md).

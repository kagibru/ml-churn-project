# Rapport d’industrialisation MLOps — Prédiction du churn télécom

**Projet :** ML Churn Prediction  
**Cadre :** Master 2 — Data Science et Intelligence Artificielle  
**Version :** 1.0 — 6 novembre 2026  
**Dépôt :** `ml-churn-project`  
**Nature du système :** prototype pédagogique conteneurisé, validé en environnement local

## Résumé exécutif

Prédire le départ d’un abonné n’a de valeur pour un opérateur télécom que si cette information peut être obtenue au bon moment, de manière fiable et dans un système que l’on sait maintenir. C’est le point de départ de ce projet. Au-delà de l’entraînement d’un algorithme, le travail a consisté à construire une chaîne MLOps permettant de transformer un modèle de classification en service accessible par API, testable, reproductible et observable.

Le prototype développé combine un modèle Random Forest, une API FastAPI, Docker Compose, une chaîne d’intégration continue GitHub Actions et deux dispositifs d’observabilité : ELK pour les journaux et Prometheus/Grafana pour les métriques. Des tests automatisés et des campagnes de charge JMeter ont permis d’évaluer son fonctionnement.

Les validations locales ont confirmé le fonctionnement des principales briques. Les tests de charge n’ont enregistré aucune erreur dans les campagnes présentées. Ils ont toutefois mis en évidence un débit voisin de **25 à 26 requêtes par seconde** et une forte dégradation des temps de réponse lorsque la concurrence augmente. Le projet démontre donc une **industrialisation partielle et fonctionnelle**, mais ne peut pas encore être qualifié de solution prête pour la production.

## 1. Contexte et objectif métier

Dans les télécommunications, le *churn* désigne le départ d’un client. Anticiper ce phénomène peut aider une équipe commerciale à mieux cibler ses actions de fidélisation. L’objectif du modèle n’est pas de décider automatiquement qu’un client quittera l’opérateur, mais de fournir une estimation de risque susceptible d’éclairer une décision humaine.

Le problème a été formulé comme une **classification binaire** : `churn = 1` pour un client considéré comme partant et `churn = 0` dans le cas contraire. Six variables alimentent le modèle : l’âge, l’ancienneté, le revenu, le nombre de réclamations, le montant mensuel et le nombre de retards de paiement.

Le jeu de données est **synthétique**. Il permet de valider les méthodes et l’architecture sans exposer de données personnelles réelles. En contrepartie, les performances observées ne peuvent pas être transposées directement à la clientèle d’un opérateur.

## 2. Démarche d’industrialisation retenue

Le projet a été conduit par étapes, afin de vérifier chaque composant avant de l’intégrer à l’ensemble :

1. Constituer un jeu de données et comprendre ses caractéristiques.
2. Entraîner plusieurs modèles et mesurer leurs performances.
3. Sérialiser un modèle et le rendre accessible par une API REST.
4. Ajouter des tests automatiques pour détecter les régressions.
5. Conteneuriser l’application et organiser les services annexes.
6. Automatiser les contrôles de qualité dans GitHub Actions.
7. Mesurer le comportement de l’API sous charge.
8. Centraliser les journaux et exposer des métriques de supervision.
9. Rassembler les preuves de validation et documenter la solution.

Cette progression a été importante : une prédiction correcte dans un notebook ne garantit ni qu’une API démarrera sur une autre machine, ni qu’elle restera exploitable en cas d’incident.

## 3. Données, entraînement et choix du modèle

### 3.1. Préparation et exploration

Le fichier `data/raw/churn.csv` constitue le jeu de données de référence. L’analyse exploratoire est documentée dans `notebooks/exploration.ipynb` : structure du tableau, types des variables, valeurs manquantes, statistiques descriptives, distribution de la cible et corrélations.

Cette étape permet notamment de vérifier que les données correspondent aux variables attendues par l’API. Elle ne remplace toutefois pas une procédure de validation de schéma automatisée pour les données futures.

### 3.2. Entraînement reproductible

Le script `src/train.py` sépare les données en **80 % pour l’entraînement** et **20 % pour le test**, avec `random_state=42` et une stratification sur la cible. Trois algorithmes sont comparés :

| Algorithme | Paramétrage principal | Rôle dans l’expérimentation |
|---|---|---|
| Régression logistique | Standardisation, `max_iter=1000` | Modèle de référence interprétable |
| Arbre de décision | Profondeur maximale de 5 | Alternative simple, non linéaire |
| Random Forest | 200 arbres, `random_state=42` | Modèle retenu pour l’API du TP |

Pour chaque modèle, le script calcule l’accuracy, la précision, le rappel et le F1-score sur le jeu de test. La stratification et la graine fixe facilitent la comparaison d’une exécution à l’autre.

**Point de méthode :** le Random Forest est sélectionné explicitement dans le code pour répondre au périmètre du TP, puis sauvegardé dans `models/model.pkl`. Le script ne choisit **pas automatiquement le meilleur modèle** en fonction du F1-score ou d’une autre métrique. Une industrialisation plus avancée exigerait un critère de sélection formalisé, une validation croisée et un suivi des expérimentations.

### 3.3. Limites scientifiques

La séparation entraînement/test constitue une première protection contre une évaluation trop optimiste, mais elle ne suffit pas à établir la robustesse métier. Il manque notamment une validation sur des données réelles représentatives, une analyse des erreurs par segment de clientèle, une calibration des probabilités et une étude du coût des faux positifs et des faux négatifs.

## 4. Mise à disposition du modèle par API

L’application `api/main.py` utilise **FastAPI** et charge le modèle sérialisé au démarrage. Son contrat de prédiction accepte les six variables métier et renvoie une classe (`CHURN` ou `NO_CHURN`), une probabilité estimée, la version du modèle (`1.0.0`) et une durée de traitement en millisecondes.

Les routes principales sont :

| Route | Usage |
|---|---|
| `GET /` | Informations générales sur l’application |
| `GET /health` | Contrôle simple de disponibilité |
| `GET /model/info` | Identification du modèle et de sa version |
| `POST /predict` | Inférence à partir des caractéristiques d’un client |
| `GET /metrics` | Exposition des métriques Prometheus |

La validation Pydantic contrôle les types attendus. En revanche, le schéma fourni ne définit pas de contraintes métier détaillées, par exemple une plage d’âge autorisée ou l’interdiction de montants négatifs. Ces contrôles seront nécessaires avant l’utilisation sur des données opérationnelles.

Le temps `duration_ms` renvoyé par l’API mesure le traitement côté application ; il ne doit pas être confondu avec le temps de réponse de bout en bout mesuré par JMeter sous charge.

## 5. Conteneurisation et orchestration

Le `Dockerfile` construit l’API à partir de `python:3.11-slim`, installe les dépendances et lance Uvicorn sur le port 8000. Docker Compose orchestre six services :

| Service | Fonction | Port local |
|---|---|---|
| FastAPI | Service de prédiction | `8000` |
| Elasticsearch 8.17.3 | Indexation des journaux | `9200` |
| Logstash 8.17.3 | Réception et transformation des logs | `5000` interne |
| Kibana 8.17.3 | Consultation des journaux | `5601` |
| Prometheus 3.5.0 | Collecte de métriques | `9090` |
| Grafana 11.6.0 | Tableaux de bord | `3001` |

Les volumes `esdata`, `prometheus_data` et `grafana_data` conservent les données des outils concernés entre redémarrages. Les configurations Prometheus et Logstash sont montées depuis le dépôt.

L’orchestration facilite la reproduction du laboratoire, mais elle ne constitue pas un mécanisme complet de haute disponibilité. Le service API dépend du démarrage de Logstash dans Compose, sans que cela prouve sa disponibilité immédiate ; Elasticsearch possède, lui, un contrôle de santé explicite. Aucun dispositif d’autoscaling ou de redondance n’est configuré.

## 6. Intégration continue et gestion du code

Le projet utilise Git et GitHub pour versionner le code et organiser les évolutions en branches. Le workflow `.github/workflows/ci.yml` se déclenche sur les *push* et les *pull requests* visant `main` ou `develop`.

Deux jobs s’exécutent dans cet ordre :

1. **Tests automatisés** : environnement Ubuntu avec Python 3.11, installation des dépendances et exécution de `tests/test_api.py` avec Pytest.
2. **Construction Docker** : après réussite des tests, création de l’image `ml-churn-api:1.0` à l’aide de Docker Buildx.

Cette automatisation apporte une garantie utile : une modification qui fait échouer les tests API bloque le job de construction. En revanche, le workflow actuel **ne publie pas l’image** (`push: false`) et **ne déploie pas** l’application. Les tests Selenium et JMeter restent en dehors de cette CI.

Il s’agit donc d’une **intégration continue avec validation de build**, et non d’une chaîne de livraison et de déploiement continus complète.

## 7. Stratégie de tests et validation

La validation ne s’est pas limitée à une requête réussie. Plusieurs niveaux de contrôle ont été mobilisés :

| Niveau | Outil | Ce qui a été vérifié |
|---|---|---|
| Tests API | Pytest | Comportement attendu des routes et de la prédiction |
| Test fonctionnel | Selenium | Parcours automatisé retenu dans le TP |
| Construction | Docker | Possibilité de construire et d’exécuter l’image |
| Intégration | GitHub Actions | Succès des tests API puis du build |
| Charge | JMeter | Latence, débit et erreurs sous concurrence |
| Journaux | Elasticsearch / Kibana | Présence et consultation des événements |
| Supervision | Prometheus / Grafana | Collecte et affichage des métriques |

Lors des vérifications locales consignées, **4 tests API** et **1 test Selenium** ont réussi. Ces résultats décrivent les exécutions observées ; ils ne constituent pas une garantie permanente. Les captures correspondantes sont conservées dans `docs/validation/`.

## 8. Résultats et analyse des tests de charge

Trois campagnes JMeter ont été conduites avec des charges croissantes :

| Utilisateurs simulés | Requêtes | Latence moyenne | P95 | Débit | Erreurs |
|---:|---:|---:|---:|---:|---:|
| 100 | 1 000 | 2 725 ms | 3 783 ms | 26,4 req/s | 0 |
| 500 | 5 000 | 17 711 ms | 19 906 ms | 25,6 req/s | 0 |
| 1 000 | 10 000 | 36 463 ms | 39 932 ms | 25,5 req/s | 0 |

### Lecture des résultats

Le premier constat est positif : les trois campagnes présentées n’ont enregistré **aucune erreur**. Le système a donc continué à répondre pendant les scénarios exécutés.

Le second constat est plus préoccupant : **le débit reste presque constant malgré l’augmentation du nombre d’utilisateurs**, tandis que le temps d’attente augmente fortement. La latence moyenne passe d’environ **2,7 secondes** à **36,5 secondes**. Au dernier palier, 95 % des requêtes observées sont traitées en environ **39,9 secondes ou moins**.

Ces mesures suggèrent qu’une limite de capacité est atteinte dans l’environnement testé. Elles ne permettent pas, à elles seules, d’attribuer le ralentissement à FastAPI, au serveur Uvicorn, au modèle, à l’envoi des logs ou aux ressources matérielles. Une campagne instrumentée, avec suivi CPU/mémoire, profilage et variations de configuration, serait nécessaire pour isoler le facteur dominant.

**Conclusion opérationnelle :** l’absence d’erreurs est encourageante, mais une API dont le temps de réponse dépasse plusieurs dizaines de secondes sous charge ne répondrait probablement pas aux attentes d’un service interactif. Il faut définir des objectifs de service mesurables avant toute mise en production.

## 9. Journalisation et traçabilité avec ELK

L’API produit des événements avec le module Python `logging`. Un gestionnaire spécifique les convertit en JSON et tente de les transmettre par TCP à Logstash sur le port interne 5000. Logstash les achemine vers Elasticsearch, où ils deviennent consultables dans Kibana via les index `ml-churn-logs-*`.

Le journal d’une prédiction peut contenir la version du modèle, la classe produite, la probabilité et la durée de traitement. Ces informations facilitent l’analyse d’un incident et la reconstitution du comportement du service.

Le code prévoit une tolérance aux indisponibilités réseau de Logstash : certaines erreurs d’envoi sont ignorées afin que l’inférence puisse continuer. Ce compromis privilégie la continuité du service, mais peut entraîner une **perte de traces**. Un système de production devrait prévoir une mise en file, des tentatives de réémission et des mécanismes de contrôle des pertes.

Les vérifications du projet ont permis de retrouver des événements dans Elasticsearch et Kibana. Elles démontrent le fonctionnement de la chaîne de journalisation dans le laboratoire, sans établir une garantie d’exhaustivité.

## 10. Supervision avec Prometheus et Grafana

L’API expose `/metrics` grâce à `prometheus-fastapi-instrumentator`. Prometheus interroge cet endpoint et Grafana visualise les séries temporelles : volume des requêtes, succès, erreurs et durées de réponse.

Lors d’une validation, la requête Prometheus `up{job="ml-churn-api"}` a renvoyé `1`, confirmant que la cible était accessible au moment du contrôle. Le tableau de bord a également affiché des compteurs de prédictions réussies et d’erreurs.

La complémentarité des outils est essentielle : **Prometheus indique qu’un indicateur se dégrade**, tandis que **Kibana aide à retrouver les événements associés**. Il manque néanmoins des règles d’alerte, des seuils de service formalisés et un processus de traitement des incidents.

## 11. Bilan de maturité MLOps

| Domaine | État atteint | Prochaine étape |
|---|---|---|
| Gestion du code | Dépôt Git, branches, workflow CI | Politique de revue et de publication |
| Données | Jeu synthétique et analyse exploratoire | Validation de schéma, versionnement des données |
| Entraînement | Script reproductible, trois algorithmes | Suivi des expériences, validation croisée |
| Artefact modèle | Fichier Joblib versionné dans le projet | Registre de modèles et traçabilité des versions |
| Service | API FastAPI opérationnelle | Validation métier, sécurité et authentification |
| Conteneurisation | Image Docker et six services Compose | Durcissement, déploiement contrôlé et résilience |
| CI/CD | Tests API et build automatisés | Publication d’image et déploiement automatisé |
| Tests | API, Selenium, JMeter | Tests de charge récurrents et critères d’acceptation |
| Observabilité | Logs ELK et métriques Prometheus/Grafana | Alertes, rétention et supervision de dérive |

Cette grille met en évidence un résultat cohérent avec le cadre du TP : la chaîne technique existe et ses principaux composants ont été vérifiés, mais plusieurs fonctions nécessaires à une exploitation durable restent manuelles ou absentes.

## 12. Risques et limites

### 12.1. Données et qualité prédictive

Le jeu synthétique peut présenter des relations différentes de celles rencontrées en production. Aucune mesure de dérive des données ou des performances du modèle n’est mise en place. Une prédiction probabiliste ne doit pas être assimilée à une certitude sur le comportement futur d’un client.

### 12.2. Sécurité

Le laboratoire désactive la sécurité Elasticsearch et utilise les identifiants Grafana `admin/admin`. L’API ne présente pas de mécanisme d’authentification dans le code examiné. Ces choix facilitent la démonstration locale mais seraient inacceptables en l’état pour une exposition publique. Les accès, les secrets, les communications et les données personnelles devraient être protégés.

### 12.3. Fiabilité et montée en charge

La forte augmentation des latences sous concurrence, l’absence de redondance et la transmission directe des logs sont des points de vigilance. Le système ne dispose pas de stratégie de reprise, de sauvegarde documentée ni d’engagement de disponibilité.

### 12.4. Gouvernance du modèle

La version `1.0.0` est déclarée dans le code, mais le projet ne possède pas de registre de modèles, de mécanisme de promotion contrôlée, ni de procédure de retour arrière. Le réentraînement n’est pas automatisé.

## 13. Feuille de route pour une exploitation réelle

### Priorité 1 — Fiabiliser et sécuriser

Définir les seuils acceptables de latence et de disponibilité, sécuriser Grafana et Elasticsearch, ajouter l’authentification de l’API, imposer des validations métier sur les entrées et diagnostiquer le plafonnement du débit observé par JMeter.

### Priorité 2 — Renforcer l’intégration et la livraison

Étendre la CI aux contrôles pertinents, automatiser la publication d’images versionnées dans un registre, introduire des environnements distincts de validation et de production, puis prévoir un déploiement réversible avec vérifications de santé.

### Priorité 3 — Industrialiser le cycle de vie ML

Versionner les jeux de données, enregistrer les métriques et paramètres des expériences, définir une règle objective de sélection du modèle, suivre la dérive et la qualité des prédictions, puis encadrer le réentraînement et l’approbation des nouvelles versions.

### Priorité 4 — Structurer l’exploitation

Configurer des alertes, une politique de conservation des journaux, des sauvegardes, des procédures d’incident et un suivi régulier des indicateurs techniques et métier. Ces mécanismes transformeront les tableaux de bord en véritable dispositif de pilotage opérationnel.

## 14. Enseignements tirés et conclusion

Ce projet a montré qu’un modèle de Machine Learning ne devient pas un service exploitable par le seul fait d’avoir été entraîné. Il faut pouvoir le charger de manière prévisible, contrôler ses entrées, tester son comportement, reproduire son environnement et comprendre ce qui se passe lorsqu’il est sollicité.

L’apport le plus significatif du travail réside dans cette **mise en relation de disciplines habituellement étudiées séparément** : Data Science, développement d’API, tests logiciels, conteneurisation, intégration continue, performance et observabilité. Les campagnes de charge ont été particulièrement instructives : elles ont confirmé que la réussite fonctionnelle ne suffit pas à caractériser la qualité de service.

Le résultat est un **prototype MLOps cohérent, démontrable et documenté**, qui répond aux objectifs d’apprentissage de l’industrialisation. Il serait toutefois excessif de le présenter comme une plateforme de production achevée. La suite logique consiste à renforcer la sécurité, à maîtriser les performances sous charge et à automatiser la gestion du cycle de vie des modèles.

## 15. Traçabilité des livrables et preuves

| Élément | Emplacement dans le dépôt |
|---|---|
| Données et exploration | `data/raw/churn.csv`, `notebooks/exploration.ipynb` |
| Entraînement et modèle | `src/train.py`, `models/model.pkl` |
| API et tests | `api/main.py`, `tests/` |
| Conteneurisation | `Dockerfile`, `docker-compose.yml` |
| Intégration continue | `.github/workflows/ci.yml` |
| Performance | `performance/jmeter/churn_load_test.jmx`, `performance/jmeter/results/` |
| Journalisation et supervision | `monitoring/logstash/`, `monitoring/prometheus/`, `monitoring/grafana/` |
| Documentation d’architecture | `docs/architecture.md` |
| Documentation technique | `docs/documentation_technique.md` |
| Guide d’installation | `docs/guide_installation.md` |
| Preuves de validation | `docs/validation/26-01-...` à `26-17-...` |

**Note de lecture :** les résultats numériques présentés sont ceux consignés lors des validations locales du projet. Ils ne constituent pas des mesures certifiées en environnement de production. Le rapport doit être mis à jour si de nouvelles campagnes modifient les résultats.

# Sprint S6 : décembre 2026

> **Mise à jour du 21 octobre 2026 (soir), après la passe de préparation.** Fait avant le sprint, sur la branche `fix/pre-sprint-audit` :
> - **S6-01** papier et README : erreurs E1 et E2 retirées, papier v2 réécrit sur le run élargi (références APA en cours de vérification), README, méthodologie, limites et spécification corrigés avec notes datées.
> - **S6-02** vocabulaire stable, **S6-03** `--file redteam`, **S6-08** M4a/M4b à l'échelle et déterministes : faits et testés.
> - **S6-04** mesuré sans téléchargement, sur les partitions étiquetées du run élargi : 275 des 282 positifs du test et 50 des 52 campagnes viennent de C17693 (`split_summary.json`).
> - Défauts D1 à D11 corrigés, plus cinq trouvés en écrivant les tests (D12 à D16 : F5 sur fenêtre vide, `min_count` de node2vec, clustering et Adamic-Adar identiquement nuls sur graphe biparti, exclusions absentes de `skipped_models.json`).
>
> **Nouveaux tickets issus de l'audit :**
> - **S6-20 · M · 3 pts** : normalisation de M1 ajustée sur l'entraînement (fonction de répartition empirique par règle) au lieu du rang dans la partition notée, qui rend M1 transductif.
> - **S6-21 · S · 5 pts** : F6 en flux pour LANL (la matérialisation de 32 retards par événement ne tient pas en mémoire), condition de S6-12.
> - **S6-22 · S · 2 pts** : réécrire le marqueur de début froid (indicateur explicite au lieu de la valeur 0, la plus « normale »), avec le run n° 1.
>
> S6-09 (Euler), S6-05/06 (registre et courbe exacte), S6-07, S6-10, S6-11 restent le cœur du sprint.

> Établi le 21 octobre 2026 à partir de la lecture du dépôt (code, spécification, papier, sorties LANL publiées), de la suite de tests exécutée en local et de trois recherches documentaires sur sources primaires. Chaque constat ci-dessous porte sa preuve : fichier et ligne, commande exécutée ou source citée. Ce qui n'a pas pu être vérifié est marqué **[non vérifié]**.

## 0. Objectif du sprint

**Répondre à RQ1 sur LANL sous le protocole AuthBench, avec des résultats qu'un relecteur ne peut pas attaquer, et repositionner le papier face à la littérature de 2025-2026.**

Concrètement, à la fin du sprint :
1. le papier ne contient plus aucune affirmation fausse (deux trouvées, §1.1) ;
2. au moins un modèle profond de référence de la littérature (Euler) et les autoencodeurs M4a/M4b sont évalués sur LANL au budget d'alertes ;
3. la figure principale devient la courbe exacte du rappel par campagne en fonction du budget, avec intervalles de confiance ;
4. une version v1.1.0 du logiciel et une v2 du préprint sont prêtes, et le choix de la revue ou de la conférence est fait.

## 1. État des lieux vérifié

### 1.1 Deux erreurs dans le papier publié (Zenodo `10.5281/zenodo.23159768`)

| # | Affirmation du papier | Ce que dit la source | Preuve |
|---|---|---|---|
| E1 | Tuor et al. rapportent « 87.0 average precision » (`reports/paper/main.tex`, l. 135-136) | Leur « AP » est l'**Average Percentile** : « we use the Average Percentile (AP) metric » | arXiv 1712.00557, texte intégral extrait le 21 oct. |
| E2 | « To our knowledge, no published work on LANL reports a budget-constrained operational metric alongside the standard aggregate metric on the same models » (l. 150-151) | **Faux.** UltraLMD++ (Larroche, arXiv 2504.13527, avril 2025) rapporte AUC, AP et Rec@B pour Argus et UltraLMD++ sur LANL. Dans sa table « avec doublons », Argus a la meilleure AUC (0,9749 contre 0,8938) et le plus faible Rec@B : une forme de notre anti-corrélation est déjà publiée. DECADES (arXiv 2103.15708) rapporte AUC@1 % et DR-k. ADSAGE (arXiv 2007.06985) rapporte un rappel cumulé à budget quotidien sur LANL, jours 1-8 / 9-13, et sa règle « known source pc » fait jeu égal avec le modèle profond (CR-12000 : 0,890 contre 0,877). | arXiv HTML de 2504.13527 ; texte intégral de 2007.06985, table 5 |

### 1.2 Deux travaux concurrents de 2026, à citer et dont il faut se distinguer

- **Larroche, « On fair and realistic performance evaluations for graph-based lateral movement detectors »**, arXiv 2607.29390, accepté à ANUBIS '26. Il harmonise le prétraitement et l'étiquetage de LANL et d'OpTC, puis réévalue Pikachu, Euler et Argus : leurs AP s'effondrent (Argus 32,27 publié contre 0,09 ± 0,04 réévalué, sur une échelle 0-100). Il signale aussi que presque tous les événements malveillants de LANL viennent d'un seul hôte source (C17693) **[à vérifier sur nos 715 événements : S6-04]**.
- **Zhao et Hassan, « A First-Principles Evaluation of Graph-Based Network Intrusion Detection Systems »**, CCS 2026, arXiv 2609.12263. Cadre GIDS-Eval ; un contrôle sans encodeur (PCA, « GIDS-Lite ») arrive premier en AP. Le dépôt d'artefacts renvoie une erreur 404 au 21 oct.

**Ce qui reste propre à AuthBench**, et que ni l'un ni l'autre ne fait : rappel **par campagne** à **budget quotidien** au niveau **événement**, avec bootstrap stratifié par campagne et correction de Holm ; encadrement exact des égalités de score ; garde-fou mécanique contre la fuite ; contrôle de circularité synthétique ; comparaison règles et ML classique contre profond sous le même protocole. Le papier doit le dire en ces termes, pas en revendiquant une primauté.

### 1.3 Un résultat déjà calculé et jamais exploité

`reports/lanl/wider_window_results/metrics_summary.json` contient `budget_for_first_detection` (rang minimum exact dans le jour, `src/authbench/evaluate/budget.py` l. 239-251 : ce n'est pas un plafond) :

| Modèle | ROC-AUC | Budget quotidien pour une 1ʳᵉ détection |
|---|---:|---:|
| M1 règles | 0,563 | **999** |
| M2b PCA | 0,930 | 1 569 |
| M2a rareté de paire | **0,941** | 2 852 |
| M0a aléatoire | 0,488 | 85 070 |
| M3b HBOS | 0,884 | 104 963 |
| M3a Isolation Forest | 0,876 | 141 522 |
| M0b échecs | 0,496 | 5 505 419 |

Sur 52 campagnes réelles, **Isolation Forest et HBOS ont besoin de plus d'alertes que le score aléatoire** pour trouver une première campagne, et le classement par ROC-AUC est presque l'inverse du classement opérationnel. Le même fichier ferme aussi l'objection sur le 0,547 de M1 : sur le run élargi, M1 a 54 684 831 scores distincts pour 54 690 768 événements.

### 1.4 Défauts de code qui empêchent un run profond propre

| # | Défaut | Preuve | Effet |
|---|---|---|---|
| D1 | M4a/M4b ajustent sur toute la partition, sans `sample_for_fit` | `models/deep.py`, `fit` | 133 M × 21 colonnes en float64 : environ 22 Go avant même le tenseur |
| D2 | M4a/M4b n'ont pas `scores_row_locally = True` | `models/base.py` : valeur par défaut `False` | 55 M lignes notées en un seul tenseur |
| D3 | Le score du VAE est **aléatoire** : `forward` tire `z` avec `randn_like` même en évaluation | `models/deep.py`, `VAEScorer.score` | deux exécutions donnent deux classements ; viole US-132 |
| D4 | Le score du VAE mélange une moyenne (reconstruction) et une somme (KL) sur les dimensions, et l'entraînement utilise une autre pondération | idem | le score noté n'est pas l'ELBO entraîné |
| D5 | **Codes catégoriels instables entre processus** : `to_physical()` dépend de l'ordre de lecture | expérience du 21 oct., Polars 1.44.2 : `C9` vaut 2 dans un processus, 0 dans l'autre | `sequence_dst` (écrit par `build_features`) et les cibles de M4c (calculées dans `train_eval`) ne parlent pas le même vocabulaire |
| D6 | M4c envoie les destinations inconnues sur le dernier identifiant réel, pas sur un identifiant réservé, contrairement à sa docstring | `models/sequence.py`, `clamp(max=vocab_size - 1)` | une machine nouvelle est confondue avec une machine connue |
| D7 | Le poids des arêtes de F5 vaut toujours 1 | `features/graph.py`, `build_daily_graph` itère sur `.unique()` | pondération sans effet |
| D8 | Aucun modèle n'a de sélection d'hyperparamètres sur la validation, sauf M1 (200 tirages sur les étiquettes de validation) | `pipeline/train_eval.py` l. 151-154 | asymétrie qu'un relecteur relèvera |
| D9 | Les régimes R1/R2 (spec §6.2) ne sont pas implémentés ; le run publié est de fait en R1 | clé `regimes` lue nulle part (`grep`) | |
| D10 | Le README dit M4, M5 et F5 « implemented and unit-tested » ; aucun test ne touche `deep.py`, `models/sequence.py`, `models/graph.py` ni `features/graph.py` | `grep` sur `tests/` | affirmation fausse dans le README |
| D11 | La docstring de `train_eval.py` renvoie à `authbench train model=ae`, commande absente du CLI | `cli.py` | |

### 1.5 Contraintes de l'environnement

- Machine locale : 8 Go de RAM, 12 cœurs, aucun GPU, torch non installé, **disque C: plein à 99 % (4,3 Go libres)**. Aucun run LANL n'est possible en local.
- Dépôt : local et GitHub synchronisés, CI verte, aucune issue, release v1.0.0. Suite de tests : succès, 1 test sauté volontairement (exécution du 21 oct.).
- Données LANL converties présentes en local (`data/interim/auth`, jours 0-13, 5,5 Go) ; `data/raw` et le feature store sont vides.
- Red team : 162 campagnes (écart de 24 h), dont 66 d'un seul événement ; les jours 14 à 29 contiennent 54 campagnes jamais utilisées.

## 2. Capacité

| Période | Contrainte | Heures AuthBench |
|---|---|---:|
| 2-15 déc. | HKPFS (1ᵉʳ), Macao (3), Saskatchewan (10), embouteillage du 15 | ≈ 6 |
| 16 déc.-3 janv. | Terme 2 en cours | ≈ 35 |
| 4-7 janv. | finaux du terme 2 | 0 |

**Total réaliste : environ 40 heures de travail humain.** Claude écrit le code et les tests ; le temps humain va à la relecture, aux runs sur machine louée et aux décisions.

## 3. Backlog du sprint

Priorités : **M** = doit, **S** = devrait, **C** = pourrait. Points : même échelle que la spécification. Chaque ticket = une branche `s6/<id>-<slug>`, une PR, une relecture `/code-review`, CI verte avant fusion.

### Bloc A : intégrité (avant tout nouveau modèle)

**S6-01 · M · 3 pts : corriger le papier et le README.**
- Papier : E1 (Tuor = average percentile), E2 (supprimer la revendication de primauté, citer UltraLMD++, DECADES, ADSAGE).
- README : D10 et D11 ; fermer l'objection du 0,547 avec les chiffres de `score_resolution` du run élargi.
- *Fait quand* : chaque phrase modifiée cite sa source ; `make lint test` passe.

**S6-02 · M · 5 pts : vocabulaire catégoriel stable (D5, D6).**
- Dans `pipeline/build_features.py`, construire le vocabulaire sur la seule partition d'entraînement : liste triée des valeurs de `src_user`, `src_computer`, `dst_computer`, `auth_type`.
- L'écrire dans `<feature_store>/vocab/<colonne>.parquet` et caster en `pl.Enum`. Une valeur hors vocabulaire va vers un identifiant réservé unique.
- La politique de vocabulaire entre dans le hachage `feature_store_version`.
- F6 et M4c lisent ce vocabulaire, jamais `to_physical()`.
- *Fait quand* : un test lance deux sous-processus qui lisent les partitions dans des ordres opposés et obtient des codes identiques ; un test vérifie qu'une destination inconnue reçoit l'identifiant réservé.

**S6-03 · M · 2 pts : télécharger `redteam.txt.gz` seul.**
- Ajouter une option `--files redteam` à `authbench data download` : 4,8 Ko au lieu de 7,2 Go, vérifié par SHA-256 contre `conf/dataset/lanl.yaml`.

**S6-04 · M · 3 pts : indépendance des campagnes.**
- Mesurer la part des 715 événements par hôte source (affirmation C17693 de Larroche) et le nombre de campagnes distinctes qui partagent un hôte.
- Si un hôte domine, ajouter au bootstrap une strate alternative (composante connexe utilisateur-hôte source) et publier les deux jeux d'intervalles.
- *Fait quand* : `docs/limitations.md` contient le chiffre mesuré et sa conséquence sur l'unité de rééchantillonnage.

### Bloc B : exploiter ce qui existe (presque sans calcul)

**S6-05 · M · 5 pts : registre des positifs.**
- Après la notation dans `train_eval`, écrire `reports/tables/positives_ledger.parquet`. Une ligne par (modèle, événement malveillant) : `event_id`, `campaign_id`, `day`, `time`, `src_user`, `src_computer`, `dst_computer`, `rank_in_day`, `rank_best`, `rank_worst` (bornes exactes dans le groupe d'égalité). Plus une table `day_sizes`.
- Environ 715 lignes par modèle : tout ce qui suit (courbes, regroupement des campagnes, bootstrap des métriques de budget) se recalcule sans relancer un seul modèle.
- *Fait quand* : sur la démo, le rappel par campagne recalculé depuis le registre est égal bit à bit à celui de `compute_budget_curve`, pour chaque budget et chaque bris d'égalité. Test de propriété (`hypothesis`) : une transformation monotone des scores laisse le registre inchangé.

**S6-06 · M · 5 pts : courbe exacte rappel-budget et son résumé.**
- `R(k)` = fraction de campagnes dont le rang minimum dans le jour est ≤ k, pour k de 1 au plus grand jour, en échelle log. C'est la nouvelle figure principale.
- Résumés :
  - budget médian de détection d'une campagne ;
  - aire normalisée sous la courbe en échelle log sur [1, K], avec K = 500 (budget opérationnel maximal de la spec). L'aire log donne plus de poids aux petits budgets que le CR-k linéaire d'ADSAGE, qu'on cite comme précédent.
- Intervalles par bootstrap stratifié par campagne, en réutilisant `paired_campaign_bootstrap`.
- *Fait quand* : les quatre points actuels (10/50/100/500) tombent exactement sur la courbe ; tests sur cas analytiques (classement parfait, classement inversé).

**S6-07 · S · 3 pts : sensibilité à l'écart de regroupement des campagnes (US-106, limite 6).**
- Écarts {1 h, 6 h, 24 h, 72 h}, recalculés depuis le registre. Publier le nombre de campagnes et `R(k)` pour chaque écart.

### Bloc C : RQ1, les modèles profonds

**S6-08 · M · 8 pts : M4a/M4b à l'échelle et déterministes (D1 à D4, D8 partiel).**
- `sample_for_fit` avec `fit_sample_size` ; `scores_row_locally = True` ; notation par lots de taille fixe.
- Arrêt précoce sur la perte de reconstruction d'un échantillon de validation, sans étiquettes.
- `torch.use_deterministic_algorithms(True)`, générateur du `DataLoader` initialisé avec la graine.
- VAE : score = ELBO négatif calculé sur `mu` (aucun tirage), reconstruction et KL sommées sur les dimensions, cohérentes avec la perte d'entraînement.
- Branchés dans `build_model_catalog` quand l'extra `deep` est présent, configurés par `conf/model/ae.yaml` et `vae.yaml`.
- *Fait quand* : deux exécutions donnent des scores identiques au bit près ; noter par morceaux donne le même résultat que noter d'un bloc ; tests sur la démo.

**S6-09 · M · 13 pts : Euler (GCN + GRU, prédiction temporelle de lien), réimplémenté dans AuthBench.**
- Pourquoi Euler : c'est le détecteur profond le plus cité sur LANL. Son entraînement est temporellement propre, il est bon marché (17 min et 7,8 Go selon Zhao et Hassan) et il a déjà été réévalué par Larroche, ce qui donne des points de contrôle.
- Pourquoi réimplémenter : son dépôt n'a pas de licence, donc on ne peut ni le copier ni le redistribuer.
- Implémentation : PyTorch pur avec matrices creuses, sans PyTorch Geometric. Graphe machine vers machine, instantanés de 30 min, 2 couches GCN puis un GRU, score d'arête par produit scalaire, échantillonnage négatif.
- Le score d'une arête (source, destination, instantané) est reporté sur chacun de ses événements. Publier deux unités d'alerte : événement et arête-instantané.
- Contrôle de fidélité avant le protocole AuthBench : sur le découpage d'Euler (entraînement avant le premier événement red team, t = 150 885), retrouver l'ordre de grandeur publié (AUC ≈ 0,99) ou documenter l'écart.
- *Fait quand* : tests unitaires (formes, causalité des instantanés, déterminisme) ; ligne Euler dans la table LANL avec la courbe de S6-06.

**S6-10 · S · 3 pts : régimes R1/R2 (D9, spec §6.2).**
- R2 = `train.filter(~is_malicious)`. Deux tables, au minimum pour M1, M2a, M2b, M4a et Euler.
- Lien avec la recommandation de Larroche : entraîner avant la première action malveillante.

**S6-11 · S · 5 pts : sélection équitable des hyperparamètres (D8).**
- Même budget de configurations pour chaque modèle paramétrable, choisi sur la validation avec le même critère que M1. Sinon, l'asymétrie est déclarée en toutes lettres dans le papier.

**S6-12 · C · 8 pts : M4c LSTM de prochain événement (famille Brown et Tuor).** Seulement après S6-02.

**S6-13 · C : Argus** (licence MIT, environ 45 Go de RAM, nécessite `flows.txt`). Reporté au sprint S7 sauf si tout le reste est fini.

### Bloc D : infrastructure et calcul

**S6-14 · M · novembre, 1 h, Mahamane : comptes de calcul.**
- **Azure for Students** : 100 $, sans carte, vérification par l'adresse universitaire ; à demander en novembre, car la validation peut prendre du temps. Machine E8as_v5 (64 Go) à 0,452 $/h, environ 2 $ pour le run principal.
  - Quota de vCPU des abonnements étudiants : **[non vérifié]**.
  - Le critère « full-time student » est à confirmer honnêtement pour ta situation.
- **Kaggle** : GPU gratuit, environ 30 h par semaine, 29 Go de RAM, pour l'entraînement d'Euler et de M4.
- **Repli** : Vast.ai, dépôt minimum de 5 $, accepte la crypto.

**S6-15 · M · 3 pts : script de run reproductible sur machine louée (`scripts/run_lanl.sh`).**
- Une commande : installation verrouillée, `preflight`, téléchargement LANL directement sur la VM (jamais depuis le Niger), conversion des jours 0-13, features, `train_eval`, archivage de `reports/` et du registre des positifs.
- Le journal du run consigne la machine, la durée et le coût réel (ce qui manquait pour le run de septembre).

**S6-16 · M · Mahamane : libérer le disque C:.** Décision à prendre par toi : rien n'est supprimé sans ton accord. `data/interim/auth` (5,5 Go) se reconstruit depuis LANL.

### Bloc E : papier et publication

**S6-17 · M · 5 pts : travaux connexes et positionnement.**
- Larroche 2025 et 2026, Zhao et Hassan 2026, Euler, Argus, Bowman et al., Brown et al., Gökstorp et al. (NOMS 2024), ADSAGE, DECADES, en plus des 7 références actuelles.
- Chaque chiffre cité est relu dans la source primaire, avec son unité (événement, arête, utilisateur-jour) et son découpage.

**S6-18 · M · 3 pts : résultats.** Table RQ1, courbe S6-06, R1/R2 ; les chiffres viennent du pipeline, jamais recopiés à la main.

**S6-19 · M · 2 pts : release v1.1.0 et préprint v2 sur Zenodo** (nouvelle version du même enregistrement), `CITATION.cff` à jour.

## 4. Calendrier

| Semaine | Tickets | Livrable |
|---|---|---|
| novembre | S6-14, S6-16 | comptes de calcul ouverts, disque libéré |
| 2-8 déc. | S6-01, S6-03, S6-04 | papier et README sans erreur connue ; mesure C17693 |
| 9-15 déc. | S6-02, S6-05 | vocabulaire stable ; registre des positifs sur la démo |
| 16-22 déc. | S6-06, S6-08, S6-09 (code), S6-15 | tout passe sur la démo ; script de run prêt |
| 23-29 déc. | **run LANL n° 1** (catalogue + M4a/M4b + Euler, R1 et R2), S6-07, S6-10 | tables et courbes LANL |
| 30 déc.-3 janv. | S6-17, S6-18, S6-19 ; S6-11 et S6-12 si le temps le permet | v1.1.0, préprint v2, choix de la venue |

**Engagement : 68 points en M et S** ; C hors engagement.

## 5. Définition de « terminé » (chaque ticket)

- `ruff check`, `ruff format --check`, `mypy --strict`, `pytest` : verts en local et en CI.
- Couverture ≥ 85 % sur `evaluate/` et `features/`, étendue à `models/` pour tout code nouveau.
- La démo se régénère bit à bit (`make demo` + vérification CI).
- **Style du code nouveau** : commentaire d'une ligne seulement là où l'intention n'est pas évidente ; docstring d'une ligne ; types stricts ; fonctions pures pour les métriques ; aucun code mort ni implémentation décorative. Le code existant n'est pas réécrit pour son style dans ce sprint.
- Commits en anglais, au style du dépôt, **sans ligne de co-auteur**. Aucune poussée vers GitHub sans ton accord.
- Tout chiffre du papier est produit par le pipeline et traçable jusqu'à un run.

## 6. Décisions à prendre (toi)

| Décision | Options | Ma recommandation |
|---|---|---|
| Venue cible | **TMLR** : aucun frais, soumission continue, pas de déplacement, juge la justesse plutôt que la portée · **DIMVA 2027 cycle 2** : 18 févr. 2027 (vérifié), Springer LNCS, inscription complète et présence aux Pays-Bas exigées, visa Schengen depuis le Niger · **ACSAC 2027** : piste R+R, bourses de voyage ouvertes aux étudiants de licence, date 2027 **[non vérifié]** (fin mai en 2026) | **TMLR**, puis une conférence si une bourse de voyage est obtenue. Décision au 3 janv. |
| Endossement arXiv | Seul l'endossement personnel existe (politique arXiv du 21 janv. 2026) | Écrire à **Corentin Larroche** (ANSSI) après S6-01 : son travail est le plus proche, son code LMDEval est sous BSD-2, et le contact a aussi une valeur scientifique |
| Second jeu de données | **OpTC**, version corrigée Inria (DOI 10.57745/UXCWOC, ≈ 0,94 To) avec le prétraitement LMDEval | Sprint S7 (janvier-février), pas S6 : 1 To ne se traite pas en décembre |
| Deuxième pli temporel (jours 14-29, 54 campagnes) | doublerait les preuves ; ≈ 545 M d'événements à convertir sur la VM | S7, ou fin S6 si le run n° 1 se passe bien |

## 7. Risques

| Risque | Parade |
|---|---|
| Euler ne retrouve pas ses chiffres publiés | C'est un résultat en soi (Larroche et Zhao-Hassan observent aussi de forts écarts) ; documenter le protocole exact |
| Azure for Students refuse l'adresse UoPeople | Kaggle pour le GPU ; Vast.ai à 5 $ pour la RAM |
| Décembre plus chargé que prévu | Le bloc A et S6-05/S6-06 suffisent à eux seuls à un papier corrigé et plus fort ; le bloc C peut glisser en S7 sans rien casser |
| Un troisième papier concurrent sort | Veille arXiv `cs.CR` « LANL lateral movement » au début de chaque semaine du sprint |

## 8. Sources principales

- Larroche 2026 : arxiv.org/abs/2607.29390 · code : github.com/cl-anssi/LMDEval
- Larroche 2025 (UltraLMD++) : arxiv.org/abs/2504.13527
- Zhao et Hassan 2026 : arxiv.org/abs/2609.12263
- Garchery et Granitzer (ADSAGE) : arxiv.org/abs/2007.06985, table 5
- Tuor et al. 2018 : arxiv.org/abs/1712.00557
- King et Huang (Euler), NDSS 2022 : ndss-symposium.org/wp-content/uploads/2022-107A-paper.pdf
- DIMVA 2027 : dimva.org/dimva2027/ · TMLR : jmlr.org/tmlr/editorial-policies.html · endossement arXiv : blog.arxiv.org/2026/01/21/attention-authors-updated-endorsement-policy
- Azure for Students : azure.microsoft.com/en-us/free/students · Kaggle : kaggle.com/docs/notebooks

## 9. Après la revue simulée du 21 octobre (round 1)

Trois relecteurs indépendants : deux « révision majeure », un « rejet en l'état, resoumission encouragée ». Réponse point par point : `reports/paper/review_round1.md`. Le texte a été révisé (titre, cadrage en étude de cas, loi nulle exacte de b\*, corrélations avec p exactes et sans M1, diagnostics de M2a, comparaison avec DECADES, ADSAGE et UltraLMD++). Ce qui exige un nouveau run devient le cœur du sprint.

**Nouveaux tickets**

- **S6-23 · M · 3 pts** : bootstrap révisé. R = 10 000 ; rééchantillonnage par compte plutôt que par campagne ; intervalles BCa ou studentisés ; erreur de Monte Carlo publiée ; docstrings « stratified » corrigées.
- **S6-24 · S · 3 pts** : variance d'ajustement. Cinq graines et sous-échantillons pour PCA, Isolation Forest, HBOS et M1, et bootstrap à grappes côté bénin (par utilisateur source) en analyse de sensibilité.
- **S6-25 · S · 3 pts** : variante à seuil fixe. Seuil calibré sur la validation pour viser k alertes par jour ; volume d'alertes et rappel réels sur le test.
- **S6-26 · M · 5 pts** : découpage harmonisé de Larroche (2026) en configuration principale. Entraînement sur les 41 premières heures, test sur toute l'activité red team ensuite, jours 30 à 57 pour les faux positifs.
- **S6-27 · S · 2 pts** : rareté invariante dans le temps (dénominateur fixé sur l'entraînement), pour supprimer le classement « le plus tardif d'abord » de M2a.
- **S6-28 · M · 5 pts** : côté source. Nouveauté et rareté (utilisateur, source) et (source, destination), règle « machine source connue » (ADSAGE) et règle UA de Bowman.
- **S6-29 · M · 5 pts** : univers et unités. Univers LogOn seul, en plus de l'univers complet ; budgets en événements, en arêtes (source, destination, heure) et en utilisateurs-jours ; un modèle d'effectif SOC sourcé pour fixer la plage de budgets.

**Priorité de décembre (capacité d'environ 40 h)** : S6-26, S6-28, S6-05/06, S6-20, puis S6-09 (Euler) si le temps le permet. S6-23 et S6-29 tournent dans le même run. S6-24, S6-25 et S6-27 passent en S7 si nécessaire.

**Recommandation sur la v2 publique (décision à Mahamane)** : ne pas publier la v2 sur Zenodo avant ce run. Le texte révisé est honnête, mais la revue montre que sa contribution empirique n'est pas encore au niveau d'une publication ; la v2 publique doit être la version d'après le sprint.

## 10. Avancement (22 octobre)

- **S6-05/06, S6-20, S6-27, S6-28** : faits sur `s6/sprint` (registre des positifs, normalisation de M1 ajustée sur l'entraînement, rareté ancrée au jour, côté source avec M2c et R8, M1 évalué ligne par ligne).
- **S6-26** : terminé. Calcul des variables jour par jour (`features/incremental.py`, `build_store`), égal bit pour bit au calcul sur la fenêtre entière ; bornes en heures, historique `window` ou `cumulative`, `conf/split/harmonised.yaml`. Évaluation jour par jour (`evaluate/statistics.py`) : chaque métrique est calculée à partir de statistiques suffisantes accumulées par jour et testée égale à sa définition sur la trame entière ; la mémoire suit le plus gros jour (872 octets par événement, mesuré sur le jour 12 réel). Bootstrap sur statistiques suffisantes : 10 000 rééchantillons de neuf modèles en 0,7 s ; sur des scores identiques, mêmes verdicts que l'ancien schéma sur 28 paires sur 28.
- **Défauts corrigés en chemin (22 octobre)** : cinq modèles n'étaient pas locaux à la ligne au bit près (PCA, M4a, M4b par l'ordre de sommation BLAS et torch ; M0a tiré par position ; ECOD transductif), masqués par un test à tolérance 1e-12 ; rappel à FPR fixe sous-estimé par `drop_intermediate` ; précision@k départagée par l'ordre des lignes ; modèle mémoire de `preflight` obsolète. Chaque correction a son test.
- **S6-29** : univers `all`, `remote_logon`, `remote_logon_user` faits ; unités (arêtes, utilisateurs-jours) et modèle d'effectif à faire.
- **Mesure sur la machine de 8 Go (jours 0 à 2 réels)** : univers LogOn distant, pic de 4,7 Go et environ 50 s par jour ; univers complet, échec d'allocation dès le jour 0. Le run complet reste prévu sur Hetzner (`preflight` : environ 55 Go de disque et 19 Go de RAM pour LANL entier).
- **Après le run** : le papier devra décrire ECOD ajusté sur l'entraînement, M0a haché, le bootstrap sur statistiques suffisantes, et les nouvelles définitions de précision@k, rappel à FPR fixe et résolution du score.

# Anomaly Explainer

Détection et explication des fraudes bancaires avec un **Transformer
Auto-Encodeur** et un **LLM local (Ollama)**.

Le système détecte les transactions anormales via l'erreur de reconstruction d'un
auto-encodeur entraîné **uniquement sur des transactions normales**, puis un LLM
génère une explication en langage naturel : attributs responsables,
interprétation, niveau de risque, pistes d'analyse. Tout s'exécute en local —
aucune donnée de transaction ne quitte la machine.

---

## Résultats

Jeu de **test** (42 894 transactions, 246 fraudes), utilisé une seule fois après
calibration du seuil sur la validation :

| Métrique | Valeur |
|----------|--------|
| Précision | **0,842** |
| Rappel | **0,715** |
| F1 | **0,774** |
| PR-AUC | **0,742** *(baseline aléatoire : 0,0057 — ×129)* |
| ROC-AUC | 0,964 |
| Faux positifs | **33** sur 42 648 transactions normales (**0,08 %**) |

Seuil : 99,9ᵉ percentile des erreurs de reconstruction normales (`1,2634`),
choisi par balayage sur la **validation** puis évalué une seule fois sur le
**test**.

> **Pourquoi PR-AUC et non l'exactitude ?** Avec 0,17 % de fraudes, prédire
> « normal » partout donne 99,83 % d'exactitude. La PR-AUC est la métrique
> honnête ici ; la ROC-AUC (0,964) paraît flatteuse car les vrais négatifs
> dominent.

Deux vérifications de rigueur ont été menées au-delà du cahier des charges (voir
[Vérifications de rigueur](#vérifications-de-rigueur) plus bas) : l'attention
apporte-t-elle réellement quelque chose face à un auto-encodeur ordinaire, et le
découpage aléatoire des données embellit-il les résultats.

---

## Architecture

```
  creditcard.csv
        │
        ▼
┌───────────────────┐  validation de schéma, split normal/fraude,
│  data/            │  scaler ajusté sur le train UNIQUEMENT
└─────────┬─────────┘
          ▼
┌───────────────────┐  attention multi-têtes → latent (16) → décodeur
│  model/           │  entraîné SUR LES TRANSACTIONS NORMALES SEULEMENT
└─────────┬─────────┘  → artefacts : model.pt, scaler.pkl, config.json
          ▼
┌───────────────────┐  erreur par attribut + erreur totale
│  detection/       │  seuil statistique, score de sévérité (erreur / seuil)
└─────────┬─────────┘
          ▼
┌───────────────────┐  balayage de seuil sur la validation → threshold.json
│  evaluation/      │  P / R / F1 / PR-AUC sur le test
└─────────┬─────────┘
          ▼
┌───────────────────┐  valeurs d'origine + reconstruction + erreur + sévérité
│  explain/         │  → prompt → Ollama → explication structurée
└─────────┬─────────┘  (repli déterministe si le LLM est indisponible)
          ▼
┌───────────────────┐  file d'attente triée par sévérité, justification
│  app/ + service   │  de chaque détection
└───────────────────┘
```

| Module | Rôle |
|--------|------|
| `data/` | Chargement, validation de schéma, prétraitement (immutable) |
| `model/` | Transformer auto-encodeur, baseline dense, entraînement, persistance |
| `detection/` | Reconstruction, erreur par attribut, seuil, sévérité |
| `evaluation/` | Métriques (P/R/F1/PR-AUC) et balayage de seuil |
| `explain/` | Contexte, prompt, client Ollama, explication structurée |
| `service.py` | Point de composition : toute la logique du tableau de bord |
| `app/` | Interface Streamlit (présentation uniquement) |

---

## Installation

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

### Données (Kaggle)

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
— 284 807 transactions, 492 fraudes (≈ 0,17 %).

```bash
# Option A — token API Kaggle dans ~/.kaggle/kaggle.json
PYTHONPATH=src ./.venv/bin/python scripts/download_data.py
# Option B — télécharger creditcard.csv manuellement dans data/
```

### LLM local (Ollama)

```bash
curl -fsSL https://ollama.com/install.sh | sh   # nécessite sudo
ollama pull llama3.2:3b                          # ~2 Go

ollama list                                      # vérifier le modèle
```

Performance mesurée (CPU uniquement, ~10 tokens/s) : **6-9 s par explication** à
chaud, ~19 s à froid.

> Ollama décharge un modèle inactif au bout de 5 min, ce qui rendrait la première
> explication après une pause 3× plus lente : `ExplainerConfig.keep_alive` est
> donc fixé à `30m`. Les modèles sont stockés sur la partition **racine**
> (`/usr/share/ollama/.ollama/models`), pas dans `/home`.

---

## Utilisation

Les étapes sont **ordonnées** : l'évaluation produit le seuil dont dépendent
l'explication et l'interface.

```bash
# 1. Entraînement (détecte CUDA automatiquement, sinon CPU)
PYTHONPATH=src ./.venv/bin/python scripts/train_model.py

# 2. Évaluation + calibration du seuil → data/artifacts/threshold.json
PYTHONPATH=src ./.venv/bin/python scripts/evaluate.py

# 3a. Expliquer une transaction en ligne de commande
PYTHONPATH=src ./.venv/bin/python scripts/explain_transaction.py
PYTHONPATH=src ./.venv/bin/python scripts/explain_transaction.py --false-positive
PYTHONPATH=src ./.venv/bin/python scripts/explain_transaction.py --no-llm

# 3b. …ou via l'interface analyste
PYTHONPATH=src ./.venv/bin/streamlit run app/streamlit_app.py
```

Tous les hyperparamètres (taille du modèle, époques, seuil, modèle LLM) sont
centralisés dans `src/anomaly_explainer/config.py`.

### Interface analyste

Le tableau de bord affiche les indicateurs globaux, une **file d'attente triée
par sévérité** (le plus suspect en premier), et pour chaque transaction : le
verdict, la sévérité par rapport au seuil, les attributs responsables, et
l'explication du LLM avec son niveau de risque.

> Les écarts sont tracés en **écarts-types** : les unités brutes ne sont pas
> comparables entre attributs (`Amount` en milliers, composantes PCA proches de
> zéro). Les valeurs exactes en unités d'origine restent dans le tableau associé.

L'interface fonctionne aussi **sans labels** : la colonne `Class` n'existe que
parce qu'il s'agit d'un jeu de référence, et une interface qui s'effondrerait
sans elle ne serait pas réutilisable en production.

---

## Vérifications de rigueur

### L'attention apporte-t-elle quelque chose ?

```bash
PYTHONPATH=src ./.venv/bin/python scripts/compare_baseline.py --epochs 150
```

Le cahier des charges impose un Transformer au motif que l'attention capte les
interactions entre variables. C'est une **affirmation**, donc elle est testée
contre un témoin : mêmes splits, même graine, même boucle d'entraînement, même
goulot latent, même règle de seuil. Seule l'architecture change.

| Modèle | Params | PR-AUC | F1 | Faux positifs |
|--------|-------:|-------:|---:|--------------:|
| **Transformer AE (spec)** | 269 137 | **0,7422** | **0,774** | **33** |
| Dense AE (goulot équivalent) | 9 198 | 0,6635 | 0,624 | 52 |
| Dense AE (capacité équivalente) | 302 638 | 0,7138 | 0,687 | 62 |

Le Transformer l'emporte, y compris face à un modèle dense **plus gros** — donc
ce n'est pas un simple effet de capacité — et surtout avec **~2× moins de faux
positifs**.

> **Une meilleure reconstruction n'implique pas une meilleure détection.** Le
> dense large atteint une erreur de reconstruction bien plus faible (0,031 vs
> 0,126) mais détecte moins bien : trop de capacité lui permet de reconstruire
> aussi les fraudes, ce qui annule l'écart d'erreur sur lequel repose la
> détection. Le goulot n'est pas une contrainte à minimiser, c'est le mécanisme.

### Le découpage aléatoire est-il trop optimiste ?

```bash
PYTHONPATH=src ./.venv/bin/python scripts/chronological_check.py --arch dense
```

Le jeu de données est un enregistrement continu de 48 h. Un découpage aléatoire
permet au modèle d'être entraîné sur l'heure 40 et évalué sur l'heure 3 — ce
qu'un système en production ne peut jamais faire.

| Découpage | Fraudes (test) | PR-AUC | ROC-AUC | F1 |
|-----------|---------------:|-------:|--------:|---:|
| Aléatoire (rapporté) | 246 | 0,7138 | 0,9545 | 0,687 |
| Chronologique | 52 | **0,5314** | 0,9184 | 0,562 |

**La PR-AUC chute de 26 %.** Le découpage aléatoire embellit donc nettement les
résultats. À noter : la ROC-AUC bouge à peine (0,954 → 0,918) et masque
l'essentiel du problème — confirmation indépendante que la PR-AUC est la bonne
métrique de référence.

> ⚠️ **Limites.** Le découpage chronologique ne conserve que **52 fraudes de test
> contre 246** (384 fraudes tombent dans la fenêtre d'entraînement) : barres
> d'erreur larges, résultat *indicatif*. Mesuré sur l'auto-encodeur **dense**
> comme substitut rapide ; l'équivalent Transformer reste à exécuter
> (`--arch transformer`, ~3-5 h sur CPU).

Cette limite n'invalide pas la comparaison ci-dessus, menée sous un découpage
unique et cohérent.

---

## Entraîner sur une autre machine (GPU recommandé)

L'entraînement du Transformer sur CPU est lent (~2-4 min/époque) :

1. Copier le projet **+ `data/creditcard.csv`** sur la machine cible.
2. `pip install -r requirements.txt` — de préférence une build PyTorch CUDA.
3. `PYTHONPATH=src python scripts/train_model.py` (affiche le *device* utilisé).
4. Récupérer **`data/artifacts/`** et le copier sur la machine d'analyse, puis y
   relancer `scripts/evaluate.py` pour produire `threshold.json`.

---

## Tests

169 tests (+ 5 tests d'interface optionnels), **99 % de couverture**.

```bash
./.venv/bin/pytest --cov=src/anomaly_explainer

# Test d'intégration de l'interface (charge le dataset complet, ~1 min)
RUN_APP_TEST=1 ./.venv/bin/pytest tests/test_streamlit_app.py
```

`pytest.ini` définit déjà `pythonpath = src` : inutile de préfixer par
`PYTHONPATH=src` pour lancer les tests.

Le journal de développement — décisions, mesures et impasses — est dans
[`PROGRESS.md`](PROGRESS.md).

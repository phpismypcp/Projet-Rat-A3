# Anomaly Explainer

Détection et explication des fraudes bancaires avec un **Transformer Auto-Encodeur**
et un **LLM local (Ollama)**.

Le système détecte automatiquement les transactions anormales via l'erreur de
reconstruction d'un auto-encodeur entraîné uniquement sur des transactions
normales, puis un LLM génère une explication en langage naturel (attributs
responsables, interprétation, niveau de risque, pistes d'analyse).

## Architecture

```
Prétraitement ──▶ Transformer Auto-Encodeur ──▶ Moteur de détection ──▶ LLM Explainer
  (scaling,          (attention multi-têtes,        (erreur + seuil,        (Ollama local,
   split normal)      entraîné sur normal)           score de sévérité)      explication)
```

Modules (`src/anomaly_explainer/`):

| Module        | Rôle                                                        |
|---------------|-------------------------------------------------------------|
| `data/`       | Chargement, validation de schéma, prétraitement (immutable) |
| `model/`      | Transformer auto-encodeur, entraînement, persistance        |
| `detection/`  | Erreur de reconstruction, seuil statistique, sévérité       |
| `evaluation/` | Métriques (P/R/F1/PR-AUC) et balayage de seuil               |
| `explain/`    | Prompt, client Ollama, explication structurée               |
| `app/`        | Interface analyste Streamlit                                |

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

### Données (Kaggle)

Dataset: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(284 807 transactions, 492 fraudes ≈ 0,17 %).

```bash
# Option A — token API Kaggle dans ~/.kaggle/kaggle.json
PYTHONPATH=src ./.venv/bin/python scripts/download_data.py
# Option B — télécharger creditcard.csv manuellement dans data/
```

### LLM local (Ollama)

```bash
curl -fsSL https://ollama.com/install.sh | sh   # nécessite sudo
ollama pull llama3.2:3b                          # ~2 Go

# Vérifier que le modèle est disponible
ollama list
curl -s http://localhost:11434/api/tags
```

Performance mesurée sur cette machine (CPU uniquement, ~10 tokens/s) :
**6-9 s par explication** à chaud, ~19 s à froid (rechargement du modèle en RAM).

> Ollama décharge un modèle inactif au bout de 5 min par défaut, ce qui rendrait
> la première explication après une pause 3x plus lente. `ExplainerConfig.keep_alive`
> est donc fixé à `30m`. Les modèles sont stockés sur la partition **racine**
> (`/usr/share/ollama/.ollama/models`), pas dans `/home`.

## Utilisation

```bash
# Entraînement (détecte automatiquement le GPU/CUDA, sinon CPU)
PYTHONPATH=src ./.venv/bin/python scripts/train_model.py

# Évaluation + calibration du seuil (écrit threshold.json dans data/artifacts/)
PYTHONPATH=src ./.venv/bin/python scripts/evaluate.py

# Détecter + expliquer une transaction (pipeline complet, LLM local)
PYTHONPATH=src ./.venv/bin/python scripts/explain_transaction.py
PYTHONPATH=src ./.venv/bin/python scripts/explain_transaction.py --false-positive
PYTHONPATH=src ./.venv/bin/python scripts/explain_transaction.py --no-llm

# Interface analyste
PYTHONPATH=src ./.venv/bin/streamlit run app/streamlit_app.py
```

> `scripts/evaluate.py` doit être lancé **après** l'entraînement : il calibre le
> seuil de détection sur le jeu de validation et l'enregistre dans
> `data/artifacts/threshold.json`, que le détecteur et l'interface réutilisent.

## Résultats (jeu de test, 42 894 transactions dont 246 fraudes)

| Métrique | Valeur |
|----------|--------|
| Précision | 0,842 |
| Rappel | 0,715 |
| F1 | 0,774 |
| PR-AUC | 0,742 *(baseline aléatoire : 0,0057)* |
| ROC-AUC | 0,964 |
| Faux positifs | 33 sur 42 648 transactions normales (0,08 %) |

Seuil retenu : 99,9ᵉ percentile des erreurs de reconstruction normales
(`1,2634`), choisi par balayage sur le jeu de **validation** puis évalué une
seule fois sur le jeu de **test**.

### Entraîner sur une autre machine (GPU recommandé)

L'entraînement du Transformer sur CPU est lent (~2 min/époque). Pour l'exécuter
sur une machine plus puissante (GPU) :

1. Copier le projet **+ `data/creditcard.csv`** sur la machine cible.
2. Installer les dépendances (`pip install -r requirements.txt`) — de préférence
   une build PyTorch avec CUDA.
3. Lancer `PYTHONPATH=src python scripts/train_model.py`. Le script affiche le
   *device* utilisé (`cuda` si disponible) et entraîne avec early stopping.
4. Récupérer le dossier **`data/artifacts/`** (model.pt, scaler.pkl, config.json)
   et le copier dans `data/artifacts/` de cette machine pour la détection et
   l'interface.

Les hyperparamètres (taille du modèle, époques, seuil) se règlent uniquement
dans `src/anomaly_explainer/config.py`.

## Tests

```bash
PYTHONPATH=src ./.venv/bin/pytest --cov=src/anomaly_explainer
```

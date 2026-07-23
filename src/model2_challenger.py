"""
Modelo 2 (desafiante): XGBoost treinado com application_train + o book de
variaveis do bureau (ABT completa, 375 features).

Mesma metodologia do Modelo 1 (holdout 80/20 estratificado, random_state=42,
mesma familia de hiperparametros) -- a unica diferenca deliberada e a
presenca das variaveis do bureau, pra a comparacao de AUC-ROC isolar esse
efeito.

Nota de execucao: no ambiente local de demonstracao, treinar direto (ler
parquet + splitar + treinar, tudo no mesmo processo) estourou memoria.
Rodar em dois passos -- (1) gerar e salvar train/val em parquet, (2) treinar
lendo esses arquivos num processo novo -- resolveu, porque cada etapa fica
com um pico de memoria bem menor. E a mesma logica ja aplicada no join da
camada Gold (ver docs/04_gold.md).

Uso:
    python3 model2_challenger.py --step split --gold-dir data/gold --split-out-dir data/gold_stage
    python3 model2_challenger.py --step train --data-dir data/gold_stage --models-dir models
"""
from __future__ import annotations

import argparse
import gc
import json

import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

PARAMS = dict(n_estimators=150, max_depth=5, learning_rate=0.08)


def step_split(gold_dir: str, out_dir: str) -> None:
    abt = pd.read_parquet(f"{gold_dir}/abt")
    y = abt["target"].astype(int)
    X = abt.drop(columns=["application_id", "target"])
    del abt
    gc.collect()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    del X, y
    gc.collect()

    X_train["__y__"] = y_train.values
    X_val["__y__"] = y_val.values
    X_train.to_parquet(f"{out_dir}/model2_train.parquet", index=False)
    X_val.to_parquet(f"{out_dir}/model2_val.parquet", index=False)
    print(f"split salvo: train={X_train.shape}, val={X_val.shape}")


def step_train(data_dir: str, out_dir: str) -> None:
    train = pd.read_parquet(f"{data_dir}/model2_train.parquet")
    val = pd.read_parquet(f"{data_dir}/model2_val.parquet")
    y_train = train.pop("__y__")
    y_val = val.pop("__y__")

    model = xgb.XGBClassifier(
        tree_method="hist",
        enable_categorical=True,
        eval_metric="auc",
        n_jobs=2,
        random_state=42,
        **PARAMS,
    )
    model.fit(train, y_train, verbose=False)
    model.save_model(f"{out_dir}/model2_challenger.json")

    auc_train = roc_auc_score(y_train, model.predict_proba(train)[:, 1])
    val_preds = model.predict_proba(val)[:, 1]
    auc_val = roc_auc_score(y_val, val_preds)
    print(f"AUC train: {auc_train:.4f} | AUC val: {auc_val:.4f}")

    fpr, tpr, _ = roc_curve(y_val, val_preds)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(f"{out_dir}/model2_roc.csv", index=False)

    with open(f"{out_dir}/model2_metrics.json", "w") as f:
        json.dump(
            {
                "auc_train": float(auc_train),
                "auc_val": float(auc_val),
                "n_features": int(train.shape[1]),
                "n_train": int(len(train)),
                "n_val": int(len(val)),
                "params": PARAMS,
            },
            f,
            indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["split", "train"], required=True)
    parser.add_argument("--gold-dir", default="data/gold")
    parser.add_argument("--split-out-dir", default="data/gold_stage")
    parser.add_argument("--data-dir", default="data/gold_stage")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()

    if args.step == "split":
        step_split(args.gold_dir, args.split_out_dir)
    else:
        step_train(args.data_dir, args.models_dir)


if __name__ == "__main__":
    main()

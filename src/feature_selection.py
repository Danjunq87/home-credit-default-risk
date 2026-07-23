"""
Selecao de variaveis via Feature Importance (XGBoost, importancia por gain).

Roda em duas versoes, pra comparar o ranking e o AUC:
  1. so application_train (silver) -- 126 features.
  2. ABT completa (application_train + book do bureau) -- 375 features.

Nota sobre amostragem: no ambiente local de demonstracao (2 vCPU / ~2.8GB
RAM), treinar XGBoost com as 375 colunas da ABT completa (307 mil linhas)
nao coube no tempo/memoria disponiveis de uma vez. Para ESTA etapa
(ranking de importancia, nao o modelo final da etapa 8/9), uma amostra
aleatoria estratificada de 150 mil linhas e suficiente para estabilizar o
ranking de importancia -- e o que este script faz para a versao "full".
O modelo final (etapas 8 e 9) usa a base inteira.

Uso:
    python3 feature_selection.py --which app_only
    python3 feature_selection.py --which full_abt
"""
from __future__ import annotations

import argparse

import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

SAMPLE_SIZE_FULL = 150_000


def run(which: str, silver_dir: str, gold_dir: str, out_dir: str) -> None:
    if which == "app_only":
        df = pd.read_parquet(f"{silver_dir}/application_train_clean.parquet")
        params = dict(n_estimators=200, max_depth=5, learning_rate=0.08)
    else:
        df = pd.read_parquet(f"{gold_dir}/abt")
        df = df.sample(n=SAMPLE_SIZE_FULL, random_state=42).reset_index(drop=True)
        params = dict(n_estimators=100, max_depth=4, learning_rate=0.1)

    y = df["target"].astype(int)
    X = df.drop(columns=["application_id", "target"])
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        tree_method="hist",
        enable_categorical=True,
        eval_metric="auc",
        n_jobs=2,
        random_state=42,
        **params,
    )
    model.fit(X_train, y_train, verbose=False)

    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    print(f"[{which}] AUC validacao: {auc:.4f}")

    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    imp = pd.Series(gain).sort_values(ascending=False)
    cols = list(X_train.columns)
    imp.index = [
        cols[int(c[1:])] if c.startswith("f") and c[1:].isdigit() else c for c in imp.index
    ]
    imp.to_csv(f"{out_dir}/importance_{which}.csv", header=["gain"])
    print(imp.head(20))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["app_only", "full_abt"], required=True)
    parser.add_argument("--silver-dir", default="data/silver")
    parser.add_argument("--gold-dir", default="data/gold")
    parser.add_argument("--out-dir", default="data/gold_stage")
    args = parser.parse_args()
    run(args.which, args.silver_dir, args.gold_dir, args.out_dir)


if __name__ == "__main__":
    main()

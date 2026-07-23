"""
Modelo 1 (baseline): XGBoost treinado so com application_train (silver),
126 features -- sem nenhuma informacao do bureau.

Existe pra dar a linha de base contra a qual o Modelo 2 (desafiante, com o
book do bureau) e comparado. Mesma metodologia de split (holdout 80/20,
estratificado, random_state=42) e mesma familia de hiperparametros do
Modelo 2, pra a comparacao de AUC-ROC ser justa.

Uso:
    python3 model1_baseline.py --silver-dir data/silver --out-dir models
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

PARAMS = dict(n_estimators=150, max_depth=5, learning_rate=0.08)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--silver-dir", default="data/silver")
    parser.add_argument("--out-dir", default="models")
    args = parser.parse_args()

    df = pd.read_parquet(f"{args.silver_dir}/application_train_clean.parquet")
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
        **PARAMS,
    )
    model.fit(X_train, y_train, verbose=False)
    model.save_model(f"{args.out_dir}/model1_baseline.json")

    auc_train = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    val_preds = model.predict_proba(X_val)[:, 1]
    auc_val = roc_auc_score(y_val, val_preds)
    print(f"AUC train: {auc_train:.4f} | AUC val: {auc_val:.4f}")

    fpr, tpr, _ = roc_curve(y_val, val_preds)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(f"{args.out_dir}/model1_roc.csv", index=False)

    with open(f"{args.out_dir}/model1_metrics.json", "w") as f:
        json.dump(
            {
                "auc_train": float(auc_train),
                "auc_val": float(auc_val),
                "n_features": int(X.shape[1]),
                "n_train": int(len(X_train)),
                "n_val": int(len(X_val)),
                "params": PARAMS,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()

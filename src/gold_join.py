"""
Junta as pecas do book (geradas por gold_build.py) com application_train
(silver) na ABT final.

Por que pandas e nao PySpark aqui: o join final envolve tabelas ja bem
menores (~306 mil linhas x ate 249 colunas cada), e nesse tamanho o overhead
de um join distribuido em Spark (serializacao, shuffle, JVM) custou mais do
que fazer o merge localmente. Em um cluster real (mais memoria/executors)
faria sentido manter tudo em Spark; aqui, no ambiente local de demonstracao
(2 vCPU / ~2.8GB RAM), o pandas com processamento em chunks foi o caminho
que efetivamente rodou sem estourar memoria. Essa e uma decisao de
engenharia real, documentada em docs/04_gold.md, nao um atalho escondido.

Uso:
    python3 gold_join.py --silver-dir data/silver --stage-dir data/gold_stage --gold-dir data/gold
"""
from __future__ import annotations

import argparse
import csv
import gc
import os

import pandas as pd


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
        elif df[c].dtype == "int64":
            df[c] = pd.to_numeric(df[c], downcast="integer")
    return df


def build_bureau_book(stage_dir: str, gold_dir: str) -> pd.DataFrame:
    book = downcast(pd.read_parquet(f"{stage_dir}/meta")).set_index("application_id")
    for name in ["agg_all", "agg_active", "agg_closed", "cats"]:
        part = downcast(pd.read_parquet(f"{stage_dir}/{name}")).set_index("application_id")
        book = book.join(part, how="left")
        del part
        gc.collect()
    book = book.reset_index()

    os.makedirs(gold_dir, exist_ok=True)
    book.to_parquet(f"{gold_dir}/bureau_book.parquet", index=False)

    with open(f"{gold_dir}/../feature_dictionary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "origin"])
        for c in book.columns:
            if c != "application_id":
                writer.writerow([c, "bureau_book"])

    print(f"[book] {book.shape[0]:,} linhas, {book.shape[1]} colunas -> {gold_dir}/bureau_book.parquet")
    return book


def build_abt(silver_dir: str, gold_dir: str, n_chunks: int = 6) -> None:
    book = pd.read_parquet(f"{gold_dir}/bureau_book.parquet")
    count_cols = [c for c in book.columns if c != "application_id" and "_count" in c]

    app = pd.read_parquet(f"{silver_dir}/application_train")
    app = app.drop(columns=["_ingested_at", "_source_file"], errors="ignore")
    for c in app.columns:
        if app[c].dtype == "float64":
            app[c] = app[c].astype("float32")
        elif app[c].dtype == "object":
            app[c] = app[c].astype("category")

    out_dir = f"{gold_dir}/abt"
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        os.remove(f"{out_dir}/{f}")

    chunk_size = (len(app) + n_chunks - 1) // n_chunks
    for i in range(n_chunks):
        part = app.iloc[i * chunk_size : (i + 1) * chunk_size]
        merged = part.merge(book, on="application_id", how="left")
        merged["has_bureau_history"] = merged["bureau_credit_count"].fillna(0) > 0
        for c in count_cols:
            merged[c] = merged[c].fillna(0)
        merged.to_parquet(f"{out_dir}/part-{i:02d}.parquet", index=False)
        print(f"[abt] chunk {i}: {merged.shape}")
        del part, merged
        gc.collect()

    print(f"[abt] concluido -> {out_dir} (colunas: {len(app.columns) + len(book.columns) - 1})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--silver-dir", default="data/silver")
    parser.add_argument("--stage-dir", default="data/gold_stage")
    parser.add_argument("--gold-dir", default="data/gold")
    args = parser.parse_args()

    build_bureau_book(args.stage_dir, args.gold_dir)
    build_abt(args.silver_dir, args.gold_dir)


if __name__ == "__main__":
    main()

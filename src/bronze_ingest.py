"""
Camada Bronze — ingestao bruta.

Le application_train.csv e bureau.csv exatamente como vieram do Kaggle
(sem renomear coluna, sem limpar nada) e grava em Parquet, particionado
por tabela, com metadados minimos de ingestao (timestamp e arquivo de
origem). Esta e a unica responsabilidade da camada Bronze: ser uma copia
fiel do dado bruto em um formato colunar mais eficiente que CSV, servindo
de base imutavel para as camadas seguintes (Silver/Gold).

Uso:
    spark-submit bronze_ingest.py --raw-dir data/raw --bronze-dir data/bronze
"""
from __future__ import annotations

import argparse

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

TABLES = ["application_train", "bureau"]


def build_spark() -> SparkSession:
    return (
        SparkSession.builder.appName("home-credit-bronze")
        .master("local[*]")
        .config("spark.driver.memory", "1500m")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


def ingest_table(spark: SparkSession, name: str, raw_dir: str, bronze_dir: str) -> None:
    src = f"{raw_dir}/{name}.csv"
    dst = f"{bronze_dir}/{name}"

    df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(src)
    )
    df = df.withColumn("_ingested_at", F.current_timestamp()).withColumn(
        "_source_file", F.lit(f"{name}.csv")
    )
    df.write.mode("overwrite").parquet(dst)

    n_rows = df.count()
    n_cols = len(df.columns)
    print(f"[bronze] {name}: {n_rows:,} linhas, {n_cols} colunas -> {dst}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--bronze-dir", default="data/bronze")
    args = parser.parse_args()

    spark = build_spark()
    try:
        for table in TABLES:
            ingest_table(spark, table, args.raw_dir, args.bronze_dir)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

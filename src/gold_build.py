"""
Camada Gold -- book de variaveis do bureau agregado na granularidade de
application_train.

A ideia: cada application_id pode ter 0, 1 ou N registros em bureau (um por
credito anterior relatado a central de credito). Um modelo de risco precisa
de UMA linha por application_id, entao o historico do bureau precisa virar
"features resumo" -- min/max/media/soma de cada indicador numerico, tanto
no geral quanto quebrado por status do credito (ativo/encerrado), mais
contagens e proporcoes por categoria (status, tipo de credito).

Isso gera 248 variaveis derivadas do bureau (o "book"), documentadas em
docs/feature_dictionary.csv. Junto com as 128 colunas do application_train
(silver, sem as 2 colunas de metadado de ingestao), a ABT final tem 377
colunas.

Executa em estagios (--stage) para caber em ambientes com pouco recurso
(o ambiente de demonstracao deste projeto tem 2 vCPU / ~2.8GB RAM); cada
estagio le/escreve parquet intermediario e pode ser re-executado
(idempotente) sem refazer o trabalho dos estagios anteriores. Em um cluster
Spark de verdade (mais memoria/executors), os estagios all/active/closed/cats
rodariam em paralelo sem essa quebra manual.

Uso:
    python3 gold_build.py --stage meta
    python3 gold_build.py --stage all
    python3 gold_build.py --stage active
    python3 gold_build.py --stage closed
    python3 gold_build.py --stage cats

Depois rode gold_join.py para montar o book final e a ABT (feito em pandas
nesta demo por limite de memoria do ambiente local -- ver docs/04_gold.md).
"""
from __future__ import annotations

import argparse
import json
import os

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

NUMERIC_COLS = [
    "days_credit", "days_overdue", "days_to_credit_end", "days_enddate_fact",
    "max_overdue_amount", "credit_prolongation_count", "credit_amount",
    "credit_debt_amount", "credit_limit_amount", "credit_overdue_amount",
    "days_credit_update", "annuity_amount", "days_since_credit_started",
    "days_since_credit_closed", "days_since_credit_update",
    "debt_to_credit_ratio", "overdue_to_credit_ratio", "limit_usage_ratio",
]
KNOWN_STATUSES = ["Active", "Closed", "Sold", "Bad debt"]
TOP_N_CREDIT_TYPES = 8

SILVER_DIR = "data/silver"
STAGE_DIR = "data/gold_stage"


def build_spark() -> SparkSession:
    spark = (
        SparkSession.builder.appName("home-credit-gold")
        .master("local[*]")
        .config("spark.driver.memory", "1500m")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def add_ratio_columns(bureau: DataFrame) -> DataFrame:
    return (
        bureau.withColumn(
            "debt_to_credit_ratio",
            F.when(F.col("credit_amount") > 0, F.col("credit_debt_amount") / F.col("credit_amount")),
        )
        .withColumn(
            "overdue_to_credit_ratio",
            F.when(F.col("credit_amount") > 0, F.col("credit_overdue_amount") / F.col("credit_amount")),
        )
        .withColumn(
            "limit_usage_ratio",
            F.when(F.col("credit_limit_amount") > 0, F.col("credit_debt_amount") / F.col("credit_limit_amount")),
        )
    )


def load_bureau(spark) -> DataFrame:
    bureau = spark.read.parquet(f"{SILVER_DIR}/bureau")
    return add_ratio_columns(bureau)


def agg_segment(bureau: DataFrame, prefix: str) -> DataFrame:
    aggs = []
    for c in NUMERIC_COLS:
        aggs.append(F.min(c).alias(f"bureau_{prefix}_{c}_min"))
        aggs.append(F.max(c).alias(f"bureau_{prefix}_{c}_max"))
        aggs.append(F.mean(c).alias(f"bureau_{prefix}_{c}_mean"))
        aggs.append(F.sum(c).alias(f"bureau_{prefix}_{c}_sum"))
    return bureau.groupBy("application_id").agg(*aggs)


def stage_meta(spark):
    bureau = load_bureau(spark)
    meta = bureau.groupBy("application_id").agg(
        F.count("*").alias("bureau_credit_count"),
        F.sum(F.when(F.col("credit_status") == "Active", 1).otherwise(0)).alias("bureau_active_count"),
        F.sum(F.when(F.col("credit_status") == "Closed", 1).otherwise(0)).alias("bureau_closed_count"),
        F.countDistinct("credit_type").alias("bureau_distinct_credit_types"),
        F.max("days_since_credit_started").alias("bureau_days_since_first_credit"),
        F.min("days_since_credit_started").alias("bureau_days_since_last_credit"),
    )
    meta.write.mode("overwrite").parquet(f"{STAGE_DIR}/meta")

    top_types = [
        row["credit_type"]
        for row in bureau.groupBy("credit_type").count().orderBy(F.desc("count")).limit(TOP_N_CREDIT_TYPES).collect()
        if row["credit_type"] is not None
    ]
    with open(f"{STAGE_DIR}/top_types.json", "w") as f:
        json.dump(top_types, f)
    print("[meta] done, top_types=", top_types)


def stage_segment(spark, segment: str):
    bureau = load_bureau(spark)
    if segment == "active":
        bureau = bureau.filter(F.col("credit_status") == "Active")
    elif segment == "closed":
        bureau = bureau.filter(F.col("credit_status") == "Closed")
    out = agg_segment(bureau, segment)
    out.write.mode("overwrite").parquet(f"{STAGE_DIR}/agg_{segment}")
    print(f"[{segment}] done")


def stage_cats(spark):
    bureau = load_bureau(spark)
    with open(f"{STAGE_DIR}/top_types.json") as f:
        top_types = json.load(f)

    total = bureau.groupBy("application_id").agg(F.count("*").alias("_total"))

    status_counts = (
        bureau.groupBy("application_id", "credit_status")
        .agg(F.count("*").alias("_cnt"))
        .groupBy("application_id")
        .pivot("credit_status", KNOWN_STATUSES)
        .agg(F.first("_cnt"))
    )
    for status in KNOWN_STATUSES:
        col = status.lower().replace(" ", "_")
        status_counts = status_counts.withColumnRenamed(status, f"bureau_status_{col}_count")

    type_bucketed = bureau.withColumn(
        "_type_bucket",
        F.when(F.col("credit_type").isin(top_types), F.col("credit_type")).otherwise("Other"),
    )
    type_counts = (
        type_bucketed.groupBy("application_id", "_type_bucket")
        .agg(F.count("*").alias("_cnt"))
        .groupBy("application_id")
        .pivot("_type_bucket", top_types + ["Other"])
        .agg(F.first("_cnt"))
    )
    for t in top_types + ["Other"]:
        safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in t)
        type_counts = type_counts.withColumnRenamed(t, f"bureau_type_{safe}_count")

    out = total.join(status_counts, "application_id", "left").join(type_counts, "application_id", "left")
    count_cols = [c for c in out.columns if c not in ("application_id", "_total")]
    for c in count_cols:
        out = out.withColumn(c, F.coalesce(F.col(c), F.lit(0)))
        out = out.withColumn(c.replace("_count", "_share"), F.col(c) / F.col("_total"))
    out = out.drop("_total")
    out.write.mode("overwrite").parquet(f"{STAGE_DIR}/cats")
    print("[cats] done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["meta", "all", "active", "closed", "cats"])
    args = parser.parse_args()

    os.makedirs(STAGE_DIR, exist_ok=True)
    spark = build_spark()
    try:
        if args.stage == "meta":
            stage_meta(spark)
        elif args.stage in ("all", "active", "closed"):
            stage_segment(spark, args.stage)
        elif args.stage == "cats":
            stage_cats(spark)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

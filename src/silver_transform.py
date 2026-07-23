"""
Camada Silver -- nomes de coluna com sentido de negocio (em ingles, padrao
da industria), tipos corrigidos e limpeza de anomalias conhecidas do
dataset Home Credit.

Regras aplicadas:
  - Todo nome de coluna vira snake_case legivel (application_id em vez de
    SK_ID_CURR, total_income em vez de AMT_INCOME_TOTAL, etc). Colunas nao
    mapeadas explicitamente caem em um fallback (so lowercase), nunca
    quebra por causa de uma coluna nova/renomeada na fonte.
  - Campos DAYS_* (sempre negativos, "dias antes da aplicacao") viram
    campos positivos e nomeados em dias/anos, na direcao que faz sentido
    de negocio (ex.: DAYS_BIRTH -> age_years). Campos que podem ser
    positivos OU negativos por natureza (ex.: DAYS_CREDIT_ENDDATE) NAO tem
    o sinal invertido, so o nome muda.
  - DAYS_EMPLOYED tem um valor sentinela conhecido (365243, ~1000 anos) que
    na verdade significa "sem emprego atual" (aposentados) -- vira null,
    com uma flag auxiliar indicando a anomalia.
  - Strings 'XNA' (sentinela de "desconhecido" usado pela Home Credit) viram
    null em todas as colunas de texto.
  - Flags FLAG_* viram booleano de verdade (0/1 -> false/true).

Uso:
    spark-submit silver_transform.py --bronze-dir data/bronze --silver-dir data/silver
"""
from __future__ import annotations

import argparse

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

DAYS_PER_YEAR = 365.25

# ---------------------------------------------------------------------
# application_train
# ---------------------------------------------------------------------

APPLICATION_RENAME = {
    "SK_ID_CURR": "application_id",
    "TARGET": "target",
    "NAME_CONTRACT_TYPE": "contract_type",
    "CODE_GENDER": "gender",
    "FLAG_OWN_CAR": "owns_car",
    "FLAG_OWN_REALTY": "owns_realty",
    "CNT_CHILDREN": "num_children",
    "AMT_INCOME_TOTAL": "total_income",
    "AMT_CREDIT": "credit_amount",
    "AMT_ANNUITY": "annuity_amount",
    "AMT_GOODS_PRICE": "goods_price",
    "NAME_TYPE_SUITE": "accompanied_by",
    "NAME_INCOME_TYPE": "income_type",
    "NAME_EDUCATION_TYPE": "education_type",
    "NAME_FAMILY_STATUS": "family_status",
    "NAME_HOUSING_TYPE": "housing_type",
    "REGION_POPULATION_RELATIVE": "region_population_density",
    "DAYS_BIRTH": "days_birth",  # -> derived: age_years
    "DAYS_EMPLOYED": "days_employed",  # -> derived: employment_years (com fix de anomalia)
    "DAYS_REGISTRATION": "days_registration",  # -> derived: years_since_registration_change
    "DAYS_ID_PUBLISH": "days_id_publish",  # -> derived: years_since_id_change
    "OWN_CAR_AGE": "car_age_years",
    "FLAG_MOBIL": "has_mobile_phone",
    "FLAG_EMP_PHONE": "has_work_phone",
    "FLAG_WORK_PHONE": "has_home_phone",  # nome original sugere "work", mas o dicionario oficial da Home Credit descreve como home phone -- inconsistencia conhecida do dataset, mantida e documentada aqui
    "FLAG_CONT_MOBILE": "mobile_phone_reachable",
    "FLAG_PHONE": "has_second_home_phone",  # dicionario oficial tambem descreve como "home phone"; ver nota acima
    "FLAG_EMAIL": "has_email",
    "OCCUPATION_TYPE": "occupation_type",
    "CNT_FAM_MEMBERS": "family_members_count",
    "REGION_RATING_CLIENT": "region_rating",
    "REGION_RATING_CLIENT_W_CITY": "region_rating_with_city",
    "WEEKDAY_APPR_PROCESS_START": "application_weekday",
    "HOUR_APPR_PROCESS_START": "application_hour",
    "REG_REGION_NOT_LIVE_REGION": "region_mismatch_permanent_vs_contact",
    "REG_REGION_NOT_WORK_REGION": "region_mismatch_permanent_vs_work",
    "LIVE_REGION_NOT_WORK_REGION": "region_mismatch_contact_vs_work",
    "REG_CITY_NOT_LIVE_CITY": "city_mismatch_permanent_vs_contact",
    "REG_CITY_NOT_WORK_CITY": "city_mismatch_permanent_vs_work",
    "LIVE_CITY_NOT_WORK_CITY": "city_mismatch_contact_vs_work",
    "ORGANIZATION_TYPE": "organization_type",
    "EXT_SOURCE_1": "external_score_1",
    "EXT_SOURCE_2": "external_score_2",
    "EXT_SOURCE_3": "external_score_3",
    "FONDKAPREMONT_MODE": "building_renovation_fund_mode",
    "HOUSETYPE_MODE": "building_house_type_mode",
    "TOTALAREA_MODE": "building_total_area_mode",
    "WALLSMATERIAL_MODE": "building_walls_material_mode",
    "EMERGENCYSTATE_MODE": "building_emergency_state_mode",
    "OBS_30_CNT_SOCIAL_CIRCLE": "social_circle_observed_30dpd",
    "DEF_30_CNT_SOCIAL_CIRCLE": "social_circle_defaulted_30dpd",
    "OBS_60_CNT_SOCIAL_CIRCLE": "social_circle_observed_60dpd",
    "DEF_60_CNT_SOCIAL_CIRCLE": "social_circle_defaulted_60dpd",
    "DAYS_LAST_PHONE_CHANGE": "days_last_phone_change",  # -> derived: days_since_phone_change
    "AMT_REQ_CREDIT_BUREAU_HOUR": "bureau_inquiries_last_hour",
    "AMT_REQ_CREDIT_BUREAU_DAY": "bureau_inquiries_last_day",
    "AMT_REQ_CREDIT_BUREAU_WEEK": "bureau_inquiries_last_week",
    "AMT_REQ_CREDIT_BUREAU_MON": "bureau_inquiries_last_month",
    "AMT_REQ_CREDIT_BUREAU_QRT": "bureau_inquiries_last_quarter",
    "AMT_REQ_CREDIT_BUREAU_YEAR": "bureau_inquiries_last_year",
}

# as ~50 colunas normalizadas do predio (APARTMENTS_AVG, ELEVATORS_MODE, ...)
# nao valem renomeacao 1-a-1 -- viram building_<nome original em minusculo>
BUILDING_SUFFIXES = ("_AVG", "_MODE", "_MEDI")

# FLAG_DOCUMENT_2 .. FLAG_DOCUMENT_21
DOCUMENT_FLAGS = {f"FLAG_DOCUMENT_{i}": f"document_{i}_provided" for i in range(2, 22)}

APPLICATION_RENAME.update(DOCUMENT_FLAGS)

APPLICATION_BOOL_COLUMNS = [
    "owns_car",
    "owns_realty",
    "has_mobile_phone",
    "has_work_phone",
    "has_home_phone",
    "mobile_phone_reachable",
    "has_second_home_phone",
    "has_email",
    "target",
] + list(DOCUMENT_FLAGS.values())

# ---------------------------------------------------------------------
# bureau
# ---------------------------------------------------------------------

BUREAU_RENAME = {
    "SK_ID_CURR": "application_id",
    "SK_ID_BUREAU": "bureau_credit_id",
    "SK_BUREAU_ID": "bureau_credit_id",  # nome varia entre versoes do dataset
    "CREDIT_ACTIVE": "credit_status",
    "CREDIT_CURRENCY": "credit_currency",
    "DAYS_CREDIT": "days_credit",  # -> derived: days_since_credit_started
    "CREDIT_DAY_OVERDUE": "days_overdue",
    "DAYS_CREDIT_ENDDATE": "days_to_credit_end",  # sinal mantido (pode ser negativo ou positivo)
    "DAYS_ENDDATE_FACT": "days_enddate_fact",  # -> derived: days_since_credit_closed
    "AMT_CREDIT_MAX_OVERDUE": "max_overdue_amount",
    "CNT_CREDIT_PROLONG": "credit_prolongation_count",
    "AMT_CREDIT_SUM": "credit_amount",
    "AMT_CREDIT_SUM_DEBT": "credit_debt_amount",
    "AMT_CREDIT_SUM_LIMIT": "credit_limit_amount",
    "AMT_CREDIT_SUM_OVERDUE": "credit_overdue_amount",
    "CREDIT_TYPE": "credit_type",
    "DAYS_CREDIT_UPDATE": "days_credit_update",  # -> derived: days_since_credit_update
    "AMT_ANNUITY": "annuity_amount",
}


def humanize_fallback(col_name: str) -> str:
    """Para qualquer coluna nao mapeada explicitamente: so normaliza p/ snake_case."""
    name = col_name.lower()
    for suf in BUILDING_SUFFIXES:
        if name.upper().endswith(suf):
            return f"building_{name}"
    return name


def rename_columns(df: DataFrame, mapping: dict[str, str]) -> DataFrame:
    for original, new in mapping.items():
        if original in df.columns:
            df = df.withColumnRenamed(original, new)
    # fallback para qualquer coluna que sobrou sem mapeamento
    for col in df.columns:
        if col.isupper() or "_" in col and col != col.lower():
            df = df.withColumnRenamed(col, humanize_fallback(col))
    return df


def clean_xna(df: DataFrame) -> DataFrame:
    string_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, T.StringType)]
    for c in string_cols:
        df = df.withColumn(c, F.when(F.trim(F.col(c)) == "XNA", None).otherwise(F.col(c)))
    return df


YN_BOOL_COLUMNS = ["owns_car", "owns_realty"]  # vem como 'Y'/'N' no Kaggle, nao 0/1


def to_bool(df: DataFrame, columns: list[str]) -> DataFrame:
    for c in columns:
        if c not in df.columns:
            continue
        if c in YN_BOOL_COLUMNS:
            df = df.withColumn(c, F.col(c) == "Y")
        else:
            df = df.withColumn(c, F.col(c).cast("int") == 1)
    return df


def transform_application(df: DataFrame) -> DataFrame:
    df = rename_columns(df, APPLICATION_RENAME)
    df = clean_xna(df)
    df = to_bool(df, APPLICATION_BOOL_COLUMNS)

    # DAYS_EMPLOYED tem sentinela 365243 (~1000 anos) para "sem emprego atual"
    df = df.withColumn(
        "is_employment_anomaly", F.col("days_employed") == 365243
    ).withColumn(
        "days_employed",
        F.when(F.col("days_employed") == 365243, None).otherwise(F.col("days_employed")),
    )

    df = (
        df.withColumn("age_years", (-F.col("days_birth") / DAYS_PER_YEAR))
        .withColumn("employment_years", (-F.col("days_employed") / DAYS_PER_YEAR))
        .withColumn(
            "years_since_registration_change", (-F.col("days_registration") / DAYS_PER_YEAR)
        )
        .withColumn("years_since_id_change", (-F.col("days_id_publish") / DAYS_PER_YEAR))
        .withColumn("days_since_phone_change", -F.col("days_last_phone_change"))
    )
    return df


def transform_bureau(df: DataFrame) -> DataFrame:
    df = rename_columns(df, BUREAU_RENAME)
    df = clean_xna(df)
    df = (
        df.withColumn("days_since_credit_started", -F.col("days_credit"))
        .withColumn("days_since_credit_closed", -F.col("days_enddate_fact"))
        .withColumn("days_since_credit_update", -F.col("days_credit_update"))
    )
    return df


def build_spark() -> SparkSession:
    return (
        SparkSession.builder.appName("home-credit-silver")
        .master("local[*]")
        .config("spark.driver.memory", "1500m")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bronze-dir", default="data/bronze")
    parser.add_argument("--silver-dir", default="data/silver")
    args = parser.parse_args()

    spark = build_spark()
    try:
        app = spark.read.parquet(f"{args.bronze_dir}/application_train")
        app_silver = transform_application(app)
        app_silver.write.mode("overwrite").parquet(f"{args.silver_dir}/application_train")
        print(f"[silver] application_train -> {args.silver_dir}/application_train")

        bureau = spark.read.parquet(f"{args.bronze_dir}/bureau")
        bureau_silver = transform_bureau(bureau)
        bureau_silver.write.mode("overwrite").parquet(f"{args.silver_dir}/bureau")
        print(f"[silver] bureau -> {args.silver_dir}/bureau")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

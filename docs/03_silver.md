# 3. Camada Silver — nomenclatura, tipos e limpeza

## Decisão de idioma

Nomes de coluna e código em **inglês** — é o padrão da indústria e o projeto também mira vagas remotas nos EUA, então manter o repositório 100% legível para quem não lê português era mais importante do que a nomenclatura em português cogitada inicialmente. A narrativa (README, comentários de decisão, post do LinkedIn) continua em português.

Script: [`src/silver_transform.py`](../src/silver_transform.py).

## O que foi feito

1. **Renomeação com sentido de negócio.** Nomes crípticos do Kaggle (`SK_ID_CURR`, `AMT_INCOME_TOTAL`, `DAYS_BIRTH`, ...) viraram nomes legíveis (`application_id`, `total_income`, `age_years`, ...). As ~50 colunas normalizadas sobre o prédio onde o cliente mora (`APARTMENTS_AVG`, `ELEVATORS_MODE`, etc.) ganharam o prefixo `building_` em vez de tradução individual — são muitas e de baixo uso direto na análise principal.
2. **Sinal dos campos de tempo.** Todo campo `DAYS_*` do Kaggle é relativo à data da aplicação (quase sempre negativo). Quando o campo só pode ir em uma direção (ex.: `DAYS_BIRTH`, sempre no passado), o sinal foi invertido para virar algo positivo e legível (`age_years`). Quando o campo pode ser positivo ou negativo por natureza (ex.: `DAYS_CREDIT_ENDDATE` — prazo restante de um crédito no bureau, que pode já ter vencido antes da aplicação), o sinal foi mantido, só o nome mudou (`days_to_credit_end`).
3. **Anomalia conhecida em `DAYS_EMPLOYED`.** Esse campo tem um valor sentinela de 365243 dias (~1000 anos) usado pela Home Credit para indicar "sem emprego atual" (majoritariamente aposentados). Isso virou `null` + uma flag `is_employment_anomaly`. Confirmado: **55.374 linhas** (~18% da base) têm essa anomalia — bate com o que é publicamente conhecido sobre esse dataset.
4. **Sentinela `'XNA'` → null.** A Home Credit usa a string `'XNA'` como "desconhecido" em várias colunas categóricas. Todas viraram `null`. Ex.: `gender` tinha 4 linhas com `'XNA'`.
5. **Flags viraram booleano de verdade** (0/1 → false/true), incluindo os 20 `FLAG_DOCUMENT_2..21` (renomeados para `document_2_provided` ... `document_21_provided`).

## Nota sobre uma inconsistência do dataset original

O dicionário oficial da Home Credit descreve **duas colunas diferentes** (`FLAG_WORK_PHONE` e `FLAG_PHONE`) como "did client provide home phone" — mesma descrição para duas colunas com nomes de variável diferentes. Isso é uma inconsistência do dataset original, não um erro deste projeto. Mantive os dois campos (`has_home_phone` e `has_second_home_phone`) documentando aqui a ambiguidade, em vez de adivinhar qual delas é "work phone" de verdade.

## Correção (encontrada durante a EDA)

`FLAG_OWN_CAR` e `FLAG_OWN_REALTY` vêm como string `'Y'`/`'N'` no Kaggle — diferente dos outros `FLAG_*`, que já são `0`/`1`. A primeira versão do script tratou todos igual (`cast("int") == 1`), o que jogou `owns_car` e `owns_realty` inteiros para `null` (100% de missing). Corrigido: essas duas colunas agora comparam direto com `'Y'`. Pego durante a EDA (abaixo) — fica registrado aqui porque é exatamente o tipo de erro que checagem de missing values pega, e vale mostrar que foi pego e corrigido, não só entregar código bonito.

## Resultado

| Tabela | Linhas | Colunas |
|---|---|---|
| application_train | 307.511 | 130 |
| bureau | 1.716.428 | 22 |

## Como reproduzir

```bash
spark-submit src/silver_transform.py --bronze-dir data/bronze --silver-dir data/silver
```

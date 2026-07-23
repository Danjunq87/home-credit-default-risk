# 2. Camada Bronze — ingestão bruta

## O que é

A camada Bronze é uma cópia fiel dos dados brutos, sem nenhuma transformação de negócio — só uma troca de formato (CSV → Parquet) para ganhar performance/compactação nas camadas seguintes, e a adição de duas colunas técnicas de rastreabilidade: `_ingested_at` (timestamp da ingestão) e `_source_file` (arquivo de origem).

Script: [`src/bronze_ingest.py`](../src/bronze_ingest.py). Roda em PySpark local (`local[*]`), lendo `data/raw/application_train.csv` e `data/raw/bureau.csv` com inferência de schema e gravando em `data/gold/<tabela>` particionado em Parquet/Snappy.

## Execução e validação

Rodado localmente (ambiente equivalente a Colab, dentro do sandbox usado nesta conversa). Resultado:

| Tabela | Linhas | Colunas (com metadados) |
|---|---|---|
| application_train | 307.511 | 124 |
| bureau | 1.716.428 | 19 |

Contagens batem com o dataset oficial da competição (application_train tem 307.511 propostas; bureau tem ~1,7M registros de crédito em outras instituições, 0 a N por proposta).

## Nota de segurança

O arquivo `data/raw/kaggle.json` (credencial da API do Kaggle) **não deve ir para o GitHub**. Já está coberto pelo `.gitignore` do projeto, mas vale conferir antes do primeiro `git push` — e considerar revogar/gerar uma nova chave no Kaggle se ela já tiver sido exposta em algum lugar.

## Como reproduzir

```bash
pip install -r requirements.txt
spark-submit src/bronze_ingest.py --raw-dir data/raw --bronze-dir data/bronze
```

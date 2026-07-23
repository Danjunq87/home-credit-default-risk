# 4. Camada Gold — book de variáveis + ABT final

## O book de variáveis do bureau

O `bureau.csv` tem granularidade de **crédito anterior** (0 a N linhas por `application_id`). Pra virar uma feature usável num modelo, precisa de **uma linha por `application_id`** — isso é o que a camada Gold entrega.

Para cada `application_id`, calculei:

- **Estatísticas gerais** (todos os créditos do bureau): min/max/média/soma de 18 indicadores numéricos (valores, dias, taxas de uso/atraso) → 72 variáveis.
- **As mesmas estatísticas, só com créditos ativos** → 72 variáveis (`bureau_active_*`).
- **As mesmas estatísticas, só com créditos encerrados** → 72 variáveis (`bureau_closed_*`).
- **Contagens e proporções por status do crédito** (ativo/encerrado/vendido/prejuízo) → 8 variáveis.
- **Contagens e proporções pelos 8 tipos de crédito mais comuns** (+ "outros") → 18 variáveis.
- **Metadados gerais**: quantidade total de créditos anteriores, quantos ativos/encerrados, tipos distintos de crédito, dias desde o crédito mais antigo/mais recente → 6 variáveis.

Total: **248 variáveis** do book do bureau (lista completa em [`docs/feature_dictionary.csv`](feature_dictionary.csv)). Junto com as 128 colunas do `application_train` (silver, sem as colunas técnicas de ingestão), a **ABT final tem 377 colunas** e 307.511 linhas — uma por proposta de crédito.

Três variáveis derivadas (razões) entraram na lista de indicadores agregados: `debt_to_credit_ratio`, `overdue_to_credit_ratio` e `limit_usage_ratio` — medem, respectivamente, quanto da dívida está pendente sobre o crédito total, quanto está em atraso, e quanto do limite está sendo usado. São indicadores clássicos de risco de crédito.

## Validação

- 85,7% dos clientes têm pelo menos um crédito anterior no bureau (`has_bureau_history`).
- Distribuição do `target` na ABT: **8,07% de inadimplência** — bate com o valor publicamente conhecido desse dataset, o que dá confiança de que a base não foi corrompida em nenhuma das transformações Bronze → Silver → Gold.

## Uma decisão de engenharia real: Spark para agregação, pandas para o join final

Rodando tudo localmente (ambiente de demonstração com 2 vCPU / ~2,8GB de RAM), o Spark deu conta bem das agregações pesadas sobre as ~1,7 milhões de linhas do `bureau` (`src/gold_build.py`, rodado em estágios — `meta`, `all`, `active`, `closed`, `cats` — cada um gravando parquet intermediário em `data/gold_stage/`).

O join final dessas peças (agora já agregadas, ~306 mil linhas cada) com o `application_train` é onde o Spark parou de compensar: nesse volume, o overhead de um join distribuído (serialização, JVM, shuffle) custava mais do que fazer o merge localmente. A tentativa direta em pandas também estourou memória de uma vez só (o processo foi morto pelo kernel, sem erro do Python) — resolvido processando a ABT em **6 lotes (chunks)** de ~51 mil linhas cada, com tipos numéricos convertidos para `float32` antes do merge (`src/gold_join.py`). Cada lote é gravado como uma parte parquet separada em `data/gold/abt/`, formato padrão para datasets particionados (o mesmo que o Spark geraria).

Isso não é um atalho escondido — é a mesma decisão que se toma em produção quando um estágio do pipeline não cabe no cluster disponível: medir, quebrar o problema e documentar por quê. Num cluster Spark real (mais executores/memória), essa quebra manual não seria necessária.

## Como reproduzir

```bash
spark-submit src/gold_build.py --stage meta
spark-submit src/gold_build.py --stage all
spark-submit src/gold_build.py --stage active
spark-submit src/gold_build.py --stage closed
spark-submit src/gold_build.py --stage cats
python3 src/gold_join.py --silver-dir data/silver --stage-dir data/gold_stage --gold-dir data/gold
```

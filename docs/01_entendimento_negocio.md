# 1. Entendimento de Negócio — Home Credit Default Risk

## O problema

A [Home Credit](https://www.kaggle.com/competitions/home-credit-default-risk/overview) é uma financeira internacional (atua em 9 países) especializada em conceder crédito a pessoas com pouco ou nenhum histórico bancário — o público "underbanked". É gente que, num banco tradicional, seria recusada por falta de histórico de crédito, mesmo tendo capacidade de pagar. Esse vácuo é historicamente ocupado por agiotas e financeiras predatórias.

O desafio de negócio é simples de enunciar e difícil de resolver: **como conceder crédito para quem não tem histórico, sem aumentar a inadimplência?** Errar em qualquer direção custa caro:

- **Aprovar demais** (falso negativo do risco) → aumento de calote, perda direta de capital.
- **Negar demais** (falso positivo do risco) → exclui gente pagadora, perde receita de juros e reforça a exclusão financeira que a empresa diz querer resolver.

A resposta da Home Credit é usar dados alternativos (telco, transacional, histórico em outras centrais de crédito) para enxergar risco onde o score tradicional é cego.

## O que é o TARGET

No `application_train`, a variável `TARGET` é binária:

- `0` — cliente pagou em dia (ou sem atraso relevante).
- `1` — cliente teve dificuldade de pagamento: atraso maior que X dias em pelo menos uma das primeiras Y parcelas do empréstimo.

É um problema de **classificação binária supervisionada**. A métrica oficial da competição é **AUC-ROC** — mede a capacidade do modelo de ranquear corretamente quem tem mais chance de inadimplir, independente do ponto de corte escolhido depois. Isso importa porque, na prática, o ponto de corte (quem aprova/nega) é uma decisão de negócio (apetite de risco), não do modelo — o modelo entrega o ranking, a política de crédito decide o corte.

O dataset é desbalanceado (a maioria dos clientes paga em dia), o que é esperado num produto de crédito saudável — mas exige cuidado na modelagem e na escolha de métrica (por isso AUC-ROC em vez de acurácia simples).

## As duas fontes usadas neste projeto

Dos arquivos disponíveis na competição, este projeto usa dois, propositalmente, para contar uma história de ganho incremental:

1. **`application_train.csv`** — uma linha por proposta de crédito (`SK_ID_CURR`), com dados cadastrais e socioeconômicos do cliente no momento da aplicação: renda, valor do crédito pedido, tipo de contrato, composição familiar, escolaridade, tipo de habitação, informações do imóvel onde mora, flags de documentos fornecidos, scores normalizados de fontes externas (`EXT_SOURCE_1/2/3`), entre outras. É o retrato do cliente *agora*.

2. **`bureau.csv`** — histórico de crédito do cliente em **outras** instituições, reportado à Central de Crédito (Credit Bureau), granularidade de 0, 1 ou N registros por `SK_ID_CURR` (um cliente pode ter vários créditos anteriores em outros lugares). Traz status do crédito (ativo/encerrado), atraso em dias, valores de crédito/dívida/limite, tipo de crédito, etc. É o retrato do comportamento *passado* do cliente em outras carteiras.

A ideia central do projeto (e a "história" que vai virar post de LinkedIn) é: **um modelo só com `application_train` já é útil, mas um book de variáveis agregado do `bureau` — resumindo o comportamento do cliente em outras instituições para a granularidade de cada proposta — deveria melhorar o poder preditivo**. É isso que os modelos 1 (baseline) e 2 (desafiante) vão comparar via AUC-ROC.

## Por que isso importa pro portfólio

Esse é o desenho de um problema real de crédito: dado desbalanceado, múltiplas fontes em granularidades diferentes, necessidade de agregar histórico sem vazar informação futura, e uma métrica de negócio (AUC-ROC) que reflete como o modelo será usado (ranking de risco, não decisão binária direta). O pipeline Bronze/Silver/Gold + comparação de modelos existe para demonstrar isso de ponta a ponta, não só treinar um modelo solto.

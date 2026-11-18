# Invariantes da Fase 7

Este documento registra os contratos que não podem mudar durante a consolidação
estrutural dos experimentos.

## Causalidade

- Cada exemplo autoregressivo usa somente o prefixo observado para construir
  `context_ids`, `context_syllable_ids` e `context_feature_vectors`.
- Uma sílaba parcialmente observada não expõe o ID da sílaba completa; usa
  `<UNK_SYL>`.
- `<END>`, separadores e targets sem alinhamento usam `<UNK_SYL>` quando não há
  uma sílaba causalmente concluída disponível.
- A avaliação por classe e os buckets da Fase 5.3 usam metadata causal de C/D,
  nunca informação futura ou IDs mascarados de A/B.
- Alterar o sufixo não observado de uma palavra não pode alterar a representação
  causal do prefixo.

## Representação e dimensões

Na configuração expandida validada, o vocabulário possui 51 tokens de caractere
e 3.932 IDs silábicos. O modelo usa:

```text
char_embedding_dim      = 8
syllable_embedding_dim  = 8
orthographic_feature_dim= 10
representation_dim      = 26
context_size             = 8
```

A representação híbrida concatena embeddings de caractere, embedding silábico e
features ortográficas; o pooling do contexto é uma média. As cabeças produzem
logits para caractere, sílaba e fronteira.

Todas as variantes A–F mantêm a mesma arquitetura e **139.432 parâmetros
traináveis**. A/B/C/D diferem somente pelos canais mascarados; E/F usam a
representação parcial causal sem alterar a tabela de parâmetros.

A contagem deve continuar sendo obtida por `count_trainable_parameters()` e
verificada nos relatórios, não inferida apenas de uma constante documental.

## Loss multitarefa

A loss total é a soma ponderada das três componentes:

```text
total_loss = 1.0 * char_loss
            + 0.7 * syllable_loss
            + 0.3 * boundary_loss
```

Os pesos são parte do contrato matemático. Uma mudança exige nova comparação
controlada e não pode ser introduzida como simples refatoração de execução.

## Treino e avaliação

- Treino pode alterar parâmetros apenas via `train_step`/backward/step.
- Validação usa `evaluate_multitask_examples()` e não chama backward nem step.
- Avaliação deve preservar embeddings, pesos, biases e estado observável do
  modelo.
- A mesma seed, split, ordem de exemplos, épocas, learning rate, contexto,
  variantes e pesos da loss deve produzir o mesmo histórico matemático dentro da
  precisão exata usada pelo runner.

## Execução e checkpoints

- `ExperimentConfig` contém controles matemáticos/reprodutíveis.
- `ExecutionConfig` contém controles operacionais, incluindo `workers`.
- `workers` pode mudar sem invalidar um checkpoint compatível.
- Seeds antigas podem ser ampliadas; seeds já `completed` não são reexecutadas.
- O checkpoint contém exatamente um registro por seed solicitada.
- `running`, `pending` e `failed` são normalizados para retry como um único
  registro `pending`; `completed` é preservado.
- O processo principal é o único responsável pela escrita do checkpoint.
- `mean/std` e agregados de métricas consideram apenas registros `completed`.
- `workers=1` é a referência determinística; multiprocessamento só altera a
  ordem/tempo da execução, não os resultados matemáticos por seed.

## Runners históricos

`run_phase5_3_expanded.py`, `run_phase5_4_expanded.py` e
`run_phase5_5_expanded.py` permanecem como adaptadores históricos e exemplos de
uso do contrato consolidado. Eles não devem receber novas refatorações
específicas. Novos experimentos devem usar `training.multi_seed_runner`.

A equivalência foi validada em
`experiments/phase7_deterministic_validation.json` comparando o histórico
`a686e32` com a consolidação `d897341`, além de `workers=1` contra `workers=2`.

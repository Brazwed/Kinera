# Kinera B009 — M1.5 baselines multiclasses

## Lacuna e decisão arquitetural

A auditoria B008 (PR #76, base `b9200eeafd4ca4929987366ae1164d6f27e28d26`) classificou M1-04 como pendente: o relatório comparativo tinha baselines para a head binária de fronteira, mas não referências majoritárias para as heads multiclasses de caracteres e sílabas. Esta tranche adiciona essas referências sem mudar a definição das métricas anteriores.

A aritmética pura vive em `kinera.monitor.metrics.baselines`: `multiclass_class_counts()` valida e conta targets e `evaluate_multiclass_majority_baseline()` seleciona a classe majoritária de uma distribuição de treino e mede sua acurácia numa distribuição de avaliação. O empate escolhe o menor ID numérico. As funções não abrem datasets nem importam modelo, trainer, runner, NumPy ou backend.

O runner comparativo prepara e limita os exemplos de cada variante conforme `max_examples`, calcula as distribuições uma vez a partir do prefixo de treino preparado e as passa ao avaliador. A preparação A/B/C/D modifica features de entrada, não targets nem sua ordem; por isso, as distribuições de targets são comuns às variantes e o mesmo conjunto de treino alimenta cada avaliação inicial, intermediária e final. A API do evaluator recebe histogramas já calculados, não o dataset de treino.

## Definição e política dos targets

Para uma head com distribuição de treino `T`, avaliação `E` e classe escolhida `c*`,

```text
c* = menor ID que maximiza T[id]
accuracy = E[c*] / soma(E.values())
```

As contagens vêm exclusivamente de targets de treino selecionados. Targets de validação são usados apenas para `E` e não podem alterar `c*`.

| Campo do relatório | Treino usado na escolha | Avaliação | Política |
| --- | --- | --- | --- |
| `char_baseline_majority` | Todos os `target_char_id` válidos | Todos os targets char válidos | Inclui START, END e UNK se legitimamente presentes no split; acompanha `char_accuracy`. |
| `syllable_baseline_majority_all` | Todos os `target_syllable_id` válidos | Todos os targets silábicos válidos | Inclui UNK, compatível com o escopo de `syllable_accuracy_all`. |
| `syllable_baseline_majority_known` | Targets com `target_syllable_id != syllable_unknown_id` e `target_char_id != vocabulary.end_id` | O mesmo recorte known na avaliação | Exclui UNK e exemplos END, conforme a política M1-04 desta tranche. |

Em `syllable_baseline_majority_known`, previsões de modelo iguais a UNK para um target conhecido continuam sendo erros. O denominador é o número de targets conhecidos da avaliação, não o número de previsões conhecidas.

O alias histórico `syllable_accuracy_known` permanece inalterado: atualmente ele exclui targets cujo ID silábico é UNK e não aplica filtro explícito de `target_char_id == END`. A baseline nova declara e usa o recorte M1-04 próprio. Essa diferença não é aplicada retroativamente à métrica legada.

Todos os targets char válidos, inclusive IDs especiais, participam tanto da distribuição de treino quanto do denominador de avaliação, mantendo o alinhamento com `char_accuracy`. Targets inválidos, booleanos, IDs negativos e IDs fora do vocabulário são rejeitados, sem conversão permissiva. Contagens também rejeitam valores negativos, booleanos e tipos não inteiros.

## Suporte e schema

Os três descritores são publicados em `metric_results` com `METRICS_SCHEMA_VERSION` existente (v1), unidade de fração `[0, 1]` e campos `value`, `availability`, `reason` e `support`.

- Sem distribuição de treino fornecida: `unavailable`, motivo `training_distribution_missing`.
- Distribuição fornecida, mas sem targets elegíveis no treino: `unavailable`, motivo `no_training_targets` (ou `no_known_training_targets` para o recorte known).
- Treino com suporte, mas avaliação sem targets: `not_applicable`, motivo `no_evaluation_targets` (ou `no_known_targets`).
- Se treino known e avaliação known estiverem ambos vazios, prevalece `unavailable/no_known_training_targets`, pois não existe classe de referência a selecionar.
- Um resultado válido de acurácia `0.0` permanece `available`; ausência nunca é substituída por zero.

`support` informa as contagens de treino e avaliação usadas por cada baseline. `baseline_selection` registra os IDs escolhidos, a política de empate e os IDs excluídos do recorte conhecido; IDs de classe não são apresentados como valores de acurácia. Os aliases existentes, distribuições, losses, métricas de fronteira e schema v1 permanecem iguais.

Na avaliação de treino feita pelo trainer, `include_multiclass_baselines=False` preserva o formato de `training.history` e evita publicação de baselines de treino contra o próprio conjunto como se fossem observações de validação. Outros consumidores do evaluator que não passam distribuições explícitas recebem descritores `unavailable` conforme o contrato; não há inferência implícita de que o split avaliado seja treino.

## Agregação, deltas e checkpoints

O runner inclui os três novos descritores nas agregações por variante, usando somente observações disponíveis e mantendo missing/invalid segundo as regras existentes. Baselines não são incluídos em `paired_deltas`: são referências derivadas da distribuição comum de treino e não representam deltas pareados de resultados de modelos entre variantes.

Checkpoints antigos continuam legíveis. Se um registro concluído não tiver os novos descritores, o runner o preserva como histórico ausente e não inventa acurácias retroativas; expansões de seeds podem agregar novas observações sem alterar os registros anteriores. Não foi alterado o fingerprint científico, `FINGERPRINT_SCHEMA_VERSION`, `EXPERIMENT_PROTOCOL_VERSION`, `METRICS_SCHEMA_VERSION`, telemetria ou configuração de resume: os campos são observações aditivas e a identidade do experimento já cobre dados e configuração relevantes.

## Preservação científica e limites

As baselines são avaliações passivas dos targets e não participam do treinamento. Não foram alterados parâmetros, losses, gradientes, optimizer, RNG, ordem ou número de updates, máscaras de variante, threshold de boundary ou preparação de dados. O evaluator reutiliza os targets/distribuições da passagem já executada; não acrescenta forwards. Os testes com modelos determinísticos verificam que a contagem de chamadas, métricas legadas e histórico de treino permanecem iguais.

A implementação protege contra vazamento por escolha da classe ao obter a classe majoritária apenas do treino. Ela não é uma alegação de qualidade geral do modelo, nem transforma baselines em deltas de modelo. As comparações são interpretáveis em conjunto com seus suportes e vocabulários identificados no relatório experimental.

## Validação B009

Testes novos cobrem validação estrita, distribuição vazia, empate e independência da ordem dos mapeamentos; targets de validação não alteram a classe escolhida; inclusão de tokens especiais, UNK e exclusão de END no recorte conhecido; suportes e disponibilidade; prefixo efetivo de `max_examples`; variantes; agregações; resume histórico; propagação aos workers CPU; e ausência de campos de baseline no histórico de treino.

Na HEAD validada `aaa9863f2a6a874fe0ac96072209062491c56e85`, no ambiente Python 3.15:

- `tests/test_monitor_metrics_m1.py`: 45 passed, 10 subtests.
- `tests/test_multitask_training.py`: 17 passed.
- `tests/test_comparison.py`: 44 passed, 10 subtests.
- Fingerprint, checkpoint/telemetry, package layout, learning state, schema e diagnósticos: 94 passed, 213 subtests.
- Suíte configurada na CI: 793 passed, 61 skipped, 5 deselected, 792 subtests passed. Os cinco deselections são exatamente os perfis/pilotos e benchmark excluídos no workflow CI. O warning único é `PytestRemovedIn10Warning` emitido pelo plugin AnyIO que importa o alias `CallSpec2`.
- `python -m compileall -q src tests`, `git diff --check`, imports CPU-only e parse AST com `feature_version=(3, 10)`: aprovados.

Runtime Python 3.10 não estava disponível (`UNVERIFIED`); a compatibilidade sintática foi conferida pela gramática AST 3.10. Não foram executados testes CUDA. Os dois testes RX 550/OpenCL são opt-in por `KINERA_RUN_OPENCL_TESTS=1` e foram skips; nenhum dispositivo físico foi iniciado. A suíte não substitui teste físico em backend GPU.

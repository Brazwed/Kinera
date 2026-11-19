# Issue #26 — M2.1: integração dos diagnósticos

Auditoria e implementação B004, em 2026-11-06.

## Referências e escopo

- Branch: `feature/issue-26-metrics-b004`.
- SHA inicial da `main` e merge-base: `3a9cd4df5a67557e5f8b2de0926f4b9f32b35106`
  (integração M1.4/PR #64).
- Durante a tranche, a `main` avançou pelo PR #65 (OpenCL F1b). A branch foi
  atualizada por merge normal; merge-base atual: `685fa68ea11b1cd500c489b64f7df900920b7f44`.
- O diff do B004 contra a base atual continua em 11 arquivos de métricas,
  diagnósticos, documentação e testes; nenhum backend foi alterado pelo B004.
- A implementação não altera training, optimizer, losses, RNG, modelos,
  datasets, protocolo A/B/C/D, telemetry ou backends.
- O algoritmo de diferenças finitas em `diagnostics/gradients.py` continua
  específico do diagnóstico e não foi migrado.

## Auditoria e contrato legado

Antes da mudança, `diagnostics/syllables.py` e
`diagnostics/orthography.py` já usavam `safe_ratio`, mas repetiam a subtração
orientada de loss/acurácia, a soma e divisão para médias por observação e as
frações de melhoria/regressão.
`summarize_paired_deltas()` não é apropriado para esses relatórios: ele agrega
pares por seed, enquanto os dois diagnósticos primeiro comparam observações de
validação dentro de cada seed.

O teste `test_diagnostic_metrics_m2.py` foi adicionado e passou contra a
implementação prévia à refatoração. Ele fixa os valores numéricos, campos
legados, política de suporte vazio, ordenação lexical dos grupos e estabilidade
dos empates nos rankings. Um caso de três observações fixa:

- deltas de loss `[0.3, -0.2, 0.0]`, com média exatamente
  `0.033333333333333326`;
- deltas de acurácia `[1, -1, 0]`, com média `0.0`;
- melhoria e regressão estritas de `1/3` cada; o empate não entra em nenhuma;
- probabilidades base `[0.1, 0.2, 0.3]`, com média legada
  `0.19999999999999998`.

O contrato preservado é:

| Saída | Fórmula e população | Suporte vazio legado |
| --- | --- | --- |
| `mean_delta_loss` | `sum(loss_base - loss_aumentada) / exemplos` | `0.0` no resumo silábico; ausente no ortográfico |
| `accuracy_delta` / `delta_accuracy` | `sum(acertos_aumentados - acertos_base) / exemplos` | `0.0` no resumo silábico; ausente no ortográfico |
| `improved_loss_fraction` | `count(delta_loss > 0) / exemplos` | ausente no resumo silábico e ortográfico |
| `regressed_loss_fraction` | `count(delta_loss < 0) / exemplos` | ausente no resumo silábico e ortográfico |
| probabilidades corretas | média por exemplo da probabilidade do target | campos ausentes no ortográfico |
| `active_fraction` | `active / exemplos` | alias histórico `0.0` |

`sum(values) / count` continua sendo a operação e a ordem de soma. As médias
globais são calculadas diretamente sobre as observações; não se tira a média
simples de grupos com tamanhos diferentes. O agregado legado da Fase 5.3
continua sendo uma média entre seeds dos resumos por seed, com suporte em seeds.

## Matriz de responsabilidades

| Consumidor | Aritmética centralizada | Responsabilidade que permanece local |
| --- | --- | --- |
| `diagnostics/syllables.py` | médias, deltas e frações por exemplo; `safe_ratio` para cobertura | pares A/C e B/D, buckets, cobertura linguística, frequência, posição, fronteira, caracteres e seleção/ranking de exemplos |
| `diagnostics/orthography.py` | médias de loss, acurácia e probabilidade; deltas e frações por exemplo | `classify_target`, classes, associação aos grupos e pares A/C e B/D |
| `scripts/legacy/phase5_3.py` | média dos valores por seed via helper escalar | escolha dos campos e definição da população como seeds completas |
| `scripts/legacy/phase5_5.py` | recebe descritores dos resumos por classe | execução retomável por seed, sem agregado científico próprio |
| `diagnostics/gradients.py` | nenhuma | diferenças finitas e comparação de gradientes continuam no diagnóstico |

Não houve mudança no número nem na ordem dos forwards. Os resumos são
calculados sobre as linhas já produzidas pelos avaliadores.

## Schema v1 nos relatórios

O pacote `monitor/metrics/` fornece `oriented_delta()`, `summarize_values()` e
`summarize_deltas()` em `core.py`, além do `metric_result()` e
`METRICS_SCHEMA_VERSION` existentes. O schema global continua na versão 1.

Cada nova execução silábica ou ortográfica contém `metrics_schema_version: 1`.
`metric_results` fica junto ao resumo geral, à cobertura, a cada bucket/classe
e a cada efeito agregado por target. Cada descritor usa `support.examples` do
grupo específico. A cobertura inclui também `support.active`. No relatório
retomável da Fase 5.3, cada resumo de par no agregado usa
`metrics_schema_version: 1` e `support.seeds`.

Um grupo vazio produz descritores `not_applicable`, valor nulo e motivo
`empty_support`; os aliases escalares vazios continuam iguais aos históricos.
Um resultado observado igual a zero permanece `available`. Inputs diagnósticos
sem exemplos são rejeitados antes da avaliação, então M2.1 não inventa uma
categoria `unavailable` para esse caso. Os descritores usam números finitos e
são serializáveis em JSON.

Versões são adicionadas às novas execuções, não retroativamente a registros
antigos. No wrapper retomável, a versão dentro de `aggregates` descreve o
formato e suporte do agregado recém-calculado; ela não afirma que todos os
registros de `runs` tenham sido emitidos no schema v1.

## Checkpoints, identidade e telemetry

`run_syllable_gain_analysis()` e `run_orthographic_class_analysis()` não têm
argumentos de checkpoint ou resume. Os entrypoints históricos
`scripts/legacy/phase5_3.py` e `phase5_5.py` usam o runner genérico
`run_seed_jobs()`: o runner compara configuração e sementes, conserva seeds
completas e trata telemetry do mesmo modo. Esse caminho não usa o fingerprint
M1.4 do runner comparativo. Um checkpoint histórico pode, portanto, reutilizar
uma seed concluída sem verificar o conteúdo do dataset ou do vocabulário.

B004 não altera o comportamento de resume nem reescreve esses checkpoints. Os
testes de integração de resume verificam a leitura dos registros legados; os
registros antigos não recebem `metrics_schema_version` por efeito colateral.
A integridade de conteúdo desses dois runners permanece um risco residual e
requer tranche própria antes de se alegar a garantia M1.4 para eles. Telemetry
continua sob seu schema e fluxo append-only existentes.

## Validação

- Pré-refatoração: os três testes congelados passaram contra os resumos legados.
- Regressão direcionada após a centralização final: `58 passed` e
  `7 subtests passed` nos testes de diagnósticos, métricas e resume.
- CI da branch após atualizar a base, com os cinco testes de piloto excluídos
  conforme o workflow: `724 passed`, `48 skipped`, `5 deselected` e
  `519 subtests passed`.
- `compileall` de `src/kinera` e `scripts/legacy/phase5_3.py`: aprovado.
- Import CPU-only de `diagnostics.syllables`, `diagnostics.orthography` e
  `monitor.metrics`: aprovado; módulos `pycuda` e `pyopencl` não foram
  carregados.
- `git diff --check`: aprovado.
- Ambiente local: Python 3.15.0. Python 3.10 não está instalado; a sintaxe foi
  mantida compatível com a declaração `requires-python >=3.10`, mas o runtime
  mínimo não pôde ser executado localmente.
- Aviso não bloqueante do ambiente: plugin do AnyIO usa o alias pytest
  `CallSpec2`, marcado como deprecado.

Nenhuma execução científica, benchmark ou uso de GPU foi realizado para esta
tranche.

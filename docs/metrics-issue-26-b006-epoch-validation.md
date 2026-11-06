# B006 — M2.3: validação opcional por época

## Referência e auditoria

- Base inicial da worktree e merge-base inicial:
  `f3887fc33a1e7c77372f3ffeaaf4c33771669e21`.
- Base atualizada da branch e merge-base para integração:
  `36f7fb9e5433e295b8556328db4e89ab669245af` (PR #69). A atualização contém
  documentação e testes OpenCL sem sobreposição com os arquivos B006.
- PR #68 está integrado nessa base.
- A M1.4 existente usa `experiment_fingerprint` para validar dados, vocabulário,
  protocolo e configuração científica antes do resume do runner comparativo.
- `train_multitask_examples()` executa SGD em ordem, avalia o split de treino
  após cada época e retorna `training.history`.
- `_execute_single_experiment()` avalia validação antes do treino e ao final;
  `final_validation` alimenta agregações e deltas pareados.
- `evaluate_multitask_examples()` chama `AutoregressiveModel.multitask_eval_step()`.
  Esse caminho executa forward/loss/predição, não chama backward nem atualiza
  parâmetros, e libera o estado de forward em `finally`. Não há estado de
  treinamento/avaliação alternado no modelo multitarefa.

O ponto de integração é um callback opcional de fim de época no trainer e uma
política de agenda no runner. A política não foi colocada no trainer porque o
trainer não deve escolher o split nem conhecer relatórios ou checkpoints. O
runner constrói o callback dentro de `_execute_single_experiment()`, portanto
workers CPU recebem somente um inteiro simples, sem serializar closures.

## Contrato

`validation_every_n_epochs: int | None` é aceito por
`run_controlled_experiments()` e pela CLI `--validation-every-n-epochs N`.

- `None`: caminho legado. Não há callback, `validation_history` ou
  `observation_config` no relatório.
- `N > 0`: após épocas múltiplas de N, o runner avalia o split de validação.
  Booleanos, valores não inteiros, zero e negativos são rejeitados.
- Se N exceder o total de épocas, não há observações intermediárias e o histórico
  contém lista vazia.
- Cada entrada de `validation_history` possui `epoch`, `split: "validation"` e
  todos os campos retornados pelo avaliador, inclusive schema v1, suportes,
  disponibilidade e distribuições.
- Métricas intermediárias não entram em `metrics.aggregates` nem
  `metrics.paired_deltas`; ambos continuam baseados em `final_validation`.
- `training.history`, `initial_validation` e `final_validation` mantêm os campos
  e valores legados.

Quando a última época é múltipla de N, o runner não faz uma avaliação extra no
callback. Ele registra `final_validation` como observação da última época. No
telemetry, avaliações intermediárias usam estágio `epoch_validation`, dentro de
`training`; a avaliação final continua no estágio `final_validation`. As etapas
`epoch_sgd` e `epoch_evaluation` continuam medindo apenas seus intervalos
anteriores. Sem telemetry, nenhum coletor ou relógio é inicializado pela nova
opção.

## Checkpoint, fingerprint e resume

A cadência é uma configuração observacional e fica em
`observation_config.validation_every_n_epochs`. Ela não altera
`FINGERPRINT_SCHEMA_VERSION`, `EXPERIMENT_PROTOCOL_VERSION` nem o fingerprint
científico, pois não muda a preparação dos dados, updates ou hipótese de
aprendizado. O checkpoint registra a agenda habilitada e o resume compara a
seção antes de qualquer reutilização, criação de backend ou escrita no arquivo.
Uma divergência falha explicitamente e preserva os bytes do checkpoint.

Ausência de `observation_config` equivale a `None`: checkpoints anteriores ao
B006 que continuam elegíveis segundo o fail-closed da M1.4 podem ser retomados
com a opção desativada. Um checkpoint sem essa seção não pode ser retomado com
cadência ativa. Checkpoints históricos sem fingerprint científico continuam
sujeitos à rejeição já existente da M1.4; esta tranche não introduz bypass.

Seeds e variantes podem continuar sendo expandidas com a mesma cadência e
identidade científica. Runs concluídos preservam seu histórico ao serem
reutilizados. Como o runner não persiste estado parcial de treino por época,
uma falha no meio da execução não publica parcialmente `validation_history`; o
retry reexecuta aquela seed e cria um novo histórico completo, sem combinar
observações de tentativas distintas. As tentativas operacionais de telemetry
continuam obedecendo ao contrato existente.

## Baseline determinístico

A fixture sintética A, seed 7, um exemplo, três épocas e `learning_rate=0.05`
congela a sequência legada de `training.history.total_loss`:
`[2.9987254639778302, 2.931094460057473, 2.8646954739655675]`.
O `total_loss` de validação inicial é `3.067645245925604` e o final é
`2.8646954739655675`. O teste compara todos os campos de treino, validação,
agregações e deltas entre cadência desativada e habilitada, além de fixar esses
valores estáveis do caminho desativado.

## Garantia científica e limites

A observação ocorre depois da avaliação do split de treino de cada época. Ela
não chama o treino, altera parâmetros, cria updates, muda a ordem de exemplos,
usa RNG ou afeta losses, otimização, seeds ou protocolo. Testes CPU determinísticos
comparam o caminho desativado e o habilitado, incluindo parâmetros, ordem e
quantidade de updates, histórico de treino, resultados finais, agregados, deltas
e estado do RNG global. A igualdade demonstrada vale para o backend CPU e o
ambiente testado; não é uma alegação bit a bit entre hardwares.

A validação ativa aumenta o custo operacional pelo número de forwards no split
de validação; não foi executado benchmark nem treinamento ampliado. Nenhum teste
GPU/CUDA ou OpenCL foi executado. Não foram alterados backends, modelos, losses,
gradientes, datasets, RNG, protocolos ou políticas de backend.

## Arquivos e evidências

Arquivos alterados nesta tranche:

- `src/kinera/training/trainer.py`: callback passivo após a avaliação de treino.
- `src/kinera/experiments/comparison.py`: política, histórico, checkpoint/resume,
  propagação sequencial/paralela e estágio de telemetry.
- `scripts/compare.py`: flag opcional da CLI.
- `tests/test_multitask_training.py`, `tests/test_comparison.py` e
  `tests/test_telemetry_controlled_runner.py`: regressões determinísticas.
- `README.md`: contrato resumido para consumidores.

Validação local sobre a base atualizada: 750 testes aprovados, 54 skips e 5
testes deselecionados conforme a CI, mais 715 subtestes. O ambiente usa Python
3.15; Python 3.10 não está instalado. `compileall`, `git diff --check` e import
CPU-only passaram. Nenhum teste CUDA/OpenCL foi executado. O workflow remoto
associado à PR e seu status são apresentados na entrega de publicação.

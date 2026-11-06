# B008 — Auditoria final de aceite da Issue #26

Data: 2026-11-06
Escopo: auditoria estática, testes determinísticos e documentação. Nenhum código de produção foi alterado.

## 1. Resumo executivo

**Parecer: PARTIALLY_COMPLETE. A Issue #26 não deve ser encerrada ainda.**

As tranches M0–M2.4 estão integradas, os checks tests das nove PRs estão verdes, e a suíte configurada pela CI foi reproduzida nesta worktree. A auditoria não encontrou alteração indevida em losses, gradientes, optimizer, RNG, dados, protocolos ou backends. Há, porém, um requisito funcional explícito sem implementação: o avaliador publica baselines constantes e majoritário para a classificação binária de fronteiras, mas não baselines de caracteres e sílabas derivados dos targets de treino, com política documentada para o recorte conhecido. A Issue #26 não registra decisão posterior que tenha removido esse requisito.

Também faltam evidências diretas de execução no Python mínimo declarado (3.10) e do gate de hardware CUDA. Esses pontos ficam UNVERIFIED; nem pyproject.toml nem CI em CPU/Python 3.14 substituem tais execuções. Não são defeitos observados, mas impedem afirmar validação integral dos critérios correspondentes.

A lacuna de identidade da Fase 5.4 está registrada como exceção histórica limitada: a Issue #26 exclui explicitamente scripts legacy da migração geral, e o README conserva essa fase como runner de reprodução explícita, separado do runner ativo. A limitação continua oferecendo risco se alguém retomar um checkpoint 5.4 para sustentar novos resultados; checkpoints desse runner não devem ser tratados como identidade verificada. A exceção não amplia a garantia do B007.

## 2. SHA e escopo auditado

- Repositório: Brazwed/Kinera.
- Branch B008: audit/issue-26-metrics-b008.
- SHA inicial de main: 1bcc664df5e2660b600e732ec75549cfe1c09f3a.
- Merge-base inicial: 1bcc664df5e2660b600e732ec75549cfe1c09f3a.
- HEAD inicial: 1bcc664df5e2660b600e732ec75549cfe1c09f3a.
- Main final auditada: eb2108ac549f98954cef36171e4b42b8acc97617.
- Merge-base final após fast-forward: eb2108ac549f98954cef36171e4b42b8acc97617.
- Avanço durante a auditoria: fast-forward de quatro commits até o merge do PR #74 (OpenCL F2.1). A revisão dos arquivos alterados confirmou que são backend/kernel OpenCL, testes OpenCL e documentação dessa tranche; nenhum critério de metrics mudou. Esses arquivos não foram editados pela B008. A suíte foi repetida no novo HEAD.
- PR #73 integrado: merge commit 1bcc664df5e2660b600e732ec75549cfe1c09f3a.
- Issues #25, #26 e #27 consultadas com corpo e comentários/histórico disponíveis; na auditoria, as três permaneciam abertas.
- Worktree: Worktrees/Kinera_B008, criada a partir de origin/main; estado inicial limpo. Nenhuma outra worktree foi modificada.
- Alteração B008: somente este relatório.

## 3. Matriz completa de rastreabilidade

PASS significa que código e evidência de teste/documentação foram encontrados; PARTIAL que a evidência ou cobertura deixa uma parte por demonstrar; FAIL que um requisito obrigatório não está implementado; UNVERIFIED que a verificação não foi executada; EXCEPTION identifica uma exclusão histórica formalmente registrada. O status não é inferido do merge isoladamente.

| ID | Requisito verificável | Tranche | Código e testes/evidência | PR | Status | Risco e ação |
| --- | --- | --- | --- | --- | --- | --- |
| M0-01 | Congelar e manter resultados legados, aliases, ordem de soma, conversões, denominadores e política de zero. | M0 | monitor/metrics/core.py (mean_from_total, safe_ratio, summarize_values) e aggregation.py (mean_std); tests/test_metrics_m0_compatibility.py::test_population_statistics_are_bitwise_equal_to_legacy_formula; testes de rotas remanescentes; docs/metrics-issue-26-m2-1.md congela valores floating-point antes/depois dos diagnósticos. | #35, #66 | PASS | Evidência cobre fixtures determinísticas e consumidores testados; não é alegação de equivalência universal para toda entrada/hardware. Nenhuma ação corretiva indicada. |
| M0-02 | Centralizar aritmética compartilhada sem estado global nem mistura entre épocas, splits, seeds ou workers. | M0 | monitor/metrics/{core,aggregation,confusion,segmentation}.py; acumuladores permanecem na avaliação/execução; testes de agregação e comparação cobrem instâncias independentes e invariantes. | #35, #50, #51 | PASS | Ownership observado é local; sem ação. |
| M0-03 | Preservar adapters, campos e imports legados; manter _multitask_targets no adaptador apropriado. | M0 | training/evaluator.py importa _multitask_targets de training.metrics; APIs/reexports em monitor/metrics/__init__.py; tests/test_package_layout.py e testes M0. | #35 | PASS | Reexports compatíveis permanecem. |
| M0-04 | Manter losses diferenciáveis, gradientes, treinamento, seleção/ordem de exemplos e backend fora do pacote de métricas. | M0 | nn/losses/, training/trainer.py, training/metrics.py e backends/ mantêm seus donos; métricas recebem resultados numéricos/IDs. Os PRs de metrics não alteraram kernels ou fórmulas de loss. | #35–#73 | PASS | Não foi observada mudança científica nesses componentes. |
| M0-05 | Incluir segmentação no inventário e preservar boundary_accuracy e o fallback histórico 1.0. | M0/M1 | language/syllables/tokenizer.py::evaluate_records; alias plano usa safe_ratio(..., zero_denominator=1.0); tests/test_syllables.py e tests/test_monitor_metrics_m1.py::OffsetSegmentationMetricsTests. | #35, #50, #51 | PASS | Schema novo expressa suporte vazio sem alterar o alias. |
| M0-06 | Documentar API pública, fontes de métricas, consumidores e exceções reais de domínio. | M0/M2 | monitor/metrics/__init__.py, docs/architecture.md, README.md, relatórios B002/B004/B005/B006/B007; inventário na seção 8 deste documento. ETL, gradient checking e losses permanecem exceções justificadas. | #35, #66 | PASS | O inventário B008 registra a exceção de identidade da Fase 5.4. |
| M0-07 | Não sobrescrever relatórios/checkpoints históricos com campos que aleguem dados novos retroativamente. | M0–M2 | Escritas acrescentam campos; metrics_schema_version descreve envelope/observações novas, sem converter registros internos antigos; resume mantém aliases e telemetry prévia. Testes de checkpoint e docs B002/B003/B005/B006. | #62, #64, #68, #70 | PASS | Metadados ausentes em registros legados continuam ausentes. |
| M1-01 | Expor TP/TN/FP/FN, distribuições previstas, suportes, precision, recall, specificity e balanced accuracy binária. | M1.1 | monitor/metrics/confusion.py::BinaryConfusion e binary_classification_metrics; training/evaluator.py::evaluate_multitask_examples; tests/test_monitor_metrics_m1.py e tests/test_multitask_training.py. | #50, #51, #62 | PASS | A classe positiva e o threshold mantêm a semântica da head de fronteira. |
| M1-02 | Definir zero suporte por métrica: F1 pode ser zero mesmo com precision indefinida; balanced accuracy de duas classes exige suporte das duas classes. | M1.1 | confusion.py::f1_score calcula 2TP/(2TP+FP+FN) diretamente; balanced_accuracy retorna indisponível se faltar classe. Testes test_zero_positive_support, test_no_predicted_positives, test_f1_score_indisponivel, test_zero_negative_support. | #50, #62 | PASS | Valor zero e ausência de suporte são distintos. |
| M1-03 | Baselines constantes 0/1 e majoritário determinístico usam treino; baseline é avaliado no split/modelo e não usa distribuição de validação para escolher classe. | M1.1 | monitor/metrics/baselines.py (evaluate_constant_baseline, majority_class, evaluate_majority_baseline); comparison.py constrói train_boundary_distribution de treino; test_majority_class_unbalanced_and_deterministic_tie_break, test_evaluator_majority_baseline_without_leakage, test_majority_baseline_distribution_comes_only_from_training_examples. | #50, #51 | PASS | Evidência aplica-se às classes binárias da head de fronteira. |
| M1-04 | Fornecer baseline de caracteres/sílabas usando apenas targets de treino e declarar explicitamente o recorte conhecido da baseline silábica. | M1 | Busca por baseline em src/kinera/monitor/metrics, training/evaluator.py, experiments/comparison.py, testes e README encontrou somente baselines boundary_baseline_*; distribuições de caracteres/sílabas atuais são do conjunto avaliado e não formam baselines de treino. Não há teste desses baselines. | #50–#73 | FAIL | Lacuna obrigatória não reconciliada na Issue. Definir inclusão/exclusão de <UNK>/<END> e suporte conhecido; calcular classes somente do treino; testar desempate, ausência de vazamento e avaliação no split escolhido. Não implementado no B008. |
| M1-05 | Publicar suporte/acertos e distribuição alvo/prevista, separando targets silábicos known, UNK e END quando recuperável. | M1.1/M1.3 | training/evaluator.py mantém char_*_distribution, syllable_*_distribution e syllable_target_status_distribution; END é identificado pelo target de caractere antes de classificar UNK. tests/test_multitask_training.py::test_versioned_metric_distributions_and_end_unknown_separation. | #50, #62 | PASS | IDs dependem dos vocabulários identificados no relatório/fingerprint. |
| M1-06 | Segmentar por offsets: exact match por palavra, precision/recall/F1 com TP/FP/FN e scored/skipped; sem confundir com a head binária. | M1.1 | monitor/metrics/segmentation.py (evaluate_word_offsets, evaluate_corpus_offsets, offset_f1) define posições internas candidatas L-1; tokenizer.py::evaluate_records emite métricas boundary_offset_* distintas. Testes de cortes ausentes/extras, suporte vazio, monosílabos e registros inválidos. | #50, #51, #62 | PASS | O universo de negativos é explicitamente o conjunto de offsets internos candidatos; não se presume que TN seja “todas as posições do texto”. |
| M1-07 | Preservar F1 de offsets: zero com suporte positivo gold/previsto sem interseção; indefinido somente sem suporte de ambos. | M1.2/B001 | tokenizer.py::evaluate_records calcula 2*matched/(gold+predicted); teste de regressão de F1 sem matches e suporte positivo em tests/test_syllables.py. Precision/recall seguem separadamente não aplicáveis quando seu próprio denominador é zero. | #51 (corrigido B001) | PASS | Sem defeito atual observado. |
| M1-08 | Preservar classe positiva, limiar target >= 0.5, argmax/desempate e unidades; não ajustar thresholds a partir da avaliação. | M1.1 | training/evaluator.py converte target soft da fronteira com int(target >= 0.5); predictions vêm de multitask_eval_step; baselines de acurácia declaram unidade fraction; testes de avaliação e baselines. | #50–#62 | PASS | Nenhum tuning é executado. |
| M1-09 | Publicar schema v1 com value, availability, reason, support; rejeitar NaN/Infinity, tipos inválidos e suportes negativos; preservar aliases. | M1.3 | monitor/metrics/schema.py::metric_result, METRICS_SCHEMA_VERSION=1; tests/test_monitor_metrics_schema.py cobre valores finitos, booleans, suportes e combinações inválidas; serializers JSON e testes de checkpoint. | #62 | PASS | Versões de metrics, telemetry e fingerprint continuam independentes. |
| M1-10 | Preservar resultados de loss por head e ponderada recebidos de nn/losses; não recalcular CE/BCE/MSE nem alterar gradientes. | M1 | training/evaluator.py consome result["losses"]; nn/losses/ conserva matemática; test_multitask_training.py fixa resultados e avaliação não atualiza parâmetros. | #50–#70 | PASS | Avaliação e apresentação não controlam aprendizado. |
| M1-11 | Publicar resultados por seed, média/desvio populacional, unidade de população e suportes; std zero com uma seed não é alegação de estabilidade. | M1.2 | aggregation.py::mean_std, aggregate_metric_values; comparison.py::_build_aggregates; runs individuais ficam em runs; experiments/partial.py agrega por seed. Testes de uma seed, missing e population std. | #51, #66 | PASS | Agregado entre seeds não é média por exemplo; o relatório preserva os runs. |
| M1-12 | Contabilizar valores presentes/missing/invalid e preservar invariantes; ausências/falhas não viram zero nem NaN/Infinity JSON. | M1.2 | MetricAggregateSummary exige conservação; _build_aggregates inclui execuções concluídas sem final_validation como missing; status de falha fica separado do resultado científico. Testes de conservação e serialização. | #51 (B001), #62 | PASS | O denominador científico agregado são observações de runs concluídos; falhas operacionais não são impostas como métrica zero. Status geral mantém failed_runs. |
| M1-13 | Comparar deltas somente em seeds comuns concluídas e valores válidos; informar orientação, unidade, pares e causas missing/invalid. | M1.2 | aggregation.py::summarize_paired_deltas; comparison.py::_build_paired_deltas; teste test_paired_deltas_use_completed_common_seeds_and_correct_units; loss = base−candidato, qualidade = candidato−base. | #51 (B001), #68 | PASS | A mesma seed não basta fora do runner; a conclusão é limitada à identidade compartilhada validada no checkpoint comparativo. |
| M1-14 | Não combinar resultados de splits/configurações/dados diferentes como pares comparáveis; preservar diferenças previstas entre variantes. | M1.2/M1.4 | _build_experiment_fingerprint identifica treino, validação, vocabulários, configuração e inputs preparados por variante; validação precede backend, reutilização e escrita; variantes/seeds novas podem ser adicionadas sem misturar identidades. test_checkpoint_rejects_changed_data_before_execution_and_preserves_bytes, test_checkpoint_expands_seeds_and_variants_without_retraining_old_runs. | #64, #70 | PASS | SHA-256 identifica apenas componentes declarados; não é assinatura nem prova universal de equivalência. |
| M1-15 | Schema e envelope permanecem legíveis com registros/checkpoints históricos; não atribuir v1 a cada registro antigo por inferência. | M1.3/M2.2 | _checkpoint_report põe versões no envelope/seção; runs[*].final_validation mantém sua própria presença/ausência de schema; fixtures de resume e documentação B002/B005. | #62, #68 | PASS | Envelope v1 não afirma que todo registro interno foi emitido em v1. |
| M2-01 | Diagnósticos reutilizam aritmética comum, mantendo buckets, categorias, pareamento e seleção no domínio. | M2.1 | diagnostics/syllables.py (_summarize_pair, _grouped_pair_summary, _coverage, _target_char_effects, _representative_examples) e orthography.py (classify_target, _summarize) usam helpers/schema compartilhados; test_diagnostic_metrics_m2.py congela valores, nomes, tipos, ordem e empates. | #66 | PASS | Gradient checking permanece em diagnostics/gradients.py; não há forward adicional. |
| M2-02 | Preservar sinais/denominadores: delta loss base−aumentada, delta acurácia aumentada−base; sem média de médias com suporte desigual. | M2.1 | core.oriented_delta, summarize_values, summarize_deltas; documento B004 congela deltas, média legada, frações estritas e probabilidade média. | #66 | PASS | Deltas por exemplo não são deltas por seed; APIs de uma população não são substituídas pelas da outra. |
| M2-03 | Cadência opcional de validação por época, default legado preservado, final sem duplicação, observações fora dos agregados finais. | M2.3 | trainer.py::train_multitask_examples callback passivo; comparison.py::_validate_validation_cadence, observe_epoch e final_validation; CLI scripts/compare.py; testes de N=1/N=2/N>epochs e aliases/agregados legados. | #70 | PASS | Validação adiciona forwards/custo operacional somente quando habilitada. |
| M2-04 | A avaliação opcional não altera parâmetros, updates, RNG, losses ou histórico de treino. | M2.3 | test_epoch_end_evaluation_preserves_training_updates_and_model_state compara ordem/quantidade de updates, parâmetros, RNG, histórico e resultados finais no CPU; avaliação usa multitask_eval_step sem backward/step e libera estado de forward em finally. | #70 | PASS em CPU | Não há evidência equivalente com cadência habilitada em hardware CUDA; ver UNVERIFIED TEST-06. Não se alega equivalência bit a bit entre hardwares. |
| M2-05 | Resume verifica identidade científica e configuração observacional antes de reutilizar ou escrever; checkpoint legado falha fechado quando identidade ausente. | M1.4/M2.3/M2.4 | comparison.py (_validate_experiment_fingerprint, _matches_validation_observation_config); fingerprint.py::validate_legacy_checkpoint_identity; testes de mudança de dados, versão/digest, null, {}, preservação byte a byte e seed expansion. | #64, #70, #73 | PASS nos fluxos cobertos | Fingerprint v1 não hashia automaticamente código/protocolo; a versão do protocolo é manual. |
| M2-06 | A Fase 5.3 e 5.5 falham fechado com checkpoints sem identidade, verificam dados no preflight e no worker e permitem expansão compatível. | M2.4 | scripts/legacy/phase5_3.py, phase5_5.py; fingerprint.py (build_legacy_experiment_fingerprint, load_legacy_experiment_inputs); test_legacy_checkpoint_identity.py e test_legacy_experiment_fingerprint.py. | #73 | PASS | Checkpoint não é reescrito quando rejeitado; seeds completas só são reutilizadas após validação. |
| M2-07 | A Fase 5.4 sem fingerprint é avaliada segundo escopo histórico, sem estender a garantia B007. | M2.4 | scripts/legacy/phase5_4_expanded.py usa run_seed_jobs() e load_seed_checkpoint() (configuração/seeds, sem hash de dados); README, linhas 544–546 e 747–763, a declara runner histórico para reprodução explícita e fora do runner ativo scripts/compare.py; a Issue #26 declara scripts legacy como exceções de migração. | #73 / exceção da #26 | EXCEPTION | Não retomar checkpoint 5.4 de conteúdo desconhecido para produzir novo claim. Se a fase voltar a ser usada para novas alegações, abrir follow-up de identidade ou tornar esse resume inelegível. A exceção não é evidência de identidade verificada. |
| M2-08 | Publicar seção metrics v1 com aliases de topo e telemetry opcional independente, sem misturar métricas com tempos/status/fingerprint. | M2.2 | comparison.py (_checkpoint_report, _serialize_telemetry_envelope, _preserve_run_telemetry); metrics mantém metrics_schema_version, telemetry telemetry_schema_version, fingerprint no nível científico, status no domínio de experiments. Testes round-trip/resume/attempts. | #68, #70 | PASS | Métricas continuam utilizáveis sem telemetry; campos históricos não são substituídos. |
| M2-09 | Preservar tentativas/segmentos operacionais no retry/resume; falha de coleta não altera sucesso científico. | M2.2 | monitor/telemetry/records.py, multi_seed.py, _preserve_run_telemetry; tests/test_telemetry_checkpoint.py e test_telemetry_controlled_runner.py cobrem round-trip, retry, append-only e falhas fail-soft. | #49, #68 | PASS no contrato compartilhado | T2b/T3/T4 avançados pertencem à #27, não são gate para métrica científica individual. |
| M2-10 | Não alocar confusion multiclasses densa nem reter logits/gradientes completos para métricas básicas. | M2 | O núcleo é de confusão binária; evaluator consome losses/IDs escalares; test_evaluator_uses_scalar_model_step_without_exporting_head_vectors; docs/cuda-backend.md delimita caminhos single-task legados. | #50–#66 | PASS em código/fixture CPU; CUDA UNVERIFIED | Não atribuir esta garantia a diagnósticos de probabilidade legados ou a todo entrypoint. |
| TEST-01 | Testar contagens, fronteiras perfeitas/erradas, suportes vazios, classe ausente, F1=0, baselines e desempates. | M1 | test_monitor_metrics_m1.py, test_syllables.py, test_multitask_training.py, test_monitor_metrics_schema.py. | #50, #51, #62 | PASS | Casos determinísticos e CPU. |
| TEST-02 | Testar média/std, uma seed, seeds ausentes/falhas, missing/invalid, pares e invariantes de conservação. | M1.2 | test_comparison.py, test_monitor_metrics_m1.py, test_telemetry_checkpoint.py; inclui execução concluída sem final_validation. | #51, #62 | PASS | Falhas são status, não valor zero. |
| TEST-03 | Testar relatórios/checkpoints novos e históricos, round-trip, resume compatível/incompatível, telemetria e retry. | M1.3–M2.4 | test_comparison.py, test_experiment_fingerprint.py, test_legacy_checkpoint_identity.py, test_telemetry_checkpoint.py, test_telemetry_controlled_runner.py. | #62–#73 | PASS nos fluxos cobertos | Exceção 5.4 permanece explícita. |
| TEST-04 | Testar package layout e imports CPU-only sem CUDA/OpenCL ou dependência numérica obrigatória. | M0–M2 | test_package_layout.py; import direto de metrics, telemetry, evaluator, comparison e diagnósticos nesta auditoria sem kinera.backends.cuda/opencl em sys.modules; dependências obrigatórias do pyproject.toml são vazias. | #35–#73 | PASS no ambiente testado | Não substitui execução no Python mínimo. |
| TEST-05 | Executar distribuição/testes no Python >=3.10 mínimo declarado. | Todos | pyproject.toml declara >=3.10; .github/workflows/ci.yml configura somente Python 3.14; ambiente B008 tem Python 3.15 e não tem python3.10. AST parse com feature_version=(3,10) passou em 166 arquivos src/scripts; inspeção direcionada de APIs recentes não encontrou usos conhecidos. | CI #35–#73 em 3.14 | UNVERIFIED | Compatibilidade sintática foi verificada; runtime, instalação e dependências de plataforma no 3.10 não. Executar suite/package layout em 3.10 ou adicionar matriz aprovada antes de marcar completo. |
| TEST-06 | Manter gates CUDA de avaliação residente e equivalência/tolerâncias aprovadas, sem D2H de vetores completos. | M1/M2.3 | Existe gate opt-in tests/test_cuda_multitask_training.py (KINERA_RUN_CUDA_TESTS=1) com teste trainer/evaluator e runner CPU/CUDA; a avaliação multitarefa retorna perdas/IDs escalares e libera forward state. O CI atual não habilita CUDA; a suíte B008 não executou GPU. | #50–#73 | UNVERIFIED para execução física atual | Código/gate herdados existem; o resultado não foi reproduzido nesta auditoria e a cadência M2.3 não tem execução CUDA documentada. Rodar o gate em runner autorizado e, se necessário, cobrir cadência ativa em fixture CUDA antes de declarar validação completa desta garantia. |

### Histórico de PRs integrados

Todos os PRs abaixo foram consultados no GitHub; cada head apresenta o check tests concluído com success. Isso comprova o workflow configurado em cada head, não os gates opt-in de hardware que o workflow omite.

| Tranche | PR | HEAD do PR | Merge commit | Check tests |
| --- | --- | --- | --- | --- |
| M0 | [#35](https://github.com/Brazwed/Kinera/pull/35) | 1c03b94409ef8eb396007a25a25756569c91036c | dafd0505dabe21b9b7c5add2e79a86c515a37d9a | success |
| M1.1 | [#50](https://github.com/Brazwed/Kinera/pull/50) | 66cb1238ba2282c7e0592cf2d65cc31b7536eaa6 | 5ab05239ddb40efe1862e4efac6a499373ce9121 | success |
| M1.2 | [#51](https://github.com/Brazwed/Kinera/pull/51) | c672bd8982fd1906d88b417e278eb5dc59562a50 | 373c9a56e3a3ffb698d7f820f6a170e17304573c | success |
| M1.3 | [#62](https://github.com/Brazwed/Kinera/pull/62) | a8342f4eddaca6aa61188a4eaeb05a2cf5e461b6 | c72502db84f7e638dc58bbf2626f57e61636fbaa | success |
| M1.4 | [#64](https://github.com/Brazwed/Kinera/pull/64) | 70285957f4fd56bf6ca286d9b75c60bcc8603c58 | 3a9cd4df5a67557e5f8b2de0926f4b9f32b35106 | success |
| M2.1 | [#66](https://github.com/Brazwed/Kinera/pull/66) | 705fa3b3bc86b5b2f52e1907554e7e68b53e9fe1 | ed31cf90df257a18bfd02536a99470c46717d025 | success |
| M2.2 | [#68](https://github.com/Brazwed/Kinera/pull/68) | d406f55231b1751d6ec21f63afbc8c5cb4612220 | f3887fc33a1e7c77372f3ffeaaf4c33771669e21 | success |
| M2.3 | [#70](https://github.com/Brazwed/Kinera/pull/70) | a9f710190e59b5510d0f89f7d162423438376b3d | a3c0dd2da678ecab29852dd8ccaefc8a4be10398 | success |
| M2.4 | [#73](https://github.com/Brazwed/Kinera/pull/73) | 0c8e21216d194150cfe62948297f4370878b324d | 1bcc664df5e2660b600e732ec75549cfe1c09f3a | success |

## 4. Auditoria M0

monitor/metrics/ é a fonte das operações aritméticas reutilizadas: divisão e fallback compatível, estatísticas populacionais, deltas orientados, validação de confusão binária e contadores de segmentação. A ordem sum(values)/n e a fórmula populacional de média/desvio são explícitas. Os consumidores não compartilham acumuladores globais.

Os snapshots M0 e B004 registram igualdade numérica nas fixtures cobertas, e os testes de rotas remanescentes verificam updates/RNG e restauração de estado. Isso suporta equivalência para os casos observados; não demonstra igualdade universal para todas as distribuições ou hardwares. Losses continuam em nn/losses, cálculo de targets em training, controle de execução em experiments, regras linguísticas em diagnostics/language e gradient checking em diagnostics/gradients.py.

## 5. Auditoria M1

### M1.1 — confusão, baselines e segmentação

As métricas de fronteira usam 0 = sem corte, 1 = corte; specificity e balanced accuracy seguem a política de duas classes. A implementação calcula F1 diretamente dos contadores, preserva a distinção entre precision indefinida e F1 igual a zero, e dá suporte a contagens de classe alvo/prevista. Segmentação por offsets possui nomes próprios e denominador de posições internas candidatas; o alias legado boundary_accuracy preserva fallback 1.0.

Os baselines constantes e majoritário existentes são binários e se aplicam à head de fronteira. train_boundary_distribution é derivada de exemplos de treino; os testes verificam que a validação não escolhe a classe majoritária. Não há baseline de caractere ou de sílaba no avaliador nem no relatório do runner. O código de histogramas char_*_distribution e syllable_*_distribution descreve targets/predictions do split avaliado; não é baseline de treino. Esse requisito permanece FAIL.

### M1.2 — agregados e pares

Agregados são por seed, com desvio populacional e contagens de valores, ausências e inválidos. Observações concluídas sem final_validation contam como missing; falhas operacionais permanecem em failed_runs e não entram como zeros. Deltas usam seeds comuns concluídas, classificação missing/invalid, unidades e sinais declarados. O fingerprint do runner comparativo limita a comparação a um experimento com dados/configuração comuns e identidade de entrada por variante.

### M1.3 — schema

metrics_schema_version=1 é independente de telemetry e fingerprint. metric_result() valida campos, suportes, finitude e coerência de availability/value. Aliases escalares continuam legíveis. O envelope atual não reescreve registros históricos nem lhes atribui descritores retroativos.

### M1.4 — identidade

O fingerprint comparativo inclui dados selecionados e preparados, splits, vocabulários, configuração científica e inputs por variante; exclui seeds da identidade comum e controles operacionais. Resume incompatível ou legado sem fingerprint falha antes de reutilização/escrita e mantém o checkpoint intacto. SHA-256 não é assinatura, não detecta alteração do código automaticamente e não prova equivalência universal; versões de protocolo são responsabilidade explícita do projeto.

## 6. Auditoria M2

- **M2.1:** diagnósticos silábico e ortográfico delegam resumos numéricos e schema, mantendo buckets, classes, pareamento, seleção e ranking no domínio. Os snapshots congelam valores, aliases, suportes, ordem e desempates.
- **M2.2:** relatório comparativo publica metrics e telemetry em seções independentes e versionadas. experiment_fingerprint não é nenhuma das duas. Telemetry preserva attempts/segments em resume e falhas de coleta não mascaram estado científico.
- **M2.3:** validation_every_n_epochs=None preserva o formato legado. Cadências positivas publicam época/split e observações intermediárias não entram nos agregados finais. A validação agendada na última época reutiliza final_validation. Resume compara observation_config; campo ausente e null/{} explícitos têm semânticas diferentes e configurações inválidas são rejeitadas sem alterar bytes. Equivalência de treinamento foi demonstrada com fixture CPU determinística, não em CUDA.
- **M2.4:** Fases 5.3 e 5.5 usam manifesto v1 compatível com o algoritmo canônico SHA-256 da M1.4, hashes de registros efetivos, vocabulários e configuração científica. Preflight e worker verificam identidade; seeds podem ser expandidas sob a mesma identidade.
- **Fase 5.4:** exceção explícita de protocolo histórico, registrada acima; a garantia fail-closed do B007 não cobre esse caminho.

## 7. Validação reproduzida em B008

Ambiente local: Python 3.15.0; PYTHONPATH=src. A primeira tentativa sem PYTHONPATH falhou na coleta por ModuleNotFoundError: kinera em 65 módulos; ela não exercitou testes. A repetição com o caminho de fonte correto passou:

    774 passed, 57 skipped, 5 deselected, 1 warning, 760 subtests passed in 41.37s

Comando: PYTHONPATH=src python -m pytest -q tests com as cinco deselections idênticas a .github/workflows/ci.yml. As cinco são entradas históricas de profiling, piloto ampliado, resume de piloto e microbenchmark explicitamente excluídas pelo workflow; não foram executadas. Os 57 skips são gates opt-in CUDA/OpenCL (KINERA_RUN_CUDA_TESTS=1 / KINERA_RUN_OPENCL_TESTS=1). O teste direcionado atualizado de arquivos CUDA/OpenCL confirmou 169 passes e os mesmos 57 skips. Nenhum GPU foi inicializado. Houve um aviso de depreciação do alias CallSpec2 no plugin externo AnyIO.

Verificações adicionais:

- python -m compileall -q src scripts tests: passou.
- git diff --check: passou após a edição documental final.
- Import CPU-only de metrics, telemetry, evaluator, comparison e diagnósticos: passou; módulos kinera.backends.cuda e kinera.backends.opencl ausentes em sys.modules.
- Parser AST com feature_version=(3, 10): sintaxe de 166 arquivos Python em src/ e scripts/ compatível com a gramática 3.10. A inspeção direcionada não encontrou APIs conhecidas introduzidas depois de 3.10.
- Instalação/testes do pacote no runtime mínimo 3.10: UNVERIFIED; esse interpretador não está disponível e a CI configura apenas Python 3.14.
- CUDA/OpenCL opt-in, equivalência de hardware e cadência M2.3 em CUDA: UNVERIFIED; não executados nesta auditoria.
- Nenhum treino ampliado, benchmark ou experimento científico foi executado.

A reprodução final B008, no SHA eb2108a, tem 774 passes com as mesmas cinco deselections da CI. O relatório B007 registra 772 passes em uma base anterior e sem essas cinco exclusões; os números não são diretamente comparáveis porque base e seleção diferem.

## 8. Inventário final de consumidores

| Consumidor | Classificação | Evidência / fronteira |
| --- | --- | --- |
| training/evaluator.py | Integrado | Usa BinaryConfusion, helpers de schema e baselines binários; produz distribuições e suporta losses escalares. Não move fórmulas de loss. |
| training/trainer.py | Integrado | Aritmética existente delega a metrics; atualização e sequência SGD permanecem locais. Callback de época é observador passivo. |
| experiments/comparison.py | Integrado | Seleciona runs/seeds; chama agregação e deltas compartilhados; compõe metrics, telemetry opt-in e fingerprint. |
| experiments/partial.py | Integrado para métricas | Agrega por seed via aggregate_metric_values; saída parcial e temporização não viram envelope comparativo. |
| experiments/config.py | Delegação compatível | Reexporta estatística compartilhada; RunRecord/status são contratos operacionais. |
| experiments/runner_common.py, multi_seed.py | Responsabilidade de execução | Checkpoint, retries e status; runner genérico não tem identidade científica salvo quando os entrypoints a fornecem. |
| diagnostics/syllables.py | Integrado | Helpers compartilhados para resumos; buckets, cobertura, efeitos, pares e ranking são domínio. |
| diagnostics/orthography.py | Integrado | Aritmética/schema compartilhados; categorias e pareamento permanecem locais. |
| diagnostics/gradients.py | Exceção de domínio | Diferenças finitas e comparação de gradientes não são qualidade do modelo. |
| language/syllables/tokenizer.py | Integrado | Usa contadores e schema para avaliar offsets; regras de segmentação permanecem linguísticas. |
| scripts/compare.py | Integrado | Entry point ativo do runner comparativo, cadência e telemetry opt-in. |
| scripts/train.py e avaliação simples | Delegação compatível | Usa trainer/evaluator; não produz seção de comparação. O evaluator single-task legado não herda automaticamente as garantias de transferência do caminho multitarefa. |
| scripts/legacy/phase5_3.py | Integrado com identidade | Métricas diagnósticas e manifesto B007; resume fail-closed. |
| scripts/legacy/phase5_5.py | Integrado com identidade | Diagnóstico ortográfico e manifesto B007; resume fail-closed. |
| scripts/legacy/phase5_4_expanded.py | Exceção histórica | Continua usando runner genérico sem hash de conteúdo; apenas reprodução explícita de protocolo antigo. Não reutilizar checkpoint desconhecido para nova alegação. |
| scripts/legacy/train_hybrid_5min.py e harnesses históricos | Exceção histórica | Mapeados pela Issue #26 como scripts/harnesses cujo protocolo não deve ser migrado incidentalmente. |

Pesquisa de run_seed_jobs() encontrou entrypoints 5.3, 5.4 e 5.5, além de testes. A API genérica não foi alterada no B007; só 5.3/5.5 validam identidade. Ocorrências de sum, Counter, timers, ETL ou loss não foram tratadas como duplicações sem confirmar ownership.

## 9. Compatibilidade e regressões

Não foram encontrados conflitos de integração na main atual: o PR #73 está integrado, cada PR #35/#50/#51/#62/#64/#66/#68/#70/#73 tem check tests concluído com sucesso, e a suíte determinística local passou com as deselections do workflow. Compatibilidade CPU-only e a gramática Python 3.10 foram inspecionadas. Não há evidência direta de execução runtime Python 3.10 ou de equivalência real em hardware CUDA/OpenCL nesta auditoria.

O relatório do B005 registrou seções metrics/telemetry, aliases de topo e round-trip. As evidências de B006 cobrem as configurações None, cadências ativas e rejeição de observação ausente/incompatível. A correção B006 rejeita observation_config: null e {} explícitos no modo desativado, preservando o checkpoint byte a byte.

## 10. Riscos residuais

| Risco | Severidade | Escopo/impacto | Bloqueia aceite? | Mitigação mínima |
| --- | --- | --- | --- | --- |
| Baselines de caractere/sílaba ausentes | Alta, requisito científico | Falta referência majoritária/constante de treino para saídas multiclasses e definição de recorte known; diagnósticos ficam sem esse contexto. | Sim | Implementar somente baselines exigidos, sem atualizar modelo; definir targets/UNK/END/suporte, treino-only e desempate determinístico; testar vazamento e ausências. |
| Python 3.10 não executado | Média, evidência de compatibilidade | Declaração >=3.10, workflow é 3.14; parser sintático passou, runtime/package não. | Sim para afirmação de aceite integral | Rodar testes/package layout no 3.10 disponível ou aprovar matriz de CI. |
| Gates de hardware opt-in não executados | Média, garantia de backend | 54 testes CUDA/OpenCL são skipped pela variável opt-in; M2.3 foi comparada em CPU, não com cadência CUDA. | Não prova defeito; impede alegação de equivalência CUDA completa | Executar gate em hardware autorizado e, se exigido para M2.3, incluir cadência ativa no fixture CUDA correspondente. |
| Fase 5.4 sem manifesto B007 | Média, checkpoint legado | Resume pode comparar apenas configuração/seeds e reutilizar resultado se o conteúdo no caminho mudar. | EXCEPTION de escopo da #26; risco não pode ser ignorado para novos claims | Não usar checkpoint antigo para novos claims; abrir follow-up se o runner continuar produzindo evidência atual. |
| Versões manuais de protocolo/fingerprint | Baixa a média | Mudança semântica no código não altera digest automaticamente. | Não bloqueia o aceite do contrato documentado | Atualizar explicitamente phase_protocol_version, EXPERIMENT_PROTOCOL_VERSION/versão de preparação quando protocolo mudar; revisão de release. |
| SHA-256 sem assinatura | Baixa para divergência acidental | Não protege contra alteração deliberada do checkpoint nem prova identidade científica universal. | Não | Manter essa limitação nos docs; não tratar hash como autenticidade. |
| Issue #27 incompleta | Operacional, fora da #26 | T2b/T3/T4 não concluídos; não invalida métricas científicas já emitidas. | Não bloqueia #26; bloqueia #25 | Concluir #27 por critérios próprios e validar integração conjunta antes de fechar #25. |

## 11. Critérios não verificados

1. Execução direta, instalação e testes no Python 3.10.
2. Execução opt-in do gate CUDA, incluindo avaliação por época M2.3; execução OpenCL também não foi feita e pertence a runners/hardware próprios.
3. Reprodutibilidade ampla em configurações/hardwares além das fixtures determinísticas executadas.
4. Autenticidade de checkpoints; fingerprint é hash de integridade de conteúdo, não assinatura criptográfica.

Os itens 1 e 2 não foram convertidos em PASS. A suíte CI verde na plataforma configurada não executa automaticamente esses gates.

## 12. Dependências externas

- #25 permanece aberta e exige aceite próprio de #26 e #27, mais integração conjunta validada.
- #27 permanece aberta; T2b/T3 e adoção adicional T4 continuam pertencendo à telemetria. A seção conjunta do PR #68 satisfaz somente a fronteira de composição já entregue.
- Issues #52/#59 e backends de outros agentes não foram modificados nem são dependências para implementar os baselines faltantes.
- Nenhuma decisão formal posterior foi encontrada na Issue #26 que retire o baseline de caracteres/sílabas ou o requisito Python mínimo.

## 13. Parecer final

**Issue #26: PARTIALLY_COMPLETE. Não recomendar fechamento agora.**

Bloqueio funcional: baselines de caracteres/sílabas exigidos no item M1 da Issue não estão presentes nem formalmente adiados/obsoletos. Evidências ainda pendentes: runtime mínimo Python 3.10 e gate CUDA relevante. A Fase 5.4 foi classificada como exceção histórica de escopo, com risco limitado e aviso contra novos claims baseados em resume de identidade desconhecida.

**Issue #27 — Telemetry:** aberta; entregas T0/T1, T2a e integração C observadas, com relatório conjunto M2.2 integrado. T2b/T3 e partes de T4 continuam pendentes. Não declarar T3/T4 concluídos por causa da composição M2.2.

**Issue #25 — Coordenação:** aberta e não apta a encerramento. Seus gates exigem #26 e #27 aceitas separadamente e validação conjunta posterior.

### Menor sequência de aceite

1. Implementar em tranche separada somente os baselines de caracteres/sílabas, definindo recorte known/UNK/END e suporte antes da fórmula final.
2. Validar o pacote/testes em Python 3.10.
3. Executar o gate CUDA em runner autorizado; registrar explicitamente se a cadência M2.3 ativa participa do gate ou precisa de fixture adicional.
4. Repetir a matriz de aceite. Manter #25 e #27 abertas até seus próprios critérios serem satisfeitos.

Este B008 não alterou código de produção, schemas, checkpoints, protocolos ou checklists do GitHub. O relatório é evidência de auditoria, não autorização para integrar ou fechar Issues.

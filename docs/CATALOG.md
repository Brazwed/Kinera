# Catálogo de documentos

Registro mestre dos Markdown de `docs/`. IDs identificam documentos, não caminhos: continuam iguais após movimentação. Os caminhos abaixo são atuais; os Lotes 1–3 e C1–C3 organizaram os 32 documentos originais por assunto.

**Como ler:** `Tipo` descreve a natureza; `Situação documental` descreve o ciclo de vida do arquivo; `Resultado/parecer` registra a conclusão técnica da fonte, quando houver. Para documentos existentes, situação formal ausente fica como `não identificado (legado)`. Ausência de parecer não significa resultado inconclusivo.

**Cobertura do inventário D0:** 34 documentos: 32 existentes no baseline D0.1 e os dois índices. Os IDs 0001–0032 foram propostos no inventário D0.1 e foram preservados; os caminhos anteriores dos documentos movimentados nos Lotes 1–3 e C1 permanecem nos respectivos registros. README e catálogo usam os IDs 0033 e 0034. **Atualização R2.5 (Issue #92):** acrescentado KIN-DOC-0035, totalizando **35 documentos catalogados**, sem renumerar ou alterar os registros históricos.

**Exceções restritas de escopo nos testes:** a Issue #88 originalmente definia zero alterações em testes. D0.4/C3 autorizou somente atualizar caminhos/chaves dos relatórios P3 em `tests/test_opencl_host_conversion_analysis.py` e `tests/test_opencl_p3e_device_profiling.py`; D0.5/F1 autorizou adicionalmente resolver o caminho histórico do relatório da Fase 7 em `tests/equivalence/test_cpu_reference_equivalence.py`. As autorizações e limites estão formalmente registrados na [Issue #88](https://github.com/Brazwed/Kinera/issues/88). SHA-256 esperados, fingerprints, assertions, critérios, demais fixtures e cobertura permanecem preservados. O manifesto histórico permanece intacto. Os links canônicos para os novos caminhos em `main` continuam pendentes do merge documental.

## KIN-DOC-0001 — Arquitetura interna — Fase 8

- **Caminho:** [architecture/architecture.md](architecture/architecture.md)
- **Tipo:** `GUIDE` / referência de arquitetura.
- **Situação documental:** não identificado (legado); candidato a referência ativa, sujeito a revisão de vigência.
- **Resultado/parecer:** não aplicável como resultado único; o texto descreve contratos e evidências de fases.
- **Relações:** Issues #8 e #12; relacionado a [cuda-backend.md](backends/cuda/cuda-backend.md), a README raiz e aos artefatos Phase 10/11 citados no próprio texto.
- **Proveniência:** SHAs a02379ca…, a6c278a…, c21aca55… são citados como revisões/evidências; não há um único baseline atribuído a todo o documento.
- **Caminho anterior:** `docs/architecture.md` → caminho atual `docs/architecture/architecture.md`.
- **Limites:** o texto combina arquitetura e histórico de fases; validar se todo conteúdo ainda representa contratos atuais antes de promovê-lo como norma.

## KIN-DOC-0002 — CUDA Runtime, ABI e Buffers — Phases 9–11

- **Caminho:** [cuda-backend.md](backends/cuda/cuda-backend.md)
- **Tipo:** `SPEC` / guia de backend.
- **Situação documental:** não identificado (legado); referência de runtime publicada.
- **Resultado/parecer:** resultados são específicos às fases e evidências listadas; sem parecer geral único.
- **Relações:** Issues #13/#20; relacionado a [architecture/architecture.md](architecture/architecture.md) e [opencl-rx550-preflight.md](backends/opencl/opencl-rx550-preflight.md); PR #23 cita este documento.
- **Proveniência:** referência estrutural `a64b2373…`; outros SHAs e artefatos Phase 9–12 estão listados no documento.
- **Caminho anterior:** `docs/cuda-backend.md` → caminho atual `docs/backends/cuda/cuda-backend.md`. **Permalink histórico:** https://github.com/Brazwed/Kinera/blob/da860a7b79c3dcc5ff37ce0336f0a88b8e00e05a/docs/cuda-backend.md.
- **Limites:** hardware e fases têm escopos próprios; não inferir equivalência geral CUDA/CPU além das evidências citadas.

## KIN-DOC-0003 — Issue #26 audit — B002

- **Caminho:** [metrics-issue-26-b002-audit.md](metrics/metrics-issue-26-b002-audit.md)
- **Tipo:** `AUDIT`.
- **Situação documental:** não identificado (legado); auditoria histórica.
- **Resultado/parecer:** status de requisitos “após B002” está na matriz; não é parecer final da Issue #26.
- **Relações:** Issues #25/#26/#27/#50/#51/#52/#59; PRs #50/#51; PR #62 cita o caminho.
- **Proveniência:** baseline `373c9a56…`; PR #51 HEAD `c672bd89…`; merge-base `9a79fef…`.
- **Artefatos relacionados:** código/testes de métricas listados na matriz interna.
- **Caminho anterior:** `docs/metrics-issue-26-b002-audit.md` → caminho atual `docs/metrics/metrics-issue-26-b002-audit.md`. **Permalink histórico:** https://github.com/Brazwed/Kinera/blob/a8342f4eddaca6aa61188a4eaeb05a2cf5e461b6/docs/metrics-issue-26-b002-audit.md. **Limites:** resultado por tranche, não aceite global.

## KIN-DOC-0004 — Issue #26 — M1.4: identidade e resume de checkpoints

- **Caminho:** [metrics/metrics-issue-26-b003-identity.md](metrics/metrics-issue-26-b003-identity.md)
- **Tipo:** `AUDIT` / decisão de identidade.
- **Situação documental:** não identificado (legado); registro M1.4.
- **Resultado/parecer:** descreve contrato de fingerprint e política de resume; parecer final único não identificado.
- **Relações:** Issue #26; vinculado às trilhas B002/B003 e à auditoria de runners apresentada em outros documentos metrics.
- **Proveniência:** SHA de baseline não identificado no inventário; registrar como não identificado (legado) até confirmação no texto.
- **Caminho anterior:** `docs/metrics-issue-26-b003-identity.md` → caminho atual `docs/metrics/metrics-issue-26-b003-identity.md`. **Limites:** não inferir que cobre todos os runners legados.

## KIN-DOC-0005 — Issue #26 — auditoria final B005 e composição M2.2

- **Caminho:** [metrics/metrics-issue-26-b005-final-audit.md](metrics/metrics-issue-26-b005-final-audit.md)
- **Tipo:** `AUDIT`.
- **Situação documental:** não identificado (legado); auditoria datada.
- **Resultado/parecer:** `PARTIALLY_COMPLETE` para #26; #27 descrita como aberta/parcial no momento registrado.
- **Relações:** Issues #25/#26/#27 e outras listadas no texto; PR #67 referida. Relaciona-se aos relatórios B006–B008.
- **Proveniência:** data 2026-11-06; worktree B005 e vários SHAs citados no corpo, sem um SHA único atribuído aqui.
- **Caminho anterior:** `docs/metrics-issue-26-b005-final-audit.md` → caminho atual `docs/metrics/metrics-issue-26-b005-final-audit.md`. **Limites:** estado histórico das Issues não equivale ao estado atual.

## KIN-DOC-0006 — B006 — M2.3: validação opcional por época

- **Caminho:** [metrics/metrics-issue-26-b006-epoch-validation.md](metrics/metrics-issue-26-b006-epoch-validation.md)
- **Tipo:** `AUDIT` / relatório de tranche.
- **Situação documental:** não identificado (legado).
- **Resultado/parecer:** evidências de validação e limites registrados; status consolidado não identificado como parecer único.
- **Relações:** Issues/PRs #68/#69; relacionado a B005 e à identidade M1.4.
- **Proveniência:** base inicial `f3887fc3…`, base atualizada `36f7fb9e…`.
- **Caminho anterior:** `docs/metrics-issue-26-b006-epoch-validation.md` → caminho atual `docs/metrics/metrics-issue-26-b006-epoch-validation.md`. **Limites:** execução CPU-only e limites de workflow descritos no documento.

## KIN-DOC-0007 — B007 — Integridade de checkpoints dos runners históricos

- **Caminho:** [metrics/metrics-issue-26-b007-legacy-checkpoint-identity.md](metrics/metrics-issue-26-b007-legacy-checkpoint-identity.md)
- **Tipo:** `AUDIT`.
- **Situação documental:** não identificado (legado).
- **Resultado/parecer:** critérios de identidade e elegibilidade de resume são descritos; não converter os resultados dos testes em status global sem consultar a seção de evidências.
- **Relações:** PR #70; Issues #35/#50/#51/#62/#64/#66/#68/#70; relacionado a B008.
- **Proveniência:** base `a3c0dd2d…`; HEAD de origem reportada `a9f71019…`.
- **Caminho anterior:** `docs/metrics-issue-26-b007-legacy-checkpoint-identity.md` → caminho atual `docs/metrics/metrics-issue-26-b007-legacy-checkpoint-identity.md`. **Limites:** exceções de runners legados precisam ser lidas separadamente.

## KIN-DOC-0008 — B008 — Auditoria final de aceite da Issue #26

- **Caminho:** [metrics-issue-26-b008-final-acceptance.md](metrics/metrics-issue-26-b008-final-acceptance.md)
- **Tipo:** `AUDIT` / decisão de aceite.
- **Situação documental:** não identificado (legado); auditoria histórica.
- **Resultado/parecer:** `PARTIALLY_COMPLETE`; o próprio relatório recomenda não encerrar #26 naquele momento.
- **Relações:** Issue #26; PR #76 e PRs M0–M2 listados no documento; PR #76 referencia o arquivo.
- **Proveniência:** vários baselines, merges e SHAs estão em suas matrizes; não reduzir a um único baseline.
- **Caminho anterior:** `docs/metrics-issue-26-b008-final-acceptance.md` → caminho atual `docs/metrics/metrics-issue-26-b008-final-acceptance.md`. **Permalink histórico:** https://github.com/Brazwed/Kinera/blob/ff26f0f85e95dc8dfff302cc28178be809f113bd/docs/metrics-issue-26-b008-final-acceptance.md. **Limites:** status histórico, não status remoto atual da issue.

## KIN-DOC-0009 — Kinera B009 — M1.5 baselines multiclasses

- **Caminho:** [metrics/metrics-issue-26-b009-multiclass-baselines.md](metrics/metrics-issue-26-b009-multiclass-baselines.md)
- **Tipo:** `SPEC` / relatório de tranche.
- **Situação documental:** não identificado (legado).
- **Resultado/parecer:** especifica baselines majoritários de caracteres/sílabas, critérios e testes; parecer global não identificado.
- **Relações:** Issue #26; PR #76 é a auditoria B008 citada como origem da lacuna. Não foi identificada PR própria de publicação do relatório B009.
- **Proveniência:** auditoria B008 e base `b9200eea…` mencionadas no texto.
- **Caminho anterior:** `docs/metrics-issue-26-b009-multiclass-baselines.md` → caminho atual `docs/metrics/metrics-issue-26-b009-multiclass-baselines.md`. **Limites:** não alterar sem preservar políticas de targets e desempate.

## KIN-DOC-0010 — B010 — Compatibilidade Python e gates científicos CUDA

- **Caminho:** [metrics/metrics-issue-26-b010-compatibility-validation.md](metrics/metrics-issue-26-b010-compatibility-validation.md)
- **Tipo:** `AUDIT`.
- **Situação documental:** não identificado (legado).
- **Resultado/parecer:** Issue #26 `BLOCKED` no escopo daquele relatório; Python 3.10 `FAIL`, Python 3.14 `PASS`, CUDA físico `BLOCKED_ENVIRONMENT`, outras dimensões `UNVERIFIED` conforme matriz.
- **Relações:** Issues #25/#26/#27/#78–#81; PRs #79/#80/#81.
- **Proveniência:** código auditado `78501f4d…`; commit de CI Python 3.14 `f0aa2f6b…`.
- **Caminho anterior:** `docs/metrics-issue-26-b010-compatibility-validation.md` → caminho atual `docs/metrics/metrics-issue-26-b010-compatibility-validation.md`. **Limites:** resultados ambientais e históricos, não validação da revisão atual.

## KIN-DOC-0011 — Issue #26 — M2.1: integração dos diagnósticos

- **Caminho:** [metrics-issue-26-m2-1.md](metrics/metrics-issue-26-m2-1.md)
- **Tipo:** `AUDIT` / relatório de implementação.
- **Situação documental:** não identificado (legado).
- **Resultado/parecer:** evidências de migração e testes constam no texto; não há parecer único para #26.
- **Relações:** Issues #26/#64/#65; PRs #64/#65; PR #66 cita o caminho; relacionado a B005.
- **Proveniência:** base `3a9cd4df…`, atualização `685fa68e…`, commits indicados no relatório.
- **Caminho anterior:** `docs/metrics-issue-26-m2-1.md` → caminho atual `docs/metrics/metrics-issue-26-m2-1.md`. **Permalink histórico:** https://github.com/Brazwed/Kinera/blob/705fa3b3bc86b5b2f52e1907554e7e68b53e9fe1/docs/metrics-issue-26-m2-1.md. **Limites:** mudanças de telemetry e contrato ficam fora do escopo descrito.

## KIN-DOC-0012 — F2.0 — Auditoria funcional de representação e perdas OpenCL

- **Caminho:** [opencl-f2-capability-audit.md](backends/opencl/opencl-f2-capability-audit.md)
- **Tipo:** `AUDIT` / planejamento de gates.
- **Situação documental:** não identificado (legado).
- **Resultado/parecer:** pareceres por operação e critérios de GO/NO-GO; documento não autoriza implementação por si só.
- **Relações:** Issues #59/#61/#69/#70; PRs #69/#70; PR #71 publica a auditoria.
- **Proveniência:** baseline funcional `36f7fb9e…`; outras bases por seção.
- **Caminho anterior:** `docs/opencl-f2-capability-audit.md` → caminho atual `docs/backends/opencl/opencl-f2-capability-audit.md`. **Permalink histórico:** https://github.com/Brazwed/Kinera/blob/2ab4e2256b7a0e489c4c58e2b0ba546b2fe621ee/docs/opencl-f2-capability-audit.md. **Limites:** tabela distingue existência de método de suporte a fluxo.

## KIN-DOC-0013 — Kinera OpenCL F1a — Embedding.forward na RX 550

- **Caminho:** [opencl-rx550-f1a-embedding-forward.md](backends/opencl/opencl-rx550-f1a-embedding-forward.md)
- **Tipo:** `EXPERIMENT`.
- **Situação documental:** não identificado (legado); evidência histórica.
- **Resultado/parecer:** `PASS` delimitado aos casos F1a documentados; ressalva do runtime Rusticl mantida.
- **Relações:** Issue/PR #62; PR #63 referencia relatório; relacionado a gates F1b–F1d.
- **Proveniência:** baseline `373c9a56…`; ambiente e fingerprints no texto.
- **Artefatos:** testes e hashes referidos no próprio relatório.
- **Caminho anterior:** `docs/opencl-rx550-f1a-embedding-forward.md` → caminho atual `docs/backends/opencl/opencl-rx550-f1a-embedding-forward.md`. **Permalink histórico:** https://github.com/Brazwed/Kinera/blob/cb2ae2dacd2380d9208ba681f3386dfb4de4eb29/docs/opencl-rx550-f1a-embedding-forward.md. **Limites:** não generaliza para treino/modelo completo.

## KIN-DOC-0014 — Kinera OpenCL F1b — Embedding.backward na RX 550

- **Caminho:** [opencl-rx550-f1b-embedding-backward.md](backends/opencl/opencl-rx550-f1b-embedding-backward.md)
- **Tipo:** `EXPERIMENT`.
- **Situação documental:** não identificado (legado).
- **Resultado/parecer:** resultados separados para RX 550 e fake API; limites de lifecycle e falhas permanecem no documento.
- **Relações:** Issues/PRs #63/#64/#65; PR #65 referencia relatório; série F1a–F1d.
- **Proveniência:** baseline `061d4973…` (merge #63); ambiente registrado.
- **Caminho anterior:** `docs/opencl-rx550-f1b-embedding-backward.md` → caminho atual `docs/backends/opencl/opencl-rx550-f1b-embedding-backward.md`. **Permalink histórico:** https://github.com/Brazwed/Kinera/blob/7dfbeba3d5eea3df68f555356bf7d5700ab09f03/docs/opencl-rx550-f1b-embedding-backward.md. **Limites:** não extrapolar a treinamento híbrido.

## KIN-DOC-0015 — OpenCL F1c — Embedding.step na RX 550

- **Caminho:** [backends/opencl/opencl-rx550-f1c-embedding-step.md](backends/opencl/opencl-rx550-f1c-embedding-step.md)
- **Tipo:** `EXPERIMENT`.
- **Situação documental:** não identificado (legado).
- **Resultado/parecer:** testes F1c e limites de falha/ownership descritos; ressalva de Rusticl preservada.
- **Relações:** Issues #61/#65; PR #67; relacionado a F1a/F1b/F1d.
- **Proveniência:** baseline solicitado `685fa68e…`; branch rebased sobre `ed31cf90…`; merge PR #67 `1cc06fc1…` (HEAD revisado `735db69e…`), conforme o relatório/Issue #61.
- **Caminho anterior:** `docs/opencl-rx550-f1c-embedding-step.md` → caminho atual `docs/backends/opencl/opencl-rx550-f1c-embedding-step.md`. **Limites:** não declarar equivalência geral sem os gates listados.

## KIN-DOC-0016 — Kinera OpenCL F1d — Embedding integration na RX 550

- **Caminho:** [opencl-rx550-f1d-embedding-integration.md](backends/opencl/opencl-rx550-f1d-embedding-integration.md)
- **Tipo:** `EXPERIMENT` / relatório de gate.
- **Situação documental:** não identificado (legado).
- **Resultado/parecer:** matriz relata resultados PASS no ciclo coberto; falhas após enqueue e ressalva do runtime são limites.
- **Relações:** Issues/PRs #67/#68; PR #69 cita o arquivo; relacionado a F1a–F1c.
- **Proveniência:** baseline `1cc06fc1…` (merge PR #67).
- **Caminho anterior:** `docs/opencl-rx550-f1d-embedding-integration.md` → caminho atual `docs/backends/opencl/opencl-rx550-f1d-embedding-integration.md`. **Permalink histórico:** https://github.com/Brazwed/Kinera/blob/17fd8fb54e4e77d2f9fa3b1b0bc0d6c1b9223212/docs/opencl-rx550-f1d-embedding-integration.md. **Limites:** cobre Embedding, não modelo/harness completo.

## KIN-DOC-0017 — F2.1 — HybridRepresentation.forward em OpenCL na RX 550

- **Caminho:** [opencl-rx550-f2-1-hybrid-forward.md](backends/opencl/opencl-rx550-f2-1-hybrid-forward.md)
- **Tipo:** `EXPERIMENT`.
- **Situação documental:** não identificado (legado).
- **Resultado/parecer:** resultados PASS para os critérios registrados; aviso Rusticl e limitações de rollback/performance permanecem.
- **Relações:** Issues #61/#71/#72/#73/#74; PRs #71/#73/#74; PR #74 cita o relatório.
- **Proveniência:** baseline mínimo `faf211c3…` (merge PR #71); ambiente RX 550 declarado.
- **Caminho anterior:** `docs/opencl-rx550-f2-1-hybrid-forward.md` → caminho atual `docs/backends/opencl/opencl-rx550-f2-1-hybrid-forward.md`. **Permalink histórico:** https://github.com/Brazwed/Kinera/blob/a6867bb1cbb6a0aa6c1d7746887262b4d0e508ea/docs/opencl-rx550-f2-1-hybrid-forward.md. **Limites:** não cobre pooling/backward/losses/treino.

## KIN-DOC-0018 — F2.2 — mean_pool_forward em OpenCL na RX 550

- **Caminho:** [opencl-rx550-f2-2-mean-pool-forward.md](backends/opencl/opencl-rx550-f2-2-mean-pool-forward.md)
- **Tipo:** `EXPERIMENT`.
- **Situação documental:** não identificado (legado).
- **Resultado/parecer:** resultados PASS nos casos F2.2 documentados; ressalva de divisão FP64 dinâmica relacionada a investigação separada.
- **Relações:** Issues #61/#75/#76/#78; PR #77 cita o arquivo; relacionado a [opencl-rx550-fp64-division-investigation.md](backends/opencl/opencl-rx550-fp64-division-investigation.md).
- **Proveniência:** baseline mínimo `eb2108ac…`; CI/commit e ambiente listados no relatório.
- **Caminho anterior:** `docs/opencl-rx550-f2-2-mean-pool-forward.md` → caminho atual `docs/backends/opencl/opencl-rx550-f2-2-mean-pool-forward.md`. **Permalink histórico:** https://github.com/Brazwed/Kinera/blob/65707f79c308974f279b9602ac61617a65418547/docs/opencl-rx550-f2-2-mean-pool-forward.md. **Limites:** resultado é específico à operação e ambiente.

## KIN-DOC-0019 — Issue #78 — Investigação de divisão FP64 na RX 550

- **Caminho:** [opencl-rx550-fp64-division-investigation.md](backends/opencl/opencl-rx550-fp64-division-investigation.md)
- **Tipo:** `EXPERIMENT` / `AUDIT`.
- **Situação documental:** não identificado (legado); registro de investigação.
- **Resultado/parecer:** `NO-GO` para presumir precisão da divisão FP64 dinâmica no ambiente testado; não é parecer de toda a OpenCL.
- **Relações:** Issue #78 e roadmap #61; PR #77 é a revisão de produção usada como baseline; PR #80 publica a investigação.
- **Proveniência:** baseline `35882d8c…`; fonte/JSON com fingerprints documentados no relatório.
- **Artefatos:** `experiments/results/opencl-rx550-fp64-division.json` é consumido por teste de integridade.
- **Caminho anterior:** `docs/opencl-rx550-fp64-division-investigation.md` → caminho atual `docs/backends/opencl/opencl-rx550-fp64-division-investigation.md`. **Permalink histórico:** https://github.com/Brazwed/Kinera/blob/83b211517f5c748d90f2820c1e9b8a76e43ecbbd/docs/opencl-rx550-fp64-division-investigation.md. **Limites:** hipóteses e operações não testadas estão separadas no relatório.

## KIN-DOC-0020 — Gate P1 — Linear.forward OpenCL na RX 550

- **Caminho:** [opencl-rx550-p1-linear-forward.md](backends/opencl/opencl-rx550-p1-linear-forward.md)
- **Tipo:** `EXPERIMENT`.
- **Situação documental:** não identificado (legado).
- **Resultado/parecer:** `PASS` específico ao kernel e aos casos listados; ressalva Rusticl mantida.
- **Relações:** sequência P0/P1/P2; links para P2 backward/step e preflight; PRs #52/#53 fazem referência ao documento.
- **Proveniência:** baseline registrado `5ab05239…`; data de execução no texto.
- **Caminho anterior:** `docs/opencl-rx550-p1-linear-forward.md` → caminho atual `docs/backends/opencl/opencl-rx550-p1-linear-forward.md`. **Permalink histórico:** https://github.com/Brazwed/Kinera/blob/35de98db02ce665d603da09974701f23aa210786/docs/opencl-rx550-p1-linear-forward.md. **Limites:** não extrapolar para backend ou treinamento geral.

## KIN-DOC-0021 — Gate P2 — Linear.backward OpenCL na RX 550

- **Caminho:** [opencl-rx550-p2-linear-backward.md](backends/opencl/opencl-rx550-p2-linear-backward.md)
- **Tipo:** `EXPERIMENT`.
- **Situação documental:** não identificado (legado).
- **Resultado/parecer:** `PASS` limitado aos casos executados; driver continua como ressalva.
- **Relações:** P1/P2 step/P2 integration; PRs #52/#53.
- **Proveniência:** baseline `5ab05239…` conforme registro da série P2.
- **Caminho anterior:** `docs/opencl-rx550-p2-linear-backward.md` → caminho atual `docs/backends/opencl/opencl-rx550-p2-linear-backward.md`. **Permalink histórico:** https://github.com/Brazwed/Kinera/blob/35de98db02ce665d603da09974701f23aa210786/docs/opencl-rx550-p2-linear-backward.md. **Limites:** equivalência específica; não declara suporte amplo.

## KIN-DOC-0022 — Gate P2 — integração Linear.forward → backward → step na RX 550

- **Caminho:** [opencl-rx550-p2-linear-integration.md](backends/opencl/opencl-rx550-p2-linear-integration.md)
- **Tipo:** `EXPERIMENT` / integração.
- **Situação documental:** não identificado (legado).
- **Resultado/parecer:** `PASS` para ciclo e casos declarados; ressalva Rusticl registrada.
- **Relações:** P1/P2 subgates e preflight; PRs #52/#53.
- **Proveniência:** baseline `5ab05239…` declarado na série; fixture e ambiente no corpo.
- **Caminho anterior:** `docs/opencl-rx550-p2-linear-integration.md` → caminho atual `docs/backends/opencl/opencl-rx550-p2-linear-integration.md`. **Permalink histórico:** https://github.com/Brazwed/Kinera/blob/d6e29b62f1afaacdf35ba1d599cb97ed443b8a34/docs/opencl-rx550-p2-linear-integration.md. **Limites:** não equivale a validação de modelo completo.

## KIN-DOC-0023 — Gate P2 — Linear.step OpenCL na RX 550

- **Caminho:** [opencl-rx550-p2-linear-step.md](backends/opencl/opencl-rx550-p2-linear-step.md)
- **Tipo:** `EXPERIMENT`.
- **Situação documental:** não identificado (legado).
- **Resultado/parecer:** `PASS` delimitado; resultados e limites numéricos no relatório.
- **Relações:** P2 integration; links internos; PRs #52/#53.
- **Proveniência:** baseline `5ab05239…` associado à série P2.
- **Caminho anterior:** `docs/opencl-rx550-p2-linear-step.md` → caminho atual `docs/backends/opencl/opencl-rx550-p2-linear-step.md`. **Permalink histórico:** https://github.com/Brazwed/Kinera/blob/35de98db02ce665d603da09974701f23aa210786/docs/opencl-rx550-p2-linear-step.md. **Limites:** manter a ligação à integração ao mover.

## KIN-DOC-0024 — Kinera Issue #52 — P3 CPU vs. OpenCL benchmark

- **Caminho:** [opencl-rx550-p3-benchmark.md](backends/opencl/opencl-rx550-p3-benchmark.md)
- **Tipo:** `EXPERIMENT`.
- **Situação documental:** não identificado (legado); resultado histórico imutável.
- **Resultado/parecer:** benchmark executado em protocolo e shapes delimitados; parecer agregado está no assessment P3.
- **Relações:** Issues #13/#52/#53; PR #53; relacionado a P3b, P3c, P3e e assessment.
- **Proveniência:** baseline `781f0238…`; SHA-256 do harness e resultados no relatório.
- **Consumidores/artefatos:** `tests/test_opencl_host_conversion_analysis.py` fixa fingerprint/caminho; JSON de resultado P3 em `experiments/results/`.
- **Caminho anterior:** `docs/opencl-rx550-p3-benchmark.md` → caminho atual `docs/backends/opencl/opencl-rx550-p3-benchmark.md`. **Permalink histórico:** https://github.com/Brazwed/Kinera/blob/f4435a80de34117bf84c38ee851190b5e95b9141/docs/opencl-rx550-p3-benchmark.md. **Limites:** não comparar com CPU vetorizada/native se não registrado.

## KIN-DOC-0025 — Kinera Issue #52 — P3b auditoria da decomposição de custos

- **Caminho:** [opencl-rx550-p3-cost-analysis.md](backends/opencl/opencl-rx550-p3-cost-analysis.md)
- **Tipo:** `AUDIT` / análise experimental.
- **Situação documental:** não identificado (legado); parcialmente corrigido por documento relacionado, não substituído integralmente.
- **Resultado/parecer:** números P3/P3b preservados; a decomposição histórica de subfases com timers inclusivos não deve ser usada como soma aditiva.
- **Relações:** Issues #52/#53; [correção P3b](backends/opencl/opencl-rx550-p3b-cost-accounting-correction.md); PR #53.
- **Proveniência:** base `781f0238…`; hashes de harness/relatório no texto.
- **Consumidores/artefatos:** `tests/test_opencl_host_conversion_analysis.py` fixa fingerprint/caminho; resultados JSON sob `experiments/results/`.
- **Caminho anterior:** `docs/opencl-rx550-p3-cost-analysis.md` → caminho atual `docs/backends/opencl/opencl-rx550-p3-cost-analysis.md`. **Permalink histórico:** https://github.com/Brazwed/Kinera/blob/f4435a80de34117bf84c38ee851190b5e95b9141/docs/opencl-rx550-p3-cost-analysis.md. **Limites:** preservar junto com a correção.

## KIN-DOC-0026 — Kinera Issue #52 — Avaliação científica final P3

- **Caminho:** [opencl-rx550-p3-final-assessment.md](backends/opencl/opencl-rx550-p3-final-assessment.md)
- **Tipo:** `DECISION` / `AUDIT`.
- **Situação documental:** não identificado (legado); decisão histórica.
- **Resultado/parecer:** recomendação B: manter OpenCL opcional e restrito às operações/ambiente verificados; sem alegação de suporte geral ou speedup universal.
- **Relações:** Issues #13/#52/#54/#56–#59; PRs #54/#57/#58 e publicação #60.
- **Proveniência:** revisão `main` no commit `9a79fef1…`; assessment usa P3/P3b/P3c/P3e sem repetir medições.
- **Caminho anterior:** `docs/opencl-rx550-p3-final-assessment.md` → caminho atual `docs/backends/opencl/opencl-rx550-p3-final-assessment.md`. **Permalink histórico:** https://github.com/Brazwed/Kinera/blob/8c6956f37a2a66c27c2c316e4a7dba445af2a206/docs/opencl-rx550-p3-final-assessment.md. **Limites:** coortes e ambientes distintos não foram agregados.

## KIN-DOC-0027 — Kinera Issue #52 — P3b correção da contabilidade do profiler

- **Caminho:** [opencl-rx550-p3b-cost-accounting-correction.md](backends/opencl/opencl-rx550-p3b-cost-accounting-correction.md)
- **Tipo:** `AUDIT` / correção documental.
- **Situação documental:** não identificado (legado); correção relacionada, documento original preservado.
- **Resultado/parecer:** corrige a contabilidade de timers exclusivos/inclusivos; resultados P3 e medições totais preservados.
- **Relações:** Issue #52; [P3 cost analysis](backends/opencl/opencl-rx550-p3-cost-analysis.md); testes de integridade P3c.
- **Proveniência:** base P3 `781f0238…`; SHA-256 de profiler, resultados e harness declarados no relatório.
- **Caminho anterior:** `docs/opencl-rx550-p3b-cost-accounting-correction.md` → caminho atual `docs/backends/opencl/opencl-rx550-p3b-cost-accounting-correction.md`. **Permalink histórico:** https://github.com/Brazwed/Kinera/blob/f4435a80de34117bf84c38ee851190b5e95b9141/docs/opencl-rx550-p3b-cost-accounting-correction.md. **Limites:** correção não sobrescreve o registro histórico original.

## KIN-DOC-0028 — P3c — Custos de conversão Python ↔ OpenCL

- **Caminho:** [opencl-rx550-p3c-host-conversion-analysis.md](backends/opencl/opencl-rx550-p3c-host-conversion-analysis.md)
- **Tipo:** `EXPERIMENT` / análise diagnóstica.
- **Situação documental:** não identificado (legado).
- **Resultado/parecer:** identifica custos host/transferência para caminhos e shapes observados; probe diagnóstico não equivale a fase de produção.
- **Relações:** Issue/PR #54; série P3 e fingerprints de backend posteriores.
- **Proveniência:** base sincronizada `43474579…`; artefato `experiments/results/opencl-rx550-p3c-host-conversions.json`.
- **Consumidores:** `tests/test_opencl_host_conversion_analysis.py` referencia caminho, fingerprints e JSON.
- **Caminho anterior:** `docs/opencl-rx550-p3c-host-conversion-analysis.md` → caminho atual `docs/backends/opencl/opencl-rx550-p3c-host-conversion-analysis.md`. **Permalink histórico:** https://github.com/Brazwed/Kinera/blob/fa8aeb606f5db0ba06ec6c0d21c0d7aabcc318c9/docs/opencl-rx550-p3c-host-conversion-analysis.md. **Limites:** medições específicas de máquina/protocolo.

## KIN-DOC-0029 — P3e — OpenCL device profiling on the RX 550

- **Caminho:** [opencl-rx550-p3e-device-profiling.md](backends/opencl/opencl-rx550-p3e-device-profiling.md)
- **Tipo:** `EXPERIMENT`.
- **Situação documental:** não identificado (legado).
- **Resultado/parecer:** timestamps de profiling válidos para comandos medidos; não constitui comparação universal de desempenho CPU/OpenCL.
- **Relações:** Issue #52; P3/P3b/P3c e assessment final.
- **Proveniência:** baseline `1759bd19…`; fingerprint SHA-256 do código/harness declarado.
- **Consumidores/artefatos:** `tests/test_opencl_p3e_device_profiling.py` fixa path e integridade; JSON de profiling em `experiments/results/`.
- **Caminho anterior:** `docs/opencl-rx550-p3e-device-profiling.md` → caminho atual `docs/backends/opencl/opencl-rx550-p3e-device-profiling.md`. **Permalink histórico:** https://github.com/Brazwed/Kinera/blob/753f7366970f7dfada0eda89f3fcea065223020c/docs/opencl-rx550-p3e-device-profiling.md. **Limites:** runtime, driver, shape e condições delimitados.

## KIN-DOC-0030 — Preflight OpenCL — AMD Radeon RX 550

- **Caminho:** [opencl-rx550-preflight.md](backends/opencl/opencl-rx550-preflight.md)
- **Tipo:** `DECISION` / `AUDIT`.
- **Situação documental:** não identificado (legado).
- **Resultado/parecer:** `GO` para scaffold P1 e gate inicial `Linear.forward`, com ressalva explícita do runtime.
- **Relações:** série P1/P2; links para [cuda-backend.md](backends/cuda/cuda-backend.md) e P2 integration; Issues/PRs #52/#53 citam o documento.
- **Proveniência:** ambiente observado em 2026-11-05; SHA de baseline não identificado na triagem.
- **Caminho anterior:** `docs/opencl-rx550-preflight.md` → caminho atual `docs/backends/opencl/opencl-rx550-preflight.md`. **Permalink histórico:** https://github.com/Brazwed/Kinera/blob/35de98db02ce665d603da09974701f23aa210786/docs/opencl-rx550-preflight.md. **Limites:** smoke não demonstra treino ou equivalência completa.

## KIN-DOC-0031 — Invariantes da Fase 7

- **Caminho:** [phase7-invariants.md](architecture/phase7-invariants.md)
- **Tipo:** `SPEC` / referência histórica.
- **Situação documental:** não identificado (legado); não classificar como arquivado sem decisão.
- **Resultado/parecer:** não aplicável como resultado experimental; define invariantes de causalidade, shapes, losses e runners.
- **Relações:** README raiz e JSONs `experiments/phase8_0_refactor_baseline.json` e `experiments/phase9_0_cuda_baseline.json` referenciam este caminho.
- **Proveniência:** baseline de origem não identificado no inventário.
- **Caminho anterior:** `docs/phase7-invariants.md` → caminho atual `docs/architecture/phase7-invariants.md`. **Permalink histórico:** https://github.com/Brazwed/Kinera/blob/f04343756b1dd07fd5b76a4feff43b37128c0223/docs/phase7-invariants.md. **Limites:** manifests preservam o caminho e hash da revisão histórica.

## KIN-DOC-0032 — Gate R0 — auditoria do núcleo CPU

- **Caminho:** [recovery/r0-cpu-core-audit.md](recovery/r0-cpu-core-audit.md)
- **Tipo:** `AUDIT`.
- **Situação documental:** não identificado (legado); auditoria encerrada como registro.
- **Resultado/parecer:** `PARTIAL — CLOSED WITH LIMITATIONS`; não representa validação integral do núcleo.
- **Relações:** Issue #86; PR #87; baseline R0 `74f06ff498de65efacf2b803d526b37acf7b755f`.
- **Proveniência:** commit de integração #87 `734d202…`; relatório preserva baseline auditado distinto.
- **Caminho anterior:** `docs/r0-cpu-core-audit.md` → caminho atual `docs/recovery/r0-cpu-core-audit.md`. **Limites:** checkpoint/resume híbrido e matriz Python permanecem lacunas conforme relatório.

## KIN-DOC-0033 — Documentação da Kinera

- **Caminho:** [README.md](README.md)
- **Tipo:** `GUIDE`.
- **Situação documental:** `ACTIVE`.
- **Resultado/parecer:** não aplicável.
- **Relações:** índice de leitura e convenções; aponta para o catálogo e documentos temáticos.
- **Proveniência:** criado nesta etapa sobre baseline `734d202ce8e1bbcdc01ffcea20ec841e23c1945c`.
- **Caminhos anteriores:** nenhum.
- **Limites:** a estrutura temática é implementada em lotes; `backends/cuda/` e `archive/` mostradas no README continuam planejadas.

## KIN-DOC-0034 — Catálogo de documentos

- **Caminho:** [CATALOG.md](CATALOG.md)
- **Tipo:** `REFERENCE` / registro mestre.
- **Situação documental:** `ACTIVE`.
- **Resultado/parecer:** não aplicável.
- **Relações:** contém registros KIN-DOC-0001–0035 e remete a [README.md](README.md) para navegação/convenções.
- **Proveniência:** criado nesta etapa sobre baseline `734d202ce8e1bbcdc01ffcea20ec841e23c1945c`; os dados legados derivam do inventário D0.1.
- **Caminhos anteriores:** nenhum.
- **Limites:** IDs 0001–0032 são propostas D0.1 agora fixadas; metadados não encontrados permanecem explicitamente não identificados.


## KIN-DOC-0035 — Gate R2 — checkpoint híbrido CPU e retomada por época

- **Caminho:** [r2-hybrid-checkpoint-and-resume.md](recovery/r2-hybrid-checkpoint-and-resume.md)
- **Tipo:** `SPEC` / `GUIDE` / `AUDIT`.
- **Situação documental:** `ACTIVE`; contrato de persistência e guia de reprodução do Gate R2.
- **Resultado/parecer:** `PASS` técnico do checkpoint e da retomada determinística na **CPU**, restritos a fronteira de época e aos dois treinadores híbridos canônicos; decisão administrativa da R2.5 registrada na Issue #92 após integração documental.
- **Relações:** [Issue #92](https://github.com/Brazwed/Kinera/issues/92); PRs [#93](https://github.com/Brazwed/Kinera/pull/93), [#94](https://github.com/Brazwed/Kinera/pull/94) e [#95](https://github.com/Brazwed/Kinera/pull/95); [R0](recovery/r0-cpu-core-audit.md) / Issue #86 e R1 / Issue #90.
- **Proveniência:** base R2.5 `d400f8b2b15d0c5bf1b69c699cb1c69a38cadf14` (merge da PR #95); [CI pós-merge #44538301297](https://github.com/Brazwed/Kinera/actions/runs/44538301297), Python 3.10/3.14 `822 passed, 62 skipped, 5 deselected, 1159 subtests passed` em cada ambiente. Evidências intermediárias no próprio relatório e na Issue.
- **Caminhos anteriores:** nenhum; documento novo em `docs/recovery/`.
- **Limites:** sem checkpoint GPU, meio-época, CLI automática, integração com replay ou validação de hardware; `sha256` não garante autenticidade; registro histórico R0 `PARTIAL` preservado.

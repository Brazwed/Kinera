# Documentação da Kinera

Esta pasta contém referências de arquitetura, contratos, auditorias e evidências experimentais. Use [CATALOG.md](CATALOG.md) para o registro mestre: cada documento tem um ID permanente, caminho, situação documental separada do resultado técnico e proveniência disponível.

## Por onde começar

- **Arquitetura:** leia [architecture/architecture.md](architecture/architecture.md) para a organização do pacote e os limites entre linguagem, modelo, backends e treino. [phase7-invariants.md](architecture/phase7-invariants.md) registra invariantes históricos de causalidade, representação e execução.
- **Núcleo CPU:** consulte [architecture/architecture.md](architecture/architecture.md) para o contrato de referência e [recovery/r0-cpu-core-audit.md](recovery/r0-cpu-core-audit.md) para a auditoria R0, cujo parecer é `PARTIAL` e cujo baseline é `74f06ff498de65efacf2b803d526b37acf7b755f`.
- **Recuperação:** leia [o relatório R0](recovery/r0-cpu-core-audit.md) e acompanhe a issue [#86](https://github.com/Brazwed/Kinera/issues/86). R0 foi encerrado administrativamente com limitações; isso não inicia nem aprova R1.
- **Métricas:** comece por [metrics-issue-26-b008-final-acceptance.md](metrics/metrics-issue-26-b008-final-acceptance.md) para a visão de aceite e siga para os relatórios B002–B010 e M2.1 conforme o tema. B002–B010 e M2.1 estão em [`metrics/`](metrics/). Eles preservam decisões e lacunas das Issues #25–#27.
- **CUDA:** use [cuda-backend.md](backends/cuda/cuda-backend.md) como referência de runtime/ABI e [architecture/architecture.md](architecture/architecture.md) para os limites arquiteturais. Evidência de hardware continua específica ao ambiente e à revisão declarados em cada relatório.
- **OpenCL:** comece pela [auditoria funcional F2.0](backends/opencl/opencl-f2-capability-audit.md) e pelo [preflight da RX 550](backends/opencl/opencl-rx550-preflight.md). A trilha P1/P2 cobre [Linear.forward](backends/opencl/opencl-rx550-p1-linear-forward.md), [backward](backends/opencl/opencl-rx550-p2-linear-backward.md), [step](backends/opencl/opencl-rx550-p2-linear-step.md) e [integração](backends/opencl/opencl-rx550-p2-linear-integration.md). F1 documenta [Embedding.forward](backends/opencl/opencl-rx550-f1a-embedding-forward.md), [backward](backends/opencl/opencl-rx550-f1b-embedding-backward.md), [step](backends/opencl/opencl-rx550-f1c-embedding-step.md) e [integração](backends/opencl/opencl-rx550-f1d-embedding-integration.md); F2 registra [HybridRepresentation.forward](backends/opencl/opencl-rx550-f2-1-hybrid-forward.md), [mean pooling](backends/opencl/opencl-rx550-f2-2-mean-pool-forward.md) e a [investigação FP64](backends/opencl/opencl-rx550-fp64-division-investigation.md). A trilha P3 segue [benchmark](backends/opencl/opencl-rx550-p3-benchmark.md) → [análise de custos](backends/opencl/opencl-rx550-p3-cost-analysis.md) → [correção P3b](backends/opencl/opencl-rx550-p3b-cost-accounting-correction.md) → [conversão host P3c](backends/opencl/opencl-rx550-p3c-host-conversion-analysis.md) → [profiling P3e](backends/opencl/opencl-rx550-p3e-device-profiling.md) → [avaliação final](backends/opencl/opencl-rx550-p3-final-assessment.md). Consulte o catálogo para proveniência e pareceres.
- **Histórico e experimentos:** relatórios datados, resultados `NO-GO`, correções e medições são registros históricos. Leia o parecer, baseline, ambiente e limitações no documento original; resultado experimental não define a situação documental.

## Referência, contrato, auditoria e experimento

- **Referência/guia** orienta navegação ou descreve arquitetura e uso.
- **Contrato/especificação** registra invariantes e interfaces que consumidores devem respeitar.
- **Auditoria/decisão** apresenta inspeção, critérios, lacunas e um parecer para um escopo delimitado.
- **Experimento** registra protocolo, configuração, ambiente, resultados e limites de uma execução específica.

Um documento pode ter mais de um aspecto; o catálogo registra um tipo principal sem apagar sua natureza secundária. Resultado (`PASS`, `FAIL`, `PARTIAL`, `NO-GO` etc.) é independente da situação documental (`DRAFT`, `ACTIVE`, `SUPERSEDED`, `ARCHIVED`). Um relatório antigo ou negativo não vira `ARCHIVED` só por idade ou resultado.

Ao cadastrar, escolha o tipo que melhor descreve o uso principal e registre aspectos secundários nas observações do catálogo. Use `DRAFT` enquanto o documento ainda aguarda revisão; `ACTIVE` quando for mantido como orientação ou registro consultável; `SUPERSEDED` somente quando houver substituto identificado; e `ARCHIVED` somente após decisão explícita de retirar o documento da manutenção corrente. A situação não atesta que cada conclusão histórica continua válida para qualquer revisão: baseline, escopo e limitações delimitam essa validade. Um relatório pode permanecer acessível como registro sem ser uma especificação normativa.

## Estrutura temática

A estrutura temática está sendo implementada em lotes. Os Lotes 1–3 e C1–C3 criaram `architecture/`, `metrics/`, `backends/cuda/`, `backends/opencl/` e `recovery/`. A raiz contém apenas este README e o catálogo; `archive/` permanece planejada e ainda não foi criada.

```text
docs/
├── README.md
├── CATALOG.md
├── architecture/
│   ├── architecture.md
│   └── phase7-invariants.md
├── metrics/
│   └── (documentação B002–B010 e M2.1)
├── backends/
│   ├── cuda/
│   │   └── cuda-backend.md
│   └── opencl/
│       ├── opencl-f2-capability-audit.md
│       ├── opencl-rx550-preflight.md
│       ├── opencl-rx550-p3-benchmark.md
│       ├── opencl-rx550-p3-cost-analysis.md
│       ├── opencl-rx550-p3b-cost-accounting-correction.md
│       ├── opencl-rx550-p3c-host-conversion-analysis.md
│       ├── opencl-rx550-p3e-device-profiling.md
│       ├── opencl-rx550-p3-final-assessment.md
│       └── (relatórios P1/P2, F1 e F2; ver CATALOG.md)
├── recovery/
│   └── r0-cpu-core-audit.md
└── archive/ (planejada)
```

`archive/` é reservado a documentos cuja situação documental tenha sido decidida como arquivada. Não é uma pasta para resultados experimentais negativos.

## Convenções de manutenção

1. **Organize por assunto**, não por resultado ou idade. Mantenha o nome do arquivo descritivo, estável e em minúsculas, usando hífens para separar termos; não renomeie um relatório só para fazê-lo parecer atual.
2. **Classifique e identifique cada documento.** Escolha um tipo principal e uma situação usando as definições acima; registre resultado literal quando aplicável e `não identificado (legado)` se a fonte não permitir determinar um campo. Atribua um ID permanente `KIN-DOC-NNNN` e registre-o uma vez em `CATALOG.md`; não o recicle nem o altere ao mover um arquivo. O ID não substitui caminho ou permalink.
3. **Inclua cada documento novo no catálogo no mesmo PR.** Antes de escolher o próximo ID, confira a `main` atualizada e os PRs concorrentes. Revalide unicidade antes de integrar. Se dois PRs ainda não integrados reivindicarem o mesmo ID, o PR integrado primeiro o mantém; o outro atualiza sua entrada para o próximo ID livre antes de integrar. IDs já publicados nunca são renumerados ou reutilizados.
4. **Atualize documentos existentes mantendo o ID.** Edições comuns não criam nova identidade; atualize no mesmo PR os metadados do catálogo quando título, tipo, situação, proveniência ou relações mudarem. Registre a alteração de situação e seu motivo no catálogo/PR. Se houver substituição substantiva, use novo registro/ID para a nova identidade e relacione os dois documentos; não reescreva a evidência original.
5. **Atualize o catálogo junto com mudanças documentais** que criem, movam, renomeiem, substituam ou arquivem documentos. Preserve `caminho anterior → caminho atual` sob o mesmo ID quando houver movimentação.
6. **Declare lacunas sem inferir.** Use `não identificado (legado)` para metadados ausentes; use `não aplicável` para campos que realmente não se aplicam. Registre baseline, issue/PR, artefatos, comandos, seed, ambiente e limitações apenas quando a fonte os informar.
7. **Preserve evidências congeladas.** Não sobrescreva resultados, seeds, configurações, hashes, pareceres ou conclusões de relatórios publicados. Para errata substantiva, crie registro/revisão identificável, explique a correção, relacione-a ao original e mantenha o documento anterior acessível.
8. **Registre substituições explicitamente.** Só use `SUPERSEDED` e `supersedes`/`superseded-by` quando uma decisão ou documento afirmar a substituição; uma correção parcial não apaga o original nem o torna integralmente obsoleto.
9. **Revise o impacto documental de mudanças de código.** Ao alterar interfaces, arquitetura, backend, métricas, telemetria ou protocolo, identifique documentos afetados e atualize-os e o catálogo no mesmo PR, ou registre uma tarefa vinculada quando a atualização depender de evidência futura.
10. **Trate legado e links com cuidado.** Preserve APIs, nomes e caminhos usados por README, código, testes, scripts, workflows, Issues, PRs ou artefatos congelados até que consumidores e links tenham destino verificado. Antes de mover, atualize links editáveis, use permalinks para evidência congelada quando disponíveis e registre links externos não preserváveis; se um caminho externo não puder ser atualizado, mantenha o arquivo ou uma ponte aprovada. IDs não redirecionam URLs; não conte com redirecionamento automático do GitHub.

O catálogo é a fonte mestre dos metadados e relações. Este README oferece orientação de leitura e regras, sem duplicar a lista de proveniência.

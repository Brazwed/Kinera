# S0.1 — Inventário, arquitetura e responsabilidades

## Estado, escopo e proveniência

**Estado:** relatório inicial para revisão. A inspeção foi somente leitura; nenhum código, dataset, teste ou artefato experimental foi alterado ou executado.

**Baseline:** `main` em `569a1ca0b41ab25a2f2c8513229c3d8b64bf49c9` (merge pós-R2, PR #98), confirmado em 2026-11-19. Worktree `Worktrees/S0-1`, branch local `audit/s0-1`. Ambiente observado: Python 3.15.2. O número de versão registra o ambiente desta inspeção, não uma matriz de compatibilidade.

**Comandos de inventário:** `git rev-parse HEAD`; `git status --short`; `git ls-files`; `rg --files`; consultas `rg -n` a símbolos/entrypoints; leituras dos arquivos referenciados abaixo; `python --version`. Não foram executados `pytest`, `unittest`, CLIs de treino/preparação, benchmarks, código CUDA/OpenCL ou experimentos.

**Referências de coordenação:** Issue [#100](https://github.com/Brazwed/Kinera/issues/100) (S0), D0 [#88](https://github.com/Brazwed/Kinera/issues/88), R2 [#92](https://github.com/Brazwed/Kinera/issues/92), R3 [#99](https://github.com/Brazwed/Kinera/issues/99) e pós-R2 [#97](https://github.com/Brazwed/Kinera/issues/97)/[PR #98](https://github.com/Brazwed/Kinera/pull/98). A Issue #99 já define uma R3.1 de auditoria do pipeline. Este relatório mapeia as fronteiras arquiteturais e sinaliza a sobreposição; não substitui nem duplica a auditoria detalhada de requisitos R3.1.

## Resumo do repositório

O código vigente é um pacote Python sob `src/kinera`, organizado em linguagem/dados, rede neural, backends, treinamento, experimentos, diagnósticos e monitoramento. A execução padrão do modelo seleciona CPU; existem implementações separadas de CPU, CUDA e OpenCL. O modelo híbrido combina embeddings de caracteres e sílabas, features ortográficas e três heads (`char`, `syllable`, `boundary`). Os entrypoints não oferecem todos os mesmos modos: `scripts/train.py` treina o caminho `char`; APIs de treino multitarefa e checkpoint/resume por época existem em módulos separados.

O repositório contém um grande histórico de fixtures, resultados e relatórios experimentais. Em contraste, entre os arquivos versionados em `data/`, só está presente `data/manifests/sources.json`; os caminhos de dados brutos descritos no manifesto não são arquivos rastreados nesta revisão. Portanto, este inventário confirma contratos e fluxos no código, mas não verifica o conteúdo nem a qualidade de um corpus disponível localmente.

## Estrutura observada

| Área | Conteúdo e papel observado | Classificação inicial |
|---|---|---|
| `src/kinera/language/` | Registros, normalização, vocabulários, ortografia/features, alinhamento, tokenização silábica e datasets/adaptação | Núcleo de preparação e representação de dados |
| `src/kinera/nn/` | Layers, embeddings, representação híbrida, heads, losses e modelo autoregressivo | Núcleo do modelo; consome abstração de backend |
| `src/kinera/backends/` | Contrato comum, CPU, CUDA e OpenCL, buffers e kernels | Execução numérica; default CPU; evidência física é específica a cada gate/ambiente |
| `src/kinera/training/` | Treinadores `Neuron`/híbridos, avaliação, configuração, checkpoints, retomada e métricas auxiliares | APIs de treino/eval; capacidades distintas por tipo de modelo e runner |
| `src/kinera/experiments/` | Comparação controlada, variantes, multi-seed, replay, fingerprints, validação científica e estado experimental | Experimentação; não equivale automaticamente ao fluxo oficial da CLI |
| `src/kinera/monitor/` | Métricas e telemetria passiva | Subsistemas transversais com contratos e consumidores próprios (#26/#27) |
| `src/kinera/diagnostics/` | Gradientes, sílabas e ortografia | Ferramentas de análise |
| `src/kinera/scripts/` | Implementações Python de importação, anotação, construção de vocabulário/dataset e benchmarks | Ferramentas de dados e benchmarks |
| `scripts/` | Wrappers de treino/comparação/preparação e runners diretos; `scripts/legacy/` guarda compatibilidade e experiências históricas | Entry points públicos e superfície legada; consumidor precisa ser rastreado antes de remoção |
| `experiments/` | JSONs de validação/benchmarks, resultados, protocolos e relatórios versionados | Evidência histórica e protocolos; cada resultado depende de SHA/configuração/ambiente próprios |
| `docs/` | Arquitetura, contratos, recuperação, métricas e evidência CUDA/OpenCL; índice D0 em `README.md`/`CATALOG.md` | Documentação temática; vigência técnica não decorre apenas do catálogo |
| `tests/` | 104 arquivos `test_*.py`, incluindo grupos de CPU, contrato, dados, checkpoints, métricas, telemetria, CUDA/OpenCL e experimentos | Evidência de testes existente; este inventário não executou a suíte nem reclassificou seus resultados |
| `data/` | Manifesto de fontes | Dados de origem não verificados nesta inspeção |

`pyproject.toml` declara pacote `kinera`, Python `>=3.10`, sem dependências runtime, e extra `test` com pytest. A CI executa a suíte pytest em Python 3.10 e 3.14 com cinco seleções de testes de piloto explicitamente removidas; esta é configuração versionada, não resultado de execução nesta worktree.

## Fluxo real observado no código

```text
fonte externa / arquivo do usuário
  → importador e normalização de registros
  → anotação ortográfica/alinhamento
  → agrupamento por palavra + split lexical
  → vocabulário + exemplos/contextos JSONL
  → loader + adapter de IDs/features
  → AutoregressiveModel (backend CPU padrão)
  → treino char ou API multitarefa
  → avaliação, relatório JSON e/ou geração gulosa
  → checkpoint híbrido e retomada CPU por fronteira de época (API separada)
```

Detalhamento estático dos caminhos:

1. **Entrada e normalização:** `kinera.scripts.import_silabas_pt` converte CSV em JSONL através de `language.datasets.records.normalize_silabas_row`; há também importador opcional Portal da Língua. O manifesto `data/manifests/sources.json` registra fontes, caminhos, hashes/contagens declarados e restrições de licença. Os arquivos citados não estão versionados no HEAD auditado.
2. **Anotação e preparação:** `annotate_syllable_dataset` aplica anotação/alinhamento; `build_syllable_dataset` agrupa registros por palavra e pode separar grupos de treino/validação com seed; `build_syllable_vocabulary` cria vocabulário silábico; `build_character_dataset` verifica interseção de palavras entre splits e gera vocabulário de caracteres e exemplos autoregressivos. As funções são ferramentas separadas, não uma única preparação automática invocada pela CLI de treino.
3. **Representação dos exemplos:** `language.datasets.loader` lê/escreve objetos JSONL. `datasets.adapter.example_to_inputs` alinha IDs de caracteres, IDs silábicos e vetores de 10 features; exemplos sem contexto persistido de sílaba usam ID desconhecido e features derivadas do caractere quando aplicável. `datasets.builder.build_autoregressive_examples` constrói contexto e targets e, quando há vocabulário silábico, persiste contexto causal silábico/features.
4. **Modelo:** `nn.models.hybrid.AutoregressiveModel` compõe embeddings, `HybridRepresentation` e heads `char`, `syllable`, `boundary`. `backends.resolve_backend(None)` escolhe `CPUBackend`; CUDA/OpenCL exigem seleção/injeção específica. O contrato base permite capacidades opcionais e falha explicitamente em caminhos residentes sem operação requerida.
5. **Treino:** `scripts/train.py` carrega vocabulário(s) e JSONL preparados, constrói o modelo e chama `training.trainer.train_examples`, que atualiza pelo target de caractere em ordem fixa. `training.trainer.train_multitask_examples` treina as três heads e recebe dados/targets multitarefa, mas é uma API separada. `training.hybrid_resume.train_hybrid_epochs` oferece execução em épocas completas nos modos `char` e `multitask` com `CPUBackend`; não embaralha exemplos nem oferece retomada no meio da época.
6. **Avaliação e saída:** `scripts/train.py` pode carregar validação e chamar `evaluate_examples`; emite relatório JSON no stdout e texto gerado por `generate_text` (decodificação gulosa até `<END>` ou limite). `evaluate_multitask_examples` calcula métricas para as três heads sem `backward`/`step`. `scripts/compare.py` é entrypoint distinto para comparação controlada, exige treino/validação e vocabulários e usa CPU como default configurável.
7. **Persistência:** `training.hybrid_checkpoint` serializa o modelo autoregressivo e vocabulários; `training.hybrid_resume` vincula estado de treino a modo/configuração, fingerprint de exemplos, parâmetros e época completa. A funcionalidade é CPU e via API Python, não um fluxo save/resume já conectado em `scripts/train.py`. `training.checkpoint` permanece para modelos legados simples (`Neuron`/`Linear`); não confundir os dois contratos.

O fluxo acima é um mapa estático de chamadas e contratos. Não demonstra que a execução CLI completa esteja operacional no ambiente desta worktree, nem que o dataset real exista, nem que o modelo generalize.

## Matriz de responsabilidades, consumidores, evidência e risco

| Componente | Responsabilidade / consumidores identificados | Dependências principais | Testes com nomes relacionados (não executados) | Risco / decisão preliminar |
|---|---|---|---|---|
| Importação, registros, anotação | CSV/Portal → registros normalizados/anotados; consumidos por builders | `records`, ortografia/alinhamento, JSONL | `test_silabas_import.py`, `test_portal_import.py`, `test_annotation_pipeline.py`, `test_alignment.py` | Dados brutos ausentes do repo; manter, avaliar proveniência/licenças em S0.4 |
| Dataset, split, vocabulários e adapter | Grupos por palavra, split lexical, exemplos/contextos/targets e inputs do modelo | vocabulários, features, loader, builder | `test_dataset_builder.py`, `test_character_dataset.py`, `test_syllable_vocabulary_builder.py`, `test_training_data.py`, `test_hybrid_inputs.py`, `test_vocabulary.py` | Há caminhos de preparação distintos; verificar leakage/identidade em S0.4; não simplificar sem consumidores |
| Layers, representação, heads e losses | Modelo composto consome embeddings, representação híbrida e heads | contrato de backend, buffers, vocabulários/dimensões | `test_embedding.py`, `test_linear.py`, `test_hybrid.py`, `test_heads.py`, `test_multitask_loss.py`, `test_autoregressive_model.py` | Contrato geral e capacidades opcionais elevam complexidade; S0.2 deve quantificar usos e fronteiras |
| `AutoregressiveModel` | Forward/predict, treino char e passos multitarefa; consumido por trainer/evaluator/CLI/experimentos | layers, losses, backend, adapter | `test_autoregressive.py`, `test_autoregressive_workflow.py`, `test_multitask_training.py`, `test_hybrid_integration.py` | Head/treino efetivo varia por entrypoint; documentar sem afirmar capacidade de linguagem geral |
| Treinadores e configuração | `train_examples`, `train_multitask_examples`, `fit` de `Neuron`, observers/runner | modelo, adapter, métricas, evaluator | `test_training_loop.py`, `test_training_data.py`, `test_multitask_training.py`, `test_train_5min.py` | `fit`/shuffle de `Neuron` não é contrato automático do híbrido; R3.1 trata integração |
| Avaliação e geração | Métricas `char`/multitask, baselines e geração gulosa; CLI e runners consomem | modelo, vocabulário, métricas | `test_autoregressive_workflow.py`, `test_multitask_training.py`, `test_phase46_metrics.py`, `test_monitor_metrics_m1.py` | Métrica/validação não prova utilidade; S0.5 define protocolo separado |
| Checkpoints simples e híbridos | `checkpoint.py` para `Neuron`/`Linear`; `hybrid_checkpoint.py` + `hybrid_resume.py` para modelo híbrido CPU/épocas | modelo, vocabulários, fingerprint e estado | `test_checkpoint.py`, `test_hybrid_checkpoint.py`, `test_hybrid_checkpoint_integrity.py`, `test_hybrid_resume.py`, `test_checkpoint_runner.py` | Contratos coexistem e têm limites diferentes; preservar formato v1/identidades; R2 é PASS apenas no escopo declarado |
| Backends CPU/CUDA/OpenCL | Operações numéricas, buffers e kernels; camadas/modelo consomem o protocolo | runtimes nativos/driver nos caminhos acelerados | `test_backend_contract.py`, `test_cuda_*.py`, `test_opencl_*.py`, `test_neuron_ops_equivalence.py` | Presença de código/teste/mock não é evidência de hardware físico; preservar distinção de gates em S0.2 |
| Métricas e telemetria | Métricas científicas/relatórios vs. medição operacional passiva; trainers e experimentos consomem | records, aggregation, telemetry snapshots/timing | `test_metrics_m0_*.py`, `test_monitor_metrics_*.py`, `test_telemetry_*.py` | Separar nomes/unidades/consumidores em S0.3; sem remoção nesta auditoria |
| Experimentos, diagnostics e legacy | Variantes, replay, multi-seed, pilots, benchmarks e runners históricos | modelos, dados, métricas, checkpoints | `test_comparison.py`, `test_multi_seed*.py`, `test_learning_*.py`, `test_partial_experiment.py`, `test_expanded_replay_pilot.py`, `test_scientific_validation.py`, `test_syllable_analysis.py` | Grande volume histórico pode parecer API canônica; mapear referências antes de congelar/remover |
| Documentação, CI e pacote | `docs/` orienta contratos; `pyproject.toml` empacota; CI roda matriz | catálogo D0, workflow GitHub, Python | `test_package_layout.py`, CI `.github/workflows/ci.yml` | README da raiz está desatualizado (ver achados); docs/architecture é referência mais recente, mas ainda precisa revisão de vigência |

## Achados e lacunas identificados

| ID | Tipo | Descrição | Estado / próximo passo |
|---|---|---|---|
| S0-F002 | Fato verificado | O README da raiz descreve o primeiro protótipo (`from nn.neuron import Neuron`, diretórios `nn/`/`training/`, comandos `unittest`) e apresenta a Kinera como marco inicial; o pacote atual está em `src/kinera`, declara `kinera` no `pyproject.toml` e tem CLIs/imports diferentes. Há divergência documental entre README raiz e código/docs atuais. | Confirmado por leitura. Tratar em S0.6 ou Issue documental pequena; não corrigido aqui. |
| S0-F003 | Fato verificado | `git ls-files data` lista somente `data/manifests/sources.json`; não há CSV/JSONL bruto ou preparado versionado na revisão auditada. O manifesto registra `silabas-pt` com 181.560 linhas declaradas e licença desconhecida/restrição de redistribuição, além de export opcional Portal da Língua com licença a verificar. | Confirmado para arquivos rastreados. S0.4 deve verificar disponibilidade/proveniência local antes de afirmar volume/qualidade atual. |
| S0-F004 | Fato verificado | O `scripts/train.py` usa treino `char`, avaliação opcional e geração; treino multitarefa, comparação e retomada por época existem em APIs/entrypoints separados. | Confirmado por chamadas estáticas. Conciliar com R3.1/#99 para não repetir auditoria nem antecipar mudanças. |
| S0-F005 | Fato verificado | `src/kinera/training/checkpoint.py` e o par `hybrid_checkpoint.py`/`hybrid_resume.py` atendem modelos/contratos distintos. Relatório R0 descreve baseline anterior ao R2; sua limitação de checkpoint híbrido não representa o estado pós-R2. | Confirmado por código e sequência de Issues; R2/#92 e pós-R2/#97/#98 delimitam PASS CPU em fronteira de época. |
| S0-H003 | Hipótese | A separação entre `scripts/`, `src/kinera/scripts/`, `scripts/legacy/`, `experiments/` e APIs de `training/` pode gerar sobreposição ou ambiguidade sobre o caminho oficial. | Aberta. Requer consumidores, referências externas e custos em S0.1/S0.6; não concluir duplicação/removibilidade pela organização apenas. |
| S0-H004 | Hipótese | Os artefatos experimentais registrados podem conter evidência relevante de capacidade, mas sua cobertura não demonstra por si só aprendizagem/generalização no corpus atual. | Aberta. S0.4/S0.5 devem reconciliar protocolo, dados, identidade e métricas antes de qualquer conclusão. |

## Decisões preliminares

- **Manter** o modelo híbrido, layers, contratos CPU/CUDA/OpenCL e checkpoints como componentes existentes; S0.1 não encontrou base para remoção.
- **Preservar** `scripts/legacy/`, experimentos, resultados e aliases até rastrear consumidores, identidades e referências históricas.
- **Tratar** o README raiz como documentação provavelmente desatualizada, sem editar nesta etapa; revisar seu status e corrigir por mudança documental rastreada.
- **Usar** o resultado R2 pós-merge como contexto vigente do checkpoint híbrido CPU; não repetir a conclusão limitada de R0 como estado atual.
- **Conciliar** este mapa com a R3.1 antes de produzir uma segunda descrição detalhada de dataset/CLI/treinadores. R3.2–R3.5 continuam dependentes de seu próprio gate; esta auditoria não os autoriza.

## Gate de continuação S0.1

**Recomendação: GO** para iniciar S0.2 e S0.3, com a matriz acima como baseline e mantendo CUDA/OpenCL separados da evidência CPU. A revisão do mapa deve ser conciliada com R3.1/#99, que já investiga o pipeline oficial. A ausência de corpus rastreado impede concluir sobre dataset e capacidade, mas não bloqueia inventário de arquitetura/backends/observabilidade; essas perguntas seguem para S0.4/S0.5.

### Checklist S0.1

- [x] Confirmar HEAD de `main`, estado limpo da worktree e mapa de diretórios/documentos.
- [x] Traçar estaticamente entrada → representação → modelo → treino/avaliação → geração/checkpoint.
- [x] Criar matriz resumida de responsabilidades, consumidores, dependências, testes relacionados e riscos preliminares.
- [x] Registrar achados, hipóteses, lacunas e decisões sem presumir que código legado seja descartável.
- [x] Produzir relatório inicial e recomendação GO para os próximos diagnósticos; aguarda revisão antes de marcar a etapa como encerrada na Issue #100.

## Limites desta auditoria

Não houve execução de testes, CLIs, treino, avaliação, benchmark, runtime CUDA/OpenCL ou inspeção de corpora externos/ignorados. A presença de um arquivo de teste demonstra apenas que o teste existe; resultados históricos citados em documentação pertencem aos SHAs e ambientes declarados por suas fontes. Nenhum item deste documento atesta correção global, generalização linguística ou utilidade prática.

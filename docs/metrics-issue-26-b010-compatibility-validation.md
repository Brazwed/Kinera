# B010 — Compatibilidade Python e gates científicos CUDA

## Parecer e escopo

**Issue #26: BLOCKED.** O gate obrigatório da suíte em Python 3.10 falhou em três testes. As falhas foram localizadas em expectativas de teste e no uso de `math.fma` indisponível no interpretador; elas não demonstram, por si, defeito em produção. Também falta executar os gates científicos em CUDA físico na versão atual. A auditoria B008 permanece como registro histórico.

Código auditado: `78501f4d8c7b618d5860db4b602d9698d3fa21d6`, merge do PR #79, também base e merge-base iniciais da B010. O contêiner recebeu `git archive HEAD` desse commit; nenhuma alteração de produção foi aplicada. A mudança B010 da matriz CI é posterior e não afeta a árvore Python executada.

## Matriz de evidências

| Gate | Status | Evidência e limite |
| --- | --- | --- |
| Python 3.10 runtime | **FAIL** | Python 3.10.22 real; instalação e imports passaram; suíte CI: 3 falhas, 790 aprovados, 61 skips, 5 deselections, 792 subtestes. |
| Python 3.14 | **PASS** | CI do PR #81 na HEAD `f0aa2f6ba677952ad55dad710bbb965066fc94a9`: job `tests (3.14)` aprovado. Não houve execução local nesse runtime. |
| CPU-only | **PASS** | Instalação e imports no contêiner 3.10, com assert de que os módulos dos backends CUDA/OpenCL não foram carregados. O resultado não substitui a suíte completa, que falhou. |
| CUDA hardware | **BLOCKED_ENVIRONMENT** | `nvidia-smi` e `nvcc` ausentes no PATH e nenhum `/dev/nvidia*`; zero testes físicos executados. RX 550/OpenCL não é CUDA. |
| Equivalência numérica CPU/CUDA | **UNVERIFIED** | Sem execução física no SHA auditado. Evidência histórica da T4 é descrita abaixo. |
| D2H residente | **UNVERIFIED** | Testes e callsites auditados, sem execução física atual. |
| M2.3 CUDA | **UNVERIFIED** | A suíte CUDA existente não cobre `validation_every_n_epochs`; falta fixture específica e execução física. |
| B009 | **PASS CPU / UNVERIFIED CUDA** | Os testes CPU de métricas e comparison passaram na suíte 3.10, exceto a expectativa fixa de loss da cadência; não se demonstrou paridade em GPU. |
| Checkpoint/resume | **PASS CPU / UNVERIFIED CUDA** | Testes CPU da suíte 3.10 passaram; o gate CUDA histórico requer adaptação ao contrato atual antes de verificar resume físico. |

## Python 3.10: instalação, imports e testes

Imagem isolada `docker.io/library/python:3.10-slim`, digest `sha256:7b6bd969dda3ecfca0655524e338b57a8b8cd692a9362e3d55b5a3ef49b07256`, ID `0f61d91517f6602b9d99096696845f54ddfe01a52029b4f55fdddb277ebe034a`; Python **3.10.22**, pytest 9.2.0. O primeiro bind mount não pôde ser lido pelo contêiner; a execução válida usou `git archive HEAD | podman run --rm -i ...`, extração em `/tmp/kinera`, `python -m pip install -e '.[test]'` e a suíte completa com as cinco exclusões exatas de `.github/workflows/ci.yml`. Instalação editável e imports `kinera`, API multiclasses e comparison passaram. A imagem é isolada do sistema operacional; o pip executou como root somente dentro do contêiner.

Falhas observadas, reproduzíveis no SHA acima:

1. `tests/test_comparison.py::ControlledComparisonTests::test_validation_cadence_preserves_legacy_results_and_reuses_final`: a segunda loss obtida foi `2.931094460057472`; a fixture exige `2.931094460057473`, diferença de cerca de `8.9e-16`. As comparações entre a execução com e sem cadência, anteriores à constante fixa no teste, passaram.
2. `tests/test_diagnostic_metrics_m2.py::DiagnosticMetricsLegacyTests::test_orthographic_summary_legacy_values_and_empty_policy`: média obtida `0.20000000000000004`, esperada `0.19999999999999998`, diferença de cerca de `5.6e-17`. O comportamento de `sum` varia entre versões do Python; os valores devem ser reconciliados com a política de equivalência histórica antes de modificar a fixture.
3. `tests/test_opencl_scaffold.py::OpenCLEmbeddingStepTests::test_step_matches_cpu_without_fma_and_keeps_buffer_resident`: `AttributeError: module 'math' has no attribute 'fma'` em Python 3.10. É um teste com backend simulado; nenhum dispositivo OpenCL foi executado.

As duas diferenças numéricas não são prova de identidade bit a bit entre versões de Python. Nenhuma tolerância foi ampliada nesta tranche. A ação mínima é revisar as expectativas para ambos os runtimes com oráculos científicos explícitos e substituir ou condicionar a dependência de `math.fma` no teste; validar as mudanças com os responsáveis antes de aceitar o gate. Não houve alteração de versão mínima ou dependência de runtime do projeto. A CI passa a executar Python 3.10 e 3.14 com o mesmo comando e exclusões.

## CUDA: auditoria e reprodução

`tests/test_cuda_multitask_training.py` contém três gates opt-in: um step multitarefa, trainer/evaluator por duas épocas e runner A/C/D com resume. Usa `ATOL=RTOL=1e-12`. O teste de avaliação registra shapes/dtypes e callsites das transferências e espera apenas escalares; também audita buffers e cleanup. Outros `test_cuda_*.py` cobrem forward, backward, step, integração e embeddings. `docs/cuda-backend.md` registra para a T4, em trabalho anterior, erro máximo `1.3877787807814457e-17` no step e `2.2204460492503131e-16` no gate de trainer/evaluator, sob `1e-12`; essas medições são herdadas, não medições B010 nem evidência para M2.3/B009 no SHA atual.

Foram confirmadas duas incompatibilidades do próprio gate existente sem inicializar GPU:

- `_flat({'value': 0.0})` lança `TypeError`; o teste de trainer/evaluator itera campos atuais como `metric_results`, que são dicionários, e os passa a `_assert_close`/`_flat` (`tests/test_cuda_multitask_training.py:446,461`).
- O teste de runner passa `checkpoint_path` sem `syllable_vocabulary`; `run_controlled_experiments` rejeita essa configuração antes de iniciar execução (`comparison.py:905`). A chamada de preflight reproduziu `ValueError: checkpoint exige syllable_vocabulary para verificar sua identidade`.

Essas falhas de fixture impedem interpretar uma futura execução do gate completo sem atualização dos testes. A suite não contém fixture CUDA para cadência M2.3 nem verificações explícitas dos três baselines B009; essa cobertura permanece pendente. Não foi adicionada fixture CUDA não verificável neste ambiente. A menor ação é atualizar a fixture para comparar descritores aninhados e fornecer vocabulário real, acrescentar uma fixture pequena de cadência/telemetry e baselines usando as tolerâncias existentes, e executá-las em GPU autorizada.

Reprodução física após corrigir as fixtures, sem benchmarks:

```sh
git checkout 78501f4d8c7b618d5860db4b602d9698d3fa21d6
nvidia-smi
nvcc --version
python --version
KINERA_RUN_CUDA_TESTS=1 python -m pytest -q -s tests/test_cuda_multitask_training.py
```

Executar a fixture corrigida em commit identificado, registrar SHA, modelo, compute capability, driver, runtime/toolkit, Python, ABI (o bridge atual exige ABI 7), testes executados, erros máximos e tolerâncias; identificar qualquer diferença em relação à HEAD final. Comparar cadência ativa/desativada, parâmetros, updates, histórico e telemetry; observar apenas exportações escalares D2H e ausência de backward/step na avaliação. Os testes físicos são opt-in e não foram iniciados aqui.

## Limitações e responsabilidades

A CI atual cobre CPU em runners GitHub, não prova CUDA. A identidade das baselines depende de targets e vocabulários; não há execução GPU que comprove sua integração observacional em M2.3. O fingerprint, schemas e contratos legados não foram alterados pela B010. A Issue #27 e sua telemetria operacional mantêm critérios próprios; a #25 continua dependente dos gates das duas issues. Até o P10 passar e o G10 ser executado ou formalmente reconciliado, não recomendar encerramento da #26.

Na [CI do PR #81](https://github.com/Brazwed/Kinera/actions/runs/42281758946), HEAD `f0aa2f6ba677952ad55dad710bbb965066fc94a9`, o job Python 3.10.22 reproduziu as mesmas três falhas e a mesma contagem local; Python 3.14 passou. A atualização documental posterior não altera código nem workflow; o estado dos jobs dessa HEAD posterior deve ser consultado no PR.

## Continuação B010 — correções de testes e preparação CUDA

Esta continuação mantém o histórico acima como registro do estado inicial. A worktree começou limpa em `d9f9895e06a6884e294fa84805f2a87c09467b76`; a HEAD publicada no PR era `6f1a967dbbaf4d71b886a24c5590f59cfa86a83c`, com a mesma árvore de arquivos. A `main` atualizada estava em `7bff9617a340faab342f5c844464a94d0acec6d0` (PR #80). Seu merge foi feito sem reescrita; em seguida, a referência existente do PR #81 também foi mesclada para preservar o caminho de fast-forward da publicação. Os artefatos de investigação OpenCL da #78 presentes na `main` foram mantidos. Merge-base atual com `main`: `7bff9617a340faab342f5c844464a94d0acec6d0`.

### P10 — compatibilidade dos testes

As alterações restringem-se a testes. A fórmula de produção e a ordem de `sum(values) / count` permanecem inalteradas.

1. `tests/test_comparison.py`: as losses históricas nos índices 0 e 2 continuam com igualdade exata. A loss histórica do índice 1 mantém o esperado `2.931094460057473`, com tolerância absoluta de `1e-15` somente nessa comparação. O teste ainda exige igualdade exata entre o relatório e o histórico das execuções com cadência ativada e desativada no mesmo runtime.
2. `tests/test_diagnostic_metrics_m2.py`: todos os campos discretos e as outras médias continuam com igualdade exata. Apenas `mean_base_prob_correct`, esperado historicamente como `0.19999999999999998`, usa tolerância absoluta `1e-16` para acomodar a diferença de um ULP observada em Python 3.10 (`0.20000000000000004`).
3. `tests/test_opencl_scaffold.py`: o oráculo `math.fma` foi substituído por aritmética `Decimal` com precisão 200 e `Decimal.from_float`, preservando os operandos binários. O teste verifica tanto o resultado fundido (`2**-54`) quanto o resultado separado (`0.0`) e exige que sejam diferentes. O kernel OpenCL e o backend de produção não foram modificados.

Em Python 3.15, os testes direcionados `test_comparison.py`, `test_diagnostic_metrics_m2.py`, `test_opencl_scaffold.py` e `test_cuda_multitask_training.py` passaram: **121 passed, 4 skipped, 292 subtests passed**, com um warning de depreciação do plugin `anyio`. Os quatro skips são os métodos CUDA opt-in; não houve tentativa de inicializar GPU. Este ambiente não oferece interpretadores Python 3.10 ou 3.14. Os resultados P10 definitivos nessas versões dependem dos jobs da matriz do PR #81.

### G10 — preparação das fixtures

O comparador de valores numéricos nos testes CUDA agora percorre dicionários e listas recursivamente. Floats finitos usam somente `ATOL=RTOL=1e-12`; inteiros, contagens, IDs, strings, estados de disponibilidade, motivos, versões e `None` são verificados exatamente. Chaves ausentes não são convertidas em zeros. Um teste auxiliar CPU valida a comparação dos descritores aninhados, inclusive rejeitando mudanças de availability, suporte ou valor disponível convertido para `None`.

A fixture de checkpoint/resume fornece agora um `SyllableVocabulary` real com os 3.932 IDs declarados pelo runner. Os asserts de incompatibilidade CPU↔CUDA, reutilização CUDA→CUDA, identidade, contagem/fechamento de modelos e liberação de alocações permanecem.

Foi adicionada uma fixture CUDA opt-in pequena para comparar cadência desativada e cadência 1, cobrindo parâmetros finais, ordem e quantidade dos updates, histórico de treino, reutilização da validação final, baselines multiclasses B009, exclusões de UNK/END no baseline conhecido, contagens de forwards e backward, saídas D2H escalares e liberação de buffers. A fixture usa as tolerâncias já aprovadas (`1e-12`). Ela não foi executada em dispositivo; portanto, sua validade física permanece **UNVERIFIED**. Os testes CUDA continuam opt-in e não são mascarados por skips adicionais.

### Estado atualizado dos gates

| Gate | Estado nesta continuação | Evidência/limite |
| --- | --- | --- |
| P10 em Python 3.10 | **PENDENTE DE CI** | As três expectativas e o oráculo incompatível foram corrigidos em testes; runtime 3.10 indisponível localmente. A aprovação requer job verde da nova HEAD do PR. |
| P10 em Python 3.14 | **PENDENTE DE CI** | Mesmo comando e cinco exclusões mantidos; aprovação requer job verde da nova HEAD do PR. |
| Suíte CPU local | **PASS PARCIAL** | Python 3.15: suíte configurada com cinco exclusões: 798 passed, 62 skipped, 5 deselected, 1 warning e 1.104 subtests, em 41,69 s. Quatro módulos direcionados: 121 passed, 4 skips CUDA e 292 subtests. Os skips não são evidência P10 para 3.10/3.14 nem evidência CUDA. |
| G10 preparação estática | **PASS** | Helpers e fixture M2.3/B009 revisados; nenhum código de produção alterado. |
| G10 CUDA físico | **UNVERIFIED** | Nenhum teste com `KINERA_RUN_CUDA_TESTS=1`; a RX 550/OpenCL não foi usada como substituta de CUDA. |
| Aceite da Issue #26 | **PARTIALLY_COMPLETE** | Não mudar o parecer até ambos jobs P10 passarem. Mesmo com P10 aprovado, o gate CUDA físico continua sujeito à evidência ou decisão formal de aceite; nenhuma conclusão de equivalência CUDA é inferida desta preparação. |

O diff desta continuação não altera `src/`, kernels, schemas, fingerprints, losses, treinamento ou protocolos. A B010 permanece em revisão e a Issue #26 permanece aberta.

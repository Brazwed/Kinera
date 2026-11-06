# Kinera OpenCL F1d — Embedding integration na RX 550

## Escopo e proveniência

Este relatório cobre a integração da classe `Embedding` no ciclo
`forward → backward → step`, usando `CPUBackend` como referência. O gate não
valida `AutoregressiveModel`, representação, losses, trainer, evaluator ou
runners; não houve benchmark nem treinamento de modelo.

- Baseline solicitado e confirmado em `main`: `1cc06fc1d143426dfcc77ea9df295b7447940c99` (merge do PR #67).
- Durante a auditoria, `main` avançou com o PR #68 (relatórios de métricas, sem alterações ao backend ou aos testes OpenCL). A branch F1d foi rebaseada sobre `f3887fc33a1e7c77372f3ffeaaf4c33771669e21` antes da publicação; as regressões foram repetidas nessa base.
- Branch: `feature/issue-61-f1d-opencl-embedding-integration`.
- Worktree isolada: `Worktrees/Kinera_A009`.
- Implementação F1a/F1b/F1c existente foi reutilizada; nenhum arquivo de produção ou kernel foi alterado.

## Ambiente

| Item | Valor observado |
| --- | --- |
| Sistema | Fedora Linux 45 Workstation |
| Kernel | `7.2.11-300.fc45.x86_64` |
| Python | `3.15.0` |
| GPU | AMD Radeon RX 550 / 550 Series, `radeonsi`, `polaris12`, ACO |
| PCI | `0000:03:00.0` |
| Driver | `26.2.6` |
| Runtime | Mesa Rusticl, OpenCL 3.1, OpenCL C 1.2 |
| FP64 | Disponível e exigido pelos testes |
| Aviso do runtime | `Patched Mesa libclc not detected` |

Os testes de hardware usaram `RUSTICL_ENABLE=radeonsi`,
`RUSTICL_FEATURES=fp64` e `KINERA_RUN_OPENCL_TESTS=1`. O aviso foi emitido e
permanece como ressalva do ambiente.

## Protocolo e casos

CPU e OpenCL criaram `Embedding(4, 3, seed=20261106)` e executaram as mesmas
sequências. Foram feitos três ciclos, com IDs repetidos e únicos, duas chamadas
de backward por ciclo, e taxas `0.0`, `0.125` e `0.03125`. Os gradientes incluem
valores positivos, negativos, zeros e cancelamento na ordem `1e16`, `1`,
`-1e16`. Um ciclo adicional usa IDs vazios, backward vazio e step com taxa zero.

Os checkpoints comparam saída do forward, tabela inteira após cada backward e
pesos após cada step. As comparações usam `float64`, `atol=rtol=1e-12`. A
materialização ocorre explicitamente com `buffer_to_host()` somente nos
checkpoints; `buffer_to_host()` foi monitorado durante chamadas operacionais e
não foi chamado implicitamente. O preflight já existente de F1a/F1b ainda pode
ler seus pequenos status de validação.

## Matriz de resultados

| Critério | Resultado | Evidência |
| --- | --- | --- |
| Seeds e pesos iniciais CPU/OpenCL | PASS | Inicialização integral igual |
| Saída de `forward` em ciclos consecutivos | PASS | Erro absoluto máximo `0.0` |
| Reset dos gradientes por novo `forward` | PASS | Tabela completa zerada; pesos preservados |
| Acumulação em dois `backward` consecutivos | PASS | Tabela completa após cada chamada; erro máximo `0.0` |
| Gradiente host e gradiente residente emprestado | PASS | Host upload temporário fechado; buffer residente permanece aberto |
| `step` com taxa zero e positiva | PASS | Estado consumido e pesos iguais ao CPU; erro máximo `0.0` |
| Identidade do buffer de pesos | PASS | Mesmo `OpenCLBuffer` preservado em todos os ciclos |
| IDs vazios | PASS | `[]`, sem buffer de dimensão zero e sem kernel de lookup/backward/step |
| `step` sem backward | PASS | Rejeitado pela camada |
| `release_output_buffer()` antes do backward | PASS | Output fica fechado; backward ainda funciona e coincide com CPU |
| `discard_forward_state()` | PASS | IDs/output fechados, estado removido, backward/step rejeitados |
| `close()` repetido e liberação | PASS | Buffers próprios fechados; sem residuais no manager; uma liberação por identidade de alocação |
| Falhas pré-execução simuladas | PASS | Estado e pesos/gradientes inalterados |
| Falha após submissão de kernel in-place | LIMITAÇÃO | Mutação pode ter ocorrido; não há rollback demonstrado |

Resultados numéricos máximos observados na RX 550:

| Etapa | Erro absoluto máximo |
| --- | ---: |
| `forward` | `0.0` |
| Gradientes após primeiro `backward` | `0.0` |
| Gradientes após segundo `backward` | `0.0` |
| Pesos após cada `step` | `0.0` |

O teste de IDs vazios confirmou que forward, backward e step a taxa zero não
criam nem enfileiram kernels. Os pesos permanecem iguais à referência CPU.

## `release_output_buffer()` e estado da camada

`release_output_buffer()` fecha a alocação de saída, mas mantém a referência
fechada em `layer.output`. `Embedding.backward()` exige que tenha havido um
forward e usa `last_ids`, `grad_weights` e os gradientes; o kernel F1b não lê a
saída do lookup. Por isso, o backward após `release_output_buffer()` passou na
API fake e na RX 550. Esse comportamento atual não invalida o backward e não
exigiu alteração de produção.

`discard_forward_state()` fecha IDs e output, limpa as referências e redefine
`_has_backward`. Os gradientes e pesos continuam pertencendo à camada até um
novo forward ou `close()`. Um `close()` repetido não fez liberações extras.

## Falhas e recuperação

Os testes CPU-only com fake OpenCL API injetaram incompatibilidade de shape e
manager, IDs residentes inválidos, gradientes não finitos, falha de alocação do
status, build, criação/configuração de kernel, enqueue de preflight/backward/
step e falha de release.

Falhas de validação e falhas simuladas antes da mutação preservaram IDs, output,
gradientes, parâmetros e o estado de backward; uma falha de enqueue do step
permitiu repetir a operação depois de removida a falha e obter o resultado CPU.
Os buffers temporários e os buffers da camada foram liberados. A contagem de
liberações usa identidade por geração de alocação, pois o runtime pode reutilizar
o mesmo valor de handle depois de liberar um objeto OpenCL.

As falhas de release após enqueue mostram o limite do contrato in-place: o fake
executa a atualização antes de injetar o erro de release. Assim, `grad_weights`
ou `weights` podem já ter mudado quando a camada propaga a exceção. No caso do
step, `_has_backward` continua verdadeiro; repetir `step` cegamente pode
aplicar o gradiente novamente. No backward, o gradiente pode ter acumulado sem
que a camada marque `_has_backward`. Não há sinal no contrato atual para
distinguir com segurança falha pré-submissão de erro reportado após execução,
nem rollback. Este gate não promete retry/recuperação automática; após exceção
ambígua pós-submissão, o chamador deve descartar a camada/estado ou reconstruí-lo
conforme sua aplicação.

Esse resultado é registrado como **limitação de falha pós-submissão**, prevista
no protocolo F1d, e não como falha de equivalência durante execução bem-sucedida.
Não foi feita correção de produção: alterar a política exigiria um contrato de
estado capaz de reportar se o kernel foi submetido/concluído, ou uma estratégia
de rollback, ambos fora do escopo autorizado.

## Regressões e artefatos

- Gate opt-in RX 550: **28 testes passaram** cobrindo F1a, F1b, F1c, F1d e
  `Linear.forward/backward/step` e integração. Todas as comparações reportaram
  erro máximo `0.0`.
- Fake API F1d: **6 testes passaram**, com erros máximos `0.0` no ciclo completo.
- Seleção da CI do repositório em Python 3.15: **740 passed, 54 skipped,
  5 deselected, 709 subtests passed**. Os cinco deselected são os pilotos/
  benchmarks experimentais já excluídos no workflow; nenhum treinamento ou
  benchmark foi executado. Houve apenas aviso de depreciação AnyIO/pytest.
- `py_compile` e `git diff --check`: PASS.
- Wheel `kinera-0.1.0-py3-none-any.whl`: construído; inclui os seis kernels
  OpenCL, entre eles `embedding_forward.cl`, `embedding_backward.cl` e
  `embedding_step.cl`.
- Importação CPU-only da instalação do wheel: PASS, sem importar
  `kinera.backends.opencl.api`.
- Teste de integridade P3c: PASS; JSONs, relatórios e fingerprints históricos
  preservados.

Os fingerprints SHA-256 dos dois arquivos de teste F1d são registrados aqui
para identificar exatamente os testes publicados:

| Arquivo | SHA-256 |
| --- | --- |
| `tests/test_opencl_embedding_integration.py` | `e18978073aa76bd722ae4bb30df88a5c9064c34b03ee9a77c8f2b0aea61fc9c9` |
| `tests/test_opencl_scaffold.py` | `86750f7c8f4c6ae14b041ee37c262e013d43cf82a4b3edef4c84d87e3c7e96d9` |

## Limites e parecer

F1d passou como ciclo integrado da classe `Embedding` na RX 550/Mesa Rusticl
com FP64, dentro dos casos e contratos acima. Nenhum código de produção,
kernel, CPUBackend ou CUDABackend foi modificado. A matriz não cobre
`AutoregressiveModel`, representação, ativações, losses, trainer, evaluator,
runners ou treinamento integrado. O aviso do Rusticl permanece. O estado após
erro reportado depois de submissão continua potencialmente ambíguo e não é
seguro para retry automático.

Os fingerprints SHA-256 dos arquivos de teste estão registrados acima. O SHA
canônico do commit publicado é mantido nos metadados do PR draft vinculado.

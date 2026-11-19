# F2.1 — HybridRepresentation.forward em OpenCL na RX 550

## Resultado

O gate F2.1 passou localmente na AMD Radeon RX 550 com Mesa Rusticl e FP64. A operação concatena três matrizes residentes na ordem **embedding de caractere → features → embedding de sílaba**. A saída permanece em um `OpenCLBuffer` próprio; não há transferência das matrizes completas para a CPU durante a fusão.

O resultado foi bit a bit igual à referência `CPUBackend` nos casos executados, incluindo signed-zero. O erro absoluto máximo observado foi **`0.0`**. Este gate cobre somente `HybridRepresentation.forward`; pooling, backward, losses e treinamento de modelo continuam indisponíveis ou não validados em OpenCL.

## Proveniência

- Issue de implementação: [#72](https://github.com/Brazwed/Kinera/issues/72)
- Pesquisa funcional: [#61](https://github.com/Brazwed/Kinera/issues/61)
- Baseline mínimo solicitado: `faf211c30d118a5904a4933d838cd04800afedec` (merge do PR #71).
- `main` atual confirmada antes da publicação: `1bcc664df5e2660b600e732ec75549cfe1c09f3a` (merge do PR #73). O delta entre o baseline mínimo e essa `main` atualiza checkpoints e fingerprints legacy; foi revisado como independente do backend OpenCL.
- Commit local com implementação e testes: `fde2fd34af3afa2737d1588a2cc2190f2ceb16d4` — `feat(opencl): add resident hybrid representation forward`.
- Fingerprint SHA-256 do backend testado: `f66262e1295ecaa4aa101375af66754dbe2b9d6e141be682f56b9819b4189367`.
- Fingerprint SHA-256 do kernel: `9f705968b57ca271fee6db025a4b15c34abc67fe33503da7115ee1e99b3a8b2c`.

## Implementação

`OpenCLBackend.hybrid_representation_forward()` valida os três buffers e suas dimensões, dtype, rank, layout, tamanho declarado, dispositivo e manager proprietário. Também verifica os limites de inteiros usados nos argumentos do kernel, os limites de `size_t` para o dispatch e os limites de alocação do dispositivo.

Antes de alocar ou devolver a saída, um kernel de preflight percorre os valores residentes dos três canais e detecta NaN e `+/-Infinity`. O host lê apenas um status `int32` de 4 bytes. Se o preflight falha, a saída não é alocada nem publicada.

Depois do preflight, o kernel de fusão copia os valores para um novo buffer row-major, sem operações aritméticas. As três entradas permanecem abertas e inalteradas. O manager rejeita uma saída que, por defeito do alocador, aliasse uma entrada. Aliases entre entradas de leitura são permitidos: não alteram a semântica da concatenação.

A camada `HybridRepresentation` mantém seu fluxo existente: valida as listas/features antes de substituir o estado anterior, envia features explicitamente, chama as embeddings e a fusão residente, libera o upload das features e os outputs intermediários das embeddings, e retém a matriz fusionada. O contexto vazio continua retornando `[]`, sem buffer ou kernel de dimensão zero. A chamada residente direta com dimensões vazias é rejeitada.

## Ambiente RX 550

| Campo | Valor observado |
|---|---|
| Dispositivo | AMD Radeon RX 550 / 550 Series (`radeonsi`, `polaris12`) |
| PCI | `0000:03:00.0` |
| Driver | `26.2.6` |
| Runtime OpenCL | `OpenCL 3.1` |
| OpenCL C | `OpenCL C 1.2` |
| FP64 | Disponível; gate executado com `RUSTICL_FEATURES=fp64` |
| Python | `3.15.0` |
| Sistema | Linux `7.2.11-300.fc45.x86_64` |
| Opt-in | `KINERA_RUN_OPENCL_TESTS=1`, `RUSTICL_ENABLE=radeonsi` |

O runtime emitiu `Patched Mesa libclc not detected`. Os kernels compilaram e os testes passaram, mas o aviso permanece como ressalva de compatibilidade do Rusticl.

## Matriz de validação F2.1

| Critério | Resultado | Evidência |
|---|---|---|
| Ordem char → features → syllable e dimensões distintas | PASS | Fake OpenCL e RX 550; saída integral comparada à CPU |
| Valores assinados, zeros e signed-zero | PASS | Igualdade de bits no fake e no gate RX 550 |
| Shape, rank, dtype, nbytes, layout e dimensões | PASS | Testes fake; layout row-major/C |
| Device, manager e ownership | PASS | Manager estrangeiro rejeitado; output no manager do backend |
| Limites int32, `size_t` e alocação | PASS | Dispatch fora do limite rejeitado e teste de max allocation |
| NaN e `+/-Infinity` residentes em cada canal | PASS | 9 combinações no RX 550 (3 canais × 3 valores inválidos) |
| Falha de preflight sem saída ou mutação das entradas | PASS | Inputs permanecem abertos; temporário de status liberado |
| Fusão sem D2H integral ou H2D implícito | PASS | Durante a operação, apenas status de 4 bytes é lido e nenhum buffer host é escrito |
| Lifecycle da camada | PASS | Output fusionado retido; outputs intermediários fechados; upload de features liberado |
| Contexto vazio | PASS | Retorna `[]`; sem alocações de dimensão zero |
| Erros de alocação/build/configuração/enqueue/status/release | PASS com limite | Falhas injetadas; tentativas de cleanup únicas; erro primário preservado |
| Falha antes da substituição do estado | PASS | Erro de validação das features preserva output/contexto anteriores |
| Falha durante a substituição | PASS | Falha de alocação descarta estado parcial; output antigo já liberado, sem promessa de rollback |

O status de release precisa ser interpretado com cautela: se uma chamada nativa de liberação retorna erro, o estado do recurso pode ser ambíguo. O gate tenta cada release uma vez e desarma o wrapper Python para evitar uma segunda liberação pelo destrutor. O teste simula uma liberação nativa concluída que ainda reporta erro. Não há garantia de recuperar uma alocação se um driver reportar erro antes de concluir a liberação.

## Testes executados

Comando de regressão, sem pilotos históricos:

```bash
RUSTICL_ENABLE=radeonsi RUSTICL_FEATURES=fp64 KINERA_RUN_OPENCL_TESTS=1 \
PYTHONPATH=src:tests python -m unittest -q \
  test_opencl_scaffold test_opencl_forward test_opencl_backward \
  test_opencl_step test_opencl_integration \
  test_opencl_embedding_forward test_opencl_embedding_backward \
  test_opencl_embedding_step test_opencl_embedding_integration \
  test_opencl_hybrid_forward test_hybrid test_embedding \
  test_embedding_buffers test_linear test_backend_contract \
  test_cuda_backend_ops test_cuda_runtime test_cuda_embedding_forward \
  test_cuda_embedding_backward test_cuda_embedding_step \
  test_cuda_embedding_integration
```

- **225 testes**, `OK`; 8 casos CUDA que exigem GPU NVIDIA foram pulados. As regressões CUDA via mocks passaram.
- Os gates opt-in OpenCL F1a–F1d, Linear e F2.1 executaram na RX 550. Nos casos numéricos reportados, o erro máximo foi `0.0`.
- A classe fake F2.1 cobre ordem/cópia, buffers inválidos, ownership, aliases, finitude e falhas injetadas nos caminhos de preflight e fusão.
- `tests/test_opencl_host_conversion_analysis.py`: 8 passaram, incluindo o teste de fingerprint; 13 subtestes passaram. O SHA F1c (`33ec898a…`) permanece registrado como histórico e o SHA do backend atual foi avançado para F2.1. O JSON e os relatórios P3c não foram modificados.
- P3b/P3e integridade: `tests/test_opencl_p3_cost_accounting.py` e `tests/test_opencl_p3e_device_profiling.py` passaram na verificação direcionada.
- `py_compile` e `git diff --check`: PASS.
- Wheel construído e inspecionado; contém `kinera/backends/opencl/kernels/hybrid_representation_forward.cl` (2.323 bytes). Instalação isolada do wheel seguida da seleção de CPU passou sem carregar módulos `kinera.backends.opencl`.
- Não foram executados benchmarks nem treinamento.

O `pytest` local emitiu um aviso de depreciação do plugin `anyio` (alias `CallSpec2`), sem falha nos testes.

## Limitações e decisão do gate

- O kernel de validação usa um único work-item para percorrer sequencialmente os três buffers. Isso simplifica a detecção determinística e não transfere matrizes completas, mas o custo não foi medido; este gate não faz afirmações de desempenho.
- Não há rollback demonstrado para erros assíncronos posteriores à submissão do kernel de fusão.
- O gate valida a fusão e a camada `HybridRepresentation.forward`; não valida pooling, backward, losses, `AutoregressiveModel`, trainer, evaluator ou treinamento completo.
- CI GitHub Actions do PR #74 (run `42267967497`, job `128064744591`) concluída com **success** no head revisado `7fbb38bd7962e4f16b2c30d49d4be37c9f43dddf`: **774 passed, 57 skipped, 5 deselected, 760 subtests passed**. É regressão remota CPU-only, não substitui os testes RX 550 realizados localmente. Esta atualização é apenas documental; o novo commit do relatório requer confirmação do status da CI no head final.

**Parecer local: GO para revisão técnica do F2.1.** A conclusão não marca F2.1 como concluído e não autoriza F2.2 ou integração de treinamento.

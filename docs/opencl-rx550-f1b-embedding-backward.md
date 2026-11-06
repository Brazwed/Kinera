# Kinera OpenCL F1b — Embedding.backward na RX 550

**Data:** 2026-11-06

**Escopo:** somente `Embedding.backward` OpenCL; F1c (`Embedding.step`) e integração de treinamento permanecem fora desta etapa.

## Resultado

O gate F1b passou na AMD Radeon RX 550 usando Mesa Rusticl com suporte FP64. Os resultados de `grad_weights` foram exatamente iguais aos do `CPUBackend` em todos os casos numéricos executados: **erro absoluto máximo `0.0`**, com tolerâncias `atol=rtol=1e-12`.

O backend atualiza e retorna o mesmo buffer residente de `grad_weights`. A acumulação percorre as posições dos IDs em ordem, sem atomics. Um kernel de preflight valida IDs residentes e finitude dos gradientes antes de qualquer escrita na tabela de destino.

## Proveniência e ambiente

- Repositório: `Brazwed/Kinera`.
- Branch: `feature/issue-61-f1b-opencl-embedding-backward`.
- Baseline limpo: `061d4973165fb871207305d5b9e0ba378e73a178` (merge do PR #63).
- Durante a execução, `main` avançou para `3a9cd4df5a67557e5f8b2de0926f4b9f32b35106` (merge do PR #64, sem arquivos sobrepostos); a branch F1b foi reaplicada sobre esse HEAD antes da publicação.
- Sistema: Fedora Linux 45, kernel `7.2.11-300.fc45.x86_64`.
- Python: `3.15.0`.
- Dispositivo: AMD Radeon RX 550 / 550 Series, PCI `0000:03:00.0`.
- Runtime: Mesa Rusticl, OpenCL `3.1`, OpenCL C `1.2`, driver reportado `26.2.6`, FP64 disponível.
- Execução: `RUSTICL_ENABLE=radeonsi RUSTICL_FEATURES=fp64 KINERA_RUN_OPENCL_TESTS=1`.

Fingerprints SHA-256 dos arquivos usados nos gates locais:

| Arquivo | SHA-256 |
| --- | --- |
| `src/kinera/backends/opencl/backend.py` | `0557ea395cf58f887f4a8f1e23d2aded50ab782f594d3841f59c83b65c1c3e21` |
| `src/kinera/backends/opencl/kernels/embedding_backward.cl` | `b5bc7e03a50b42d635aa9caa40edcd98d44fd55e49716bdc6aff7bc8dda17dba` |
| `src/kinera/nn/layers/embedding.py` | `ca37bc5c5c2df8f83ebbc675ba587607cc7c7af84515db52ae04e3a0d3b10dfb` |
| `tests/test_opencl_scaffold.py` | `555ec1677711809676f941da86d38623079edd6002937c8c7b14005b7c5f4745` |
| `tests/test_opencl_embedding_backward.py` | `0b5e960c6dc2053ccac59677b1a0a820f2e065399bba2c80e2497d6c5de0d75c` |

## Implementação e contratos

O novo `embedding_backward.cl` tem dois kernels:

1. `kinera_embedding_validate_backward_inputs` percorre IDs e `grad_output` residentes. IDs fora de `[0, vocab_size)` e valores NaN/`+Infinity`/`-Infinity` retornam um código `int32` de status. O host lê somente esse escalar de 4 bytes.
2. `kinera_embedding_backward_f64` atribui um work-item a cada elemento da tabela de gradientes. Cada work-item percorre os IDs na ordem original e soma contribuições correspondentes sequencialmente. Não há atomics nem redução paralela.

O host aguarda a conclusão do preflight, consulta o status e só então enfileira o kernel in-place. Nenhuma transferência da tabela de gradientes ou dos vetores completos ocorre dentro do backend. `grad_weights`, IDs e `grad_output` precisam ser buffers `OpenCLBuffer` abertos, residentes no dispositivo selecionado, pertencentes ao mesmo `OpenCLMemoryManager`, com shapes, dtypes e tamanhos compatíveis. Handles iguais entre os três argumentos são rejeitados.

O método retorna o mesmo objeto `grad_weights`; chamadas sucessivas acumulam sobre o valor existente. Linhas não referenciadas mantêm seus valores. O caso vazio usa as listas vazias já produzidas pela camada: valida `grad_weights`, não aloca buffer de dimensão zero, não compila/enfileira kernel e retorna o buffer inalterado.

Na camada `Embedding`, o caminho com gradiente residente é habilitado para OpenCL somente com o backend OpenCL concreto, IDs e gradiente como `OpenCLBuffer`s pertencentes ao manager desse backend. O caminho CUDA existente foi preservado. Gradientes fornecidos como listas continuam sendo validados na camada, transferidos explicitamente uma vez e liberados após o backward. O import de OpenCL nessa validação é lazy; o caminho CPU continua sem carregar módulos OpenCL.

## Casos e resultados

### RX 550

Foram executados 6 métodos opt-in, com dados determinísticos:

- shapes de tabela `1×1`, `3×2`, `3×3` e `4×2`;
- IDs únicos, repetidos e em ordens diferentes;
- gradientes positivos, negativos, zeros e cancelamento sensível à ordem;
- valores de magnitude `1e16` e `1e308` com resultados finitos;
- duas chamadas consecutivas acumulando sobre o estado anterior;
- caminho de camada com gradientes host e residentes;
- reset da tabela de gradientes após novo `forward`;
- IDs `-1`, `vocab_size`, `INT32_MIN` e `INT32_MAX`;
- NaN e infinitos residentes em `grad_output`;
- backward vazio, ownership e preservação do estado em falha.

Todas as comparações integrais da tabela contra `CPUBackend.embedding_backward` passaram em `atol=rtol=1e-12`. O erro absoluto máximo reportado foi `0.0` em cada caso. Os valores não finitos foram introduzidos diretamente no buffer do dispositivo apenas nos testes, contornando o upload de host que corretamente os rejeita.

### Testes simulados e regressões

- Fake OpenCL API: 11 testes F1b passaram, incluindo falhas injetadas de build, alocação, criação/configuração de kernel, enqueue, leitura do status e release.
- Regressões F1a com API fake: 9 testes passaram.
- Embedding CPU, mocks CUDA, contrato de backend, scaffold OpenCL e integridade P3c: `84 passed, 2 skipped`; os skips são os testes CUDA que exigem hardware NVIDIA.
- Gates OpenCL/Rusticl anteriores de Embedding.forward e Linear (`forward`, `backward`, `step`, integração): `22 passed`, incluindo 439.897 subtestes.
- CPU-only import: coberto pelo teste de subprocesso do scaffold; passou sem importar módulos OpenCL.
- CI local completa sobre o HEAD atual de `main`, conforme os filtros versionados em `.github/workflows/ci.yml`: `719 passed, 48 skipped, 5 deselected`.
- `py_compile`, empacotamento do kernel, CI GitHub e verificações finais de diff são registrados após a validação completa do PR.

Os testes de falhas confirmam que validação, preparação e erros antes do enqueue de acumulação deixam `grad_weights` sem alterações; os temporários de preflight e kernels são liberados em caminhos normais, e a exceção primária é preservada se a limpeza também falhar. A API fake também confirma que o backend só lê o status de 4 bytes e não escreve dados host durante o backward residente.

## Lifecycle e limites

- Em sucesso, o kernel de validação, o buffer de status e o kernel de acumulação têm cada um uma única tentativa explícita de release. Os buffers de entrada e o destino continuam sob ownership do chamador/camada.
- Falhas de validação e falhas antes do enqueue de acumulação não alteram `grad_weights`.
- O enqueue de acumulação é síncrono na API atual, mas não foi demonstrado rollback se o driver reportar falha assíncrona após submeter o kernel in-place. Uma falha de release depois da execução também pode ser reportada após os gradientes terem sido atualizados; a camada não marca então `_has_backward`. Esses casos não têm rollback validado.
- A validação de finitude cobre os elementos de entrada de `grad_output`. Somar entradas finitas ainda pode causar overflow na acumulação; o backend não promete que toda saída permaneça finita.
- O ambiente emitiu `Patched Mesa libclc not detected`. O gate passou, mas o aviso do Rusticl permanece relevante para compatibilidade e não foi suprimido.
- F1c (`Embedding.step`), F1d, treinamento integrado, benchmarks e runners científicos não foram implementados nem validados por este relatório.

## Arquivos desta etapa

- `src/kinera/backends/opencl/kernels/embedding_backward.cl`
- `src/kinera/backends/opencl/backend.py`
- `src/kinera/nn/layers/embedding.py`
- `tests/test_opencl_scaffold.py`
- `tests/test_opencl_embedding_backward.py`
- `tests/test_opencl_host_conversion_analysis.py` (fingerprint corrente do backend atualizado; fingerprints históricos P3c e artefatos publicados preservados)
- `docs/opencl-rx550-f1b-embedding-backward.md`

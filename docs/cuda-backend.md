# CUDA Runtime, ABI e Buffers — Phases 9–11

## Baseline e limites

A referência estrutural pré-CUDA é `a64b23730a4604270c0508f173715fe413704acd`. `CPUBackend` continua sendo o oracle numérico; a instalação padrão e os testes CPU não compilam nem carregam CUDA. Importar e usar apenas o backend CPU não importa os módulos `kinera.backends.cuda` nem procura `nvcc`.

Selecionar ou consultar explicitamente CUDA pode compilar/carregar o runtime nativo. Se CUDA não estiver disponível, a operação solicitada falha com `BackendUnavailableError`; não há fallback silencioso para CPU.

## Build e carregamento opcionais

`CudaBridge` procura, nesta ordem:

1. caminho fornecido em `KINERA_CUDA_LIBRARY`;
2. biblioteca já compilada no cache configurado por `KINERA_CUDA_CACHE` (por padrão `~/.cache/kinera/cuda/`);
3. compilação sob demanda dos `.cu` empacotados, usando `NVCC` ou `nvcc` encontrado no `PATH`.

O build sob demanda está habilitado para Linux. Em outras plataformas, deve ser fornecida uma biblioteca compatível por `KINERA_CUDA_LIBRARY`; falha ao construir não altera o caminho CPU. A biblioteca é carregada via `ctypes.CDLL` somente após solicitação explícita. O wheel inclui as fontes CUDA, mas não executa `nvcc` durante instalação.

O ABI C é versionado (`kinera_cuda_abi_version()`, atualmente versão `7`). Cada função retorna um status inteiro; `kinera_cuda_error_string()` converte falhas CUDA em diagnósticos legíveis. O status `-1` foi introduzido na ABI 5 para IDs da Embedding fora do vocabulário e vira `ValueError` no Python. `CudaBridge` valida todos os símbolos obrigatórios antes de publicar a biblioteca carregada. A ABI 2 adicionou `Embedding.forward`, a ABI 3 adicionou o backward determinístico, a ABI 4 adicionou `Embedding.step` in-place, a ABI 5 valida limites de IDs residentes no device antes de acessar pesos/gradientes e a ABI 6 adiciona concatenação híbrida residente, mean pooling residente e argmax residente; a ABI 7 adiciona cross-entropy e BCE-with-logits residentes com gradientes ponderados, agregação determinística dos gradientes das heads, mean-pool backward e split backward da representação híbrida. Dimensões passadas aos kernels da ABI 6 são validadas no Python contra o limite positivo de `int32` antes de alocar saídas; a representação híbrida também rejeita grids acima de `INT_MAX` blocos, e a ABI nativa repete a checagem antes de estreitar o número de blocos. Bibliotecas ABI 1–6 são rejeitadas explicitamente.

A interface nativa da versão 7 inclui:

- consulta de versão, erro, quantidade e propriedades de devices;
- seleção e sincronização explícitas do device;
- alocação/liberação de memória;
- cópias explícitas host → device e device → host;
- `kinera_cuda_linear_forward_f64`, que executa a soma afim em pesos row-major, com um thread por saída e acumulação sequencial em `float64`;
- `kinera_cuda_linear_backward_f64`, que calcula gradientes de pesos, bias e entrada sem converter buffers para listas;
- `kinera_cuda_linear_step_f64`, que aplica gradient descent diretamente aos buffers de pesos e bias residentes no device;
- `kinera_cuda_embedding_forward_f64`, que valida no device cada ID residente (`0 <= id < vocab_size`) antes do primeiro acesso aos pesos, faz lookup row-major de pesos `float64` e escreve uma saída independente. A checagem transfere ao host apenas um status `int32`, nunca o vetor de IDs; ID inválido retorna o status de validação ABI 5 e não lança o lookup;
- `kinera_cuda_embedding_backward_f64`, que aplica a mesma validação antes de ler/alterar a tabela de gradientes e, em seguida, acumula cada `(vocab_row, embedding_dimension)` percorrendo posições sequencialmente em ordem de entrada, sem `atomicAdd`. IDs inválidos falham sem modificar gradientes;
- `kinera_cuda_embedding_step_f64`, que atualiza os pesos `float64` in-place em VRAM, sem cópias implícitas;
- `kinera_cuda_hybrid_representation_forward_f64`, que concatena buffers `float64` residentes na ordem `[char | features | syllable]`, sem cópia para host;
- `kinera_cuda_mean_pool_forward_f64`, que acumula cada feature sequencialmente na ordem das posições, compatível com a soma `float64` do oracle CPU;
- `kinera_cuda_argmax_f64`, que escolhe o primeiro índice máximo e copia ao host somente um ID `int32`;
- `kinera_cuda_cross_entropy_forward_f64` e `kinera_cuda_binary_cross_entropy_with_logits_forward_f64`, que calculam losses estáveis a partir de logits residentes e copiam ao host apenas o escalar e o status numérico;
- `kinera_cuda_cross_entropy_gradient_f64` e `kinera_cuda_binary_cross_entropy_with_logits_gradient_f64`, que produzem gradientes ponderados `float64` em buffers residentes; somente o status numérico é copiado ao host;
- `kinera_cuda_sum_head_input_gradients_f64`, que agrega por feature na ordem determinística `char → syllable → boundary`;
- `kinera_cuda_mean_pool_backward_f64`, que replica o gradiente agregado dividido pelo tamanho do contexto;
- `kinera_cuda_hybrid_representation_backward_f64`, que separa a matriz na ordem `[char | features | syllable]` para o backward das embeddings.

### Evidência da Phase 11A integrada

O gate de forward residente passou na Tesla T4: 8 testes, sem falhas, erros ou skips; ambiente Python 3.13.15, fingerprint `a4827f39aab54432a943a559d41f0d5b`, manifesto dos fontes/testes `0d8c402e06294c316068dfc5f1dd8f0be828660f26aff14a3b464925b34553db`, `float64`, `atol=rtol=1e-12`. A suíte CPU passou com 377 testes e 23 skips CUDA-only. Os testes host também confirmam o retorno vazio da representação CPU e rejeição de grids CUDA não representáveis antes de alocação/estreitamento. O wheel candidato CPU-only foi construído e instalado em alvo isolado (137017 bytes, SHA-256 `1f80fa77cea3cda027eca4258e85c04e18b16467274105a4267546ceb9cca4a5`); o smoke `forward → backward → step` passou sem `nvcc` no `PATH` e sem importar módulos CUDA. Manifesto de entradas do pacote: `b0bd83a1808bb8236c47573b50c8ac866f0f5d8f2fb5535c4156788faf56b7a8`. Relatório: [`../experiments/phase11a_forward_validation.json`](../experiments/phase11a_forward_validation.json). A Phase 11A foi integrada pelo PR #20 no commit `2c66c6de2b3ac1b3301de5382a01df63f3c3c043`; o check-run pós-merge `tests` passou nesse SHA.

### Evidência candidata da Phase 11B

Na Tesla T4, o bridge compilou a biblioteca nativa com ABI 7 e detectou um device. A fixture determinística de um único `multitask_train_step` comparou logits, losses por head, gradientes weighted, agregação, pooling/backward, ambas as embeddings e todos os parâmetros atualizados CPU↔CUDA em `float64`, `atol=rtol=1e-12`; erro absoluto máximo `1.3877787807814457e-17`. A matriz dos módulos `test_cuda_*.py` executou 96 testes, sem falhas, erros ou skips. O teste confirma que durante o step só são copiados escalares de loss e status, nunca vetores de logits/gradientes. A suíte CPU executou 393 testes, com 24 skips CUDA-only. Relatório: [`../experiments/phase11b_training_validation.json`](../experiments/phase11b_training_validation.json).

A wheel CPU-only do snapshot pré-commit também compilou, instalou num alvo isolado e passou o smoke CPU sem `nvcc` no `PATH` e sem importar módulos CUDA; ela contém `losses.cu`, `multitask_backward.cu` e `runtime.cu`. A wheel final foi reconstruída do commit exato `c5f39e2ec9d3775e126f60690f041b4ed10085a2` (145340 bytes, SHA-256 `00527618b26e88e2b446147321a9401c0b02c2d40396876bd5753f357b245f54`) e também passou build, instalação isolada e smoke; detalhes no relatório. Nenhum benchmark ou treino ampliado foi executado; CI do PR e checks pós-merge continuam gates separados.

Os gates de `Linear.forward`, `Linear.backward` e `Linear.step/update` passaram na Tesla T4. O update é in-place em `float64`, sem cópias host↔device implícitas; `learning_rate=0` retorna os mesmos buffers sem chamar o kernel. A equivalência CPU↔CUDA ficou com erro absoluto máximo `0.0` após 1 e 8 updates (`atol=rtol=1e-12`).

A fixture integrada da Phase 9.8 também passou: `forward → backward → step` (input 3, output 2), comparando saída, gradientes e parâmetros finais CPU↔CUDA com erro absoluto máximo `0.0`. O step não fez transferências implícitas; os 10 buffers do fixture foram liberados e uso após liberação falhou com `BufferReleasedError`. A indisponibilidade simulada do runtime falhou explicitamente com `BackendUnavailableError`, sem fallback CPU. Relatórios: [`../experiments/phase9_7_linear_step_validation.json`](../experiments/phase9_7_linear_step_validation.json) e [`../experiments/phase9_8_linear_integrated_validation.json`](../experiments/phase9_8_linear_integrated_validation.json).

## `CudaBuffer`

A representação CUDA é `CudaBuffer`, um handle próprio para memória contígua tipada: `float64` para parâmetros/intermediários numéricos e `int32` para IDs da Embedding a partir da Phase 10.0. Cada instância registra `device`, `device_id`, `shape`, `dtype`, `nbytes`, layout e ownership. Esses metadados são somente leitura.

O buffer não implementa o protocolo de sequência do Python. Transferências são explícitas por `CudaMemoryManager.copy_to_device(...)` e `copy_to_host(...)` (ou `buffer.to_host()`). Operações CUDA devem aceitar buffers e manter intermediários na VRAM, evitando conversões implícitas para listas entre kernels.

O manager que aloca o buffer é seu owner. `close()` e o context manager liberam a alocação; um segundo `close()` não libera duas vezes, e qualquer uso após liberação gera `BufferReleasedError`. O destrutor é somente uma salvaguarda de melhor esforço; código chamador deve preferir liberar explicitamente.

As operações numéricas CUDA continuam em `float64`, mantendo correspondência com o `float` do oracle Python. Desde a Phase 10.0, o memory manager também transporta IDs como signed `int32`: bool, floats e valores fora do intervalo são rejeitados sem narrowing silencioso. Para IDs, a fronteira explícita é `CUDABackend.buffer_from_host(ids, shape=(n,), dtype="int32")`; `buffer_to_host`/`to_host()` devolvem inteiros Python sem passar pelo formato `float64`. A camada `Embedding` mantém `_vocab_size` e `_embedding_dim` explícitos e guarda pesos/gradientes como `BackendArray`. `Embedding.forward` e o backward determinístico passaram na T4: tabela de gradientes com erro máximo `0.0`; finite difference selecionada com máximo `1.3377743357523286e-11` (limites `1e-8`/`1e-5`). Na Phase 10.3, `Embedding.step` in-place passou após 1 e 8 updates com erro máximo `0.0`; `learning_rate=0` preservou os pesos exatamente. Os kernels validados mantêm parâmetros e gradientes em buffers residentes, sem `atomicAdd` no backward nem cópia implícita no step. Relatórios: [`../experiments/phase10_0_embedding_storage_validation.json`](../experiments/phase10_0_embedding_storage_validation.json), [`../experiments/phase10_1_embedding_forward_validation.json`](../experiments/phase10_1_embedding_forward_validation.json), [`../experiments/phase10_2_embedding_backward_validation.json`](../experiments/phase10_2_embedding_backward_validation.json) e [`../experiments/phase10_3_embedding_step_validation.json`](../experiments/phase10_3_embedding_step_validation.json). O gate T4 de segurança para IDs residentes passou na ABI 5: 9 testes, sem falha/erro/skip, incluindo `-1`, `vocab_size`, `INT32_MIN` e `INT32_MAX` em forward e backward; backward manteve `grad_weights` inalterado e um lookup válido após entradas inválidas também passou. A validação transfere somente 4 bytes de status ao host. Evidência: [`../experiments/phase10_6_embedding_id_bounds_validation.json`](../experiments/phase10_6_embedding_id_bounds_validation.json).

## Política numérica registrada antes do kernel

- dtype aritmético: `float64`;
- dtype de índices da Embedding (Phase 10.0–10.6): signed `int32`, sem uso como valor aritmético;
- tolerância absoluta: `atol=1e-12`;
- tolerância relativa: `rtol=1e-12`;
- comparação por elemento: `abs(cuda - cpu) <= atol + rtol * abs(cpu)`.

O maior erro absoluto observado no gate inicial de `Linear.forward` foi `4.440892098500626e-16`, abaixo do limite composto registrado. Os casos, ambiente T4/nvcc e a saída completa estão em [`experiments/phase9_5_linear_forward_validation.json`](../experiments/phase9_5_linear_forward_validation.json). A tolerância não deve ser ampliada apenas para fazer um teste passar.

## Ordem de validação

1. Runtime, ABI e buffers — device detectado, biblioteca carregada, alocação/liberação e round-trip host/device.
2. `Linear.forward` — kernel real e equivalência CPU ↔ CUDA; este é o gate formal.
3. Somente após PASS do forward: `Linear.backward`, depois `Linear.step/update`.
4. Fixture integrada curta `forward → backward → update` comparada contra CPU.

A Phase 9 está funcional e integrada em `main`. Na Phase 10.1, `Embedding.forward` passou CPU↔CUDA na Tesla T4 (erro absoluto máximo `0.0`, `atol=rtol=1e-12`). Na Phase 10.2, o backward determinístico passou com erro máximo da tabela `0.0`, finite-difference máximo `1.3377743357523286e-11` e sem `atomicAdd`. Na Phase 10.3, o step in-place passou após 1 e 8 updates com erro máximo `0.0`; `learning_rate=0` preservou os pesos exatamente e não lançou kernel. Na Phase 10.4, a fixture integrada `forward → backward → step` passou na T4 com saída, gradientes e pesos finais CPU↔CUDA com erro máximo `0.0`, buffers liberados/use-after-release validado e falha explícita sem runtime CUDA. Na Phase 10.5, o wheel foi compilado/instalado em alvo CPU-only e o smoke `forward/backward/step` passou sem importar módulos CUDA nem resolver `nvcc`; suíte CPU: 343 testes, 13 skips CUDA-only. Naquele estado da Phase 10.5, a ABI 5 estava ativa. Na Phase 10.6, o gate de segurança passou na T4 com 9 testes, sem falha/erro/skip; os quatro IDs inválidos foram rejeitados em forward e backward, `grad_weights` permaneceu inalterado e a fixture válida `forward → backward → step` seguiu equivalente à CPU. A suíte CPU no head foi 347 testes, 15 skips CUDA-only. O wheel foi reconstruído de `git archive` do commit exato `c21aca55211fb3d14299e2e3007be78f7b5a5569`, instalado em alvo isolado e passou o smoke CPU `forward → backward → step` com `PATH` sem `nvcc` e sem importar módulos CUDA; SHA-256 `4e8f04aacfc6121927751c4ed0e03515aaeaa5f8da4fd869b4a445deacf616ec` (129412 bytes). Evidências: [`../experiments/phase10_6_embedding_id_bounds_validation.json`](../experiments/phase10_6_embedding_id_bounds_validation.json) e [`../experiments/phase10_6_cpu_wheel_validation.json`](../experiments/phase10_6_cpu_wheel_validation.json), além dos relatórios versionados `experiments/phase10_1_*` a `phase10_5_*`. Nenhum benchmark ou treino ampliado foi executado.

# CUDA Runtime, ABI e Buffers — Phases 9–10

## Baseline e limites

A referência estrutural pré-CUDA é `a64b23730a4604270c0508f173715fe413704acd`. `CPUBackend` continua sendo o oracle numérico; a instalação padrão e os testes CPU não compilam nem carregam CUDA. Importar e usar apenas o backend CPU não importa os módulos `kinera.backends.cuda` nem procura `nvcc`.

Selecionar ou consultar explicitamente CUDA pode compilar/carregar o runtime nativo. Se CUDA não estiver disponível, a operação solicitada falha com `BackendUnavailableError`; não há fallback silencioso para CPU.

## Build e carregamento opcionais

`CudaBridge` procura, nesta ordem:

1. caminho fornecido em `KINERA_CUDA_LIBRARY`;
2. biblioteca já compilada no cache configurado por `KINERA_CUDA_CACHE` (por padrão `~/.cache/kinera/cuda/`);
3. compilação sob demanda dos `.cu` empacotados, usando `NVCC` ou `nvcc` encontrado no `PATH`.

O build sob demanda está habilitado para Linux. Em outras plataformas, deve ser fornecida uma biblioteca compatível por `KINERA_CUDA_LIBRARY`; falha ao construir não altera o caminho CPU. A biblioteca é carregada via `ctypes.CDLL` somente após solicitação explícita. O wheel inclui as fontes CUDA, mas não executa `nvcc` durante instalação.

O ABI C é versionado (`kinera_cuda_abi_version()`, atualmente versão `5`). Cada função retorna um status inteiro; `kinera_cuda_error_string()` converte falhas CUDA em diagnósticos legíveis. O status `-1` é reservado pela ABI 5 para IDs da Embedding fora do vocabulário e vira `ValueError` no Python. `CudaBridge` valida todos os símbolos obrigatórios antes de publicar a biblioteca carregada. A ABI 2 adicionou `Embedding.forward`, a ABI 3 adicionou o backward determinístico, a ABI 4 adicionou `Embedding.step` in-place e a ABI 5 valida limites de IDs residentes no device antes de acessar pesos/gradientes. Bibliotecas ABI 1–4 são rejeitadas explicitamente.

A interface nativa da versão 5 inclui:

- consulta de versão, erro, quantidade e propriedades de devices;
- seleção e sincronização explícitas do device;
- alocação/liberação de memória;
- cópias explícitas host → device e device → host;
- `kinera_cuda_linear_forward_f64`, que executa a soma afim em pesos row-major, com um thread por saída e acumulação sequencial em `float64`;
- `kinera_cuda_linear_backward_f64`, que calcula gradientes de pesos, bias e entrada sem converter buffers para listas;
- `kinera_cuda_linear_step_f64`, que aplica gradient descent diretamente aos buffers de pesos e bias residentes no device;
- `kinera_cuda_embedding_forward_f64`, que valida no device cada ID residente (`0 <= id < vocab_size`) antes do primeiro acesso aos pesos, faz lookup row-major de pesos `float64` e escreve uma saída independente. A checagem transfere ao host apenas um status `int32`, nunca o vetor de IDs; ID inválido retorna o status de validação ABI 5 e não lança o lookup;
- `kinera_cuda_embedding_backward_f64`, que aplica a mesma validação antes de ler/alterar a tabela de gradientes e, em seguida, acumula cada `(vocab_row, embedding_dimension)` percorrendo posições sequencialmente em ordem de entrada, sem `atomicAdd`. IDs inválidos falham sem modificar gradientes;
- `kinera_cuda_embedding_step_f64`, que atualiza os pesos `float64` in-place em VRAM, sem cópias implícitas.

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

A Phase 9 está funcional e integrada em `main`. Na Phase 10.1, `Embedding.forward` passou CPU↔CUDA na Tesla T4 (erro absoluto máximo `0.0`, `atol=rtol=1e-12`). Na Phase 10.2, o backward determinístico passou com erro máximo da tabela `0.0`, finite-difference máximo `1.3377743357523286e-11` e sem `atomicAdd`. Na Phase 10.3, o step in-place passou após 1 e 8 updates com erro máximo `0.0`; `learning_rate=0` preservou os pesos exatamente e não lançou kernel. Na Phase 10.4, a fixture integrada `forward → backward → step` passou na T4 com saída, gradientes e pesos finais CPU↔CUDA com erro máximo `0.0`, buffers liberados/use-after-release validado e falha explícita sem runtime CUDA. Na Phase 10.5, o wheel foi compilado/instalado em alvo CPU-only e o smoke `forward/backward/step` passou sem importar módulos CUDA nem resolver `nvcc`; suíte CPU: 343 testes, 13 skips CUDA-only. ABI 5 está ativo. Na Phase 10.6, o gate de segurança passou na T4 com 9 testes, sem falha/erro/skip; os quatro IDs inválidos foram rejeitados em forward e backward, `grad_weights` permaneceu inalterado e a fixture válida `forward → backward → step` seguiu equivalente à CPU. A suíte CPU no head foi 347 testes, 15 skips CUDA-only. O wheel foi reconstruído de `git archive` do commit exato `c21aca55211fb3d14299e2e3007be78f7b5a5569`, instalado em alvo isolado e passou o smoke CPU `forward → backward → step` com `PATH` sem `nvcc` e sem importar módulos CUDA; SHA-256 `4e8f04aacfc6121927751c4ed0e03515aaeaa5f8da4fd869b4a445deacf616ec` (129412 bytes). Evidências: [`../experiments/phase10_6_embedding_id_bounds_validation.json`](../experiments/phase10_6_embedding_id_bounds_validation.json) e [`../experiments/phase10_6_cpu_wheel_validation.json`](../experiments/phase10_6_cpu_wheel_validation.json), além dos relatórios versionados `experiments/phase10_1_*` a `phase10_5_*`. Nenhum benchmark ou treino ampliado foi executado.

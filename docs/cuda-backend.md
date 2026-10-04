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

O ABI C é versionado (`kinera_cuda_abi_version()`, atualmente versão `1`). Cada função retorna um status CUDA inteiro; `kinera_cuda_error_string()` converte falhas em diagnósticos legíveis. `CudaBridge` valida todos os símbolos obrigatórios antes de publicar a biblioteca carregada.

A interface nativa da versão 1 inclui:

- consulta de versão, erro, quantidade e propriedades de devices;
- seleção e sincronização explícitas do device;
- alocação/liberação de memória;
- cópias explícitas host → device e device → host;
- `kinera_cuda_linear_forward_f64`, que executa a soma afim em pesos row-major, com um thread por saída e acumulação sequencial em `float64`;
- `kinera_cuda_linear_backward_f64`, que calcula gradientes de pesos, bias e entrada sem converter buffers para listas;
- `kinera_cuda_linear_step_f64`, que aplica gradient descent diretamente aos buffers de pesos e bias residentes no device.

Os gates de `Linear.forward`, `Linear.backward` e `Linear.step/update` passaram na Tesla T4. O update é in-place em `float64`, sem cópias host↔device implícitas; `learning_rate=0` retorna os mesmos buffers sem chamar o kernel. A equivalência CPU↔CUDA ficou com erro absoluto máximo `0.0` após 1 e 8 updates (`atol=rtol=1e-12`).

A fixture integrada da Phase 9.8 também passou: `forward → backward → step` (input 3, output 2), comparando saída, gradientes e parâmetros finais CPU↔CUDA com erro absoluto máximo `0.0`. O step não fez transferências implícitas; os 10 buffers do fixture foram liberados e uso após liberação falhou com `BufferReleasedError`. A indisponibilidade simulada do runtime falhou explicitamente com `BackendUnavailableError`, sem fallback CPU. Relatórios: [`../experiments/phase9_7_linear_step_validation.json`](../experiments/phase9_7_linear_step_validation.json) e [`../experiments/phase9_8_linear_integrated_validation.json`](../experiments/phase9_8_linear_integrated_validation.json).

## `CudaBuffer`

A representação CUDA é `CudaBuffer`, um handle próprio para memória contígua tipada: `float64` para parâmetros/intermediários numéricos e `int32` para IDs da Embedding a partir da Phase 10.0. Cada instância registra `device`, `device_id`, `shape`, `dtype`, `nbytes`, layout e ownership. Esses metadados são somente leitura.

O buffer não implementa o protocolo de sequência do Python. Transferências são explícitas por `CudaMemoryManager.copy_to_device(...)` e `copy_to_host(...)` (ou `buffer.to_host()`). Operações CUDA devem aceitar buffers e manter intermediários na VRAM, evitando conversões implícitas para listas entre kernels.

O manager que aloca o buffer é seu owner. `close()` e o context manager liberam a alocação; um segundo `close()` não libera duas vezes, e qualquer uso após liberação gera `BufferReleasedError`. O destrutor é somente uma salvaguarda de melhor esforço; código chamador deve preferir liberar explicitamente.

As operações numéricas CUDA continuam em `float64`, mantendo correspondência com o `float` do oracle Python. Desde a Phase 10.0, o memory manager também transporta IDs como signed `int32`: bool, floats e valores fora do intervalo são rejeitados sem narrowing silencioso. Para IDs, a fronteira explícita é `CUDABackend.buffer_from_host(ids, shape=(n,), dtype="int32")`; `buffer_to_host`/`to_host()` devolvem inteiros Python sem passar pelo formato `float64`. A camada `Embedding` mantém `_vocab_size` e `_embedding_dim` explícitos e guarda pesos/gradientes como `BackendArray`. O gate de kernels Embedding ainda está pendente. Evidência host-only: [`../experiments/phase10_0_embedding_storage_validation.json`](../experiments/phase10_0_embedding_storage_validation.json).

## Política numérica registrada antes do kernel

- dtype aritmético: `float64`;
- dtype de índices da Embedding (Phase 10.0): signed `int32`, sem uso como valor aritmético;
- tolerância absoluta: `atol=1e-12`;
- tolerância relativa: `rtol=1e-12`;
- comparação por elemento: `abs(cuda - cpu) <= atol + rtol * abs(cpu)`.

O maior erro absoluto observado no gate inicial de `Linear.forward` foi `4.440892098500626e-16`, abaixo do limite composto registrado. Os casos, ambiente T4/nvcc e a saída completa estão em [`experiments/phase9_5_linear_forward_validation.json`](../experiments/phase9_5_linear_forward_validation.json). A tolerância não deve ser ampliada apenas para fazer um teste passar.

## Ordem de validação

1. Runtime, ABI e buffers — device detectado, biblioteca carregada, alocação/liberação e round-trip host/device.
2. `Linear.forward` — kernel real e equivalência CPU ↔ CUDA; este é o gate formal.
3. Somente após PASS do forward: `Linear.backward`, depois `Linear.step/update`.
4. Fixture integrada curta `forward → backward → update` comparada contra CPU.

A Phase 9 está funcional e integrada em `main`. A Phase 10 começou pela Embedding: o suporte de buffers `int32` está disponível, mas os gates CUDA de lookup, backward e update ainda precisam passar em sequência antes da conclusão. Não executar benchmark ou treino ampliado até esses gates, a configuração experimental e autorização explícita.

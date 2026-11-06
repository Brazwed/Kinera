# F2.2 — `mean_pool_forward` em OpenCL na RX 550

## Resultado

O gate implementa pooling médio residente em OpenCL e foi comparado ao `CPUBackend` na AMD Radeon RX 550 com Mesa Rusticl/FP64. O teste direto cobriu contextos de 1, 3, 4 e 257 linhas, diferentes números de features, cancelamento e magnitudes extremas. O erro máximo observado nos vetores diretos foi **0.0 absoluto e 0.0 relativo**. No caminho integrado de `HybridRepresentation.forward → mean_pool_forward`, o erro máximo foi **`6.938893903907228e-18` absoluto** e **`1.7841461331080236e-16` relativo**, dentro de `atol=rtol=1e-12`.

O valor pooled permanece em um `OpenCLBuffer` pertencente ao manager de origem. A chamada direta lê apenas um status de 4 bytes para validar finitude; não escreve nem materializa a matriz no host. A integração usa o helper existente do modelo, que libera a matriz fusionada depois do pooling. Este gate não valida backward, losses, treinamento nem o workflow completo do `AutoregressiveModel`.

## Proveniência

- Issue de implementação: [#75](https://github.com/Brazwed/Kinera/issues/75)
- Roadmap: [#61](https://github.com/Brazwed/Kinera/issues/61)
- Baseline mínimo solicitada: `eb2108ac549f98954cef36171e4b42b8acc97617`.
- `main` foi confirmada nesse baseline no início. Antes da publicação, ela avançou pelo PR #76 até `b9200eeafd4ca4929987366ae1164d6f27e28d26`; a worktree foi sincronizada com esse HEAD. O delta adicional é documental e não entra neste PR.
- Worktree: `Worktrees/Kinera_A012`.
- Branch: `feature/issue-75-f2-2-mean-pool-forward`.
- Fingerprint SHA-256 do backend OpenCL testado: `18b9da3829c28d0301dcd852ad6ec1d11b7d1372d15aaa7a40a6c09f757306bb`.
- Fingerprint SHA-256 do kernel: `2bf61a0e24a7d6a4a824e35ed0bb1db7f834041b28697133a50b26999fd9844e`.
- Commit publicado de implementação e testes: `e6d633abf302ddb339cfbc2af3c6b13e90c85eb2`; o fingerprint acima identifica diretamente o arquivo de produção testado.

## Contrato e implementação

`OpenCLBackend.mean_pool_forward()` exige uma `OpenCLBuffer` aberta, `float64`, rank 2, layout C e shape positivo. O método valida o tamanho declarado, manager/contexto, dispositivo, limites de `int32`, `size_t`, dispatch e alocação do dispositivo. O resultado tem shape `(feature_dim,)`, é alocado no mesmo `OpenCLMemoryManager` e não pode aliasar a entrada. A entrada permanece aberta e inalterada.

Um kernel de preflight percorre os dados residentes e rejeita NaN e `+/-Infinity` antes de alocar o output. O host recebe somente um status `int32`. Se o preflight for aprovado, um work-item por feature acumula as linhas sequencialmente, mantendo a ordem do `CPUBackend`. A flag `_PYTHON_FLOAT_SUM_COMPENSATED` seleciona a soma compensada conforme a política do CPython em uso; `#pragma OPENCL FP_CONTRACT OFF` impede contração implícita. O preflight é sequencial e esta entrega não contém medições de desempenho.

### Semântica numérica e observação do Rusticl

No Python 3.15 usado pelo gate, `_PYTHON_FLOAT_SUM_COMPENSATED` vale `True`. A implementação CUDA também usa acumulação sequencial compensada, mas ela foi tratada como referência de desenho, não como prova de equivalência. Os casos automatizados cobrem `[1e16, 1, -1e16]`, `[1e16, -1e16, 1]`, pequenas parcelas entre magnitudes grandes, alternância de sinais e um contexto de 257 linhas. Um diagnóstico numérico adicional com 60.000 sequências determinísticas de doubles finitos não encontrou divergência entre `sum()` do Python e a fórmula compensada usada pelo kernel; isso é evidência amostral, não uma prova para toda entrada possível.

Durante a primeira execução na RX 550, um kernel de diagnóstico que dividia por um divisor double recebido em runtime retornou `0.3333333320915699` para `1.0 / 3`, com erro relativo aproximado de `3.73e-9`. A entrada e a soma intermediária foram conferidas; a divergência estava no quociente produzido pelo caminho de divisão double deste runtime. O kernel F2.2 calcula o escalar `1.0 / context_size` no host e o envia como argumento double, multiplicando-o pela soma já concluída no dispositivo. O resultado foi comparado de novo ao `CPUBackend` e passou no limite de `1e-12`. A redução continua na GPU e nenhum tensor é transferido para formar o recíproco. A causa interna do resultado de divisão observado não foi isolada.

Com entradas finitas, a soma ainda pode sofrer overflow. Um caso `[[1e308], [1e308]]` foi aceito pelo preflight e produziu `Infinity`, igual ao `CPUBackend`; o gate valida a finitude das entradas, não promete que toda saída seja finita. O aviso `Patched Mesa libclc not detected` apareceu nas execuções e permanece como ressalva do ambiente.

## Ambiente RX 550

| Campo | Valor observado |
|---|---|
| GPU | AMD Radeon RX 550 / 550 Series (`radeonsi`, `polaris12`) |
| PCI | `0000:03:00.0` |
| Driver | `26.2.6` |
| Runtime OpenCL | `OpenCL 3.1` |
| OpenCL C | `OpenCL C 1.2` |
| FP64 | Disponível; execução com `RUSTICL_FEATURES=fp64` |
| Python | `3.15.0` |
| Sistema | Linux `7.2.11-300.fc45.x86_64` |
| Opt-in | `RUSTICL_ENABLE=radeonsi`, `KINERA_RUN_OPENCL_TESTS=1` |
| Aviso do runtime | `Patched Mesa libclc not detected` |

## Matriz do gate F2.2

| Critério | Resultado | Evidência |
|---|---|---|
| Soma em ordem e política compensada selecionada pelo Python | PASS | Testes fake e RX 550; cancelamentos e contexto longo |
| Contexto unitário e múltiplos shapes | PASS | Contextos 1, 3, 4 e 257; dimensões 2, 4 e 7 |
| Equivalência direta com CPU (`float64`, `atol=rtol=1e-12`) | PASS | Erro máximo absoluto/relativo `0.0` nas amostras RX 550 |
| Divisão double dinâmica no Rusticl | Divergência observada | Caso `1/3`: erro relativo `~3.73e-9`, acima da tolerância |
| Recíproco escalar pré-calculado e multiplicação final | PASS | Erro direto `0.0`; integração dentro da tolerância |
| NaN e `+/-Infinity` residentes | PASS | Preflight rejeita antes de alocar/publicar output; status de 4 bytes |
| Overflow da soma com entradas finitas | PASS com limite | Saída `Infinity` coincide com CPU; finitude de saída não é garantida |
| Buffer inválido, fechado, estrangeiro, device/layout/size inválido e alias | PASS | Testes CPU-only com fake OpenCL API |
| Falhas de build, alocação, criação/configuração/enqueue/read/release | PASS | Limpeza única; erro primário preservado quando cleanup também falha |
| D2H/H2D da chamada direta | PASS | Uma leitura de 4 bytes; nenhuma escrita host/device durante pooling |
| Integração limitada com `HybridRepresentation.forward` | PASS | Pooling residente; matriz fusionada e outputs de embeddings liberados pelo consumidor; pesos preservados |
| Entrada emprestada e ownership do output | PASS | Entrada inalterada/aberta; output novo no manager atual |

O wrapper existente `OpenCLApi.enqueue_1d()` chama `clFinish`; F2.2 usa essa semântica de enqueue já existente para preflight e pooling e não adiciona uma chamada independente de sincronização. Erros que cheguem depois da submissão podem deixar o resultado de saída ambíguo; a saída não é publicada quando enqueue/release falha e não há promessa de rollback ou retry automático.

## Testes executados

Testes específicos:

```bash
PYTHONPATH=src:tests python -m unittest -q \
  test_opencl_scaffold.OpenCLMeanPoolForwardTests

RUSTICL_ENABLE=radeonsi RUSTICL_FEATURES=fp64 KINERA_RUN_OPENCL_TESTS=1 \
PYTHONPATH=src:tests python -m unittest -q test_opencl_mean_pool_forward
```

- CPU-only F2.2 fake: **9 testes passaram**.
- RX 550 F2.2 opt-in: **4 testes passaram**; erro direto máximo absoluto/relativo `0.0`, integração `6.938893903907228e-18` absoluto e `1.7841461331080236e-16` relativo.
- Regressões direcionadas CPU, OpenCL/Rusticl, Linear, Embedding F1a–F1d, F2.1, contratos e CUDA via mocks: **226 passaram**, **43 ignorados** na execução CPU-only; **261 passaram**, **8 ignorados** na execução RX 550, com **440.460 subtestes** aprovados.
- Os skips CUDA correspondem a casos que exigem GPU NVIDIA; o backend CUDA foi exercitado por mocks.
- O aviso pytest/anyio `CallSpec2` apareceu sem falha.
- CI completa local, usando as cinco exclusões já configuradas para evitar pilotos e microbenchmarks históricos: **783 passaram**, **61 ignorados**, **5 deselecionados** e **782 subtestes** aprovados.
- Wheel `kinera-0.1.0-py3-none-any.whl` construído e inspecionado; inclui `mean_pool_forward.cl` (1.632 bytes). Instalação isolada e execução de `CPUBackend` passaram sem carregar `kinera.backends.opencl.api`.
- `py_compile`, verificação de fingerprints P3c e `git diff --check`: PASS.
- Não foram executados benchmarks, pilotos históricos ou treinamento.

O resultado da CI do GitHub será acompanhado no PR draft. A JSON e os relatórios históricos P3c permanecem inalterados; o teste de integridade atualiza somente o fingerprint do backend para F2.2 e mantém o SHA F2.1 no histórico.

## Limites

- Não houve medição de desempenho; o preflight sequencial percorre a matriz no dispositivo.
- O desvio observado no divisor double runtime é específico ao ambiente testado e sua origem interna não foi determinada. O aviso de libclc continua relevante.
- Falhas após submissão podem deixar o estado de saída não confiável; o output não é devolvido nesse caminho, sem garantia de rollback de recursos nativos em erro de release.
- O gate cobre somente `mean_pool_forward` e sua chamada após a representação F2.1. Não cobre `mean_pool_backward`, losses F2.3+, backward de representação, treinamento integrado ou execução completa do modelo.

**Estado:** resultados locais para revisão técnica. Este relatório não conclui a Issue #75 nem inicia F2.3.

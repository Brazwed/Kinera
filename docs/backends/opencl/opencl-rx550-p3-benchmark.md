# Kinera Issue #52 — P3 CPU vs. OpenCL benchmark

- **Data:** 2026-11-05
- **Estado:** primeira análise comparativa, executada na AMD Radeon RX 550 local
- **Branch:** `feature/issue-52-p3-benchmark`
- **Base sincronizada com `main`:** `781f02386868a2e7d2001c001e6727433d88fb1d` (merge do PR #53)

## Resumo

O gate numérico passou em todos os quatro shapes: CPU e OpenCL produziram resultados idênticos em `float64` para inicialização, `forward`, os três gradientes e parâmetros após `step` (erro absoluto máximo `0.0`, `atol=rtol=1e-12`).

As medições dependem fortemente do tamanho e orientação da matriz. `1×1` favorece CPU em todas as medições. Em `51×26`, o ciclo residente favorece OpenCL, mas o ciclo end-to-end ficou próximo da paridade. Nos dois shapes com cerca de 102 mil pesos, o ciclo residente e o end-to-end favorecem OpenCL; `forward` isolado em `26×3932` ficou próximo da paridade e teve dispersão alta. Esses números não medem tempo puro de kernel e não justificam uma alegação geral de que OpenCL é sempre mais rápido.

## Código e fingerprint

- Commit base do Kinera efetivamente medido: `781f02386868a2e7d2001c001e6727433d88fb1d`.
- Fingerprint SHA-256 do benchmark e das fontes CPU/OpenCL incluídas no harness: `a08a863613b13222dc591929d749ca29e90721a7c9c7d0c2390f5ccb9060446f`.
- SHA-256 de `experiments/benchmark_opencl_linear.py`: `abffdeb154743a03b4937b4b557e3d1ce05d1c38cc69c7882d6cbdfd6fc04cbc`.
- O backend, os kernels e os contratos CPU/CUDA não foram modificados nesta etapa.

Os resultados estruturados estão em `experiments/results/opencl-rx550-p3-benchmark.json`. Essa cópia preserva protocolo, medições, shapes, seeds, equivalência e limites do dispositivo, e omite snapshots transitórios e caminhos locais da coleta.

## Ambiente medido

| Item | Valor |
| --- | --- |
| CPU | Intel Xeon E5-2680 v4 @ 2.40 GHz, 28 processadores lógicos |
| GPU | AMD Radeon RX 550 / 550 Series, 4 GiB |
| Runtime OpenCL | Mesa Rusticl, driver `26.2.6`, OpenCL `3.1`, OpenCL C `1.2` |
| Precisão | `cl_khr_fp64`; `CL_DEVICE_DOUBLE_FP_CONFIG=63` |
| Memória reportada | 4 GiB global; máximo de alocação individual 1 GiB |
| Sistema | Fedora 45, kernel `7.2.11-300.fc45.x86_64` |
| Python | CPython `3.15.0`, GCC `16.2.1` |
| Variáveis OpenCL | `RUSTICL_ENABLE=radeonsi`, `RUSTICL_FEATURES=fp64` |

O governor da CPU estava em `performance`; clocks de CPU/GPU não foram fixados. Snapshots de carga, afinidade, utilização e VRAM foram observacionais e não atribuíveis ao benchmark, então não são publicados como dados de máquina. Nenhum limite de memória foi aproximado: o maior tensor de pesos teve `817,856` bytes.

O runtime imprimiu o aviso **`Patched Mesa libclc not detected`**. O teste não substituiu o `libclc` nem alterou drivers. As medições caracterizam somente esta combinação de GPU, Rusticl/Mesa, firmware/driver e estado do sistema.

## Protocolo

O harness é `experiments/benchmark_opencl_linear.py`. Ele usa dados determinísticos, `Linear` com seeds fixas, dtype `float64`, learning rate `0.03125`, relógio monotônico de alta resolução (`time.perf_counter_ns`), 3 warmups e 15 repetições temporizadas por medição. Inicialização/compilação usa 5 amostras. CPU e OpenCL são medidos em pares, alternando qual backend vem primeiro. As estatísticas são mediana e IQR (`Q3−Q1`); a coluna de speedup mostra a mediana e `[Q1,Q3]` da razão pareada `CPU / OpenCL`, em que valor acima de 1 favorece OpenCL.

Shapes executados:

- `1×1`: gate escalar pequeno.
- `51×26`: dimensão de uma head já usada nos gates de equivalência.
- `3932×26`: dimensão ampla usada na fixture P2.
- `26×3932`: mesma ordem de quantidade de pesos com dimensões invertidas; expõe o efeito do tamanho da redução serial no kernel `forward`.

O script verifica primeiro equivalência entre CPU e OpenCL e compara saída, `grad_weights`, `grad_bias`, `grad_inputs`, pesos e biases após `step`. Para `step` isolado, cada amostra parte dos mesmos pesos e biases: o reset dos buffers OpenCL é feito por H2D bloqueante antes do cronômetro; essa cópia não entra na latência da operação e seu custo é medido separadamente. O ciclo residente mantém os dados e estado entre etapas. O ciclo end-to-end constrói e fecha `Linear`, prepara parâmetros e entradas, executa `forward → backward → step`, materializa as saídas OpenCL em host e consome os resultados de ambos os backends com `math.fsum`.

Métricas separadas:

1. Descoberta do runtime/dispositivo, criação do contexto/queue e build de cada programa.
2. H2D/D2H bloqueantes em buffers previamente alocados.
3. Caminhos públicos `buffer_from_host` e `buffer_to_host`, incluindo validação/marshal Python, alocação e materialização.
4. Chamadas residentes de `CPUBackend`/`OpenCLBackend`, incluindo validação, alocações temporárias da API OpenCL, enqueue, `clFinish` e liberação do kernel. Limpeza dos resultados temporários e transfers de host ficam fora desta métrica.
5. Ciclo `Linear` residente e ciclo end-to-end descritos acima.

O command queue não habilita profiling. Portanto, não há timestamps de dispositivo para separar launch de execução nem medição de kernel puro. Os builds foram feitos em contextos novos, mas o cache de compilação do driver Mesa não foi limpo. Clock e estado térmico não foram fixados; os snapshots de sistema são observacionais.

O oracle `CPUBackend` executa os loops/list comprehensions Python do Kinera, sem BLAS ou paralelismo de CPU nesta implementação. Os resultados comparam o backend OpenCL ao `CPUBackend` existente, não à capacidade teórica máxima do Xeon nem a uma implementação CPU vetorizada.

### Comando de reprodução

Com um ambiente Python que tenha as dependências de teste instaladas:

```bash
RUSTICL_ENABLE=radeonsi RUSTICL_FEATURES=fp64 PYTHONPATH=src \
  python experiments/benchmark_opencl_linear.py \
  --warmups 3 --repetitions 15 --setup-samples 5 \
  --output /tmp/opencl-rx550-p3-benchmark.json
```

O benchmark foi executado duas vezes durante a preparação. A primeira execução foi descartada: a medição D2H isolada lia um buffer prealocado sem inicializá-lo. O harness passou a preencher esse buffer com dados conhecidos e a medição completa foi repetida com os mesmos parâmetros; somente os resultados da segunda execução aparecem abaixo. Nenhum piloto histórico ou módulo de treinamento foi importado ou executado.

## Inicialização e compilação

Tempos em ms, `mediana [IQR]`, 5 amostras:

| Fase | Tempo |
| --- | ---: |
| Descoberta runtime + seleção do dispositivo | `3.498 [0.042]` |
| Criação de contexto + command queue | `0.083 [0.005]` |
| Build `linear_forward.cl` | `1.824 [0.103]` |
| Build `linear_backward.cl` | `3.001 [0.123]` |
| Build `linear_step.cl` | `1.528 [0.004]` |

O máximo observado no build do `forward` foi `7.954 ms`; os demais builds tiveram máximos de `3.409 ms` (`backward`) e `1.582 ms` (`step`). Cache do driver não foi limpo, portanto esses números não representam necessariamente compilação fria.

## Comparação das operações

Tempos em ms, `mediana [IQR]`, 15 amostras. O speedup pareado é `CPU / OpenCL`; `1.00×` significa tempos equivalentes pela mediana.

| Shape | Medição | CPU ms | OpenCL ms | Speedup `[Q1,Q3]` |
| --- | --- | ---: | ---: | ---: |
| `1×1` | `forward` residente | `0.004 [0.003]` | `0.133 [0.006]` | `0.03× [0.01,0.03]` |
| `1×1` | `backward` residente | `0.006 [0.004]` | `0.181 [0.016]` | `0.03× [0.01,0.03]` |
| `1×1` | `step` residente | `0.003 [0.002]` | `0.113 [0.002]` | `0.03× [0.02,0.03]` |
| `1×1` | ciclo `Linear` residente | `0.151 [0.034]` | `0.561 [0.102]` | `0.31× [0.21,0.32]` |
| `1×1` | ciclo end-to-end | `0.198 [0.034]` | `1.276 [0.081]` | `0.14× [0.13,0.15]` |
| `51×26` | `forward` residente | `0.085 [0.004]` | `0.149 [0.061]` | `0.58× [0.40,0.59]` |
| `51×26` | `backward` residente | `0.180 [0.014]` | `0.296 [0.044]` | `0.58× [0.52,0.64]` |
| `51×26` | `step` residente | `0.138 [0.003]` | `0.111 [0.003]` | `1.24× [1.22,1.29]` |
| `51×26` | ciclo `Linear` residente | `3.770 [0.024]` | `0.684 [0.103]` | `5.49× [5.16,6.01]` |
| `51×26` | ciclo end-to-end | `5.384 [0.055]` | `5.108 [0.241]` | `1.05× [1.04,1.08]` |
| `3932×26` | `forward` residente | `6.492 [0.100]` | `0.683 [0.097]` | `9.45× [8.91,10.27]` |
| `3932×26` | `backward` residente | `16.497 [0.576]` | `4.845 [0.066]` | `3.36× [3.32,3.48]` |
| `3932×26` | `step` residente | `10.890 [0.192]` | `0.319 [0.021]` | `34.30× [32.69,34.95]` |
| `3932×26` | ciclo `Linear` residente | `278.691 [0.959]` | `6.507 [0.828]` | `42.74× [41.08,46.92]` |
| `3932×26` | ciclo end-to-end | `399.208 [3.764]` | `222.567 [2.642]` | `1.80× [1.79,1.81]` |
| `26×3932` | `forward` residente | `4.580 [0.166]` | `4.873 [1.762]` | `0.96× [0.93,1.43]` |
| `26×3932` | `backward` residente | `14.441 [2.813]` | `0.832 [0.069]` | `17.72× [16.60,19.50]` |
| `26×3932` | `step` residente | `7.913 [1.228]` | `0.363 [0.091]` | `23.41× [21.13,26.00]` |
| `26×3932` | ciclo `Linear` residente | `254.396 [2.190]` | `5.024 [1.764]` | `50.52× [40.00,54.92]` |
| `26×3932` | ciclo end-to-end | `364.532 [1.675]` | `219.254 [5.008]` | `1.67× [1.65,1.70]` |

O ganho residente não deve ser confundido com o ganho end-to-end: os primeiros tempos mantêm dados no backend e não incluem transferências de resultados. Em `51×26`, por exemplo, o ciclo residente OpenCL foi mais rápido, enquanto o end-to-end ficou perto da paridade. Em `26×3932`, a dispersão do `forward` OpenCL foi alta e sua mediana não mostra vantagem clara.

## Transferências e memória

Bytes transferidos no ciclo end-to-end: `1×1` H2D/D2H `32/48`; `51×26` `11,632/22,648`; `3932×26` `880,976/1,730,288`; `26×3932` `849,728/1,667,792`.

A tabela abaixo soma medianas medidas separadamente para cada tensor. D2H usa a composição de um ciclo: duas matrizes de pesos, dois vetores de bias, um vetor de entrada/gradiente e um vetor de gradiente de saída. Por isso, é uma aproximação composta, não uma amostra cronometrada como uma única sequência.

| Shape | H2D bruto (4 tensors) ms | D2H bruto composto ms | H2D API pública composto ms | D2H API pública composto ms |
| --- | ---: | ---: | ---: | ---: |
| `1×1` | `0.080` | `0.435` | `0.183` | `0.474` |
| `51×26` | `0.134` | `0.465` | `1.928` | `1.215` |
| `3932×26` | `0.509` | `0.972` | `123.901` | `46.519` |
| `26×3932` | `0.465` | `1.122` | `129.703` | `40.462` |

As transferências brutas usam buffers prealocados e chamadas bloqueantes da API. As chamadas públicas incluem alocação e preparação/materialização Python: conversão/validação dos valores no H2D e leitura/reshape para listas no D2H. Esse custo de host é relevante nos tensors grandes e está incluído no ciclo end-to-end. A maior alocação foi `817,856` bytes, muito abaixo do máximo individual de 1 GiB reportado pelo dispositivo. A leitura de VRAM é global, não um pico isolado do benchmark.

## Equivalência e comandos executados

Todos os quatro casos passaram na verificação CPU↔OpenCL. Em cada caso, os valores comparados foram pesos/bias iniciais, saída do forward, `grad_weights`, `grad_bias`, `grad_inputs` e pesos/bias finais. O erro absoluto máximo foi `0.0` em cada resultado; tolerâncias permaneceram `atol=rtol=1e-12`. Nenhuma tolerância ou matemática de referência foi alterada.

Executado na preparação do relatório:

- `python -m py_compile experiments/benchmark_opencl_linear.py` — PASS.
- `git diff --check` — PASS.
- Benchmark completo no runtime Rusticl RX 550 com `--warmups 3 --repetitions 15 --setup-samples 5` — PASS, 4 shapes; primeira execução descartada pelo motivo de buffer D2H descrito acima.
- Nenhuma suíte de testes CPU foi executada nesta etapa de medição. Nenhum piloto histórico, benchmark científico existente ou treinamento foi executado.

## Leitura dos resultados e limites

- Para `1×1`, launch, sincronização e preparação OpenCL dominam o trabalho; CPU foi claramente mais rápida.
- Para `51×26`, `step` e ciclo residente favoreceram OpenCL, mas o ganho end-to-end mediano foi pequeno (`1.05×`).
- Para `3932×26`, todas as operações residentes e o ciclo end-to-end favoreceram OpenCL nesta execução (`1.80×` end-to-end), mesmo com custos altos de marshal H2D/D2H.
- Para `26×3932`, backward, step e ciclo favoreceram OpenCL, mas o forward ficou aproximadamente empatado. O kernel `forward` acumula sequencialmente a dimensão de entrada, que é 3,932 neste shape; isso é compatível com a diferença observada, mas o benchmark não isola causalidade por timestamp de dispositivo.
- O end-to-end aqui é um único ciclo de uma camada `Linear` com runtime e programas já inicializados; não é cold-start, treino completo ou modelo Kinera ampliado.
- Rusticl advertiu sobre ausência de patched libclc. A compilação do driver usa cache não limpo; GPU/CPU clocks não foram fixados; medições de uso de memória/utilização são snapshots globais.

Esta é uma primeira análise de uma única máquina e de operações `Linear`. Ela registra em que casos cada backend venceu sem propor mudança de kernel, thresholds automáticos ou scheduler. A decisão de uso mais ampla permanece condicionada a revisão e a medições adicionais controladas.

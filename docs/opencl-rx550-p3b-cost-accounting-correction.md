# Kinera Issue #52 — P3b correção da contabilidade do profiler

- **Data:** 2026-11-05
- **Branch:** `feature/issue-52-p3-benchmark`
- **Base P3:** `781f02386868a2e7d2001c001e6727433d88fb1d`
- **Baseline P3:** `experiments/benchmark_opencl_linear.py` SHA-256 `abffdeb154743a03b4937b4b557e3d1ce05d1c38cc69c7882d6cbdfd6fc04cbc`
- **Profiler corrigido:** `experiments/analyze_opencl_linear_costs.py` SHA-256 `93f7e896f067e20150031926b9b04ca1bc29727a4f63d66b3be45d352c15868b`
- **Resultados:** `experiments/results/opencl-rx550-p3b-accounting-diagnostic.json`

## Correção

A instrumentação P3b original guardava apenas o tempo inclusivo de cada wrapper. `OpenCLMemoryManager.allocate` chama `require_device` por meio da validação do dispositivo e da propriedade `device`; por isso, parte de `opencl.device_check` também estava dentro de `opencl.allocate`. O cálculo anterior somava esses tempos inclusivos e subtraía ambos do método backend, contando esse intervalo duas vezes e subestimando `opencl_backend_residual_ms`.

O profiler agora mede, para cada evento, tempo inclusivo, tempo dos timers filhos e tempo exclusivo (`inclusivo − filhos`). Os componentes OpenCL usam somente o tempo exclusivo. Para cada amostra, a soma dos componentes exclusivos e do residual recompõe exatamente a duração inclusiva do método backend. `allocation` e `device_check` aparecem como partes distintas sem sobreposição.

O perfil também verifica automaticamente que:

- cada evento satisfaz `inclusive = children + exclusive`, sem durações negativas;
- componentes exclusivos mais residual recompõem o método backend em cada amostra;
- backend, validação, cópias host e residual recompõem o ciclo residente;
- contagem de amostras, quartis, mediana, IQR, média e limites das estatísticas correspondem às observações.

Os valores apresentados abaixo são medianas calculadas separadamente. A mediana de cada componente não precisa somar à mediana do método; a identidade é verificada amostra a amostra e os dados de amostra estão no JSON estruturado.

## Diagnóstico executado

Foi repetido somente o perfil instrumentado necessário para medir os componentes corrigidos: um processo, ordem CPU primeiro, 3 warmups e 5 amostras por backend e shape. Não foi repetido o benchmark P3, a série de estabilidade P3b, nem o gate numérico. Os tempos originais P3/P3b e a estabilidade entre quatro processos foram preservados. Os checks puros de contabilidade não dependem de GPU.

```bash
RUSTICL_ENABLE=radeonsi RUSTICL_FEATURES=fp64 PYTHONPATH=src:experiments \
  python experiments/analyze_opencl_linear_costs.py --diagnostic-only \
  --profile-warmups 3 --profile-repetitions 5 --order cpu-first \
  --output experiments/results/opencl-rx550-p3b-accounting-diagnostic.json
```

O runtime voltou a exibir `Patched Mesa libclc not detected`. A fila OpenCL continua sem profiling de dispositivo; `enqueue + clFinish` não separa duração do kernel e espera no host. Os tempos são diagnósticos instrumentados, não substituem os valores de benchmark P3/P3b.

## Componentes residentes OpenCL corrigidos

Tempos em ms, mediana `[IQR das cinco amostras instrumentadas]`. `Método` é inclusivo. As demais colunas mostram somente o próprio tempo exclusivo daquele evento, não somas de wrappers inclusivos.

| Shape / operação | Método | Alocação | Verificação de dispositivo | `enqueue + clFinish` | Residual backend |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1×1` forward | `0.3416 [0.0155]` | `0.0357 [0.0023]` | `0.0063 [0.0002]` | `0.1978 [0.0075]` | `0.0539 [0.0012]` |
| `1×1` backward | `0.3650 [0.0233]` | `0.0697 [0.0065]` | `0.0107 [0.0004]` | `0.1580 [0.0161]` | `0.0815 [0.0009]` |
| `1×1` step | `0.2122 [0.0163]` | `0.0000 [0.0000]` | `0.0032 [0.0001]` | `0.0874 [0.0051]` | `0.0679 [0.0074]` |
| `51×26` forward | `0.3831 [0.0565]` | `0.0353 [0.0015]` | `0.0064 [0.0005]` | `0.2358 [0.0613]` | `0.0536 [0.0006]` |
| `51×26` backward | `0.4557 [0.0689]` | `0.0702 [0.0018]` | `0.0108 [0.0002]` | `0.2371 [0.0771]` | `0.0812 [0.0079]` |
| `51×26` step | `0.2713 [0.0088]` | `0.0000 [0.0000]` | `0.0031 [0.0000]` | `0.1606 [0.0074]` | `0.0647 [0.0022]` |
| `3932×26` forward | `0.8632 [0.0460]` | `0.0605 [0.0258]` | `0.0081 [0.0011]` | `0.6792 [0.0488]` | `0.0659 [0.0023]` |
| `3932×26` backward | `4.5036 [0.4029]` | `0.0810 [0.0012]` | `0.0116 [0.0002]` | `4.2243 [0.4430]` | `0.0987 [0.0182]` |
| `3932×26` step | `0.5431 [0.0264]` | `0.0000 [0.0000]` | `0.0039 [0.0001]` | `0.4022 [0.1255]` | `0.0812 [0.0079]` |
| `26×3932` forward | `3.3990 [0.1127]` | `0.0599 [0.0173]` | `0.0076 [0.0009]` | `3.1615 [0.0873]` | `0.0668 [0.0089]` |
| `26×3932` backward | `0.9012 [0.0103]` | `0.0818 [0.0011]` | `0.0118 [0.0004]` | `0.6479 [0.0095]` | `0.0983 [0.0017]` |
| `26×3932` step | `0.5510 [0.0550]` | `0.0000 [0.0000]` | `0.0039 [0.0001]` | `0.4087 [0.0573]` | `0.0851 [0.0016]` |

O JSON também registra lookup de programa, validação dos buffers, configuração e liberação de kernels, liberação de memória e amostras individuais. Esses itens permanecem no cálculo do residual somente quando seu próprio timer não foi instrumentado.

## Exemplo end-to-end da hierarquia de timers

Tempos em ms, medianas de cinco amostras. No construtor, `buffer_from_host` mede o caminho completo; suas chamadas filhas de alocação, verificação e H2D são eventos exclusivos. Na materialização, o residual host é a parede da fase menos as chamadas `read_buffer` medidas.

| Shape | Total ciclo | Construção `Linear` | Cópia de parâmetros `buffer_from_host` inclusiva | Self do caminho host | Alocação self | Device check self | H2D self | Materialização | `read_d2h` self | Residual host D2H | Consumo `math.fsum` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `3932×26` | `222.257` | `143.143` | `118.448` | `117.505` | `0.192` | `0.009` | `0.764` | `48.030` | `2.233` | `45.939` | `18.573` |
| `26×3932` | `216.139` | `146.072` | `123.513` | `122.610` | `0.405` | `0.009` | `0.539` | `41.513` | `2.272` | `39.155` | `17.692` |

Essas novas amostras explicam a composição, mas não substituem o end-to-end P3/P3b: instrumentação, ordem e quantidade de amostras são diferentes. As chamadas de transferência são bloqueantes. O residual host D2H inclui validação de buffer, alocação host, conversão e reshape, e não deve ser interpretado como somente custo de Python.

## Integridade dos artefatos e validação

- `experiments/results/opencl-rx550-p3-benchmark.json` preserva os resultados P3 e remove snapshots transitórios e caminhos locais.
- `experiments/results/opencl-rx550-p3b-original-results.json` mantém os resultados originais P3b, identificando os campos de subfase OpenCL supersedidos; os hashes das fontes JSON originais estão registrados nesses arquivos.
- `experiments/results/opencl-rx550-p3b-accounting-diagnostic.json` contém o profiler corrigido, estatísticas e cinco amostras por shape/backend.
- `PYTHONPATH=src:experiments python -m unittest -v tests/test_opencl_p3_cost_accounting.py` — 6/6 PASS.
- `python -m py_compile experiments/benchmark_opencl_linear.py experiments/analyze_opencl_linear_costs.py tests/test_opencl_p3_cost_accounting.py` — PASS.
- Diagnóstico instrumentado na RX 550 / Mesa Rusticl — PASS em quatro shapes; invariantes de componentes, residuais e estatísticas — PASS.
- Sem mudanças em `src/kinera`, CPU, CUDA, OpenCL, kernels ou contratos científicos. Nenhum piloto histórico, microbenchmark científico ou treinamento executado.

# Kinera Issue #52 — P3b auditoria da decomposição de custos

- **Data:** 2026-11-05
- **Branch:** `feature/issue-52-p3-benchmark`
- **Base:** `781f02386868a2e7d2001c001e6727433d88fb1d` (merge do PR #53)
- **Baseline P3 preservado:** SHA-256 de `experiments/benchmark_opencl_linear.py`: `abffdeb154743a03b4937b4b557e3d1ce05d1c38cc69c7882d6cbdfd6fc04cbc`; SHA do relatório como coletado originalmente: `fde628345b91ce5d56a27d721e5065f641135c49cac65b28d0c5f58982ffa682`.
- **Harness P3b original:** SHA-256 na coleta inicial: `df29feff492e4e8fa831b7e8775a5894b2dd5fd3396bb02a0dcaf85b9e4389ab`. A revisão de contabilidade e seu diagnóstico estão documentados em [P3b — correção da contabilidade do profiler](opencl-rx550-p3b-cost-accounting-correction.md).

> **Correção de contabilidade:** os tempos totais P3/P3b, a estabilidade entre processos e a decomposição CPU abaixo não mudam. A tabela histórica de subfases OpenCL deste relatório somou timers inclusivos e produziu residuais incorretos; mantenho seus números como registro histórico, mas eles não devem ser usados como parcelas aditivas. A medição exclusiva corrigida está no relatório de correção e nos resultados estruturados.

## Conclusão resumida

O ciclo residente CPU que parecia muito mais lento que a soma das primitivas é explicado principalmente pela camada `Linear`: em cada `forward`, o caminho CPU valida e copia listas de pesos, biases e entrada; depois valida a saída e recria listas de gradientes zerados. `backward` também valida e copia dados antes do cálculo, e percorre os gradientes produzidos. As chamadas primitivas P3 mediam `CPUBackend.linear_*` diretamente e não incluíam esse trabalho de `Linear`.

Para `3932×26`, no perfil instrumentado, o ciclo CPU residente mediu `278.070 ms`: `34.296 ms` dentro das três chamadas do backend, `156.225 ms` em validações da camada, `78.910 ms` em `CPUBackend.buffer_from_host` e `7.907 ms` de trabalho de camada restante. Para `26×3932`, os valores correspondentes foram `257.686 ms`, `28.827 ms`, `150.817 ms`, `70.224 ms` e `6.888 ms`. Isso confirma que comparar a soma das primitivas diretas com o ciclo da camada mistura fronteiras de medição diferentes.

O end-to-end OpenCL nos shapes grandes continua mais rápido nesta RX 550, mas sua maior parcela não é o enqueue do kernel: preparar os tensores de parâmetros para H2D consome cerca de `119–126 ms`, principalmente no preparo/validação Python; materializar e remodelar saídas D2H consome outros `41–48 ms`, dos quais cerca de `2 ms` são as chamadas bloqueantes de leitura e `39–46 ms` são o residual de host. O consumo explícito dos resultados por `math.fsum` custa `17–18 ms` nos dois backends.

As medições end-to-end foram relativamente estáveis entre processos. Algumas latências residentes OpenCL, especialmente `forward` de `26×3932`, tiveram grande variação entre processos; por isso, o resultado P3 isolado desse caso não sustenta uma conclusão estável sobre qual backend vence esse `forward`.

## Integridade do baseline e escopo

O arquivo e o relatório P3 originais mantiveram os hashes listados acima. O código de backend, os kernels, `Linear`, CPU, CUDA e OpenCL não foram editados. O harness P3b importa o benchmark P3 e substitui `_paired_measure` apenas em memória durante cada processo, para alternar a ordem inicial. O perfil usa uma subclasse de `Linear` e wrappers de tempo em instâncias de backend/API; nenhuma dessas alterações é gravada em fontes de produção.

Os dados e operações mantêm os seeds, shapes, `float64`, learning rate `0.03125` e matemática P3. Cada processo também executa novamente as verificações CPU↔OpenCL de saída, `grad_weights`, `grad_bias`, `grad_inputs` e parâmetros finais com `atol=rtol=1e-12`.

Uma primeira bateria P3b foi descartada: o agregador do perfil sobrescrevia a duração da primeira de duas preparações de entrada com a duração da segunda. O baseline P3 não foi afetado. Corrigi o perfil para cronometrar entrada e `grad_output` em fases distintas e repeti os quatro processos independentes; somente a bateria corrigida está nos resultados abaixo.

## Método e ambiente

O processo pai iniciou quatro subprocessos Python independentes. A ordem inicial alternou `CPU → OpenCL`, `OpenCL → CPU`, `CPU → OpenCL`, `OpenCL → CPU`; dentro de cada medida, a ordem dos pares também alternou. Cada processo repetiu o benchmark P3 com 3 warmups, 15 amostras temporizadas e 5 amostras de setup/build. O perfil instrumentado fez 3 warmups e 5 amostras por backend e shape. As estatísticas de estabilidade abaixo são calculadas sobre as quatro medianas de processo; o IQR indicado é entre processos, e o intervalo relativo é `(máximo−mínimo)/mediana`.

| Item | Valor |
| --- | --- |
| CPU | Intel Xeon E5-2680 v4 @ 2.40 GHz, 28 processadores lógicos |
| GPU | AMD Radeon RX 550 / 550 Series, 4 GiB |
| Runtime | Mesa Rusticl `26.2.6`; OpenCL `3.1`, OpenCL C `1.2` |
| FP64 | `cl_khr_fp64`, `CL_DEVICE_DOUBLE_FP_CONFIG=63` |
| Sistema | Fedora 45, kernel `7.2.11-300.fc45.x86_64`; CPython `3.15.0` |
| Variáveis | `RUSTICL_ENABLE=radeonsi`, `RUSTICL_FEATURES=fp64` |

Snapshots globais de carga, utilização e VRAM variaram entre processos, mas não identificam qual processo estava usando a GPU; esses dados transitórios não são publicados. O governor da CPU estava em `performance`, clocks de CPU/GPU não foram travados, e o nível de desempenho da GPU ficou em modo automático.

O aviso **`Patched Mesa libclc not detected`** apareceu nos processos. Não houve alteração de driver. As compilações usam cache do driver não limpo. Logs temporários e identificadores de processo foram excluídos dos artefatos publicados; as estatísticas necessárias à auditoria estão em `experiments/results/`.

### Comando executado

```bash
RUSTICL_ENABLE=radeonsi RUSTICL_FEATURES=fp64 PYTHONPATH=src \
  python experiments/analyze_opencl_linear_costs.py \
  --runs 4 --warmups 3 --repetitions 15 --setup-samples 5 \
  --profile-warmups 3 --profile-repetitions 5 \
  --baseline-json /tmp/opencl-rx550-p3-benchmark.json \
  --output /tmp/opencl-rx550-p3b-cost-analysis.json
```

## Auditoria de equivalência e fronteiras de medição

### Dados, estado e matemática

- `1×1`, `51×26`, `3932×26` e `26×3932` usam os mesmos pesos determinísticos, inputs, `grad_output`, seeds e learning rate no CPU e OpenCL.
- Cada subprocesso repetiu o gate numérico. Os quatro passaram; erro absoluto máximo `0.0` em todos os shapes e resultados, sem relaxar tolerâncias.
- Nas medições `step` diretas, CPU recebe os pesos/biases originais em cada chamada. O buffer OpenCL é restaurado para os mesmos valores por H2D bloqueante antes do cronômetro; o reset fica fora do tempo de `step`.
- O ciclo residente inicializa as duas camadas com a mesma seed e executa o mesmo número de ciclos em cada backend. As chamadas CPU/OpenCL são sequenciais; não há duas medições concorrentes.
- Cada ciclo end-to-end cria uma camada com a mesma seed, prepara as mesmas entradas e gradientes, executa `forward → backward → step`, consome saída/gradientes/parâmetros e fecha os objetos. No CPU os resultados já são listas host; no OpenCL são lidos explicitamente e então consumidos.

### Assimetria legítima dos contratos e do caminho `Linear`

Os cálculos e dados são equivalentes, mas as fronteiras de custo não são intercambiáveis:

- As medidas primitivas do P3 chamam `CPUBackend.linear_forward/backward/step` e `OpenCLBackend.linear_forward/backward/step` diretamente. Não incluem a validação/preparação da camada `Linear`.
- `Linear._prepare_backend_array` valida listas CPU e chama `CPUBackend.buffer_from_host`, que percorre, valida e copia a árvore Python. Como `Linear.forward` prepara novamente pesos, bias e input, pesos/bias residentes em listas CPU são copiados a cada forward. `backward` também prepara o `grad_output`.
- Depois das operações, `Linear` valida os arrays de saída e gradientes. No CPU a validação percorre todos os valores; em OpenCL, os buffers passam por checagens de shape, dtype, device e lifecycle.
- `Linear.forward` recria listas CPU de gradientes zerados, que o `backward` seguinte substitui pelos gradientes calculados. Esse custo está no residual de camada; a instrumentação não isolou o list comprehension da alocação zero.
- `CPUBackend.linear_step` cria novas listas de pesos e bias. OpenCL atualiza os buffers existentes in-place. O estado final é numericamente equivalente, mas custo de alocação/lifecycle não é igual.
- O ciclo residente inclui fechamento do output anterior e de `grad_inputs` em OpenCL; o CPU descarta referências a listas. O end-to-end inclui `Linear.close` e os fechamentos explícitos de buffers.
- Em operações primitivas, limpeza dos resultados ocorre depois do cronômetro nos dois backends. No ciclo integrado, fechamentos previstos fazem parte do tempo do ciclo. `write_buffer` e `read_buffer` OpenCL são bloqueantes; cada enqueue inclui `clFinish`.

Assim, somar os tempos primitivos P3 não deve reproduzir o tempo do ciclo `Linear`: a soma direta omite cópias, validações e estado da camada. Não foi encontrada diferença na matemática dos resultados.

## Decomposição do ciclo residente

Valores são mediana `[IQR entre os quatro processos]`, em ms. Os componentes são medidos por wrappers e a soma residual é calculada no mesmo ciclo. `residual` inclui bookkeeping da camada; `housekeeping` é uma parcela desse residual, portanto não deve ser somada novamente.

| Shape | Backend | Ciclo total | Métodos backend | Validação `Linear` | `buffer_from_host` CPU | Residual camada | Housekeeping observado |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1×1` | CPU | `0.214 [0.003]` | `0.013 [0.000]` | `0.083 [0.001]` | `0.026 [0.000]` | `0.092 [0.002]` | `0.009 [0.001]` |
| `1×1` | OpenCL | `0.774 [0.076]` | `0.662 [0.063]` | `0.018 [0.001]` | `0.000` | `0.090 [0.006]` | `0.029 [0.002]` |
| `51×26` | CPU | `3.828 [0.035]` | `0.410 [0.006]` | `2.169 [0.018]` | `1.045 [0.012]` | `0.200 [0.006]` | `0.013 [0.000]` |
| `51×26` | OpenCL | `0.968 [0.068]` | `0.859 [0.061]` | `0.019 [0.001]` | `0.000` | `0.088 [0.004]` | `0.028 [0.002]` |
| `3932×26` | CPU | `278.070 [2.728]` | `34.296 [0.725]` | `156.225 [1.025]` | `78.910 [1.595]` | `7.907 [0.120]` | `0.101 [0.011]` |
| `3932×26` | OpenCL | `6.287 [0.303]` | `6.140 [0.336]` | `0.030 [0.001]` | `0.000` | `0.122 [0.008]` | `0.036 [0.001]` |
| `26×3932` | CPU | `257.686 [0.930]` | `28.827 [1.726]` | `150.817 [0.971]` | `70.224 [0.500]` | `6.888 [0.259]` | `0.078 [0.004]` |
| `26×3932` | OpenCL | `4.815 [0.414]` | `4.667 [0.396]` | `0.029 [0.003]` | `0.000` | `0.120 [0.006]` | `0.036 [0.001]` |

Em `3932×26`, validações e cópias host CPU explicam aproximadamente `235 ms` dos cerca de `244 ms` de diferença entre o ciclo residente e as chamadas backend diretas. No OpenCL, `buffer_from_host=0` no ciclo residente porque os inputs e parâmetros permanecem em buffers de device.

## Fases internas das chamadas OpenCL — registro original supersedido

Os números abaixo foram preservados como apareceram na análise P3b original, mas a soma usada para calcular o residual continha tempos inclusivos aninhados. `opencl.allocate` chamava `require_device`; ambos foram subtraídos como componentes separados, reduzindo artificialmente o residual. Não use esta tabela para decomposição aditiva. A correção e as amostras diagnósticas estão em [P3b — correção da contabilidade do profiler](opencl-rx550-p3b-cost-accounting-correction.md). `enqueue_1d` chama `clEnqueueNDRangeKernel` e depois `clFinish`, então continua impossível separar execução do kernel e sincronização pela instrumentação atual.

Exemplos do registro original, mediana `[IQR entre processos]`, ms; mantidos apenas para preservar a trilha histórica:

| Shape/operação | Método backend | Alocações | `enqueue + clFinish` | Setup/liberação kernel | Residual backend |
| --- | ---: | ---: | ---: | ---: | ---: |
| `3932×26` forward | `0.790 [0.038]` | `0.063 [0.003]` | `0.637 [0.022]` | `0.042 [0.002]` | `0.041 [0.001]` |
| `3932×26` backward | `4.925 [0.267]` | `0.069 [0.002]` | `4.756 [0.257]` | `0.044 [0.001]` | `0.042 [0.002]` |
| `26×3932` forward | `3.362 [0.306]` | `0.064 [0.004]` | `3.194 [0.294]` | `0.042 [0.002]` | `0.038 [0.001]` |
| `26×3932` backward | `0.795 [0.013]` | `0.070 [0.004]` | `0.631 [0.013]` | `0.043 [0.001]` | `0.039 [0.001]` |

Os tempos de método e as medições de ciclo/end-to-end permanecem válidos; apenas a decomposição de subfases OpenCL e o residual derivado são supersedidos. O artefato original está arquivado com esses campos explicitamente identificados em `experiments/results/opencl-rx550-p3b-original-results.json`.

Para `step`, o kernel atualiza pesos/biases in-place, portanto a fase não aloca buffers de saída. Os tempos de método e suas subfases são instrumentados; em operações pequenas, o próprio wrapper é uma parcela material do tempo, como descrito na seção de estabilidade.

## End-to-end: preparo, transfers e consumo

Medianas `[IQR entre processos]`, ms. As fases públicas são deliberadamente separadas: `buffer_from_host` inclui marshal Python, alocação e H2D; D2H separa chamada bloqueante de leitura e materialização Python. A fase de operação soma as paredes das chamadas `Linear`.

| Shape | Backend | Total | Construção `Linear` | Inputs H2D | `forward+backward+step` | Materialização D2H (API + host) | `math.fsum` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `3932×26` | CPU | `401.712 [4.938]` | `103.365 [1.927]` | — | `278.319` | — | `18.566 [0.252]` |
| `3932×26` | OpenCL | `221.459 [2.583]` | `143.457 [1.767]` | `5.254 [0.072]` | `6.212` | `47.585 [0.988]` (`1.972 + 45.542`) | `18.424 [0.335]` |
| `26×3932` | CPU | `365.282 [4.485]` | `92.811 [0.911]` | — | `255.046` | — | `17.480 [0.266]` |
| `26×3932` | OpenCL | `218.540 [5.997]` | `148.636 [4.051]` | `5.133 [0.045]` | `4.923` | `40.976 [0.997]` (`2.271 + 38.536`) | `17.507 [0.125]` |

Na construção OpenCL, as duas cópias públicas de pesos/biases levaram `118.669 ms` (`3932×26`) e `125.933 ms` (`26×3932`). Em `3932×26`, desse total, `0.192 ms` foram alocação e `0.753 ms` foram as chamadas H2D bloqueantes; o residual aproximado de preparo/marshal Python foi `117.724 ms`. No shape invertido, foram `0.416 ms` de alocação e `0.580 ms` de H2D, deixando `124.937 ms` de residual host. O residual é a subtração dos spans instrumentados; não isola internamente cada validação, flatten ou conversão ctypes.

Nas entradas end-to-end, H2D público levou `5.254 ms` e `5.133 ms` nos dois shapes grandes. As chamadas H2D bloqueantes somaram somente `0.307 ms` e `0.229 ms`; a diferença é preparação/validação e custo host do caminho público. Para D2H, `read_buffer` bloqueante levou `1.972 ms` e `2.271 ms`; materializar os buffers em listas Python consumiu o restante de `45.542 ms` e `38.536 ms`. Os tempos de leitura incluem a chamada `CL_TRUE`; o residual também inclui alocação host e reshape.

Em CPU, a camada `Linear` já trabalha com listas host. A preparação ocorre dentro de `forward`/`backward`: em `3932×26`, `buffer_from_host` levou `76.109 ms` em `forward`, e `backward` teve `2.623 ms` de cópia do gradiente; validações somaram `79.008 ms` e `78.926 ms`, respectivamente. Em `26×3932`, a cópia `forward` foi `70.442 ms`, validações `75.366 ms` e `75.097 ms`; os tensores são os mesmos tamanhos, embora o shape/orientação mude as passagens e os tempos.

## Estabilidade entre processos e comparação com P3

Tempos são as medianas das 15 amostras em cada processo. Cada célula de tempo mostra `mediana entre processos [IQR entre processos]` em ms. `CPU/OpenCL` é a razão pareada mediana; acima de 1 favorece OpenCL. `Range CPU/OpenCL` é o intervalo entre a menor e a maior mediana de processo, em percentual da mediana agregada.

| Shape | Medição | CPU ms [IQR] | OpenCL ms [IQR] | Speedup [Q1,Q3] | Range CPU/OpenCL |
| --- | --- | ---: | ---: | ---: | ---: |
| `1×1` | forward | `0.0034 [0.0005]` | `0.1268 [0.0036]` | `0.02× [0.02,0.03]` | `20.3% / 8.8%` |
| `1×1` | backward | `0.0060 [0.0004]` | `0.1894 [0.0055]` | `0.03× [0.02,0.03]` | `12.9% / 9.1%` |
| `1×1` | step | `0.0031 [0.0003]` | `0.1152 [0.0043]` | `0.03× [0.03,0.03]` | `15.6% / 11.1%` |
| `1×1` | ciclo residente | `0.1534 [0.0090]` | `0.5029 [0.0210]` | `0.29× [0.27,0.30]` | `6.1% / 7.1%` |
| `1×1` | ciclo end-to-end | `0.2033 [0.0036]` | `1.2891 [0.0532]` | `0.15× [0.14,0.16]` | `3.2% / 12.8%` |
| `51×26` | forward | `0.0859 [0.0010]` | `0.1658 [0.0156]` | `0.51× [0.49,0.52]` | `2.7% / 22.9%` |
| `51×26` | backward | `0.1790 [0.0015]` | `0.2996 [0.0150]` | `0.60× [0.57,0.62]` | `1.6% / 12.8%` |
| `51×26` | step | `0.1397 [0.0039]` | `0.1205 [0.0037]` | `1.17× [1.15,1.20]` | `6.1% / 8.2%` |
| `51×26` | ciclo residente | `3.7582 [0.0255]` | `0.7285 [0.0245]` | `5.16× [5.06,5.41]` | `1.0% / 9.3%` |
| `51×26` | ciclo end-to-end | `5.4165 [0.2495]` | `4.9252 [0.2778]` | `1.10× [1.09,1.10]` | `13.0% / 16.7%` |
| `3932×26` | forward | `6.5220 [0.1772]` | `0.6769 [0.0787]` | `9.85× [9.55,10.28]` | `7.7% / 17.6%` |
| `3932×26` | backward | `16.6913 [0.2829]` | `4.7857 [0.2211]` | `3.60× [3.46,3.69]` | `4.4% / 11.5%` |
| `3932×26` | step | `11.2019 [0.3415]` | `0.3114 [0.0251]` | `36.34× [35.49,39.28]` | `5.8% / 22.1%` |
| `3932×26` | ciclo residente | `278.7619 [1.6235]` | `5.6704 [0.4734]` | `49.35× [47.64,50.85]` | `2.2% / 13.8%` |
| `3932×26` | ciclo end-to-end | `407.3093 [11.9044]` | `225.1538 [6.5460]` | `1.80× [1.80,1.81]` | `4.7% / 4.8%` |
| `26×3932` | forward | `4.5644 [0.0220]` | `3.3847 [0.4026]` | `1.34× [1.24,1.37]` | `1.7% / 47.1%` |
| `26×3932` | backward | `14.3026 [0.7148]` | `0.7574 [0.0278]` | `18.44× [18.21,18.69]` | `11.9% / 13.1%` |
| `26×3932` | step | `8.1285 [0.0852]` | `0.2883 [0.0403]` | `28.19× [25.91,30.03]` | `2.0% / 23.5%` |
| `26×3932` | ciclo residente | `255.8503 [1.7397]` | `4.5691 [0.6191]` | `56.12× [52.04,57.84]` | `1.8% / 38.1%` |
| `26×3932` | ciclo end-to-end | `366.9391 [2.4716]` | `219.0682 [2.8425]` | `1.68× [1.67,1.68]` | `1.8% / 3.6%` |

Comparação direta do end-to-end com P3:

| Shape | P3 CPU/OpenCL ms | P3b CPU/OpenCL ms | Variação P3b vs P3 |
| --- | ---: | ---: | ---: |
| `1×1` | `0.198 / 1.276` | `0.203 / 1.289` | `+2.6% / +1.0%` |
| `51×26` | `5.384 / 5.108` | `5.417 / 4.925` | `+0.6% / −3.6%` |
| `3932×26` | `399.208 / 222.567` | `407.309 / 225.154` | `+2.0% / +1.2%` |
| `26×3932` | `364.532 / 219.254` | `366.939 / 219.068` | `+0.7% / −0.1%` |

O end-to-end grande reproduziu o resultado P3 dentro de aproximadamente `3.6%`; `1×1` continuou claramente desfavorável a OpenCL. O `forward` OpenCL de `26×3932` mudou de `4.873 ms` no P3 para mediana P3b `3.385 ms`; a variação entre processos foi `47.1%`, com um processo medindo `4.765 ms`. O P3 original também tinha IQR alto nesse `forward`. Portanto, não se conclui vantagem estável nesse caso. Outros ops residentes GPU pequenos também variam mais que os ciclos end-to-end, sobretudo `step` de `3932×26` (`22.1%` de range) e ciclo residente OpenCL `26×3932` (`38.1%`).

Os fatores entre o perfil instrumentado e o P3 sem wrappers foram maiores nas operações pequenas: medianas de `1.40×` CPU e `1.54×` OpenCL no ciclo residente `1×1`; `1.02×` CPU e `1.30×` OpenCL em `51×26`. Para os ciclos residentes maiores, os fatores foram `1.00×/1.11×` (`3932×26`) e `1.00×/1.06×` (`26×3932`). No end-to-end maior os fatores ficaram próximos de 1. Isso quantifica que as leituras de componente são diagnósticas e que o wrapper distorce especialmente o tempo de chamadas pequenas.

## Custos não separados com precisão e hipóteses

**Medido diretamente:** validação `Linear`, chamada `buffer_from_host`, alocação de buffers OpenCL, `write_buffer`/`read_buffer` bloqueantes, lookup do programa, configuração/liberação do kernel, `enqueue_1d + clFinish`, materialização em listas, `math.fsum` e tempos de fechamento end-to-end.

**Não separável com a instrumentação atual:** tempo do kernel no dispositivo versus tempo de enqueue e `clFinish`; `clFinish` só é exposto como uma chamada combinada no binding usado. O residual de `buffer_from_host` não distingue recursão/shape/finiteness, flatten, conversão ctypes e overhead do wrapper. O residual D2H mistura buffer host, conversão para lista e reshape. O residual de `Linear` não isola a alocação de listas de gradientes zeradas, destruição de listas/refcount nem cada validação interna. `clCreateBuffer` mede o tempo da chamada API, não necessariamente commitment físico de memória. VRAM e utilização são snapshots globais, não picos por processo.

**Hipóteses coerentes com os dados, mas não causalmente provadas:** parte da variação em OpenCL residente vem de estado DPM/clock e atividade concorrente global; a inversão de orientação afeta `forward` porque a redução serial do kernel percorre `input_size`; marshal Python e reshape, não a banda de transferência bruta, dominam os transfers públicos grandes. A decomposição não instrumenta relógios de dispositivo nem controla tráfego de terceiros para provar essas causas.

## Próximos experimentos possíveis

1. Repetir isoladamente `forward` `26×3932` e operações pequenas com estado de GPU mais controlado, anotando clock/ocupação por amostra, para determinar se a alta dispersão continua.
2. Usar uma queue de diagnóstico com profiling habilitado, somente no harness experimental, para separar timestamps de execução de kernel e espera do host; manter a queue de produção e os kernels intocados.
3. Rodar um perfil Python de diagnóstico separado para decompor o residual de marshal em validação/flatten, conversão ctypes e reshape, sem alterar o caminho de produção.
4. Amostrar RSS do processo e uso de VRAM ao longo dos ciclos para obter pico por processo em vez de snapshots globais.

Esses são experimentos de medição; este relatório não propõe otimizações, thresholds, scheduler ou mudanças aos kernels. Nenhum piloto de treinamento foi executado.

## Validação executada

- Quatro subprocessos P3 independentes, dois com CPU primeiro e dois com OpenCL primeiro; cada um repetiu todos os 4 shapes e todas as medições P3.
- Equivalência CPU↔OpenCL: 4/4 processos passaram; erro máximo `0.0`; `float64`, `atol=rtol=1e-12`.
- Profile instrumentado: 4 shapes, 3 warmups e 5 amostras por backend/processo.
- `python -m py_compile experiments/analyze_opencl_linear_costs.py` — PASS na coleta original.
- Nenhuma suíte pytest ou piloto científico foi executado nesta etapa.
- Sem alterações a CPU/CUDA/OpenCL ou kernels durante a coleta original; os artefatos P3 foram preservados. A revisão posterior do profiler e seus testes estão registrados no relatório de correção.

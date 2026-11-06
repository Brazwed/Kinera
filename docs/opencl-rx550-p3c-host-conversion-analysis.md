# P3c — Custos de conversão Python ↔ OpenCL

**Data da coleta:** 2026-11-05

**Worktree/branch:** `Kinera_A001`, `feature/issue-52-p3c-host-conversion`

**Base sincronizada:** `434745790f984f718788bcb22b0a945df50b71a5` (merge do PR #54)

**Escopo:** diagnóstico experimental; nenhuma mudança de produção, backend ou kernel.

## Resultado

Nos shapes grandes, o custo do caminho público H2D foi dominado pela preparação no host. `OpenCLMemoryManager._flatten_float_values` (validação e flatten combinados) levou `90.989 ms` em `3932×26` e `100.264 ms` em `26×3932`; o empacotamento ctypes mediu cerca de `23 ms`. A chamada H2D bloqueante prealocada ficou abaixo de `0.40 ms`. Na volta, converter o array ctypes lido para uma lista Python levou `18.052–18.198 ms`; o reshape acrescentou `2.448 ms` com 3932 linhas ou `0.313 ms` com 26 linhas.

Isso identifica onde o tempo foi gasto neste caminho e nesta máquina. Os probes de validação e flatten separados são diagnósticos espelhados, não fases internas instrumentadas separadamente na produção; seus tempos não devem ser somados ao tempo do helper combinado nem tratados como ganhos alcançáveis.

## Auditoria estática dos caminhos

### `OpenCLBackend.buffer_from_host` → `OpenCLMemoryManager.copy_to_device`

1. `_validate_dtype` confere o dtype. Para `float64`, `_flatten_float_values` percorre a estrutura recursivamente, valida tipos/números finitos e dimensões retangulares, converte folhas para `float` e cria uma lista plana intermediária. A forma explícita é normalizada e comparada à forma inferida.
2. `allocate` valida shape, dtype e device, consulta os limites do dispositivo, garante contexto/queue e chama `clCreateBuffer`.
3. `(ctypes.c_double * n)(*flat_values)` aloca e preenche um array contíguo host. `write_buffer` usa `CL_TRUE`, portanto o retorno da chamada aguarda a transferência bloqueante.
4. Se o `write_buffer` falha, o buffer OpenCL é liberado antes de propagar a exceção. A construção ctypes ocorre depois da alocação e antes desse `try`; portanto, há uma possível ausência de liberação explícita do buffer se construir/preencher o array ctypes falhar. Essa falha não foi injetada e a produção não foi alterada nesta entrega. O buffer ainda tem lifecycle próprio/destrutor, mas isso não substitui uma verificação explícita do caminho de erro.

Há passagens intermediárias que não são intercambiáveis sem preservar semântica: cada folha vira uma lista de um elemento, cada nó agrega listas filhas em uma lista nova, e por fim os valores são copiados para ctypes. O helper também aceita tuplas; CPUBackend e `Linear._validate_backend_array` aceitam listas para esses caminhos.

`allocate` chama `_validate_device`, que faz duas chamadas a `runtime.require_device` (uma validação e outra para guardar a informação do device). As medições confirmam a chamada aninhada, mas o custo é pequeno diante do flatten/empacotamento; não se presume que ela seja removível sem revisar o contrato de descoberta.

### `OpenCLBackend.buffer_to_host` → `OpenCLMemoryManager.copy_to_host`

O caminho valida buffer aberto, ownership, dtype e device, aloca um array ctypes host, executa `read_buffer` com `CL_TRUE`, materializa `list(host_array)` e chama `_reshape_host_values`. Para rank 2, o reshape cria slices e listas aninhadas; para rank 1, devolve a lista plana.

### CPUBackend e camada `Linear`

- `CPUBackend.buffer_from_host` valida shape/dtype e, em `_copy_host_tree`, valida retangularidade e números finitos enquanto cria uma cópia de listas e converte folhas numéricas para `float`. Não há uma etapa separada de flatten ou armazenamento contíguo.
- `CPUBackend.buffer_to_host` também revalida e cria uma cópia independente da árvore Python. A forma vem dos dados; listas CPU não carregam metadados de shape/dtype.
- `Linear.__init__` gera pesos e bias em listas aninhadas/listas simples com `random.Random(seed)` e envia cada um ao backend por `buffer_from_host`. O tempo do construtor inclui geração dos parâmetros e as duas cópias.
- `_validate_backend_array` percorre listas e valida shape e valores numéricos. Para `BackendBuffer`, verifica `closed`, shape e dtype; ownership e device são validados pelos métodos de operação/gerenciamento de memória depois. `_prepare_backend_array` valida a lista e então chama `buffer_from_host`, portanto a preparação de uma lista para OpenCL percorre os dados novamente para flatten/validar.
- Com buffers residentes, `_validate_backend_array` faz apenas as verificações de metadados e `_prepare_backend_array` devolve o mesmo buffer. Esse caso não materializa listas nem transfere os dados.

## Método e ambiente

O harness separado [analyze_opencl_host_conversions.py](../experiments/analyze_opencl_host_conversions.py) reutiliza os dados/seeds de `benchmark_opencl_linear.py`. Cada processo mede um backend e um shape; foram executados quatro processos independentes para cada combinação, em ordem alternada CPU→OpenCL e OpenCL→CPU. Cada processo fez 2 warmups e 8 repetições, com `time.perf_counter_ns`. A estatística na tabela é a mediana dos quatro valores medianos por processo, acompanhada de `[Q1, Q3]` entre processos. O JSON preserva também cada amostra e os IQRs dentro de cada processo.

Foram medidos: roundtrip direto das matrizes de pesos e dos vetores de bias/inputs; validação host da `Linear`; `_prepare_backend_array` para pesos host; construção completa de `Linear`; alocação OpenCL; H2D/D2H bloqueantes com memória previamente alocada; flatten/validação, empacotamento ctypes, criação do array host, materialização ctypes→lista e reshape. O caminho público é medido separadamente dos probes de etapas. Os dados de entrada e a saída materializada foram comparados por igualdade exata com a referência determinística; todos os casos usaram `float64` e `atol=rtol=1e-12` (a checagem de roundtrip foi exata, mais estrita que a tolerância).

| Componente | Ambiente observado |
| --- | --- |
| CPU | Intel Xeon E5-2680 v4 @ 2.40 GHz, 28 processadores lógicos |
| GPU | AMD Radeon RX 550 / 550 Series, 4 GiB global; limite de alocação individual de 1 GiB |
| OpenCL | Mesa Rusticl, OpenCL 3.1 / OpenCL C 1.2, driver 26.2.6, FP64 disponível |
| Runtime | `RUSTICL_ENABLE=radeonsi`, `RUSTICL_FEATURES=fp64` |
| Python/SO | CPython 3.15.0, Fedora 45, kernel 7.2.11 |
| Aviso | `Patched Mesa libclc not detected`, observado nos 16 processos OpenCL |

Os clocks e estado térmico não foram fixados; a ordem alternada reduz parte do viés de sequência, mas não controla interferência externa. As medições foram seriais, sem workers concorrentes do harness. Timers Python e wrappers de instrumentação afetam mais os casos muito pequenos. Os tempos são custo de chamada bloqueante no host, não timestamps de dispositivo.

## API direta e preparação pela camada `Linear`

Tempos em ms. `Linear.prepare` prepara somente os pesos recebidos como lista e inclui `_validate_backend_array` mais `buffer_from_host`; `Linear.init` inclui geração determinística de pesos/bias e suas cópias. Os valores da tabela são medianas `[Q1, Q3]` entre os quatro processos.

| Shape de pesos | CPU `from_host` / `to_host` | OpenCL `from_host` (H2D) / `to_host` (D2H) | CPU `Linear` validar / prepare / init | OpenCL `Linear` validar / prepare / init |
| --- | ---: | ---: | ---: | ---: |
| `1×1` | `0.004 [0.004, 0.004]` / `0.005 [0.005, 0.005]` | `0.087 [0.085, 0.091]` / `0.108 [0.106, 0.113]` | `0.012 [0.012, 0.012]` / `0.022 [0.021, 0.022]` / `0.031 [0.031, 0.032]` | `0.024 [0.024, 0.025]` / `0.133 [0.131, 0.140]` / `0.202 [0.199, 0.206]` |
| `51×26` | `0.924 [0.923, 0.927]` / `0.931 [0.928, 0.935]` | `1.664 [1.644, 1.668]` / `0.428 [0.417, 0.442]` | `0.989 [0.986, 0.990]` / `1.941 [1.926, 1.960]` / `1.349 [1.342, 1.359]` | `1.004 [0.994, 1.009]` / `2.694 [2.689, 2.707]` / `2.188 [2.176, 2.201]` |
| `3932×26` | `74.228 [73.902, 75.007]` / `74.465 [73.844, 75.589]` | `113.924 [112.215, 115.942]` / `22.227 [22.203, 22.254]` | `73.826 [73.649, 74.485]` / `149.090 [148.536, 150.462]` / `108.068 [106.882, 109.072]` | `75.336 [74.241, 76.722]` / `188.728 [187.545, 190.118]` / `144.354 [143.331, 145.434]` |
| `26×3932` | `66.631 [66.164, 67.125]` / `67.019 [66.641, 67.569]` | `124.220 [123.553, 125.213]` / `19.514 [19.430, 19.609]` | `74.002 [73.828, 74.234]` / `141.165 [140.710, 141.602]` / `94.372 [94.057, 94.568]` | `73.974 [73.848, 74.106]` / `196.743 [196.031, 197.469]` / `147.919 [146.579, 149.389]` |

CPUBackend mede cópias de árvores Python nas duas direções. A camada adiciona sua validação antes da cópia ao preparar uma lista, o que explica por que `Linear.prepare` fica perto da soma de dois percursos em tensores grandes. Em OpenCL, para a preparação de lista grande, a validação de `Linear` acrescenta aproximadamente `74–75 ms` antes de flatten/empacotamento/alocação/transferência.

## Decomposição OpenCL

Tempos em ms, mediana `[Q1, Q3]` entre processos. Os dois campos “sombra” percorrem a estrutura separadamente: validação reproduz shape e checagem numérica; flatten opera sobre a estrutura já validada. São probes diagnósticos e conferidos contra a saída do helper de produção. A produção mede `opencl_flatten_validate` junto, em uma chamada. O array ctypes/lista/reshape também são probes isolados, não somas observadas no mesmo span end-to-end.

| Shape | Validação sombra / flatten sombra | Helper de produção (valida + flatten) | Pack ctypes | Allocate | H2D bloqueante prealocado | D2H bloqueante prealocado | `list(ctypes)` | Reshape |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1×1` | `0.002 / 0.001` | `0.006` | `0.001` | `0.026` | `0.030` | `0.082` | `0.001` | `0.001` |
| `51×26` | `0.476 / 0.463` | `1.140` | `0.271` | `0.033` | `0.083` | `0.085` | `0.215` | `0.021` |
| `3932×26` | `36.906 / 35.153` | `90.989` | `23.029` | `0.066` | `0.368` | `0.379` | `18.052` | `2.448` |
| `26×3932` | `35.548 / 33.541` | `100.264` | `22.898` | `0.070` | `0.397` | `0.389` | `18.198` | `0.313` |

Na matriz grande, flatten e empacotamento são aproximadamente 80–81% e 18–20% do tempo público H2D, respectivamente, usando esses probes separados como aproximação de componentes. A transferência bloqueante e a alocação são pequenas diante deles. Em D2H, `list(ctypes)` é a maior transformação host; o reshape é mais caro com 3932 linhas do que com 26 linhas. Essa leitura é consistente com o código que cria slices/listas por linha, mas o benchmark não atribui causalidade por perfilador de CPU.

As medições aninhadas foram preservadas no JSON como tempos inclusivos e exclusivos. Por exemplo, `memory.copy_to_device` contém os spans de flatten, allocate e write; `allocate` contém `clCreateBuffer` e as validações de device; `copy_to_host` contém read e reshape. **Não somar spans inclusivos entre si.** O span exclusivo de `copy_to_device` também contém o empacotamento ctypes e lógica Python ao redor, por isso foi medido separadamente como probe. O exclusivo de `copy_to_host` inclui alocação host e a materialização `list(ctypes)`; a leitura bloqueante e o reshape têm spans próprios.

## RSS e integridade dos valores

`resource.getrusage(RUSAGE_SELF).ru_maxrss` foi lido antes e depois das operações em cada processo. O delta é o aumento do high-water em relação ao pico já atingido após imports, runtime e criação dos dados; inclui retenção do allocator Python e não mede VRAM. A tabela mostra mediana entre quatro processos, KiB:

| Shape | CPU pico / delta RSS | OpenCL pico / delta RSS |
| --- | ---: | ---: |
| `1×1` | `24,462 / 272` | `146,650 / 0` |
| `51×26` | `24,752 / 496` | `146,762 / 0` |
| `3932×26` | `45,478 / 13,588` | `162,930 / 9,186` |
| `26×3932` | `42,238 / 12,390` | `164,168 / 13,124` |

O pico já inclui imports e, para OpenCL, Rusticl/contexto. Portanto, delta zero nos shapes menores significa apenas que as operações não excederam esse pico anterior; não significa consumo de memória zero.

Validações executadas no harness:

- 32 processos: 4 processos independentes por backend/shape; todos retornaram sucesso.
- `weights`, `bias` e `inputs`: `buffer_from_host` e `buffer_to_host` idênticos à referência em todos os processos; igualdade exata, não apenas dentro da tolerância.
- Pesos e biases inicializados por `Linear(seed)` foram iguais aos do gerador de referência CPU.
- A materialização D2H e os probes de flatten/pack/reshape preservaram valores e shapes.
- Os 16 processos CPU não carregaram `kinera.backends.opencl.api`; esse caminho executou sem biblioteca/runtime OpenCL.
- Aviso `Patched Mesa libclc not detected` apareceu nos 16 processos OpenCL. O gate mede conversão de memória e não avalia kernels, treinamento ou correção numérica de operações OpenCL.

## Gargalos e experimentos posteriores possíveis

As faixas de ganho abaixo classificam apenas a parcela de tempo medida que um experimento poderia investigar; **não são ganhos previstos ou medidos de uma otimização**.

| Experimento posterior | Parcela observada / potencial investigável | Complexidade | Risco numérico | Impacto de API | Compatibilidade CPU-only |
| --- | --- | --- | --- | --- | --- |
| Reutilizar o resultado de validação de `Linear` durante a conversão, mantendo exatamente as mesmas rejeições | Alto em matrizes grandes: validação de lista ~`74–75 ms`; representa ~40% da preparação OpenCL grande. Isso é um teto de parcela investigada, não economia demonstrada. | Média | Baixo se forma, retangularidade, tipos, finitude e conversão forem preservados | Médio: exigiria um contrato interno confiável entre validador e backend | Deve manter caminho CPU sem OpenCL |
| Achatar e converter diretamente para armazenamento contíguo, evitando a lista plana e/ou o segundo pack | Alto: helper combinado ~`91–100 ms` e pack ~`23 ms` nas matrizes grandes | Média/alta | Médio: preservar `float64`, ordem dos valores e validações | Baixo se interno ao OpenCL; médio se aceitar novos tipos de entrada | OpenCL opcional; nenhuma dependência obrigatória pode entrar no caminho CPU |
| Avaliar retorno D2H plano/array além da lista aninhada atual | Alto no percurso D2H: lista ctypes→Python ~`18 ms`, reshape até `2.45 ms` | Média/alta | Baixo para os valores, mas requer conferir representação e consumidores | Alto: a API pública atual entrega listas | Deve preservar exportação em listas no CPU |
| Reutilizar alocação/buffer para sequências de cópias | Baixo para estes tempos: alocação ~`0.07 ms` contra ~`114–124 ms` do upload público grande | Média | Baixo, com risco de lifetime/ownership | Médio | Separável, mas o backend CPU não deve importar OpenCL |

Antes de qualquer mudança, o próximo experimento poderia comparar uma implementação sombra de flatten/pack ou materialização com a atual, verificando valores exatos e repetindo este protocolo. Nenhuma proposta foi implementada ou medida como otimização aqui.

## Reprodução e fingerprints

Com o mesmo ambiente OpenCL:

```bash
RUSTICL_ENABLE=radeonsi RUSTICL_FEATURES=fp64 PYTHONPATH=src:experiments \
  python experiments/analyze_opencl_host_conversions.py \
  --warmups 2 --repetitions 8 --process-repetitions 4 \
  --output experiments/results/opencl-rx550-p3c-host-conversions.json
```

Fingerprint do código testado (SHA-256):

| Arquivo | SHA-256 |
| --- | --- |
| `experiments/analyze_opencl_host_conversions.py` (harness final) | `bf11fef67ee39b1f8432bef90bfc4619a990b5fab9d3871ca0d14ae479fe2856` |
| `experiments/results/opencl-rx550-p3c-host-conversions.json` | `d8d0b5932037320a27f585ebd6c74d7cbfd22ab81c67ac42ee8abbeb93e4adc8` |
| `src/kinera/backends/opencl/memory.py` | `19d67e70eb685c82bcb7311c2448e03ff43e9005900eb0453e1e1a3a873ec5d9` |
| `src/kinera/backends/cpu/backend.py` | `990b9422f425e0136a720bad880dbe5ebc045c275ac762751a0357c8ec9c2ffb` |
| `src/kinera/nn/layers/linear.py` | `41da794a9dd3a34d08b3548cef83da90e217c3a94edc3890e895c103e686ea81` |

Os resultados estruturados completos, inclusive tempos inclusivos/exclusivos, spans dos vetores, amostras por processo, ordem de execução e RSS, estão em [opencl-rx550-p3c-host-conversions.json](../experiments/results/opencl-rx550-p3c-host-conversions.json). Os relatórios e resultados P3/P3b existentes não foram editados. `py_compile`, integridade das 32 saídas do harness e `git diff --check` foram usados para validar esta entrega; nenhum benchmark histórico, teste de operação neuronal, piloto científico ou treinamento foi executado.

O JSON permanece byte a byte igual ao resultado experimental original, produzido pelo harness de coleta com SHA-256 `dcf52049d5e48a7a2b4f92957c802f97259b16132a0253d4221be38010454b56`. O harness final apenas substitui o clamp de duração exclusiva por erro explícito quando os filhos excedem o span inclusivo; os dados não foram coletados novamente. Testes stdlib verificam o profiler, estatísticas, estrutura/roundtrips do resultado e os fingerprints P3/P3b/P3c. O ponto de liberação explícita na falha de construção ctypes permanece uma pendência de revisão, sem correção de produção.

# Gate P2 — integração `Linear.forward → backward → step` na RX 550

**Resultado: PASS.** A fixture executou o ciclo pela camada `Linear` em
OpenCL, comparando cada etapa e os parâmetros finais com uma camada CPU criada
com o mesmo seed. Os testes rodaram na AMD Radeon RX 550 por Mesa Rusticl, em
`float64`, com `atol=rtol=1e-12`. O maior erro absoluto observado foi `0.0`.

## Identificação do código e ambiente

| Campo | Valor |
| --- | --- |
| HEAD base | `5ab05239ddb40efe1862e4efac6a499373ce9121` |
| SHA-256 final de fontes, probe e testes auditados | `2c80713f83193bfcd75739325fbae6658e0fe5a09aeae82ee3694309b988edde` |
| SHA-256 do wheel validado | `1daa24f8a1a3156df04e64a4c83ba29add897daf1162c13f9b0cd5e7bf373d88` |
| Worktree | `codigo`, sujo; fixture e relatórios ainda não commitados |
| Sistema / kernel | Fedora Linux 45 Workstation / `7.2.11-300.fc45.x86_64` |
| Python | `3.15.0` |
| GPU / PCI | AMD Radeon RX 550 / 550 Series / `0000:03:00.0` |
| Driver | Mesa Rusticl `26.2.6` |
| OpenCL | 3.1; OpenCL C 1.2; `cl_khr_fp64`; `CL_DEVICE_DOUBLE_FP_CONFIG=0x3f` |
| Tipo / tolerâncias | `float64`; `atol=1e-12`, `rtol=1e-12` |

O fingerprint atual inclui `pyproject.toml`, seleção de backend, fontes e
kernels OpenCL, todos os testes dos gates OpenCL e o probe P0:

```bash
sha256sum pyproject.toml src/kinera/backends/__init__.py \
  src/kinera/backends/opencl/*.py \
  src/kinera/backends/opencl/kernels/*.cl \
  tests/test_opencl_scaffold.py tests/test_opencl_forward.py \
  tests/test_opencl_backward.py tests/test_opencl_step.py \
  tests/test_opencl_integration.py experiments/opencl_rx550_preflight.c | sha256sum
```

## Fixture e equivalência

CPU e OpenCL criaram cada camada com o mesmo seed e dtype. Os pesos e biases
iniciais foram exportados explicitamente da GPU e coincidiram exatamente com
a inicialização CPU. Cada ciclo forneceu à CPU e à GPU os mesmos vetores
determinísticos de entrada e `grad_output`, executou `forward`, `backward` e
`step` com taxa alternada `0.125` / `0.03125`, e comparou saída, cada gradiente
e pesos/biases finais. Cada shape executou dois ciclos sobre o estado da mesma
camada, sem recriar seus parâmetros entre os ciclos.

| Shape dos pesos | Ciclos | Saída | `grad_weights` | `grad_bias` | `grad_inputs` | Pesos finais | Biases finais |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1×1` | 2 | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` |
| `51×26` | 2 | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` |
| `3932×26` | 2 | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` |

Os valores da tabela são erros absolutos máximos por etapa, separados por
ciclo. Cada comparação passou sem usar tolerância acima de `1e-12 +
1e-12·abs(referência)`. Os valores reportados foram os mesmos nos dois ciclos.
O maior shape também passou sem exceder o limite de alocação observado para a
RX 550.

## Residência, transferências e lifecycle

Antes de cada ciclo, a fixture criou explicitamente dois buffers FP64 no
device: entrada e `grad_output` (duas chamadas H2D). Durante as chamadas de
`Linear.forward`, `Linear.backward` e `Linear.step`, instrumentou a API e
confirmou zero chamadas de leitura ou escrita de buffer. Saídas, gradientes e
parâmetros permaneceram como buffers OpenCL residentes. Depois do ciclo, seis
leituras D2H explícitas exportaram saída, três gradientes e parâmetros para a
comparação com CPU.

Os buffers examinados tinham device `opencl`, device id `0`, dtype `float64`,
ownership próprio e o mesmo manager OpenCL da camada. No ciclo seguinte, os
gradientes anteriores foram fechados quando a camada substituiu seu estado.
Como a fixture ainda retinha referências Python aos buffers anteriores de
entrada e saída, fechou-os explicitamente após a substituição. `Linear.close()`
fechou parâmetros e estado corrente; uma segunda chamada foi segura, e
`forward`, `backward` e `step` após o fechamento falharam com erro explícito.

## Falhas injetadas e recuperação

A fixture injetou uma falha em `enqueue` durante `backward`. O backend
propagou o erro, liberou os três buffers de gradiente alocados, não transferiu
dados e deixou a camada sem gradientes parciais; um novo `backward` foi
executado com sucesso.

Também injetou falha de `enqueue` durante `step`. O kernel criado foi liberado,
pesos e biases permaneceram idênticos ao snapshot anterior e o estado de
backward continuou válido. Repetir `step` após remover a falha concluiu e
coincidiu com a camada CPU. A injeção valida a recuperação quando a chamada de
enqueue falha antes de submeter o kernel; não simula falhas assíncronas do
device durante a execução de um kernel já aceito.

Não foi encontrada divergência numérica e nenhuma correção de kernel foi
necessária. CPU, CUDA e protocolos permaneceram inalterados.

## Auditoria pré-PR e empacotamento

A revisão encontrou um caso de erro escondido na enumeração: um status OpenCL
de falha com contador de plataformas ou devices zerado podia ser tratado como
resultado vazio. A API agora reconhece primeiro os status definidos para
“nenhuma plataforma/device”; outros status são propagados como
`OpenCLApiError`. Testes CPU-only cobrem os dois caminhos. As docstrings que
ainda descreviam apenas o gate P1 foram atualizadas; não foi necessário mudar
os kernels ou a matemática.

O wheel `kinera-0.1.0-py3-none-any.whl` contém os três recursos
`kinera/backends/opencl/kernels/{linear_forward,linear_backward,linear_step}.cl`.
Os kernels são lidos com `importlib.resources`; todos foram abertos a partir
do wheel instalado. O SHA-256 do wheel está na tabela acima. O metadata não
declara OpenCL como dependência obrigatória; `pytest>=8.0` permanece somente
no extra `.[test]`. Uma venv CPU-only instalou o wheel com `--no-deps`,
executou uma chamada CPU enquanto `ctypes.util.find_library` estava bloqueado,
e confirmou que nenhum módulo OpenCL foi importado.

Os testes foram executados numa venv isolada com Python 3.15.0 e pytest 9.2.0.
A regressão ampla passou com `603 passed, 38 skipped`; o arquivo
`tests/test_expanded_replay_pilot.py` foi excluído deliberadamente por conter
execuções do piloto e estes dois testes que lançam benchmark: `test_cpu_vs_cuda_microbenchmark_separates_init_and_verifies_parity` e
`test_cli_runner_seeds_resume_and_verification` (este último passa
`--benchmark-backends`). Os 38 skips incluem gates de hardware CUDA/OpenCL sem
opt-in. Os dois módulos que antes falhavam ao importar pytest foram incluídos
e passaram com o extra instalado. Os gates OpenCL CPU-only passaram; no GPU,
12 gates passaram e cobriram os mesmos shapes e ciclos da seção acima.

O aviso Rusticl `Patched Mesa libclc not detected` continua sendo uma ressalva
de runtime. O preflight registrou FP64 em `cl_khr_fp64` e
`CL_DEVICE_DOUBLE_FP_CONFIG=0x3f`; esta aprovação é para a RX 550 `0000:03:00.0`
e Mesa Rusticl 26.2.6, não uma garantia para todo device OpenCL. As falhas
injetadas foram antes da submissão no enqueue; falhas assíncronas depois de o
driver aceitar um kernel não foram validadas.

O probe C P0 foi recompilado com `gcc -std=c11 -O2 -Wall -Wextra -Werror`
e `-lOpenCL`, e executado com `--pci-bdf 0000:03:00.0`; retornou
`GO_P0_SMOKE_PASS`, selecionou a RX 550 e obteve erro `float64=0` e `int32`
exato nos 16 valores. O código do probe integra o fingerprint final acima.

## Testes executados

Gates GPU existentes e fixture integrada, na RX 550:

```bash
PYTHONPATH=src \
RUSTICL_ENABLE=radeonsi \
RUSTICL_FEATURES=fp64 \
KINERA_RUN_OPENCL_TESTS=1 \
python -m unittest -v \
  tests.test_opencl_forward tests.test_opencl_backward \
  tests.test_opencl_step tests.test_opencl_integration
```

Resultado: 12 testes GPU passaram. A integração cobriu três shapes em dois
ciclos cada e um caso adicional de falha/retry. Na execução pytest usada na
auditoria pré-PR, foram 12 testes passados e 439823 subtestes passados em
75,47 s; o tempo vem das comparações determinísticas por elemento, não de um
benchmark.

Regressões CPU-only relevantes:

```bash
PYTHONPATH=src python -m unittest -v \
  tests.test_opencl_scaffold tests.test_opencl_forward \
  tests.test_opencl_backward tests.test_opencl_step \
  tests.test_opencl_integration tests.test_backend_contract tests.test_linear
```

Resultado: 52 testes descobertos, 40 passaram e os 12 testes GPU foram
ignorados por opt-in. O teste de subprocesso confirmou que selecionar CPU
não importa módulos OpenCL.

A CI foi ajustada para executar `python -m pytest -q tests
--ignore=tests/test_expanded_replay_pilot.py`. O arquivo foi excluído porque
contém o piloto científico e os testes que iniciam microbenchmark; o comando
de regressão local usou a mesma exclusão.

Uma tentativa adicional de `python -m unittest discover -s tests` percorreu
658 testes, com 38 skips e 2 erros de importação: os módulos
`test_phase46_metrics` e `test_regime1_identifiability` dependem de `pytest`,
que não está instalado no ambiente. A descoberta global também acionou um
teste preexistente que imprime uma execução CPU vs. CUDA de microbenchmark;
essa execução ocorreu incidentalmente durante a descoberta e não foi usada
como evidência deste gate. A descoberta total, portanto, não é reportada como
PASS.

## Ressalva Rusticl

O runtime imprimiu `Patched Mesa libclc not detected`. Apesar do PASS nos
kernels e casos registrados, o aviso de que o `libclc` upstream pode conter
bugs ou mudanças incompatíveis continua aplicável. Não foram feitos benchmarks
planejados, treinamento ampliado, Embedding ou integração híbrida.

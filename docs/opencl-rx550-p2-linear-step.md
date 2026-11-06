# Gate P2 — `Linear.step` OpenCL na RX 550

**Resultado: PASS.** Em 2026-11-05, `Linear.step` atualizou pesos e biases
diretamente nos buffers FP64 da AMD Radeon RX 550 via Mesa Rusticl. Os
resultados CPU↔OpenCL coincidiram: erro absoluto máximo `0.0` para pesos e
biases, com `atol=rtol=1e-12`.

## Identificação do gate

| Campo | Valor |
| --- | --- |
| HEAD | `5ab05239ddb40efe1862e4efac6a499373ce9121` |
| SHA-256 do snapshot deste gate | `0860430b39750422f7edc1fef5b1bbe122ee01d699cc7749ad3c2f15336b2eda` |
| Estado do worktree | Sujo; implementação e relatórios ainda não commitados |
| Sistema / kernel | Fedora Linux 45 Workstation / `7.2.11-300.fc45.x86_64` |
| Python | `3.15.0` |
| GPU / PCI | AMD Radeon RX 550 / 550 Series / `0000:03:00.0` |
| Driver | Mesa Rusticl `26.2.6` (`mesa-dri-drivers-26.2.6-1.fc45`) |
| OpenCL | 3.1; OpenCL C 1.2; `cl_khr_fp64`; `CL_DEVICE_DOUBLE_FP_CONFIG=0x3f` |
| FP64 / tolerância | `float64`; `atol=1e-12`, `rtol=1e-12` |

Este fingerprint identifica o snapshot usado no gate step. O fingerprint
atual da auditoria pré-PR, que também inclui o probe P0, está no relatório
integrado. O hash deste gate foi calculado com:

```bash
sha256sum pyproject.toml src/kinera/backends/__init__.py \
  src/kinera/backends/opencl/*.py \
  src/kinera/backends/opencl/kernels/*.cl \
  tests/test_opencl_scaffold.py tests/test_opencl_forward.py \
  tests/test_opencl_backward.py tests/test_opencl_step.py | sha256sum
```

## Implementação e semântica

O kernel atualiza os buffers existentes in-place, calculando em cada elemento
`parameter = parameter - learning_rate * gradient`. Pesos e biases são
atualizados por work items independentes. `FP_CONTRACT OFF` impede que o
produto e a subtração virem uma FMA, preservando o arredondamento separado da
referência CPU. `learning_rate=0` devolve os mesmos buffers sem lançar kernel.
Não há transferência implícita para CPU; cópias para comparar resultados nos
testes usam `buffer_to_host` explicitamente após a chamada.

Antes de lançar o kernel, o backend valida taxa, buffer aberto, tipo OpenCL,
dtype `float64`, rank, shapes, device selecionado e ownership. Taxas bool e
tipos não numéricos são rejeitados; taxas negativas, NaN, infinitas e inteiros
que não cabem em `float64` também falham antes do lançamento. Falhas ao criar
ou configurar o kernel e falhas de enqueue são propagadas; o recurso kernel é
liberado quando foi criado. Testes com API OpenCL falsa verificam que essas
falhas não iniciam atualização e não alteram pesos ou biases.

## Gates CPU↔OpenCL

O primeiro caso usa pesos `2×3`, biases `2`, gradientes assinados, zeros e
magnitude `1e16`, com taxa `1 + 2^-27`. Para pesos e biases, erro máximo foi
`0.0`. O witness FP64 distingue o arredondamento separado: para
`a=1+2^-27`, `b=1-2^-27`, `a*b` arredonda para `1.0`, então `1.0-a*b=0.0`;
uma FMA produziria `2^-54`. CPU e OpenCL retornaram exatamente `0.0` nos
elementos discriminantes.

O mesmo par de buffers residentes foi atualizado consecutivamente com taxas
`0.1`, `0.025` e `0.1`, comparando pesos e biases após cada atualização. Uma
chamada posterior com taxa `0.0` foi comparada ao CPU e não alterou nenhum
valor. Erros máximos observados:

| Caso | Pesos | Biases |
| --- | ---: | ---: |
| Atualização discriminante (taxa `1 + 2^-27`) | `0.0` | `0.0` |
| Consecutiva 1 (`0.1`) | `0.0` | `0.0` |
| Consecutiva 2 (`0.025`) | `0.0` | `0.0` |
| Consecutiva 3 (`0.1`) | `0.0` | `0.0` |
| Taxa zero | `0.0` | `0.0` |

O teste instrumentou as rotinas de leitura e escrita OpenCL durante a chamada
de atualização e confirmou zero transferências. As leituras para comparação
foram explícitas e ocorreram depois. O teste CPU-only validou taxas inválidas,
shapes, dtype, ownership, buffer fechado, taxa zero e limpeza do kernel nas
falhas injetadas.

## Testes executados

Na RX 550:

```bash
PYTHONPATH=src \
RUSTICL_ENABLE=radeonsi \
RUSTICL_FEATURES=fp64 \
KINERA_RUN_OPENCL_TESTS=1 \
python -m unittest -v tests.test_opencl_step
```

Resultado: dois testes de GPU passaram; erro máximo `0.0` para pesos e biases
em todos os casos. Os gates `Linear.forward` (cinco testes) e
`Linear.backward` (três testes, cinco formatos) também foram reexecutados na
mesma RX 550 e passaram com erro máximo `0.0`.

Suítes CPU-only e regressões:

```bash
PYTHONPATH=src python -m unittest -v \
  tests.test_opencl_scaffold tests.test_opencl_forward \
  tests.test_opencl_backward tests.test_opencl_step \
  tests.test_backend_contract tests.test_linear
```

Resultado: 50 testes descobertos, 40 passaram e os 10 gates exclusivos de
GPU foram ignorados por opt-in.

## Ressalva Rusticl e escopo

O runtime emitiu `Patched Mesa libclc not detected`: o programa compilou e
executou, mas o aviso indica que o `libclc` upstream pode conter bugs ou
mudanças incompatíveis. Portanto, o PASS é específico ao kernel, aos casos e à
RX 550 registrados aqui. Não foram executados benchmarks nem a fixture
integrada `forward → backward → step` durante este gate isolado. A fixture foi
autorizada depois e está registrada em
[`opencl-rx550-p2-linear-integration.md`](opencl-rx550-p2-linear-integration.md).
Embedding, scheduler e treinamento ampliado não fazem parte deste gate.

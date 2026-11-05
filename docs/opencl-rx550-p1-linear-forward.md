# Gate P1 — `Linear.forward` OpenCL na RX 550

**Resultado: PASS.** Em 2026-11-05, o primeiro kernel de `Linear.forward`
executou na AMD Radeon RX 550 (`0000:03:00.0`) via Mesa Rusticl e coincidiu
com a referência CPU nos dois casos determinísticos: erro absoluto máximo
`0`, com `atol=rtol=1e-12`.

Este gate cobre o scaffold, transferência explícita de buffers e
`Linear.forward`. O backward foi validado separadamente no
[relatório P2](opencl-rx550-p2-linear-backward.md). Este trabalho não mede
desempenho nem cobre treino completo ou execução híbrida. `Linear.step` foi
validado em gate próprio no [relatório P2 de step](opencl-rx550-p2-linear-step.md).

## Casos medidos

| Caso | Dimensões | Detalhe | Erro absoluto máximo |
| --- | --- | --- | ---: |
| Matriz pequena assinada | `3 × 3` | pesos e entradas positivos, negativos e zero | `0` |
| Soma com cancelamento | `1 × 3` | `[1e16, 1, -1e16] · [1, 1, 1]` | `0` |

O segundo caso protege a equivalência da soma interna com a referência Python:
o kernel usa acumulação compensada somente quando o interpretador sinaliza que
`sum((1e16, 1.0, -1e16)) == 1.0`. O cálculo de CPU não foi modificado.

## Auditoria numérica do forward

Auditoria em Fedora 45, Python 3.15.0, Mesa Rusticl 26.2.6, OpenCL 3.1,
OpenCL C 1.2, GPU `0000:03:00.0`, `CL_DEVICE_DOUBLE_FP_CONFIG=0x3f`. O HEAD do
checkout era `5ab05239ddb40efe1862e4efac6a499373ce9121`; o worktree estava sujo
e os arquivos P1/P2 testados ainda não estavam commitados. O forward foi
reexecutado na revisão do gate P2 step, em 2026-11-05, e passou novamente nos
cinco testes da RX 550. O fingerprint `0860430b39750422f7edc1fef5b1bbe122ee01d699cc7749ad3c2f15336b2eda`
identifica aquele snapshot histórico; o fingerprint da auditoria pré-PR atual
está no relatório integrado P2.

No CPython 3.15, o `CPUBackend` multiplica cada par como `float` e depois aplica
`sum()` aos produtos em ordem. A implementação CPython usa `hi`/`lo` com a
variante Neumaier: `next = hi + value`; acumula o erro de arredondamento em
`lo`; e retorna `hi + lo` quando `lo` é finito e diferente de zero. O kernel
OpenCL faz a mesma sequência. O teste de capacidade existente seleciona essa
sequência somente quando `sum((1e16, 1.0, -1e16)) == 1.0`; em runtimes que dão
`0.0`, usa a soma sequencial. A documentação Python registra a mudança no
algoritmo de `sum()` a partir do Python 3.12, e a fonte do CPython documenta a
implementação Neumaier ([documentação](https://docs.python.org/3.15/library/functions.html#sum),
[fonte CPython 3.15](https://github.com/python/cpython/blob/3.15/Python/bltinmodule.c)).

O `#pragma OPENCL FP_CONTRACT OFF` separa o arredondamento da multiplicação e
da soma como no oracle Python. Um caso discriminante usa `a=1+2^-27`,
`b=1-2^-27` e `c=-1`: produto arredondado seguido de soma dá exatamente `0.0`,
enquanto FMA produz `-2^-54`. RX 550 retornou exatamente `0.0`, igual ao CPU.
O `CL_DEVICE_DOUBLE_FP_CONFIG=0x3f` reportado pelo dispositivo inclui
`CL_FP_ROUND_TO_NEAREST` e suporte a subnormais; o gate também confirmou o
menor subnormal `5e-324` sem divergência. Cada redução percorreu os termos na
mesma ordem crescente que `sum(map(operator.mul, ...))`, sem árvore paralela
de redução. Os detalhes de configuração FP64 estão no
[relatório P0](opencl-rx550-preflight.md).

O gate ampliado cobre 52 saídas em dimensões `1×1`, `2×3`, `3×3`, `7×19`,
`31×5`, além de vetores de cancelamento e magnitude extrema. Inclui sinais
misto, zeros, produtos da ordem de `1e300`, cancelamentos `1e16`, `1e100` e
`1e300`, e o subnormal `5e-324`. Todas as saídas coincidiram exatamente com o
CPU: erro absoluto máximo `0.0`, mantendo `atol=rtol=1e-12`. Não houve
divergência que justificasse corrigir o kernel. A evidência cobre estes casos
determinísticos na RX 550, sem afirmar equivalência para toda entrada possível.

## Reproduzir

Na raiz do checkout, com o driver Mesa Rusticl e FP64 habilitados antes de iniciar
Python:

```bash
PYTHONPATH=src \
RUSTICL_ENABLE=radeonsi \
RUSTICL_FEATURES=fp64 \
KINERA_RUN_OPENCL_TESTS=1 \
python -m unittest -v tests.test_opencl_forward
```

Os testes CPU-only do scaffold podem rodar sem placa OpenCL:

```bash
PYTHONPATH=src python -m unittest -v tests.test_opencl_scaffold
```

O runtime exibiu o aviso `Patched Mesa libclc not detected` durante o gate. O
kernel compilou e os casos de auditoria passaram; a ressalva do runtime continua
válida para kernels e operações ainda não medidos.

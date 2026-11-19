# Gate P2 — `Linear.backward` OpenCL na RX 550

**Resultado: PASS.** Revisão complementar executada em 2026-11-05 na GPU AMD
Radeon RX 550 via Mesa Rusticl. Foram comparados separadamente
`grad_weights`, `grad_bias` e `grad_inputs` com `CPUBackend`; erro absoluto
máximo `0.0`, com `atol=rtol=1e-12`, nos cinco formatos testados.

## Identificação

| Campo | Valor |
| --- | --- |
| HEAD | `5ab05239ddb40efe1862e4efac6a499373ce9121` |
| SHA-256 do snapshot deste gate | `0860430b39750422f7edc1fef5b1bbe122ee01d699cc7749ad3c2f15336b2eda` |
| Worktree | Sujo; as mudanças OpenCL ainda não estão commitadas |
| Sistema / kernel | Fedora Linux 45 Workstation / `7.2.11-300.fc45.x86_64` |
| Python | `3.15.0` |
| GPU / PCI | AMD Radeon RX 550 / 550 Series / `0000:03:00.0` |
| Mesa | Rusticl `26.2.6` |
| OpenCL | 3.1; OpenCL C 1.2; `cl_khr_fp64`; `CL_DEVICE_DOUBLE_FP_CONFIG=0x3f` |
| Tolerância | `atol=1e-12`, `rtol=1e-12` |

## Casos e resultados

Cada formato foi executado na GPU e comparado com `CPUBackend.linear_backward`.
Os dados contêm sinais positivos/negativos, zeros e cancelamento. O caso
`4×3` e o formato head-like `51×26` incluem a sequência `[1e16, 1, -1e16]`
na redução de `grad_inputs[0]`; ambos retornam `1.0`, conforme a soma
compensada do Python.

| `weights` | `inputs` | `grad_output` | Erro máx. `grad_weights` | `grad_bias` | `grad_inputs` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `4×3` | `3` | `4` | `0.0` | `0.0` | `0.0` |
| `1×1` | `1` | `1` | `0.0` | `0.0` | `0.0` |
| `2×3` | `3` | `2` | `0.0` | `0.0` | `0.0` |
| `3×5` | `5` | `3` | `0.0` | `0.0` | `0.0` |
| `51×26` | `26` | `51` | `0.0` | `0.0` | `0.0` |

O kernel produz os três gradientes em buffers OpenCL residentes. A redução de
`grad_inputs` mantém a ordem crescente da dimensão de saída e usa a política
compensada condicional já auditada para corresponder a `sum()` no runtime
Python. Nenhuma alteração foi feita à implementação CPU ou à matemática.
Testes CPU-only também cobrem shapes, dtype, ownership, buffers fechados e
liberação das saídas parciais após erro de configuração OpenCL.

## Reproduzir

Gate GPU:

```bash
PYTHONPATH=src \
RUSTICL_ENABLE=radeonsi \
RUSTICL_FEATURES=fp64 \
KINERA_RUN_OPENCL_TESTS=1 \
python -m unittest -v tests.test_opencl_backward
```

Suítes CPU-only e regressões relevantes:

```bash
PYTHONPATH=src python -m unittest -v \
  tests.test_opencl_scaffold tests.test_opencl_forward \
  tests.test_opencl_backward tests.test_opencl_step \
  tests.test_backend_contract tests.test_linear
```

O comando CPU-only executou 50 testes: 40 passaram e os 10 testes de GPU
foram ignorados por opt-in, como esperado. O gate de backward na RX 550
executou três métodos de teste cobrindo os cinco formatos da tabela.

Este é o fingerprint do snapshot P2 usado na validação backward. A auditoria
pré-PR acrescentou teste para propagar erros de enumeração OpenCL; o fingerprint
final está registrado no relatório integrado. O hash histórico foi calculado,
sem incluir os relatórios, com:

```bash
sha256sum pyproject.toml src/kinera/backends/__init__.py \
  src/kinera/backends/opencl/*.py \
  src/kinera/backends/opencl/kernels/*.cl \
  tests/test_opencl_scaffold.py tests/test_opencl_forward.py \
  tests/test_opencl_backward.py tests/test_opencl_step.py | sha256sum
```

## Ressalva Rusticl

O runtime imprimiu `Patched Mesa libclc not detected`. O kernel compilou e os
casos passaram, mas a ressalva do driver permanece. O PASS se limita aos
formatos testados nesta RX 550; não cobre treinamento completo nem desempenho.

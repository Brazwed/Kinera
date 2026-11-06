# Issue #78 — Investigação de divisão FP64 na RX 550

## Resultado e decisão

**NO-GO para presumir equivalência de divisão FP64 com divisor dinâmico neste ambiente.** A divergência de `1/3` foi reproduzida tanto passando o divisor como argumento `double` quanto lendo-o de um buffer. O mesmo resultado apareceu no recíproco calculado no dispositivo. O argumento, o divisor carregado e o numerador retornaram os bits de entrada esperados nos casos executados. Isso enfraquece a hipótese de erro simples no transporte, mas não prova a correção de toda a ABI nem identifica a origem da aritmética observada.

O recíproco calculado no host passou nos casos investigados; isso não prova equivalência geral entre multiplicar pelo recíproco arredondado e dividir. A causa interna da divergência continua **inconclusiva**. Não há evidência para atribuí-la a Mesa, Rusticl, libclc, LLVM ou hardware isoladamente.

## Proveniência e ambiente

- Baseline de produção: `35882d8c4ffb4ba46720267e500118d2a1ce624d` (merge PR #77).
- Diagnóstico: `experiments/investigate_opencl_fp64_division.py`, SHA-256 `ef3fb4396ff6098f30bed976c157e5b99e2ee6613bfb5519a6f1574de01a867d`.
- Resultados: `experiments/results/opencl-rx550-fp64-division.json`, SHA-256 `1c73507ef7afb17a68e00122383ae0ce3bd003bc9b9facfe243606be11beb2ff`. O JSON contém o hash da fonte OpenCL efetivamente compilada, entradas, saídas, bits e resultados por variante.
- AMD Radeon RX 550 / 550 Series, PCI `0000:03:00.0`, dispositivo `1002:699f`, `amdgpu`, Mesa/Rusticl `26.2.6-1.fc45`; OpenCL 3.1, OpenCL C 1.2, FP64 habilitado.
- LLVM `23.1.4-1.fc45`; pacotes `libclc22` e `libclc22-spirv` `22.1.8-3.fc45`. Versões de pacotes instalados, sem prova de quais componentes determinaram cada operação.
- Python 3.15.0; Fedora Linux `7.2.11-300.fc45.x86_64`. `RUSTICL_ENABLE=radeonsi`, `RUSTICL_FEATURES=fp64`.
- O runtime emitiu `Patched Mesa libclc not detected`. A associação causal com o desvio não foi testada; não houve comparação com libclc modificado ou outro driver.

## Método e contratos auditados

O diagnóstico é independente de camadas e modelos. `OpenCLRuntime` descobre a GPU e exige FP64. `OpenCLApi.set_kernel_double` cria `ctypes.c_double` e chama `clSetKernelArg` com `sizeof` do valor e seu endereço. O wrapper compila a fonte com `clBuildProgram`; a produção usa opções vazias. Contexto, fila, buffers, programa e kernel são criados explicitamente e liberados em ordem inversa. Leituras e escritas são bloqueantes, e `enqueue_1d` termina com `clFinish`.

O kernel de diagnóstico faz eco do numerador, do divisor escalar e do divisor em buffer antes de escrever as cinco variantes abaixo. O kernel CUDA de mean pooling soma sequencialmente e divide pelo tamanho de contexto convertido em `double`; o OpenCL F2.2 usa recíproco escalar do host. Não se executou CUDA nem se alterou produção.

A comparação usa Python `float` como referência e, para entradas finitas e divisor não zero, uma segunda referência construída com `fractions.Fraction` a partir dos valores binários exatos, arredondada para `float`. As duas referências coincidiram em bits nos casos aplicáveis. Para valores finitos, PASS usa `math.isclose` com `atol=rtol=1e-12`, isto é, erro absoluto <= `max(atol, rtol * max(|esperado|, |observado|))`. O erro relativo é absoluto dividido pelo módulo esperado, indefinido quando esperado é zero. ULP é a distância inteira entre representações binárias ordenadas, não uma medida de ciclos ou utilização do hardware. Para NaN, infinidades e divisão por zero, a referência é classificada explicitamente: Python levanta exceção em divisor zero, portanto a sonda monta o resultado esperado IEEE por sinal; NaNs são comparados por categoria, não por payload. O sinal de zero é registrado à parte. `-cl-fast-relaxed-math` admite pressupostos sobre finitude e sinais de zero; seus casos especiais são apenas observações, não prova de conformidade. [Especificação OpenCL API](https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_API.html).

Cada processo executou 24 pares de entradas em três compilações: opções vazias (como produção), `-cl-opt-disable` e `-cl-fast-relaxed-math`. Cada caso foi repetido três vezes. Dois processos independentes, executados sequencialmente, produziram JSON idêntico byte a byte. Isso estabelece repetição apenas nestas condições. O aviso do libclc apareceu nos dois processos.

## Matriz

Os totais a seguir consideram 17 pares com entradas finitas e divisor não zero por opção; cinco deles usam divisor `3.0` e podem ser comparados à variante literal. Um resultado pode receber PASS por tolerância absoluta apesar de erro relativo grande em valores subnormais. Há sete casos adicionais com entrada especial ou divisor zero.

| Variante | Operação | Padrão | `-cl-opt-disable` | `-cl-fast-relaxed-math` |
|---|---|---:|---:|---:|
| A | `n / 3.0` literal | 5/5 PASS | 5/5 PASS | 5/5 PASS |
| B | `n / scalar` | 9/17 PASS | 9/17 PASS | 9/17 PASS |
| C | `n / buffer[1]` | 9/17 PASS | 9/17 PASS | 9/17 PASS |
| D | `n * (1/d no host)` | 17/17 PASS | 17/17 PASS | 17/17 PASS |
| E | `n * (1/d no dispositivo)` | 9/17 PASS | 9/17 PASS | 9/17 PASS |

As três opções produziram os mesmos bits nos casos realizados. Isso não demonstra que tenham gerado o mesmo código de máquina. Variantes B, C e E falharam em `1/3`, `1/7`, `2/3`, numerador ou divisor negativo com magnitude 3, `1/0.1`, `1e200/7` e `1/1e-300`. `1/8` e `2.5/0.5` passaram, mas não se generaliza a todas as potências de dois.

| Caso, opções padrão | Variante | Esperado / bits | Observado / bits | Erro absoluto | Erro relativo | Distância ULP | Estado |
|---|---|---|---|---:|---:|---:|---|
| `1/3` | A e D | `0.3333333333333333` / `3fd5555555555555` | igual | 0 | 0 | 0 | PASS |
| `1/3` | B, C e E | `0.3333333333333333` / `3fd5555555555555` | `0.3333333320915699` / `3fd5555554000000` | `1.241763414316921e-9` | `3.725290242950763e-9` | 22.369.621 | FAIL |
| `1/7` | B, C e E | `0.14285714285714285` / `3fc2492492492492` | `0.14285714365541935` / `3fc2492494000000` | `7.982765004577175e-10` | `5.587935503204022e-9` | 28.760.942 | FAIL |
| `2/3` | B, C e E | `0.6666666666666666` / `3fe5555555555555` | `0.6666666641831398` / `3fe5555554000000` | `2.483526828633842e-9` | `3.725290242950763e-9` | 22.369.621 | FAIL |
| `1/1e-300` | B, C e E | `9.999999999999999e299` / `7e37e43c8800759b` | `1.0000000272434209e300` / `7e37e43c92ec0000` | `2.7243421014068043e292` | `2.7243421014068046e-8` | 183.208.549 | FAIL |
| mínimo normal / `3` | B, C e E | `7.41691286169067e-309` | `7.41691283406052e-309` | `2.763015e-317` | `3.72529007641731e-9` | 5.592.405 | PASS por `atol` |

A maior diferença relativa observada nos 17 pares finitos da variante D foi `2.18e-16`. Os resultados completos, inclusive bits de todas as linhas, estão no JSON. O caso literal usa `3.0` fixo no código, de modo que o compilador pode tratar essa expressão de forma diferente da divisão dinâmica.

Nos testes separados com `+0`, `-0`, infinito e NaN, as categorias e sinais de zero observados coincidiram com a referência escolhida. Em `0/0`, ambos são NaN, mas os bits de sinal diferiram; payload e sinal de NaN não são aqui critério de igualdade. Isso não valida todas as regras especiais; em particular a opção fast math permite tratamento diferente.

## Hipóteses e limites

- **H1 transporte:** bits de eco corretos para os 24 pares e todos os modos; falha também com divisor em buffer. Não demonstra ausência de toda falha de ABI.
- **H2 compilação:** desvio presente com opções vazias e `-cl-opt-disable`; também com fast math. Não se inspecionou IR ou código de máquina, nem se provou que a flag desative toda transformação interna.
- **H3 runtime/bibliotecas:** versões e aviso documentados; sem controle cruzado com outra versão, causa inconclusiva.
- **H4 precisão:** referência Python e racional coincidem nos casos aplicáveis; B/C/E falham acima da tolerância em oito pares. A tolerância absoluta mascara erro relativo no caso subnormal citado.
- **H5 denominador:** 3, 7, 0.1 e `1e-300` produziram falhas sob as condições medidas; alguns divisores exatos passaram. A matriz não caracteriza o espaço completo de valores.

A observação é específica à combinação de hardware, runtime, kernel e versões registradas. A semelhança de bits ou resultados não identifica uma instrução, implementação de divisão ou biblioteca responsável. Não se realizou comparação de IR, libclc alternativo, outra GPU ou outro runtime.

## Impacto e próximos gates

O risco de divisão dinâmica é demonstrado diretamente nesta sonda. Mean pooling F2.2 mantém sua solução e seus gates previamente registrados; o experimento não garante equivalência para todas as entradas. `mean_pool_backward`, normalizações, softmax, cross-entropy e BCE podem requerer divisões ou outras funções sensíveis conforme sua implementação futura: isso é risco arquitetural, não falha já observada nessas operações. Esta investigação não valida `exp`, `log`, softmax, BCE ou cross-entropy.

Para F2.3a, **NO-GO para implementação que dependa implicitamente da precisão de divisão dinâmica neste ambiente sem um gate próprio**. Antes de decidir o gate funcional, especificar a matemática efetivamente necessária e testar divisão, `exp` e `log` separados e em composições relevantes contra CPU e referência de maior precisão, com extremos, estabilidade, erros e valores especiais. Um teste de outra versão de runtime/libclc ou de um segundo dispositivo, e inspeção de IR, ajudaria a identificar a origem; nenhum deles autoriza por si só uma substituição de produção. Recíproco no host é alternativa somente quando o denominador for conhecido no host e a diferença de arredondamento for validada para o contrato; outras técnicas exigem avaliação separada.

## Reprodução e validação

```bash
RUSTICL_ENABLE=radeonsi RUSTICL_FEATURES=fp64 PYTHONPATH=src \
  python experiments/investigate_opencl_fp64_division.py --output /tmp/kinera-fp64.json
PYTHONPATH=src python -m unittest discover -s tests -p test_opencl_fp64_division_investigation.py -v
```

A sonda exige GPU OpenCL FP64; falha explicitamente se não estiver disponível. A saída é escrita somente após conclusão, sem modificar artefatos anteriores. Os quatro testes CPU-only verificam cálculos de referência, distância de ULP, valores especiais e integridade de cada registro publicado. `py_compile` e `git diff --check` passaram. Nenhum treinamento, benchmark ou kernel de produção foi executado ou modificado nesta investigação.

# F2.3a-P0 — auditoria matemática de cross-entropy FP64

## 1. Resumo executivo

**Parecer: NO-GO para avançar diretamente com um forward F2.3a baseado nas funções matemáticas OpenCL avaliadas nesta RX 550.** O `exp` coincidiu bit a bit com Python em todos os argumentos amostrados; a soma sequencial permaneceu dentro de `atol=rtol=1e-12`; porém `log` falhou a tolerância em denominadores que surgem da própria matriz de cross-entropy. A loss composta também falhou em 3 dos 18 casos por compilação.

O NO-GO se limita à abordagem avaliada — built-ins OpenCL `exp`/`log`, acumulação sequencial FP64 e composição direta —, a esta RX 550/Rusticl e às entradas testadas. Não demonstra impossibilidade de implementar F2.3a por outra abordagem, nem se aplica automaticamente a outro dispositivo/runtime. Nenhum kernel funcional foi implementado e nada foi alterado em produção.

A referência independente também revelou uma limitação distinta do próprio CPU oracle: com logits iguais de magnitude `1e16` ou `1e300`, a expressão histórica produz loss zero, enquanto o cálculo de maior precisão produz `ln(2)`. O OpenCL reproduz o CPU nesses casos, mas isso não é precisão matemática adequada. Como o contrato não declara limite superior de magnitude para logits, essa questão precisa ser delimitada antes de qualquer gate funcional.

## 2. Proveniência e escopo auditado

- Issue #82: [F2.3a-P0](https://github.com/Brazwed/Kinera/issues/82).
- Roadmap #61: [F2 OpenCL](https://github.com/Brazwed/Kinera/issues/61); mantém F2.3a-P0, F2.3b, BCE e integração como gates separados.
- Investigação #78: [divisão FP64 dinâmica](https://github.com/Brazwed/Kinera/issues/78). Seu NO-GO para divisão dinâmica não foi usado como evidência sobre `exp` ou `log`.
- PR #80 integrado: [PR #80](https://github.com/Brazwed/Kinera/pull/80), merge `7bff9617a340faab342f5c844464a94d0acec6d0`.
- Proveniência da investigação #78 integrada em #80: script SHA-256 `ef3fb4396ff6098f30bed976c157e5b99e2ee6613bfb5519a6f1574de01a867d`; JSON SHA-256 `1c73507ef7afb17a68e00122383ae0ce3bd003bc9b9facfe243606be11beb2ff`. O artefato anterior mediu divisão dinâmica, inclusive `1/3`; não é resultado desta auditoria de CE.
- `main` no início, branch-base e merge-base: `7bff9617a340faab342f5c844464a94d0acec6d0`.
- Branch de auditoria: `research/issue-82-f2-3a-p0`.

Foram lidos os corpos das Issues #82, #78 e #61 e o diagnóstico/report/JSON já integrados em #80. A evidência #78 continua dizendo que a causa da divisão divergente é inconclusiva. Seu diagnóstico anterior registrou divisor escalar e em buffer com bits de entrada preservados, mas não mediu a composição da CE.

O relatório #78 registra para `1/3` desvio relativo `3.725290242950763e-9`, oito falhas entre 17 casos finitos com divisor dinâmico e mesmos bits nas opções de compilação testadas. Seu parecer é NO-GO para presumir precisão da divisão nesse ambiente e inconclusivo quanto à causa. Esses valores contextualizam o controle e não foram usados como evidência negativa para E1/E2/E4.

Arquivos de produção inspecionados:

- `src/kinera/backends/cpu/ops/probability.py:13-18` e `src/kinera/backends/cpu/backend.py:215-216`.
- `src/kinera/nn/losses/multitask.py:52-92`.
- `src/kinera/backends/cuda/backend.py:20-21,372-410` e `src/kinera/backends/cuda/kernels/losses.cu:12-68`.
- `src/kinera/backends/opencl/backend.py:1168-1182`, `src/kinera/backends/opencl/runtime.py:53-112` e `src/kinera/backends/opencl/api.py:344-500`.
- `docs/opencl-f2-capability-audit.md:135-138` para os gates futuros F2.3a/b e F2.4a/b.

### Sequência contratual observada

O `CPUBackend.cross_entropy` executa, nesta ordem:

```python
maximum = max(logits)
log_sum_exp = maximum + math.log(
    sum(math.exp(value - maximum) for value in logits)
)
loss = log_sum_exp - logits[target_id]
```

Não há divisão explícita no forward. O avaliador `MultiTaskLoss.forward` usa essa chamada no caminho host; o caminho residente chama `cross_entropy_forward_buffer`. O `OpenCLBackend` atual falha explicitamente para CE. O CUDA tem um kernel separado que calcula o máximo, soma exponenciais sequencialmente (com compensação Neumaier selecionada por um probe do `sum()` Python), chama `log` e forma `maximum + log(denominator) - logits[target]`; CUDA não foi executado nesta auditoria.

O diagnóstico criou kernels experimentais independentes por meio de `OpenCLRuntime` e `OpenCLApi`; não importou uma operação de CE de produção. A API fez uploads e leituras bloqueantes explícitos e `enqueue_1d` chama `clFinish`. O kernel recebeu logits por buffer, mais `double` e `int` escalares de controle para verificar a passagem de argumentos. Os intermediários foram copiados ao host porque são objeto da auditoria; isso não valida nem propõe D2H de logits numa operação funcional futura.

## 3. Ambiente e método

| Item | Ambiente observado |
|---|---|
| GPU | AMD Radeon RX 550 / 550 Series, PCI `0000:03:00.0`, vendor/device `1002:699f` |
| Runtime/driver | Mesa Rusticl `26.2.6-1.fc45`; plataforma Mesa/X.org; driver string identifica `radeonsi`, `polaris12`, ACO e DRM 3.64 |
| OpenCL | OpenCL 3.1; OpenCL C 1.2; FP64 presente (`double_fp_config=63`) |
| LLVM/libclc instalados | `llvm-libs 23.1.4-1.fc45`; `libclc22` e `libclc22-spirv 22.1.8-3.fc45`; o pacote `llvm` e `llvm-config` não estavam instalados |
| Python | CPython 3.15.0, GCC 16.2.1, Fedora 45; kernel `7.2.11-300.fc45.x86_64` |
| Variáveis | `RUSTICL_ENABLE=radeonsi`, `RUSTICL_FEATURES=fp64` |
| Aviso | `Patched Mesa libclc not detected` apareceu em cada um dos três processos |

O aviso foi registrado, mas não se atribui a ele a causa dos resultados: não houve controle com outro libclc, driver, LLVM ou hardware. As versões de pacotes identificam o que estava instalado; não provam isoladamente qual componente originou cada resultado.

Cada uma das três repetições foi executada num processo Python separado. Em cada processo foram criados novos contextos/programas para as opções `""` (padrão), `-cl-opt-disable` e `-cl-fast-relaxed-math`. Essa última opção é apenas diagnóstica e não é recomendação para produção. As saídas foram bit a bit estáveis entre processos e idênticas entre as três opções. Isso não demonstra identidade do código de máquina gerado.

Referências:

- Contrato/oracle: `CPUBackend.cross_entropy` no interpretador ativo.
- Referência independente: `decimal.Decimal` com precisão 100/120 dígitos; floats de entrada convertidos por `Decimal.from_float`, preservando o valor binário. A CE de alta precisão cancela `maximum - target` antes de somar o `ln`, para não repetir o cancelamento sob investigação.
- Para resultados finitos, PASS usa `erro <= max(1e-12, 1e-12 * max(|esperado|, |observado|))`. Erro relativo é indefinido quando o esperado é zero. Distâncias ULP comparam representações binárias ordenadas e são registradas mesmo quando a tolerância passa.
- Um resultado não finito igual dos dois lados é marcado `SPECIAL_MATCH`, não PASS numérico. A tolerância finita não se aplica a esses casos.

### Matriz numérica executada

Por opção de compilação, nas três execuções independentes:

- E1: 17 argumentos em `{0, -0, -epsilon, -1, -10, -100, -700, região de underflow, -1e6}`; 51 observações GPU por opção.
- E2: 31 entradas positivas, incluindo vizinhos de 1, valores subnormais/pequenos, magnitudes maiores e todos os denominadores da matriz CE; 93 observações por opção. Das 31, 25 são `>=1` e compatíveis com uma soma de exponenciais deslocadas; seis abaixo de 1 são sondas auxiliares de sensibilidade do `log`, fora do domínio de denominador da CE.
- E3: oito sequências positivas de 1 a 1000 termos; acumulação OpenCL sequencial e ordem reversa comparadas a `sum()` ativo, `math.fsum` e soma Decimal.
- E4: 18 vetores de logits (cardinalidade 1, 2 e múltiplas classes), empate e zeros com sinais, diferenças pequenas, alvos no início/meio/fim, magnitudes extremas, underflow e cancelamento final; máximos, diferenças, `exp`, somas parciais, `log`, log-sum-exp e loss foram capturados.
- E5: bits das entradas de buffer, do argumento `double` e do alvo inteiro foram ecoados pelo dispositivo.
- E6: compilação padrão, sem otimização e fast relaxed, construídas separadamente.
- E7: dependências de softmax/divisão, CE gradient e BCE mapeadas, sem executar nem aprovar esses gates.

## 4. Resultados E1–E7

| Gate | Status | Evidência observada |
|---|---|---|
| **E1 — `exp`** | **PASS na amostra** | 51/51 por opção contra `math.exp` e Decimal arredondado: erro absoluto/relativo 0, 0 ULP. Inclui underflow e argumentos não positivos usados pela CE. Isso é evidência amostral desta máquina, não garantia universal. |
| **E2 — `log`** | **FAIL** | No conjunto completo: 78 PASS / 15 FAIL por opção. Entre 25 entradas do domínio `>=1`: cinco denominadores da própria CE falharam em cada processo; 15/75 por opção. Erro absoluto máximo do conjunto `1.5734258340671659e-10`; relativo máximo `1.5646942742742792e-11`; distância máxima `114892` ULP. As falhas CE também falham contra Decimal FP64. |
| **E3 — soma** | **PASS dentro da tolerância, com ordem relevante** | 24/24 sequências por opção, tanto ordem direta como reversa, passaram contra `sum()` Python. Erro absoluto máximo `1.4210854715202004e-14`, relativo `1.1102230246251442e-14`, máximo 50 ULP. Algumas sequências mudaram bits com a ordem. Nas somas que entram na CE, o erro máximo foi 1 ULP (`4.440892098500626e-16`) e as três falhas compostas abaixo tinham denominadores iguais em bits. Python 3.15 ativa soma compensada segundo o probe já usado pelo Kinera (`sum((1e16, 1, -1e16)) == 1`). |
| **E4 — CE completa** | **FAIL** | Por opção: 42 PASS finitos, 9 FAIL e 3 `SPECIAL_MATCH` dos 54 resultados. Três casos finitos falham à tolerância; ver tabela abaixo. Os dois casos de cancelamento extremo reproduzem o CPU em `0.0`, mas diferem de `ln(2)` por `0.6931471805599453`. Um loss matematicamente acima do máximo FP64 gera `inf` nos dois e fica como `SPECIAL_MATCH`. |
| **E5 — dados/IO** | **PASS** | Todas as entradas ecoadas mantiveram bits; argumentos escalares `double` e `int` foram recuperados sem alteração em cada trial/opção. Isso testa o caminho da sonda/API, não toda a ABI de uma operação de produção. |
| **E6 — opções** | **PASS quanto à repetibilidade observada** | Para todas as observações, opções padrão, `-cl-opt-disable` e fast relaxed produziram os mesmos bits; também houve estabilidade bit a bit nos três processos. Não se infere causa nem equivalência de código gerado. |
| **E7 — dependências futuras** | **OUT_OF_SCOPE** | F2.3b (divisão/softmax/gradiente) e F2.4 (BCE, incluindo `log1p`) não foram executados. A #78 continua independente. |

### Falhas representativas de E4

| Caso | CPU | OpenCL | Erro absoluto | Erro relativo | ULP | Resultado |
|---|---:|---:|---:|---:|---:|---|
| `[1,-1,-2,-3]`, alvo 0 | `0.1851824526038126` | `0.18518245260256894` | `1.2436718321851004e-12` | `6.7159269936113545e-12` | 44.808 | FAIL |
| `[1000,999,998]`, alvo 1 | `1.4076059644444285` | `1.4076059644379484` | `6.480149750132114e-12` | `4.6036674423227385e-12` | 29.184 | FAIL |
| `[-1000,-1001,-1002]`, alvo 2 | `2.4076059644444285` | `2.4076059644379484` | `6.480149750132114e-12` | `2.691532520616368e-12` | 14.592 | FAIL |

Nas três falhas, máximo, logits deslocados, `exp` e soma foram iguais em bits; a discrepância aparece no `log` do mesmo denominador. Os denominadores `1.2034379904932109` e `1.5032147244080551` falham isoladamente. No denominador `1.5032147244080551`, `log` difere do CPU em `6.377787187261674e-12`, erro relativo `1.5646942742742792e-11` e 114.892 ULP.

O critério absoluto pode mascarar erro relativo quando `log(x)` está perto de zero: os resultados completos retêm cada erro, bits, ULP e limite usado. Neste conjunto, as falhas dos denominadores CE excedem tanto o limite absoluto `1e-12` quanto o relativo. Em logaritmos de magnitude grande, alguns erros absolutos maiores passam pelo limite relativo; isso está registrado por observação, sem relaxar o critério.

### Cancelamento e valores especiais

- Para logits `[1e16, 1e16]` e `[1e300, 1e300]`, CPU e OpenCL seguem a expressão em ordem e retornam `0.0`: `maximum + ln(2)` arredonda para `maximum` antes da subtração do alvo. Decimal calcula `ln(2)`. O match CPU/OpenCL, portanto, não constitui exatidão matemática nesse extremo.
- Para `[1e308, -1e308]`, alvo 1, os dois caminhos chegam a `inf` porque a loss é maior que o maior `float64` finito. Foi contado como `SPECIAL_MATCH`; não como aprovação da tolerância.
- `atol=1e-12` pode ocultar diferenças relativas quando a resposta é próxima de zero. Comparações com zero registram `relative_error: null`; os bits e erro absoluto permanecem disponíveis.

## 5. Integridade dos artefatos

- Diagnóstico: `experiments/investigate_opencl_fp64_cross_entropy_math.py` — SHA-256 `c928921c35e692e811ecde5d559ddc80a443f544d712a1a65b04edf4ea54c2a0`.
- Fonte OpenCL experimental compilada: SHA-256 `474dd522f8e8fa18f236f7689091efd914cad61b56d8c89fff2a5b521a54c0fc`.
- Resultados: `experiments/results/opencl-rx550-f2-3a-math-audit.json` — SHA-256 `82940bdeb55a4c4e221bcba1a1ad76de43ed102808363254e3c9216d15245f80`.
- O JSON registra SHA do script/fonte, versão e variáveis de ambiente, dispositivo, opções, inputs, outputs, representações hexadecimais, intermediários, comparações e contabilização por opção.
- Para evitar salvar nove cópias de traces idênticas, o JSON mantém os valores/bits completos de uma execução canônica (compilação padrão, processo 1). Cada uma das outras oito combinações de processo/opção mantém hashes SHA-256 de inputs/outputs, contagem de casos, confirmação de eco dos argumentos e confirmação explícita de igualdade bit a bit com a execução canônica. As contagens numéricas agregam as três repetições por opção.

O teste CPU-only recalcula hashes, valida JSON estrito (sem `NaN`/`Infinity` literais), confirma consistência PASS/FAIL/SPECIAL_MATCH, verifica eco de bits e reavalia os resultados CE a partir dos bits serializados.

## 6. Testes e verificações

Executados em Python 3.15.0, Fedora 45:

- Diagnóstico físico: três processos independentes, três opções por processo, na RX 550 acima; todos terminaram com sucesso.
- `PYTHONPATH=src python -m unittest tests.test_opencl_fp64_cross_entropy_math_audit -v`: **5 testes aprovados**.
- Suíte configurada em `.github/workflows/ci.yml`, mesmo comando e cinco exclusões declaradas, com `PYTHONPATH=src`: **802 passed, 61 skipped, 5 deselected, 1 warning, 1101 subtests passed** em 41,81 s. Warning: `PytestRemovedIn10Warning` emitido por `anyio` instalado; não relacionado à alteração.
- `python -m py_compile experiments/investigate_opencl_fp64_cross_entropy_math.py tests/test_opencl_fp64_cross_entropy_math_audit.py`: aprovado.
- `git diff --check`: aprovado.

Os 61 skips e 5 deselections seguem registrados como tais. O teste de hardware executado foi este diagnóstico P0; a suíte normal não habilitou gates opt-in de GPU do projeto. Não houve CUDA, treino, benchmark nem piloto histórico.

## 7. Limitações e decisão para F2.3a

1. São amostras reproduzíveis de uma GPU/driver/runtime, não prova universal de conformidade FP64 nem sobre todos os logits, tamanhos de vocabulário ou devices.
2. A divergência de `log` foi medida, mas sua causa não foi identificada. O aviso libclc não é prova causal. As três opções amostradas coincidiram em bits.
3. A redução do kernel de sonda é sequencial; não foi medida uma redução paralela. A soma Python observada é compensada nesta versão de CPython, ao contrário da soma sequencial OpenCL.
4. A expressão CPU perde `ln(2)` para logits iguais de magnitude extrema; o projeto ainda não define limite de magnitude de logits nem comportamento requerido nesse caso. Um consumidor futuro deverá resolver esse contrato sem alterar silenciosamente a referência.
5. F2.3b, BCE, treinamento, CUDA e outros dispositivos permanecem sem evidência nesta tarefa.

**Decisão:** `NO-GO` para autorizar a implementação funcional F2.3a com a abordagem OpenCL compilada e avaliada aqui, pois o requisito `atol=rtol=1e-12` falha em entradas finitas da matriz. A evidência autoriza somente continuar a investigação em escopo próprio. Um futuro gate funcional precisa validar novamente a função de `log`, a composição com o denominador e a política para cancelamento/overflow contra CPU e alta precisão, sem inferir solução a partir das opções de compilação coincidentes. Este parecer não altera nem corrige F2.2/#78 e não é aprovação de F2.3b.

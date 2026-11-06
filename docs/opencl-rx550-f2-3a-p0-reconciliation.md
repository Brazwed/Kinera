# F2.3a-P0 — Consolidação das auditorias #83 e #84

## Parecer

**NO-GO permanece para a implementação funcional F2.3a avaliada.** Na nova coleta, `log` FP64 ainda diverge do CPUBackend e da referência Decimal em entradas finitas; a composição completa da cross-entropy (CE) também falha a tolerância `atol=rtol=1e-12`. A soma Neumaier-compensada reproduziu o `sum()` do Python 3.15 nos casos testados, mas não corrigiu as divergências de `log` nem a CE que delas resulta.

O resultado não identifica a causa da divergência e não é uma afirmação universal sobre toda implementação OpenCL, outro hardware ou outro runtime. Nenhum backend de produção foi modificado.

## Baseline e proveniência

- Baseline de produção: `7bff9617a340faab342f5c844464a94d0acec6d0`.
- Auditoria PR #83: commit `f6fa46438c6b6ba68101c1cffca79fb9516d3d7b`.
- Auditoria PR #84 usada como base: commit `66f0c20457bf1f5d9a71bac7fa3016b535ff80dd`.
- A coleta consolidada é nova e foi executada a partir da matriz e do harness do #84, estendidos com fixtures selecionadas do #83. Os artefatos originais permanecem no histórico Git e o JSON original do #84 continua byte a byte inalterado neste commit.
- SHA-256 do script consolidado: `c0cb979f8d02b8a62a12dfe450182bb6083d0c41c608c2c850d957e24fae3055`.
- SHA-256 da fonte OpenCL experimental compilada: `e4b8d1a2a94fec55cf8ccb61b58ac546b15983c0dc30c536b5d295aa63a420e3`.
- SHA-256 do JSON consolidado: `1cffbd4cd625b41d5c87d8704316374f16ca4e76b27c969d034850d9d394a614`.
- JSON original #83: `68a5478aedd08bfc45089ada1cead367c78ccc55313ffbe15e5d9f6c385cf014`.
- JSON original #84: `82940bdeb55a4c4e221bcba1a1ad76de43ed102808363254e3c9216d15245f80`.

Os SHA originais são referências históricas, não hashes do artefato consolidado. O novo JSON registra commits, hashes dos dois diagnósticos originais, hash da nova fonte e do novo script.

## Ambiente RX 550

| Item | Valor observado |
|---|---|
| GPU | AMD Radeon RX 550 / 550 Series, `radeonsi`, PCI `0000:03:00.0`, vendor `0x1002` |
| Runtime/plataforma | Mesa Rusticl, Mesa/X.org |
| Driver/runtime | Mesa `26.2.6-1.fc45`, OpenCL 3.1, OpenCL C 1.2 |
| FP64 | Habilitado (`double_fp_config=63`) |
| Python | CPython 3.15.0, GCC 16.2.1 |
| Sistema | Fedora 45, kernel `7.2.11-300.fc45.x86_64`, glibc 2.44 |
| Variáveis | `RUSTICL_ENABLE=radeonsi`, `RUSTICL_FEATURES=fp64` |
| Aviso | `Patched Mesa libclc not detected` apareceu nos três processos |

Pacotes registrados: `mesa-dri-drivers` e `mesa-libOpenCL` `26.2.6-1.fc45`; `llvm-libs` `23.1.4-1.fc45`; `libclc22` e `libclc22-spirv` `22.1.8-3.fc45`. O pacote `llvm` não estava instalado. Isso identifica pacotes presentes, não prova qual componente causa os desvios. Não foi feito controle com outro Mesa, libclc, driver ou hardware.

## Reconciliação das metodologias

| Aspecto | PR #83 original | PR #84 original | Coleta consolidada |
|---|---|---|---|
| Base | Diagnóstico independente | Diagnóstico mais amplo, escolhido como base | Harness e matriz #84 preservados e ampliados |
| Opções | padrão, `-cl-opt-disable` | padrão, `-cl-opt-disable`, `-cl-fast-relaxed-math` | Opções do #84; fast-relaxed é apenas diagnóstico |
| Repetição | 3 repetições/caso dentro de cada execução; duas execuções separadas produziram JSON idêntico | 3 processos independentes sequenciais; um resultado por fixture/opção/processo | 3 processos independentes sequenciais; um resultado por fixture/opção/processo |
| E1 exp | 23 fixtures/opção | 17 | 28: 17 do #84 + 11 exclusivas do #83 |
| E2 log | 14 fixtures/opção | 31, incluindo denominadores CE | 66: 31 do #84 + 8 logs diretos + 27 denominadores derivados de fixtures #83 |
| E3 soma | 6 sequências; modos ingênuo e Neumaier-compensado | 8 sequências; ordem direta/reversa ingênua | 12: 8 do #84 + 4 do #83; direta/reversa ingênua e compensada |
| E4 CE | 31 composições, incluindo vocabulários 26/257 e cancelamentos | 18 composições | 45: 18 do #84 + 27 fixtures/targets exclusivos do #83 |
| Referência independente | Decimal com precisão 400 | Decimal com precisão 100/120 | Decimal com precisão 100 para somas e 120 para CE |

Os números agregados das auditorias originais não são combinados como se fossem uma única população. Os protocolos de repetição, opções, conjuntos de entradas e reduções diferem. A tabela de contagens abaixo separa fixtures únicos das observações repetidas.

### Origem e contabilidade da coleta nova

Cada opção compilou em três processos independentes. O JSON guarda o trace completo canônico do processo 1 com opção padrão e registra hashes, contagens, ecos e verificações bit a bit para as demais combinações. Os resumos agregam as três observações por opção.

| Matriz | Fixtures por processo/opção | Origem | Observações por opção após 3 processos |
|---|---:|---|---:|
| E1 `exp` | 28 | 17 #84 + 11 #83 | 84 |
| E2 `log` | 66 | 31 #84 + 8 entradas diretas #83 + 27 denominadores de CE #83 | 198 |
| E3 soma | 12 sequências | 8 #84 + 4 #83 | 36 por modo de acumulação |
| E4 CE | 45 vetores/targets | 18 #84 + 27 #83 | 135 composições; cada kernel produz os caminhos sequencial e compensado |

Os denominadores CE são fixtures próprios da matriz E2 e não casos independentes de CE. As linhas novas do #83 mantêm um rótulo de proveniência no JSON; nenhuma fixture da matriz original do #84 foi removida ou substituída.

## Protocolo consolidado

O oracle contratual é `CPUBackend.cross_entropy` no CPython 3.15. Referências Decimal convertem os floats de entrada por `Decimal.from_float`; o caminho CE calcula a referência em alta precisão sem repetir a subtração cancelante da expressão CPU. Também se registram `math.fsum` e uma soma Decimal exata arredondada a FP64 para os experimentos de soma.

O código OpenCL é uma sonda independente: FP64, `#pragma OPENCL FP_CONTRACT OFF`, `exp`/`log` comuns, buffers pequenos e ecos de bits para entradas e escalares. Os dois caminhos CE compartilham os mesmos logits, shifts e resultados de `exp`; um acumula sequencialmente, o outro aplica Neumaier em uma passagem sequencial fixa. A variante compensada é um controle experimental, não uma afirmação de reprodução do algoritmo interno de `sum()` do CPython. O probe `sum((1e16, 1, -1e16)) == 1` confirmou o comportamento compensado do Python ativo.

Para valores finitos, PASS usa `abs(actual-expected) <= max(1e-12, 1e-12 * max(abs(expected), abs(actual)))`. Erro absoluto, erro relativo, ULP, bits e limite são registrados por comparação. Erro relativo é `null` quando a referência é zero. Igualdade de categorias especiais/sinal de infinito recebe `SPECIAL_MATCH`; não é classificada como aprovação numérica finita. Os erros de `log` perto de zero são interpretados junto ao limite absoluto, sem deixar de registrar erro relativo e ULP.

Comando de reprodução:

```bash
RUSTICL_ENABLE=radeonsi RUSTICL_FEATURES=fp64 PYTHONPATH=src \
  python experiments/investigate_opencl_fp64_cross_entropy_math.py \
  --output /tmp/opencl-rx550-f2-3a-math-audit-consolidated.json
```

## Resultados consolidados

Os resultados das três opções foram bit a bit iguais entre si nesta amostra. Para cada opção também houve estabilidade bit a bit entre os três processos, com novos contextos e builds. Isso é repetibilidade observada, não evidência de código de máquina idêntico.

### E1 — `exp`

| Comparação | Contagem por opção (3 processos) | Erro máximo | Estado |
|---|---:|---:|---|
| OpenCL vs `math.exp` | 84 PASS / 0 FAIL | absoluto 0; 0 ULP | PASS amostral |
| OpenCL vs Decimal arredondado a FP64 | 84 PASS / 0 FAIL | absoluto 0; 0 ULP | PASS amostral |

Inclui os 17 casos #84 e 11 entradas distintas do #83, incluindo `-inf`. O resultado não prova equivalência de `exp` para todos os argumentos/dispositivos.

### E2 — `log`

Cada opção: **156 PASS / 42 FAIL** em 198 observações; são 52 PASS e 14 FAIL por processo. O mesmo conjunto de entradas falhou contra CPU e Decimal arredondado a FP64. O máximo no conjunto foi erro absoluto `1.5734258340671659e-10`, relativo `8.349416146054523e-11` e `527990` ULP.

| Entrada | CPU/Python | OpenCL | Erro absoluto | Relativo | ULP | Estado |
|---|---:|---:|---:|---:|---:|---|
| `log(257)` | `5.54907608489522` | `5.549076084930974` | `3.575362228502854e-11` | `6.443166707039962e-12` | 40.255 | FAIL |
| `log(1.191862558932969)` | `0.17551725908668359` | `0.17551725910133825` | `1.465466636929591e-11` | `8.349416146054523e-11` | 527.990 | FAIL |
| `log(1.5032147244080551)` | `0.4076059644443804` | `0.4076059644380026` | `6.377787187261674e-12` | `1.5646942742742792e-11` | 114.892 | FAIL |

`log(257)` também aparece como denominador da CE de 257 logits iguais. Os ecos de argumentos mantiveram os bits de entrada; a diferença não foi explicada por transporte de argumento no caminho medido.

### E3 — acumulação

Resultados por opção após três processos, comparados com `sum()` Python 3.15:

| Modo | PASS | FAIL | Erro absoluto máximo | Erro relativo máximo | ULP máximo |
|---|---:|---:|---:|---:|---:|
| Sequencial ingênuo, ordem direta | 30/36 | 6/36 | 1.0 | 1.0 | 4.607×10¹⁸ |
| Sequencial ingênuo, ordem reversa | 33/36 | 3/36 | 1.0 | 1.0 | 4.607×10¹⁸ |
| Neumaier, ordem direta | 36/36 | 0 | 0 | 0 | 0 |
| Neumaier, ordem reversa | 36/36 | 0 | 0 | 0 | 0 |

As falhas grandes da soma ingênua são os cancelamentos assinados adicionados do #83. Para `[1e16, 1, -1e16]`, Python retorna `1`, a ordem direta ingênua OpenCL retorna `0` e a soma Neumaier retorna `1`. No fixture assinado alternado, Python/Neumaier retornam `0`, enquanto o kernel ingênuo direto retorna `-1`. A soma Neumaier coincidiu também com `math.fsum` e com a soma Decimal arredondada nos 12 fixtures (36 observações por opção em cada comparação).

### E4 — cross-entropy completa

Por opção, 45 fixtures foram executadas em 3 processos. Os valores em cada coluna são contagens observadas agregadas entre processos; equivalem aos estados por fixture multiplicados por três.

| Caminho OpenCL | Referência | PASS | FAIL | SPECIAL_MATCH | Erro absoluto máx. | Relativo máx. | ULP máx. |
|---|---|---:|---:|---:|---:|---:|---:|
| CE com soma sequencial | CPUBackend | 105 | 27 (9 fixtures) | 3 (1 fixture) | `3.575362228502854e-11` | `8.349321264528583e-11` | 527.984 |
| CE com soma Neumaier | CPUBackend | 105 | 27 (9 fixtures) | 3 (1 fixture) | `3.575362228502854e-11` | `8.349321264528583e-11` | 527.984 |
| CE sequencial | Decimal alta precisão | 93 | 39 (13 fixtures) | 3 | `0.7586236756795135` | `1.0` | 4.605×10¹⁸ |
| CE Neumaier | Decimal alta precisão | 93 | 39 (13 fixtures) | 3 | `0.7586236756795135` | `1.0` | 4.605×10¹⁸ |

As contagens/status finais do caminho sequencial e compensado são iguais. A compensação muda os bits da soma em 7/45 fixtures e da loss final em 4/45, mas não remove nenhuma das 9 fixtures de CE que falham a tolerância contra CPU. Para a matriz positiva de exponenciais CE, a soma compensada coincidiu com `sum()` do Python em todos os 45 casos; a soma sequencial teve no máximo `7.105427357601002e-15` de erro absoluto (2 ULP), dentro da tolerância. A falha restante é compatível com o desvio observado no estágio `log`, mas isso não identifica a causa interna da implementação OpenCL.

| Caso | CPUBackend | OpenCL (ambas somas) | Erro abs. | Relativo | ULP | Observação |
|---|---:|---:|---:|---:|---:|---|
| CE de 257 logits iguais, qualquer um dos três alvos | `5.54907608489522` | `5.549076084930974` | `3.575362228502854e-11` | `6.443166707039962e-12` | 40.255 | FAIL; reproduz `log(257)` |
| `[3,-2,0,1,-10]`, alvo 0 | `0.1755172590866838` | `0.1755172591013383` | `1.4654499835842216e-11` | `8.349321264528583e-11` | 527.984 | FAIL; log do denominador `1.191862558932969` |
| `[1000,999,998]`, alvo 1 | `1.4076059644444285` | `1.4076059644379484` | `6.480149750132114e-12` | `4.6036674423227385e-12` | 29.184 | FAIL |
| `[-1000,-1001,-1002]`, alvo 2 | `2.4076059644444285` | `2.4076059644379484` | `6.480149750132114e-12` | `2.691532520616368e-12` | 14.592 | FAIL |

O estágio interno `log` da CE falha em 13/45 fixtures, mas a adição/subtração final mascara parte dessas diferenças em quatro casos; a loss final contra CPU falha em nove. A tolerância é aplicada a cada resultado final de acordo com a fórmula registrada, sem arredondamento intermediário introduzido pelo relatório.

### CPU contratual versus Decimal

Para `[1e16,1e16]` e `[1e300,1e300]`, CPU e ambas as composições OpenCL retornam `0.0`, enquanto Decimal retorna `ln(2) ≈ 0.6931471805599453`. Para `[1e16,1e16,1e16-2]`, CPU e OpenCL retornam `0.0`, enquanto Decimal retorna `0.7586236756795135`. Esses casos passam quando comparados ao oracle contratual CPU e falham contra a referência matemática Decimal; não são evidência de exatidão matemática. O caso de overflow da loss continua classificado como `SPECIAL_MATCH`, não como PASS finito.

## Matriz de reconciliação #83 / #84 / consolidada

| Evidência/fixture | PR #83 | PR #84 | Nova coleta consolidada |
|---|---|---|---|
| `log(257)` | Incluído; FAIL, erro abs. `3.575362228502854e-11`, `40255` ULP | Não era entrada direta da matriz | Incluído diretamente e como denominador CE; mesmo erro e bits entre processos/opções |
| 257 logits iguais | Incluído, três posições de alvo; FAIL nos três | Caso original era vocabulário 256 | Incluído com alvos primeiro/meio/final; 3 fixtures, todos FAIL por log |
| Python 3.15 `sum()` compensado | Probe afirmativo; soma Neumaier testada | Reconhecia compensação Python, mas kernel CE era sequencial | Probe refeito; Neumaier direto/reverso coincide bit a bit com Python, `fsum` e Decimal nas 12 sequências |
| Cancelamento `[1e16,1,-1e16]` | Incluído; ingenuidade falha, compensada passa | Não fazia parte das oito sequências positivas | Adicionado; direto ingênuo `0` versus CPU `1`, Neumaier `1` |
| Cancelamento alternado | Incluído em sequência longa | Não incluído | Adicionado; separa efeito da ordem e da compensação |
| CE mista `[3,-2,0,1,-10]` | Incluída para alvos 0/2/4; FAIL em log | Não incluída | Reexecutada com os três alvos; log continua FAIL, máximo de `527990` ULP no estágio log |
| CE com sumas de 26/257 classes | Incluiu `26` e `257` | Incluiu vocabulário 256 | Incluídas novamente; 257 classes reproduz `log(257)` |
| Acumulação compensada na composição CE | Composta em kernel usando probe Python | Não calculada | Calculada junto ao caminho sequencial. A soma alinha ao CPU, mas as falhas de `log`/loss permanecem |
| Contagem de E1/E2/E3/E4 por opção | 23 / 14 / 6 / 31 | 17 / 31 / 8 / 18 | 28 / 66 / 12 / 45 por processo/opção |

Os resultados do #83 e #84 foram preservados e não foram somados. A nova coluna é outra coleta, com suas próprias entradas, repetições, hashes e estatísticas.

## E5–E7, repetibilidade e limites

- **E5 — integridade de dados:** entradas de buffer, escalares `double` e alvos `int` ecoaram com bits corretos em todos os processos e opções. Isso reduz a hipótese de corrupção de transporte nos caminhos sondados, sem validar todo ABI/kernel de produção.
- **E6 — opções e repetição:** padrão, `-cl-opt-disable` e `-cl-fast-relaxed-math` produziram os mesmos bits para todas as matrizes observadas. Cada opção foi recompilada em três processos; hashes e ecos foram verificados. O fast-relaxed continua somente diagnóstico, não recomendação de produção.
- **E7 — outras primitivas:** a pesquisa não valida divisão, softmax, gradiente CE, BCE/`log1p`, `exp`/`log` universais ou treinamento. A Issue #78 permanece independente.
- O aviso `Patched Mesa libclc not detected` foi registrado. Não se demonstrou relação causal entre o aviso, Mesa, Rusticl, libclc, LLVM, driver ou hardware e as divergências.
- Comparação Decimal identifica perda adicional do próprio caminho CPU em casos de cancelamento extremo; não substitui CPUBackend como contrato de equivalência.
- As leituras de intermediários são parte da sonda experimental. Não estabelecem desenho de transferência para um kernel de produção.

## Execução e validação

Nova coleta RX 550: três processos Python independentes, executados sequencialmente; cada processo compilou as três opções e avaliou todas as fixtures. Não houve benchmark de desempenho, treinamento ou piloto científico.

Testes de integridade CPU-only:

```bash
PYTHONPATH=src python -m unittest tests.test_opencl_fp64_cross_entropy_math_audit -v
```

Os testes verificam integridade byte a byte do JSON original #84, hashes do JSON consolidado, hashes dos scripts/fontes, origens e contagens de fixtures, entradas ecoadas, resultados PASS/FAIL/SPECIAL_MATCH, comparadores, resumos e repetibilidade. O JSON usa apenas valores estritamente serializáveis; infinitos/NaNs são representados como strings e bits.

## Decisão

A nova coleta confirma que:

1. `exp` passou nos casos finitos e especiais amostrados, sem estabelecer equivalência universal.
2. A soma Neumaier reproduziu o `sum()` de Python 3.15 nas fixtures de soma, enquanto a soma ingênua falhou nos cancelamentos assinados.
3. A soma compensada na composição CE não removeu as divergências de `log`; ambas as opções de acumulação tiveram os mesmos nove casos CE finitos fora de tolerância.
4. `log(257)` e os três alvos da CE de 257 logits iguais reproduziram a divergência do #83 em três processos e três opções.
5. `-cl-opt-disable` não alterou os resultados observados; a causa permanece aberta.

**Parecer final desta auditoria: NO-GO para F2.3a com as abordagens e o ambiente avaliados; a conclusão funcional continua NO-GO.** Não se implementa `cross_entropy_forward_buffer`, não se relaxa a tolerância e não se propõe que a compensação resolva o erro de `log`. Qualquer nova alternativa matemática/aritmética exige protocolo e validação próprios antes de reconsiderar F2.3a.

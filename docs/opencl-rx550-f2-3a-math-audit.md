# F2.3a-P0 — Auditoria matemática de cross-entropy FP64

## Parecer

**NO-GO para a composição experimental avaliada como base de F2.3a** na RX 550/Rusticl deste teste. O `log` OpenCL ultrapassou `atol=rtol=1e-12` em 3 de 14 argumentos isolados, e a loss composta divergiu em 6 de 31 casos por opção de compilação. O máximo, as subtrações, as exponenciais e a soma da composição coincidiram em bits com CPU nos casos observados; controles de `log` isolado com as mesmas entradas reproduziram os desvios. Isso localiza o desvio observado no resultado de `log` sem identificar sua causa interna.

Não se conclui que toda abordagem de CE seja inviável. Uma alternativa exige investigação e validação próprias. O diagnóstico não implementa o kernel funcional, não resolve a divisão da #78 e não autoriza F2.3a.

## Ambiente e proveniência

- `main` no início: `7bff9617a340faab342f5c844464a94d0acec6d0`, merge do PR #80; worktree `Worktrees/Kinera_A014`.
- GPU AMD Radeon RX 550 / 550 Series (`radeonsi`, `polaris12`, ACO), PCI `0000:03:00.0`, dispositivo `1002:699f`, driver amdgpu; Mesa/Rusticl `26.2.6-1.fc45`, OpenCL 3.1 e OpenCL C 1.2, FP64 habilitado.
- Python 3.15.0, Fedora/Linux `7.2.11-300.fc45.x86_64`; LLVM instalado `23.1.4-1.fc45`, `libclc22`/`libclc22-spirv` `22.1.8-3.fc45`. Pacotes instalados não demonstram qual componente causou o resultado.
- Variáveis `RUSTICL_ENABLE=radeonsi`, `RUSTICL_FEATURES=fp64`. O aviso `Patched Mesa libclc not detected` apareceu. Não se demonstrou relação causal com o desvio.
- Diagnóstico SHA-256 `ed62afa8481a1f22808dabcfa10093982de481a58b16181e22e749359ae4d875`; fonte OpenCL embutida SHA-256 `27d2a6b812484957151cd1b2b403518d71fc3cbca53505c125c3a7fa91cd4670`; JSON SHA-256 `68a5478aedd08bfc45089ada1cead367c78ccc55313ffbe15e5d9f6c385cf014`.

## Contrato e método

`CPUBackend.cross_entropy` calcula `max(logits)`, `math.exp(value - maximum)` em ordem, `sum` sobre o gerador, `math.log` da soma, `maximum + log` e subtrai o logit alvo. **Não há divisão explícita no forward.** O caminho residente CUDA valida alvo, buffers e finitude e devolve um escalar; seu kernel usa soma sequencial com política compensada condicional. `MultiTaskLoss.forward` valida os alvos e usa esse caminho residente quando aplicável. Nenhum desses caminhos foi alterado ou executado como treinamento.

A sonda é independente, com kernels temporários de uma work-item, FP64, `FP_CONTRACT OFF` e `exp`/`log` comuns; não usa `native_exp` ou `native_log`. O wrapper OpenCL cria contexto, fila, buffers e programas explicitamente, envia entradas por buffer ou argumento escalar `ctypes.c_double`, sincroniza pelo `enqueue_1d` existente e lê os resultados para análise. As leituras integrais dos intermediários são **instrumentação de pesquisa**, não um contrato proposto para o futuro backend. Os recursos são liberados por `ExitStack`.

O oracle contratual é o CPUBackend/Python ativo. Como controle independente, `decimal` usa precisão 400 dígitos e `Decimal.from_float` para os valores binários originais; resultados decimais são arredondados para `float` no registro. Em extremos, limites de expoente e arredondamento dessa referência também são restrições; não é prova matemática universal. O JSON guarda entradas, eco, intermediários, bits, valores esperados/observados e resultados por opção.

Para finitos, PASS usa `math.isclose` com `atol=rtol=1e-12`, ou seja `|a-b| <= max(atol, rtol * max(|a|,|b|))`. O erro relativo usa `|a-b|/|CPU|`, omitido quando a referência é zero; ULP é distância entre representações binárias ordenadas. Para NaN/inf, o comparador verifica categoria ou igualdade de infinito com sinal e não calcula ULP; isso é observação de caso especial, não prova de conformidade. O `atol` pode mascarar erro relativo perto de zero. O sinal de zero é preservado nos bits, embora não seja critério numérico separado.

Cada opção (`""`, igual à produção, e `-cl-opt-disable`) executou 23 entradas isoladas de `exp`, 14 de `log`, seis sequências de soma e 31 composições CE. Cada execução foi repetida três vezes, com bits estáveis. Dois processos independentes, sequenciais, produziram JSON byte a byte idêntico. As duas opções produziram registros iguais nesta amostra; isso não prova geração de código idêntico.

## Matriz E1–E7

| Gate | Evidência por opção | Estado |
|---|---|---|
| E1 `exp` | 23/23 argumentos isolados e 1.695 exponenciais das composições iguais em bits ao Python, incluindo underflow e `-inf`; Decimal coincide para os isolados | PASS amostral |
| E2 `log` | 11/14 passam, 3 falham; cinco são iguais em bits. O máximo absoluto isolado é `3.575362228502854e-11`; máximo relativo `8.349416146054523e-11`; máximo 527.990 ULPs | **FAIL** |
| E3 soma | Python 3.15 detectado como compensado; 6/6 sequências compensadas coincidem em bits. Soma ingênua falha em dois cancelamentos; outros desvios pequenos são mascarados por tolerância | PASS amostral para compensada |
| E4 composição | 25/31 losses passam, 6 falham; máximo absoluto `3.575362228502854e-11`, máximo relativo `8.349321264528583e-11`, máximo 527.984 ULPs | **FAIL** |
| E5 dados | Ecos de buffer e escalar em 37 entradas isoladas e todos os logits das 31 composições preservam os bits | PASS amostral |
| E6 opções | Opções padrão e `-cl-opt-disable` produzem resultados iguais; três repetições por caso e dois processos | PASS de reprodução, causa inconclusiva |
| E7 dependências | Divisão/softmax/gradiente F2.3b e BCE/`log1p` permanecem sem gate nesta pesquisa | INCONCLUSIVO |

## Casos representativos

| Caso | Referência CPU | OpenCL | Erro absoluto | Erro relativo | ULPs | Estado |
|---|---:|---:|---:|---:|---:|---|
| `log(257)` | `5.54907608489522` (`40163241004e9011`) | `5.549076084930974` (`40163241004f2d50`) | `3.575362228502854e-11` | `6.443166707039962e-12` | 40.255 | FAIL |
| `log(1.191862558932969)` | `0.17551725908668359` | `0.17551725910133825` | `1.465466636929591e-11` | `8.349416146054523e-11` | 527.990 | FAIL |
| CE `[3,-2,0,1,-10]`, alvo 0 | `0.1755172590866838` | `0.1755172591013383` | `1.4654499835842216e-11` | `8.349321264528583e-11` | 527.984 | FAIL |
| CE 257 logits iguais, alvo 0 | `5.54907608489522` | `5.549076084930974` | `3.575362228502854e-11` | `6.443166707039962e-12` | 40.255 | FAIL |
| CE `[1e16,1e16,1e16-2]`, alvo 0 | `0.0` | `0.0` | 0 | indefinido | 0 | PASS CPU |

No último caso o `log` intermediário falhou, mas a adição ao máximo de grande magnitude arredondou o erro e a subtração final retornou zero. A referência Decimal para o valor matemático é aproximadamente `0.7586236756795135`: o PASS reproduz o contrato CPU, não a CE ideal. Para logits finitos `[1e308,-1e308]`, a subtração intermediária e a loss para o segundo alvo resultaram em infinito tanto em CPU quanto na sonda; o CUDA residente atual rejeita loss não finita. O tratamento futuro desse limite precisa ser definido separadamente.

As falhas de loss ocorreram nos três alvos de `[3,-2,0,1,-10]` e nos três alvos dos 257 logits iguais. O `log` intermediário falhou também nos dois alvos do caso de cancelamento final; seu efeito foi ocultado pelo arredondamento subsequente. Entre os 31 casos por opção, `max`, `sum`, 1.695 deslocamentos e 1.695 `exp` coincidiram em bits, inclusive valores que sofreram underflow. O somatório ingênuo errou os dois casos assinados adversariais, enquanto o compensado coincidiu com o `sum()` ativo; esses cancelamentos não são representativos das exponenciais positivas da CE, mas testam a política de redução.

## Implicações e limites

A amostra de `exp` passou, porém não constitui garantia para todo argumento ou outro runtime. A divergência isolada de `log` com entrada recebida corretamente demonstra risco direto para a abordagem medida da CE forward, independentemente da anomalia de divisão. `-cl-opt-disable` não mudou os valores observados; não foram examinados IR, código de máquina, outros dispositivos ou libclc alternativo. Não é possível atribuir a falha a um componente específico.

Próximos controles possíveis, sob escopo próprio: ampliar a caracterização de `log` com referências de alta precisão e outros runtimes/libclc; avaliar implementações alternativas e os contratos de arredondamento/valores especiais; repetir cada etapa e a composição contra CPU. Qualquer técnica alternativa deve ser testada, sem relaxar tolerâncias nem alterar a matemática CPU. Divisão e softmax de F2.3b, BCE com `log1p` e gradientes exigem auditorias próprias; resultados de `log` isolado não os validam.

## Reprodução e testes

```bash
RUSTICL_ENABLE=radeonsi RUSTICL_FEATURES=fp64 PYTHONPATH=src \
  python experiments/investigate_opencl_fp64_cross_entropy_math.py \
  --output /tmp/kinera-ce-math.json
PYTHONPATH=src python -m unittest discover -s tests -p test_opencl_f2_3a_math_audit.py -v
```

Os testes CPU-only verificam referências, bits, erros, ULPs, casos especiais, contagens, ecos e hash da fonte. A sonda requer GPU OpenCL FP64 e falha explicitamente quando indisponível. Não foram executados benchmarks, pilotos nem treinamento; artefatos históricos e produção foram preservados.

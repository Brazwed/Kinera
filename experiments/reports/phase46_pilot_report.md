# Relatório Experimental — Fase 46: Desambiguação Contextual

> **Issue:** [#46 — Research: Desambiguação contextual de entradas com alvos conflitantes](https://github.com/Brazwed/Kinera/issues/46)  
> **Branch de Trabalho:** `research/issue-46-contextual-disambiguation`  
> **Commit Base:** `b0502c01ab7d4ce6256fe54e9305d398f10a8702` (`main`)  
> **Artefatos:** [`scripts/run_phase46_pilot.py`](scripts/run_phase46_pilot.py), [`experiments/phase46_pilot_results.json`](experiments/phase46_pilot_results.json), [`tests/test_regime1_identifiability.py`](tests/test_regime1_identifiability.py), [`tests/test_phase46_metrics.py`](tests/test_phase46_metrics.py)

---

## 1. Resumo Executivo

A Fase 46 investigou se a adição de um sinal de condicionamento contextual de lema permite separar representações e viabilizar a coexistência funcional de alvos ortográficos conflitantes (`mala` vs. `medo` e `pato` vs. `povo` na transição `[1]`), preservando paridade estrita de orçamento (4 passos de consolidação) e isolamento total do núcleo (`src/kinera/`).

### Principais Descobertas
1. **Regime 1 (Não-Condicionado): Falha Estrutural de Identificabilidade Comprovada**  
   - Na transição `[1]`, as entradas de `mala` e `medo` (`context_ids=['<START>', 'm']`, `context_syllable_ids=['<UNK_SYL>', '<UNK_SYL>']`) são idênticas.
   - A distância de representação pré-cabeça e de logits é nula ($d_{\text{cos}} = 0{,}0000$, $d_{\text{euc}} = 0{,}0000$).
   - O modelo determinístico é forçado a um único argmax. Nas 3 seeds, a consolidação sequencial da tarefa antiga sobrescreveu a nova ($\Delta_{\text{new}} = -0{,}5000$, $0/3$ aprovações).
2. **Regime 2 (Condicionado por Lema): Separação Representacional e Coexistência Parcial**  
   - O sinal de lema separou os vetores pré-cabeça ($d_{\text{cos}} = 0{,}2618 \pm 0{,}1032$, $d_{\text{euc}} > 1{,}0$).
   - O loss logarítmico conjunto reduziu de $1{,}1701$ para $0{,}6859$.
   - **Coexistência funcional obtida em 2 de 3 seeds:** Seeds 42 e 43 atingiram $R_{\text{pos}} = 1{,}0000$ e $\Delta_{\text{new}} = +0{,}0000$.
   - **Falha na Seed 44 ($\Delta_{\text{new}} = -0{,}5000$):** Embora a separação de lema tenha ocorrido, a interferência de gradiente na cabeça compartilhada durante o replay sequencial com $\text{LR}=0{,}20$ fez `medo` perder o argmax para `'a'`. Sob o critério predefinido estrito (`all_seeds_criteria_met`), a taxa de aprovação foi de $2/3$ ($66{,}7\%$).
3. **Controle Negativo: Dependência Causal Estrita**  
   - Sob permutação de lemas na avaliação (`mala` com lema de `medo`), a acurácia colapsou ($R_{\text{pos}} = 0{,}5000 \pm 0{,}14$, $\Delta_{\text{new}} = -1{,}0000 \pm 0{,}00$, joint loss = $1{,}7373$), provando dependência causal no sinal condicional e descartando artefatos espúrios.
4. **Avaliação de Holdout Diagnóstico vs. Partição Independente:**  
   - **Holdout Diagnóstico (6 palavras conhecidas da Issue #42):** Sob condicionamento por lema, o modelo atingiu acurácia agregada de $94{,}4\%$ (preservando o controle histórico prévio), contra $50{,}0\%$ na baseline não-condicionada. Essa medição avalia a estabilidade em palavras previamente investigadas e não constitui evidência de generalização para vocabulário novo.
   - **Partição Independente (7 palavras não-vistas com `<UNK_LEMMA>`):** O modelo obteve $14{,}3\%$ de acurácia (1/7 acertos), descrevendo o desempenho empírico estrito sob a configuração avaliada, onde palavras desconhecidas recebem o token genérico `<UNK_LEMMA>`. O piloto não implementou nem avaliou uma representação composicional de lemas (baseada em raízes ou morfologia compartilhada); portanto, esse resultado delimita o limite da lookup table de IDs categóricos memorizados, sem permitir concluir que toda forma de condicionamento lexical falharia em generalizar.

---

## 2. Resultados Experimentais Consolidados

Todas as execuções foram realizadas em CPU local, sem custos de nuvem/GPU, com $N_{\text{steps}} = 4$ por execução (2 novos + 2 replay) e replay LR $= 0{,}20$.

### Tabela 1 — Comparação Agregada (3 seeds: 42, 43, 44)

| Condição | Separação Angular ($d_{\text{cos}}$) | Retenção Positiva ($R_{\text{pos}}$) | Delta Tarefa Nova ($\Delta_{\text{new}}$) | Log-Loss Conjunto | Holdout Diagnóstico | Teste Independente | Todas as Seeds Aprovadas? |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Baseline Não-Condicionada** | $0{,}0000 \pm 0{,}00$ | $1{,}0000 \pm 0{,}00$ | $-0{,}5000 \pm 0{,}00$ | $1{,}1701 \pm 0{,}01$ | $0{,}50 \pm 0{,}00$ | $0{,}14 \pm 0{,}00$ | **Não** (0/3) |
| **2. Condicionamento por Lema** | $0{,}2618 \pm 0{,}10$ | $1{,}0000 \pm 0{,}00$ | $-0{,}1667 \pm 0{,}24$ | $0{,}6859 \pm 0{,}12$ | $0{,}94 \pm 0{,}08$ | $0{,}14 \pm 0{,}00$ | **Não** (2/3) |
| **3. Controle Negativo (Permutado)** | $0{,}2618 \pm 0{,}10$ | $0{,}5000 \pm 0{,}14$ | $-1{,}0000 \pm 0{,}00$ | $1{,}7373 \pm 0{,}28$ | $0{,}67 \pm 0{,}00$ | $0{,}14 \pm 0{,}00$ | **Não** (0/3) |

---

### Tabela 2 — Resultados Individuais por Seed com Desagregação de Sondas

Para garantir total rigor matemático, a tabela explicita:
- $N_{\text{pos}}$: número de sondas internas com competência de referência $> 0{,}0$ ($N_{\text{pos}} \in \{3, 4\}$);
- $k_{\text{pos}}$: número inteiro de sondas positivas preservadas após a intervenção ($0 \le k_{\text{pos}} \le N_{\text{pos}}$);
- $R_{\text{pos}}$: valor contínuo da razão de retenção ($k_{\text{pos}} / N_{\text{pos}}$);
- $C_{\text{post}}$: acurácia total sobre as 4 sondas de controle interno (`malas`, `patos`, `sapos`, `fogos`).

#### A. Condição 1 — Baseline Não-Condicionada (Regime 1)
| Seed | $d_{\text{cos}}$ | $N_{\text{pos}}$ | Sondas Retidas ($k_{\text{pos}}$) | $R_{\text{pos}}$ | Acurácia Interna ($C_{\text{post}}$) | $C_{\text{new, acq}}$ | $C_{\text{new, post}}$ | $\Delta_{\text{new}}$ | Log-Loss | Critérios Atendidos? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **42** | $0{,}0000$ | 3 | 3 | $3/3$ ($1{,}0000$) | $0{,}75$ (3/4) | $0{,}50$ | $0{,}00$ | $-0{,}5000$ | $1{,}1654$ | Não |
| **43** | $0{,}0000$ | 3 | 3 | $3/3$ ($1{,}0000$) | $0{,}75$ (3/4) | $0{,}50$ | $0{,}00$ | $-0{,}5000$ | $1{,}1644$ | Não |
| **44** | $0{,}0000$ | 3 | 3 | $3/3$ ($1{,}0000$) | $0{,}75$ (3/4) | $0{,}50$ | $0{,}00$ | $-0{,}5000$ | $1{,}1805$ | Não |

#### B. Condição 2 — Condicionamento por Lema (Regime 2)
| Seed | $d_{\text{cos}}$ | $N_{\text{pos}}$ | Sondas Retidas ($k_{\text{pos}}$) | $R_{\text{pos}}$ | Acurácia Interna ($C_{\text{post}}$) | $C_{\text{new, acq}}$ | $C_{\text{new, post}}$ | $\Delta_{\text{new}}$ | Log-Loss | Critérios Atendidos? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **42** | $0{,}4033$ | 4 | 4 | $4/4$ ($1{,}0000$) | $1{,}00$ (4/4) | $1{,}00$ | $1{,}00$ | $+0{,}0000$ | $0{,}5273$ | **Sim** |
| **43** | $0{,}1600$ | 3 | 3 | $3/3$ ($1{,}0000$) | $1{,}00$ (4/4)* | $1{,}00$ | $1{,}00$ | $+0{,}0000$ | $0{,}8094$ | **Sim** |
| **44** | $0{,}2220$ | 3 | 3 | $3/3$ ($1{,}0000$) | $1{,}00$ (4/4)* | $1{,}00$ | $0{,}50$ | $-0{,}5000$ | $0{,}7210$ | Não |

*\*Nota: Nas seeds 43 e 44, a 4ª sonda (`fogos`) possuía competência de referência 0,0 e foi adquirida durante o treinamento, elevando a acurácia total a 4/4 enquanto $N_{\text{pos}}=3$.*

#### C. Condição 3 — Controle Negativo (Lema Permutado na Avaliação)
| Seed | $d_{\text{cos}}$ | $N_{\text{pos}}$ | Sondas Retidas ($k_{\text{pos}}$) | $R_{\text{pos}}$ | Acurácia Interna ($C_{\text{post}}$) | $C_{\text{new, acq}}$ | $C_{\text{new, post}}$ | $\Delta_{\text{new}}$ | Log-Loss | Critérios Atendidos? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **42** | $0{,}4033$ | 4 | 2 | $2/4$ ($0{,}5000$) | $0{,}50$ (2/4) | $1{,}00$ | $0{,}00$ | $-1{,}0000$ | $2{,}1141$ | Não |
| **43** | $0{,}1600$ | 3 | 1 | $1/3$ ($0{,}3333$) | $0{,}50$ (2/4) | $1{,}00$ | $0{,}00$ | $-1{,}0000$ | $1{,}4191$ | Não |
| **44** | $0{,}2220$ | 3 | 2 | $2/3$ ($0{,}6667$) | $0{,}75$ (3/4) | $1{,}00$ | $0{,}00$ | $-1{,}0000$ | $1{,}6788$ | Não |

---

## 3. Análise Detalhada dos Mecanismos

### 3.1 Por que o Regime 1 Falha Estruturalmente?
No teste formal [`tests/test_regime1_identifiability.py`](tests/test_regime1_identifiability.py), verificou-se que:
$$\vec{x}(\text{mala}, [1]) \equiv \vec{x}(\text{medo}, [1]) = \Big(\text{context\_ids}=[\text{<START>}, \text{'m'}], \text{context\_syllable\_ids}=[\text{<UNK\_SYL>}, \text{<UNK\_SYL>}]\Big)$$
Como o caractere `'m'` não é final de sílaba, o id silábico é estritamente mascarado por causalidade para evitar vazamento do alvo. Portanto, qualquer função determinística $f(\vec{x})$ produz exatamente o mesmo vetor $\vec{h}$ e os mesmos logits $\vec{z}$. Consequentemente:
$$\mathbb{I}(\hat{y}_{\text{mala}} = \text{'a'}) + \mathbb{I}(\hat{y}_{\text{medo}} = \text{'e'}) \le 1$$
Apenas um dos alvos pode vencer o argmax. A busca por ordenação e calibração de LR na Issue #45 não poderia superar essa barreira estrutural.

### 3.2 O Que a Condição de Lema Resolve e o Que Não Resolve?
1. **O que resolve:** A injeção do vetor de lema $\vec{e}_{\text{lemma}}$ no pooling pré-cabeça quebra a simetria de entrada, permitindo representações distintas ($d_{\text{cos}} \ge 0{,}16$). Isso viabilizou a coexistência simultânea em 2 das 3 seeds ($66{,}7\%$).
2. **O que não resolve:** Na Seed 44, mesmo com $d_{\text{cos}} = 0{,}2220$, a consolidação de `mala` e `pato` com $\text{LR}=0{,}20$ atualizou a matriz da cabeça linear compartilhada em direção a `'a'`, reduzindo a probabilidade de `medo` de $0{,}4995$ para $0{,}2885$ (ficando abaixo do limiar de predição). Ou seja: **o condicionamento remove o colapso representacional, mas a dinâmica de interferência retroativa por gradiente continua existindo no espaço paramétrico compartilhado da cabeça de saída.**

### 3.3 Validação Causal e Limites de Generalização
- **Validação Causal:** A inversão/colapso imediato no controle negativo demonstra que a resposta diferencial decorre especificamente do lema e não de convergência espúria de outros pesos.
- **Delimitação Metodológica sobre Generalização:**
  1. O resultado de **14,3% (1/7 acertos)** na partição independente corresponde estritamente ao desempenho observado na configuração avaliada, onde todas as palavras não vistas receberam o identificador genérico `<UNK_LEMMA>`.
  2. O piloto avaliou um embedding de identificador categórico (`lemma_id`), que opera como indexador/chave de memorização direta. O experimento **não demonstra generalização composicional de lemas**, pois essa representação (baseada em raízes compartilhadas, traços morfológicos ou semântica distribucional) não foi implementada neste piloto.
  3. Esses resultados **não autorizam concluir que toda forma de condicionamento lexical falharia em generalizar**; eles delimitam apenas o comportamento da representação categórica indexada por ID avaliada.
  4. Palavras cujas entradas permaneçam indistinguíveis (sem contexto discriminativo à esquerda nem sinal lexical diferenciador) continuam estritamente sujeitas à barreira teórica de identificabilidade do Regime 1.

---

## 4. Recomendações para a Arquitetura Kinera

1. **Formalização Teórica nos Documentos de Pesquisa:** Registrar no repositório que predição de sequências ortográficas com prefixos curtos ($k \le 2$) requer necessariamente sinal de intenção lexical ou contexto estendido se os prefixos colidirem.
2. **Separação de Camadas (Modularidade de Replay):** Para mitigar interferências como a observada na Seed 44, futuras arquiteturas devem desacoplar as projeções contextuais das cabeças lineares de saída ou utilizar regularização de projeção ortogonal durante o replay.
3. **Não mesclar em `src/kinera/`:** Manter a implementação de protótipo isolada em `scripts/` e `experiments/`, preservando a integridade da biblioteca principal da Kinera.

---

## 5. Auditoria de Divergências Metodológicas (Revisão Pré-Merge do PR #48)

Esta seção documenta a resolução formal das divergências identificadas durante as revisões técnicas do PR #48:

### 5.1 Alinhamento da Partição Independente e Restrição de Vocabulário
- **Divergência:** O protocolo conceitual listou no diagrama os exemplos `mola`/`muro`, `pala`/`pipa` e controles `selo`, `toca`, `vela`. O runner executado usou `mola`, `muda`, `pala`, `pelo`, `selo`, `toga`, `vela`.
- **Causa Raiz Identificada:** O vocabulário canônico unificado de caracteres do projeto (`_build_unified_vocabularies()`) é derivado das palavras de `_WORD_SYLLABLES` e possui apenas 20 caracteres (`['<START>', '<PAD>', '<UNK>', 'a', 'b', 'd', 'e', 'f', 'g', 'l', 'm', 'o', 'p', 'r', 's', 't', 'u', 'v', 'x', 'z']`). Os caracteres `'i'` (presente em `pipa`) e `'c'` (presente em `toca`) **não existem no vocabulário**. A tentativa de codificá-los mapeia o caractere-alvo para `<UNK>`, invalidando o teste de acurácia de letras do português.
- **Resolução:** O runner adaptou para palavras 100% contidas no vocabulário (`pelo` em vez de `pipa`, `toga` em vez de `toca`, e `muda` no par com `mola`). A semântica original do teste (pares mínimos contrastantes com prefixos em `m` e `p` e controles balanceados) foi preservada. A restrição foi formalizada no código, no protocolo e coberta pelo teste automatizado `test_independent_holdout_vocabulary_compatibility` em [`tests/test_phase46_metrics.py`](tests/test_phase46_metrics.py). Os resultados numéricos foram preservados integralmente sem substituições post hoc.

### 5.2 Resolução da Notação e Cálculo de $R_{\text{pos}}$ no Controle Negativo
- **Divergência:** A versão preliminar do relatório apresentou `R_pos = 0,5000` na Seed 42 do controle negativo com a notação `1,5/3`.
- **Causa Raiz Identificada:** O denominador $N_{\text{pos}}$ da métrica `compute_probe_partition_breakdown` não é constante: ele é determinado dinamicamente pelo número de sondas que atingiram referência prévia $> 0{,}0$. Na Seed 42, todas as 4 sondas foram aprendidas na referência ($N_{\text{pos}} = 4$). No controle negativo, exatamente 2 sondas positivas foram retidas ($k_{\text{pos}} = 2$), resultando em $R_{\text{pos}} = 2/4 = 0{,}5000$. A notação `1,5/3` era um artefato de redação que forçou o denominador 3 na tabela. Nas seeds 43 e 44, a referência foi de 3 sondas ($N_{\text{pos}} = 3$), resultando em $1/3$ ($0{,}3333$) e $2/3$ ($0{,}6667$) respectivamente.
- **Resolução:** A Tabela 2 foi reestruturada para exibir explicitamente e de forma separada: (1) o denominador real $N_{\text{pos}}$, (2) a contagem discreta de sondas retidas $k_{\text{pos}}$, (3) a razão de retenção contínua $R_{\text{pos}}$, e (4) a acurácia interna total. Foi implementado o teste formal `test_positive_retention_ratio_discrete_nature` em [`tests/test_phase46_metrics.py`](tests/test_phase46_metrics.py) para garantir que contagens fracionárias sejam formalmente rejeitadas.

### 5.3 Consistência da Tabela de Baseline (Correção de `C_new, acq` e `C_new, post`)
- **Divergência:** A versão anterior da Tabela 2.A registrou `C_new, acq = 1,00` e `C_new, post = 0,50` para as seeds 42 e 43.
- **Causa Raiz Identificada:** Erro de transcrição tipográfica na confecção da tabela Markdown. O arquivo estruturado [`experiments/phase46_pilot_results.json`](experiments/phase46_pilot_results.json) registra com fidelidade que, no Regime 1 (baseline não-condicionada), as seeds 42, 43 e 44 atingiram todas `c_new_acq = 0.50` (aquisição de apenas 1 das 2 tarefas novas devido à colisão pré-cabeça com as tarefas antigas) e `c_new_post = 0.00` pós-replay (perda total da nova competência ao sofrer sobrescrita pela consolidação das tarefas antigas), resultando identicamente em $\Delta_{\text{new}} = -0{,}5000$ nas três seeds.
- **Resolução:** A Tabela 2.A foi alinhada aos dados brutos do JSON (`0,50 | 0,00 | -0,5000` em todas as seeds), preservando a integridade estrita dos artefatos computacionais.

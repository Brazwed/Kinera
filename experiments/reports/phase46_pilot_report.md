# Relatório Experimental — Fase 46: Desambiguação Contextual

> **Issue:** [#46 — Research: Desambiguação contextual de entradas com alvos conflitantes](https://github.com/Brazwed/Kinera/issues/46)  
> **Branch de Trabalho:** `research/issue-46-contextual-disambiguation`  
> **Commit Base:** `b0502c01ab7d4ce6256fe54e9305d398f10a8702` (`main`)  
> **Artefatos:** [`scripts/run_phase46_pilot.py`](file:///home/brazwed/Projetos/Kinera/Main/scripts/run_phase46_pilot.py), [`experiments/phase46_pilot_results.json`](file:///home/brazwed/Projetos/Kinera/Main/experiments/phase46_pilot_results.json), [`tests/test_regime1_identifiability.py`](file:///home/brazwed/Projetos/Kinera/Main/tests/test_regime1_identifiability.py)

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
4. **Generalização em Palavras Não-Vistas (Teste Independente)**  
   - No holdout diagnóstico (6 palavras conhecidas da Issue #42), o modelo condicionado manteve $94{,}4\%$ de acurácia.
   - Na partição independente (7 palavras não-vistas com `<UNK_LEMMA>`), a acurácia foi de $14{,}3\%$ (nível de chance para caracteres), confirmando que a tabela de lemas opera como indexação direcionada e não confere generalização autônoma de vocabulário aberto.

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

### Tabela 2 — Resultados Individuais por Seed

#### A. Condição 1 — Baseline Não-Condicionada (Regime 1)
| Seed | $d_{\text{cos}}$ | $R_{\text{pos}}$ ($N_{\text{pos}}=3$) | $C_{\text{new, acq}}$ | $C_{\text{new, post}}$ | $\Delta_{\text{new}}$ | Log-Loss | Critérios Atendidos? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **42** | $0{,}0000$ | $3/3$ ($1{,}0000$) | $1{,}00$ | $0{,}50$ | $-0{,}5000$ | $1{,}1654$ | Não |
| **43** | $0{,}0000$ | $3/3$ ($1{,}0000$) | $1{,}00$ | $0{,}50$ | $-0{,}5000$ | $1{,}1644$ | Não |
| **44** | $0{,}0000$ | $3/3$ ($1{,}0000$) | $0{,}50$ | $0{,}00$ | $-0{,}5000$ | $1{,}1805$ | Não |

#### B. Condição 2 — Condicionamento por Lema (Regime 2)
| Seed | $d_{\text{cos}}$ | $R_{\text{pos}}$ ($N_{\text{pos}}=3$) | $C_{\text{new, acq}}$ | $C_{\text{new, post}}$ | $\Delta_{\text{new}}$ | Log-Loss | Critérios Atendidos? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **42** | $0{,}4033$ | $3/3$ ($1{,}0000$) | $1{,}00$ | $1{,}00$ | $+0{,}0000$ | $0{,}5273$ | **Sim** |
| **43** | $0{,}1600$ | $3/3$ ($1{,}0000$) | $1{,}00$ | $1{,}00$ | $+0{,}0000$ | $0{,}8094$ | **Sim** |
| **44** | $0{,}2220$ | $3/3$ ($1{,}0000$) | $1{,}00$ | $0{,}50$ | $-0{,}5000$ | $0{,}7210$ | Não |

#### C. Condição 3 — Controle Negativo (Lema Permutado na Avaliação)
| Seed | $d_{\text{cos}}$ | $R_{\text{pos}}$ ($N_{\text{pos}}=3$) | $C_{\text{new, acq}}$ | $C_{\text{new, post}}$ | $\Delta_{\text{new}}$ | Log-Loss | Critérios Atendidos? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **42** | $0{,}4033$ | $1{,}5/3$ ($0{,}5000$) | $1{,}00$ | $0{,}00$ | $-1{,}0000$ | $2{,}1141$ | Não |
| **43** | $0{,}1600$ | $1/3$ ($0{,}3333$) | $1{,}00$ | $0{,}00$ | $-1{,}0000$ | $1{,}4191$ | Não |
| **44** | $0{,}2220$ | $2/3$ ($0{,}6667$) | $1{,}00$ | $0{,}00$ | $-1{,}0000$ | $1{,}6788$ | Não |

---

## 3. Análise Detalhada dos Mecanismos

### 3.1 Por que o Regime 1 Falha Estruturalmente?
No teste formal [`tests/test_regime1_identifiability.py`](file:///home/brazwed/Projetos/Kinera/Main/tests/test_regime1_identifiability.py), verificou-se que:
$$\vec{x}(\text{mala}, [1]) \equiv \vec{x}(\text{medo}, [1]) = \Big(\text{context\_ids}=[\text{<START>}, \text{'m'}], \text{context\_syllable\_ids}=[\text{<UNK\_SYL>}, \text{<UNK\_SYL>}]\Big)$$
Como o caractere `'m'` não é final de sílaba, o id silábico é estritamente mascarado por causalidade para evitar vazamento do alvo. Portanto, qualquer função determinística $f(\vec{x})$ produz exatamente o mesmo vetor $\vec{h}$ e os mesmos logits $\vec{z}$. Consequentemente:
$$\mathbb{I}(\hat{y}_{\text{mala}} = \text{'a'}) + \mathbb{I}(\hat{y}_{\text{medo}} = \text{'e'}) \le 1$$
Apenas um dos alvos pode vencer o argmax. A busca por ordenação e calibração de LR na Issue #45 não poderia superar essa barreira estrutural.

### 3.2 O Que a Condição de Lema Resolve e o Que Não Resolve?
1. **O que resolve:** A injeção do vetor de lema $\vec{e}_{\text{lemma}}$ no pooling pré-cabeça quebra a simetria de entrada, permitindo representações distintas ($d_{\text{cos}} \ge 0{,}16$). Isso viabilizou a coexistência simultânea em 2 das 3 seeds ($66{,}7\%$).
2. **O que não resolve:** Na Seed 44, mesmo com $d_{\text{cos}} = 0{,}2220$, a consolidação de `mala` e `pato` com $\text{LR}=0{,}20$ atualizou a matriz da cabeça linear compartilhada em direção a `'a'`, reduzindo a probabilidade de `medo` de $0{,}4995$ para $0{,}2885$ (ficando abaixo do limiar de predição). Ou seja: **o condicionamento remove o colapso representacional, mas a dinâmica de interferência retroativa por gradiente continua existindo no espaço paramétrico compartilhado.**

### 3.3 Validação Causal e Limites de Generalização
- **Validação Causal:** A inversão/colapso imediato no controle negativo demonstra que a resposta diferencial decorre especificamente do lema e não de convergência espúria de outros pesos.
- **Limites:** O modelo obteve apenas $14{,}3\%$ na partição independente, demonstrando que a abordagem de tabela de identificadores de lema (`lemma_id`) atua como direcionador de grafia pré-determinada, e não como modelo gerador livre com entendimento contextual semântico.

---

## 4. Recomendações para a Arquitetura Kinera

1. **Formalização Teórica nos Documentos de Pesquisa:** Registrar no repositório que predição de sequências ortográficas com prefixos curtos ($k \le 2$) requer necessariamente sinal de intenção lexical ou contexto estendido se os prefixos colidirem.
2. **Separação de Camadas (Modularidade de Replay):** Para mitigar interferências como a observada na Seed 44, futuras arquiteturas devem desacoplar as projeções contextuais das cabeças lineares de saída ou utilizar regularização de projeção ortogonal durante o replay.
3. **Não mesclar em `src/kinera/`:** Manter a implementação de protótipo isolada em `scripts/` e `experiments/`, preservando a integridade da biblioteca principal da Kinera.

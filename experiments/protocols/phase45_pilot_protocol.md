# Protocolo Experimental: Calibração de Taxa e Agendamento no Replay Direcionado (Issue #45 — Etapa B)

## 1. Identificação e Rastreabilidade

- **Projeto:** Brazwed/Kinera
- **Issue Vinculada:** [#45 — Research — Reduzir interferência retroativa no replay direcionado (Etapa B)](https://github.com/Brazwed/Kinera/issues/45)
- **Issue Antecessora:** [#42 — Research — Experimento ampliado de replay: A/B/C em condições controladas](https://github.com/Brazwed/Kinera/issues/42)
- **Baseline de Partida Aprovado:** `d097e48208d0a2d751fad54cb8256efd0ca63c03` (Merge do PR #44 na branch `main`)
- **Fase de Investigação:** 45 — Calibração de Intensidade e Agendamento de Replay (Etapa B)
- **Ambiente de Execução:** Local CPU (Python 3.12, ambiente puramente determinístico, zero custo de nuvem/GPU)
- **Scripts Executáveis:**
  - [`scripts/run_phase45_pilot.py`](../../scripts/run_phase45_pilot.py) (Calibração de $\text{LR}_{\text{replay}}$)
  - [`scripts/run_phase45_interleaving_pilot.py`](../../scripts/run_phase45_interleaving_pilot.py) (Agendamento de Interleaving)
- **Artefatos de Dados:**
  - [`experiments/phase45_pilot_results.json`](../phase45_pilot_results.json)
  - [`experiments/phase45_interleaving_results.json`](../phase45_interleaving_results.json)

### 1.1 Rastreabilidade Cronológica e Distinção Metodológica
- **Decisões Registradas Publicamente no GitHub (*a priori*):** No corpo da [Issue #45](https://github.com/Brazwed/Kinera/issues/45) (criada em 2026-10-22T15:27:06Z), foram formalizados antes dos pilotos: a pergunta de pesquisa, a hipótese de trabalho (ajustes na intensidade e/ou agendamento de replay), o escopo geral (reproduzir a Variante C congelada como baseline, executar piloto prévio de baixa escala na condição de alta interferência antes de qualquer expansão, e comparar intervenção candidata e referência B sob orçamento comparável, reportando separadamente retenção e plasticidade sem métricas compostas ocultadoras).
- **Especificações Operacionais nas Diretrizes de Sessão (*a priori* interativo, não versionado na issue pública):** Os parâmetros operacionais específicos — a saber: a delimitação a 3 seeds determinísticas (`42`, `43`, `44`), a faixa de variação de $\text{LR}_{\text{replay}} \in [0{,}15; 0{,}18]$ com baseline congelada em $0{,}20$, o orçamento estrito de 4 passos físicos de gradiente (2 novos + 2 replay), as 4 ordens de interleaving, e os limiares quantitativos conjuntos ($R_{\text{pos}} \ge 0{,}70 \land \Delta_{\text{new}} \ge -0{,}25$ exigidos por seed individual) — foram estabelecidos nas diretrizes de trabalho interativo pré-execução com o operador, constando nos registros de diálogo/transcrição da sessão, mas não foram comitados no repositório nem comentados no corpo público da Issue #45 antes da execução dos experimentos.
- **Formalização no Repositório (*a posteriori*):** O presente documento consolida e formaliza *a posteriori* esses requisitos operacionais no repositório, garantindo rastreabilidade permanente e distinguindo com transparência o que constava na issue pública do que foi especificado nas diretrizes interativas de planejamento.

---

## 2. Pergunta de Pesquisa e Hipóteses

### 2.1 Pergunta Central
> É possível reduzir a interferência retroativa causada pelo replay direcionado (Variante C) sobre a tarefa nova, preservando sua vantagem comprovada na retenção de conhecimento antigo, sob orçamento de treinamento estritamente comparável?

### 2.2 Evidência de Partida (Etapa A — Issue #42)
No experimento ampliado de 30 seeds da Issue #42 sob a condição de alta interferência (`high`):
- **Variante A (Sem Replay):** Retenção média $0{,}0000$; degradação catastrófica total nos itens colidentes.
- **Variante B (Random Replay):** Retenção média $0{,}5000$; perda média na tarefa nova de $-0{,}2500$.
- **Variante C (Directed Replay):** Retenção média $0{,}9778$; perda média na tarefa nova de $-0{,}4833$.
- *Diagnóstico da Etapa A:* C recupera quase perfeitamente o conhecimento anterior, mas paga um custo desproporcional na tarefa nova, impedindo superioridade estrita conjunta na vasta maioria das seeds ($3{,}3\%$ de superioridade estrita).

### 2.3 Hipóteses Investigadas na Etapa B
1. **Hipótese 1 (Intensidade do Replay / Calibração de LR):**  
   A taxa de aprendizado padrão de consolidação ($\text{LR}_{\text{replay}} = 0{,}20$) é excessivamente agressiva para o modelo com pesos já adaptados, sobrescrevendo a tarefa recém-adquirida. Uma redução calibrada de $\text{LR}$ na faixa $[0{,}15, 0{,}18]$ poderia estabilizar os logits antigos sem suprimir os novos.
2. **Hipótese 2 (Agendamento / Interleaving de Replay):**  
   A consolidação sequencial padrão da Fase 42 aplica os passos de replay em bloco após os passos da tarefa nova (`[new, new, replay, replay]`). Intercalar passos da tarefa nova com passos de replay durante a consolidação poderia equilibrar a trajetória de otimização dos pesos compartilhados.

---

## 3. Metodologia Experimental e Orçamento Fixo

### 3.1 Condição Experimental
- **Cenário:** Condição de alta interferência (`high`), onde as tarefas novas e antigas compartilham exatamente os mesmos prefixos de 2 tokens:
  - Antigas ($D_{\text{old, train}}$): `mala` (`['<START>', 'm'] \to 'a'`), `pato` (`['<START>', 'p'] \to 'a'`), `sapo`, `fogo`.
  - Novas ($D_{\text{new, train}}$): `medo` (`['<START>', 'm'] \to 'e'`), `povo` (`['<START>', 'p'] \to 'o'`).
- **Amostragem:** Seeds determinísticas `42`, `43` e `44`.

### 3.2 Fases do Ciclo de Treinamento
1. **Fase 1 (Aquisição Antiga):** 20 épocas em $D_{\text{old, train}}$ com $\text{LR} = 0{,}08$. Registro de eventos de memória e competência de referência.
2. **Fase 2 (Interferência / Tarefa Nova):** 15 épocas em $D_{\text{new, train}}$ com $\text{LR} = 0{,}10$.
3. **Fase 3 (Seleção Metacognitiva de Replay):**
   - Avaliação das sondas de controle interno ($P_{\text{old, internal}}$: `malas`, `patos`, `sapos`, `fogos`).
   - Cálculo do `forgetting_score` por sonda e ordenação de prioridade.
   - Em todas as seeds na condição `high`, a Variante C selecionou estritamente `mala` (índice 0) e `pato` (índice 1), pois ambos sofreram esquecimento máximo ($1{,}0$).
4. **Fase 4 (Consolidação sob Orçamento Fixo):**
   - Orçamento estrito: **exatamente 4 passos físicos de gradiente** (2 da tarefa nova + 2 de replay).
   - Verificação física: rastreador de passos (`step_tracker`) observa e valida `actual_new_updates == 2`, `actual_replay_updates == 2` e `total_updates == 4`.

---

## 4. Métricas e Regras de Decisão Pré-definidas

### 4.1 Definições e Denominadores Explícitos
- **$R_{\text{pos}}$ (Retenção Positiva):**  
  Média da razão de retenção calculada estritamente sobre as sondas de controle interno que apresentavam competência de referência positiva ($\text{ref} > 0{,}0$).
  $$\text{Denominador efetivo:} \quad N_{\text{pos}} = 3 \quad (\text{'malas'}, \text{'patos'}, \text{'sapos'})$$
  *Equivalência com a Etapa A:* Esta métrica corresponde exatamente ao cálculo de `mean_retention_ratio` registrado em `seed_results` no artefato `phase42_pilot_results.json`, onde o particionamento de sondas avaliava retenção estritamente sobre sondas com $\text{ref} > 0{,}0$.
- **Retenção Etapa A (Total):**  
  Média da razão de retenção calculada sobre todas as 4 sondas de controle interno (denominador $N = 4$). Sob a função `calculate_retention_ratio`, sondas não aprendidas inicialmente ($\text{ref} \le 0{,}0$) mantidas em zero ou com aquisição espúria contam como fração $1{,}0$. Para a Variante B, isso gera $(0 + 0 + 0 + 1)/4 = 0{,}25$ nas seeds 42 e 44, elevando a média agregada da tabela para $0{,}5000$.
- **Competência Nova ($C_{\text{new, acq}}$, $C_{\text{new, post}}$):**  
  Média da acurácia de caracteres nas duas tarefas novas de treino (`medo` e `povo`).
  $$\text{Denominador efetivo:} \quad N_{\text{new}} = 2 \quad (\text{'medo'}, \text{'povo'})$$
- **Delta da Competência Nova ($\Delta_{\text{new}}$):**  
  $$\Delta_{\text{new}} = C_{\text{new, post}} - C_{\text{new, acq}}$$
- **Probabilidades Contínuas e Log-Loss por Tarefa:**  
  $$P(\text{alvo}) = \text{softmax}(\vec{z})[\text{alvo}], \quad \text{log\_loss} = -\ln P(\text{alvo})$$
  Calculados para cada par concorrente: `['<START>', 'm']` (`malas:a` vs `medo:e`) e `['<START>', 'p']` (`patos:a` vs `povo:o`).

### 4.2 Critérios de Decisão para Expansão
Definidos antes da execução do piloto econômico:
$$\text{Critério de Sucesso:} \quad R_{\text{pos}} \ge 0{,}70 \quad \land \quad \Delta_{\text{new}} \ge -0{,}25$$
- **Regra Operacional:** Uma variante só seria considerada para expansão a 30 seeds caso demonstrasse atender simultaneamente a ambos os critérios em seeds individuais, e não apenas por artefato de agregação de média.

### 4.3 Análise de Compatibilidade Matemática dos Critérios de Aceitação

A análise dos denominadores discretos e da formulação das tarefas demonstra que o critério conjunto $R_{\text{pos}} \ge 0{,}70 \land \Delta_{\text{new}} \ge -0{,}25$ é **estruturalmente incompatível** no cenário sintético avaliado sob predição discreta por $\text{argmax}$:

1. **Discretização de $R_{\text{pos}}$:**
   Com $N_{\text{pos}} = 3$ sondas de controle interno com referência positiva (`malas`, `patos`, `sapos`) e avaliação binária por sonda, os únicos valores possíveis para a métrica por seed individual são:
   $$R_{\text{pos}} \in \left\{ \frac{0}{3},\; \frac{1}{3},\; \frac{2}{3},\; \frac{3}{3} \right\} = \{0{,}0000;\; 0{,}3333;\; 0{,}6667;\; 1{,}0000\}$$
   Como $2/3 \approx 0{,}6667 < 0{,}70$, satisfazer o limiar $R_{\text{pos}} \ge 0{,}70$ exige estritamente $R_{\text{pos}} = 3/3 = 1{,}0000$. O modelo é forçado a acertar 100% das sondas, exigindo que as predições para as entradas colidentes sejam os alvos antigos:
   $$y^*(\text{['<START>', 'm']}) = \text{'a'} \quad \land \quad y^*(\text{['<START>', 'p']}) = \text{'a'}$$

2. **Discretização de $\Delta_{\text{new}}$:**
   Com $N_{\text{new}} = 2$ tarefas novas (`medo`, `povo`) e competência de aquisição prévia $C_{\text{new, acq}} = 0{,}5000$, os valores discretos possíveis de $\Delta_{\text{new}} = C_{\text{new, post}} - C_{\text{new, acq}}$ são:
   $$\Delta_{\text{new}} \in \{0{,}0 - 0{,}5;\; 0{,}5 - 0{,}5;\; 1{,}0 - 0{,}5\} = \{-0{,}5000;\; 0{,}0000;\; +0{,}5000\}$$
   Como $-0{,}5000 < -0{,}25$, satisfazer o limiar $\Delta_{\text{new}} \ge -0{,}25$ exige estritamente $\Delta_{\text{new}} \ge 0{,}0000 \iff C_{\text{new, post}} \ge 0{,}5000$. Ou seja, o modelo é forçado a acertar pelo menos uma das tarefas novas pós-replay, exigindo:
   $$y^*(\text{['<START>', 'm']}) = \text{'e'} \quad \lor \quad y^*(\text{['<START>', 'p']}) = \text{'o'}$$

3. **Incompatibilidade Mútua via Colisão Representacional:**
   Como as representações de entrada $\vec{h}$ são idênticas entre `mala` e `medo`, e entre `pato` e `povo`, a saída discreta $\text{argmax}(\vec{z})$ é determinística e única. Portanto:
   - Se $R_{\text{pos}} = 1{,}0000$ (satisfaz o critério de retenção), o modelo necessariamente prevê `'a'` para ambas as entradas, errando `medo` e `povo`. Logo, $C_{\text{new, post}} = 0{,}0000 \implies \Delta_{\text{new}} = -0{,}5000 < -0{,}25$ (o critério novo falha obrigatoriamente).
   - Se $\Delta_{\text{new}} \ge 0{,}0000$ (satisfaz o critério da tarefa nova), o modelo necessariamente prevê `'e'` ou `'o'` em pelo menos uma das entradas, errando a sonda antiga correspondente. Logo, no máximo 2 das 3 sondas antigas podem estar corretas: $R_{\text{pos}} \le 2/3 \approx 0{,}6667 < 0{,}70$ (o critério de retenção falha obrigatoriamente).

**Conclusão Operacional:** A ausência de execuções aprovadas ($0/42$) decorre dessa incompatibilidade matemática intrínseca sob saídas determinísticas pontuais. Os limiares pré-definidos não são alterados retroativamente, preservando a fidelidade histórica do protocolo; no entanto, a causa de não atendimento fica formalmente caracterizada como decorrência da sobreposição representacional na presença de operadores discretos.

---

## 5. Experimento 1: Calibração de Taxa de Aprendizagem ($\text{LR}_{\text{replay}}$)

### 5.1 Configurações
- Baseline $B_{\text{ref}}$: replay aleatório uniforme com $\text{LR}=0{,}20$.
- Baseline $C_{\text{baseline}}$: replay direcionado congelado com $\text{LR}=0{,}20$.
- Variantes $C$: replay direcionado com $\text{LR} \in \{0{,}15, 0{,}16, 0{,}17, 0{,}18\}$.
- Orçamento: 4 passos (2 novos + 2 replay em posições `(2, 3)`).

### 5.2 Resultados Consolidados

| Configuração | $\text{LR}$ | $R_{\text{pos}}$ (Int.) | Ret. Etapa A | $C_{\text{new, post}}$ | $\Delta_{\text{new}}$ | $P(\text{new})$ | $\text{Loss}_{\text{new}}$ | $P(\text{old\_col})$ | $\text{Loss}_{\text{old\_col}}$ | Gate Indiv. (Todas) |
|---|---|---|---|---|---|---|---|---|---|---|
| **$B_{\text{ref}}$** | $0{,}20$ | $0{,}3333 \pm 0{,}47$ | $0{,}5000$ | $0{,}3333$ | $-0{,}1667 \pm 0{,}24$ | $0{,}3461$ | $1{,}2107$ | $0{,}2952$ | $1{,}4496$ | FAIL (0/3) |
| **$C_{0.15}$** | $0{,}15$ | $0{,}0000 \pm 0{,}00$ | $0{,}2500$ | $0{,}5000$ | $+0{,}0000 \pm 0{,}00$ | $0{,}3081$ | $1{,}1885$ | $0{,}3018$ | $1{,}2049$ | FAIL (0/3) |
| **$C_{0.16}$** | $0{,}16$ | $0{,}2222 \pm 0{,}31$ | $0{,}4167$ | $0{,}5000$ | $+0{,}0000 \pm 0{,}00$ | $0{,}2943$ | $1{,}2348$ | $0{,}3171$ | $1{,}1507$ | FAIL (0/3) |
| **$C_{0.17}$** | $0{,}17$ | $0{,}7778 \pm 0{,}16$ | $0{,}8333$ | $0{,}3333$ | $-0{,}1667 \pm 0{,}24$ | $0{,}2808$ | $1{,}2824$ | $0{,}3323$ | $1{,}1023$ | FAIL (0/3)\* |
| **$C_{0.18}$** | $0{,}18$ | $1{,}0000 \pm 0{,}00$ | $1{,}0000$ | $0{,}0000$ | $-0{,}5000 \pm 0{,}00$ | $0{,}2677$ | $1{,}3310$ | $0{,}3473$ | $1{,}0593$ | FAIL (0/3) |
| **$C_{0.20}$** | $0{,}20$ | $1{,}0000 \pm 0{,}00$ | $1{,}0000$ | $0{,}0000$ | $-0{,}5000 \pm 0{,}00$ | $0{,}2429$ | $1{,}4294$ | $0{,}3759$ | $0{,}9882$ | FAIL (0/3) |

*\*Nota sobre $C_{0.17}$:* A média agregada passa numericamente nos limiares unicamente porque combina 2 seeds onde a tarefa nova venceu ($R_{\text{pos}}=0{,}6667, \Delta_{\text{new}}=0{,}0000$) com 1 seed onde a antiga venceu ($R_{\text{pos}}=1{,}0000, \Delta_{\text{new}}=-0{,}5000$). Nenhuma seed individual atendeu aos dois critérios simultaneamente.

---

## 6. Experimento 2: Agendamento de Interleaving

### 6.1 Configurações de Ordem (Orçamento Fixo = 4 Passos)
1. **Sequencial Padrão:** `[new, new, replay, replay]` (replay nas posições `(2, 3)`).
2. **Intercalado Alternado:** `[new, replay, new, replay]` (replay nas posições `(1, 3)`).
3. **Intercalado Invertido:** `[replay, new, replay, new]` (replay nas posições `(0, 2)`).
4. **Replay no Início:** `[replay, replay, new, new]` (replay nas posições `(0, 1)`).
- Taxas avaliadas: $\text{LR}_{\text{replay}} \in \{0{,}17, 0{,}20\}$ nas seeds `42`, `43` e `44`.

### 6.2 Resultados Consolidados

| Agendamento | Ordem dos Passos | Pos. Replay | $\text{LR}$ | Mean $R_{\text{pos}}$ | Mean $\Delta_{\text{new}}$ | Mean $P(\text{new})$ | Mean $P(\text{old\_col})$ | Joint Log-Loss | Gate Indiv. (Todas) |
|---|---|---|---|---|---|---|---|---|---|
| Sequencial | `[new, new, rep, rep]` | `(2, 3)` | $0{,}17$ | $0{,}7778 \pm 0{,}16$ | $-0{,}1667 \pm 0{,}24$ | $0{,}2808$ | $0{,}3312$ | **$1{,}1941$** | FAIL (0/3) |
| Intercalado | `[new, rep, new, rep]` | `(1, 3)` | $0{,}17$ | $0{,}1111 \pm 0{,}16$ | $+0{,}0000 \pm 0{,}00$ | $0{,}2949$ | $0{,}3070$ | $1{,}2153$ | FAIL (0/3) |
| Interc. Invertido | `[rep, new, rep, new]` | `(0, 2)` | $0{,}17$ | $0{,}0000 \pm 0{,}00$ | $+0{,}0000 \pm 0{,}00$ | $0{,}3446$ | $0{,}2246$ | $1{,}2968$ | FAIL (0/3) |
| Replay no Início | `[rep, rep, new, new]` | `(0, 1)` | $0{,}17$ | $0{,}0000 \pm 0{,}00$ | $+0{,}0000 \pm 0{,}00$ | $0{,}3689$ | $0{,}1831$ | $1{,}3558$ | FAIL (0/3) |
| Sequencial | `[new, new, rep, rep]` | `(2, 3)` | $0{,}20$ | $1{,}0000 \pm 0{,}00$ | $-0{,}5000 \pm 0{,}00$ | $0{,}2429$ | $0{,}4029$ | **$1{,}1695$** | FAIL (0/3) |
| Intercalado | `[new, rep, new, rep]` | `(1, 3)` | $0{,}20$ | $1{,}0000 \pm 0{,}00$ | $-0{,}5000 \pm 0{,}00$ | $0{,}2607$ | $0{,}3719$ | $1{,}1848$ | FAIL (0/3) |
| Interc. Invertido | `[rep, new, rep, new]` | `(0, 2)` | $0{,}20$ | $0{,}0000 \pm 0{,}00$ | $+0{,}0000 \pm 0{,}00$ | $0{,}3297$ | $0{,}2524$ | $1{,}2662$ | FAIL (0/3) |
| Replay no Início | `[rep, rep, new, new]` | `(0, 1)` | $0{,}20$ | $0{,}0000 \pm 0{,}00$ | $+0{,}0000 \pm 0{,}00$ | $0{,}3625$ | $0{,}1953$ | $1{,}3350$ | FAIL (0/3) |

---

## 7. Diagnóstico da Colisão Representacional

A auditoria matemática e de código confirmou que a limitação observada é de natureza representacional da entrada:
1. Para a transição `[1]` (`context_size=2`), os campos `context_ids`, `context_syllable_ids` e `context_feature_vectors` são 100% idênticos entre `mala` e `medo`, e entre `pato` e `povo`.
2. A representação híbrida pós-pooling $\vec{h}$ é idêntica em ponto flutuante: $\vec{h}(\text{mala}) \equiv \vec{h}(\text{medo})$ e $\vec{h}(\text{pato}) \equiv \vec{h}(\text{povo})$.
3. A cabeça linear determinística $\vec{z} = W \vec{h} + b$ gera logits idênticos.
4. Sob previsão discreta por $\text{argmax}(\vec{z})$, a saída é determinística e única. Como os alvos são divergentes (`'a'` vs `'e'`, e `'a'` vs `'o'`), qualquer modelo individual é forçado a escolher um alvo ou outro:
   $$\mathbb{I}(\text{predição} = \text{'a'}) + \mathbb{I}(\text{predição} = \text{'e'}) \le 1$$
5. Por conseguinte, recuperar a tarefa antiga exige $\text{predição} = \text{'a'}$, o que zera a acurácia da tarefa nova ($\Delta_{\text{new}} = -0{,}5000$). Preservar a tarefa nova exige $\text{predição} = \text{'e'}$, o que impede a retenção do alvo antigo ($R_{\text{pos}} \le 2/3 = 0{,}6667 < 0{,}70$).

---

## 8. Resultados Negativos, Limitações e Conclusões Permitidas

### 8.1 Resultados Negativos Documentados
- Nenhuma das 18 execuções do piloto de calibração de $\text{LR}$ atingiu simultaneamente os critérios pré-definidos ($0/18 = 0\%$).
- Nenhuma das 24 execuções do piloto de interleaving atingiu simultaneamente os critérios pré-definidos ($0/24 = 0\%$).

### 8.2 Limitações Metodológicas e Escopo de Generalização
- Os testes foram executados com 3 seeds (`42`, `43`, `44`) sob orçamento econômico estrito de 4 atualizações físicas.
- **Escopo Restrito do Cenário:** Resultados limitados ao cenário sintético de micro-fixtures de alta interferência da Issue #42 (n-gramas com prefixos idênticos de 2 tokens sob decodificação determinística por argmax em cabeça linear de saída única).
- **Não Generalização à Arquitetura Ampla:** Não se afirma que o interleaving ou a calibração de taxa sejam ineficazes sob outros volumes de dados, orçamentos maiores ou arquiteturas dotadas de contexto estendido, recorrência, atenção ou variáveis de tarefa. Afirma-se estritamente que, dentro da formulação representacional avaliada, a busca por ordenação e taxa não soluciona a colisão de alvos divergentes sob entradas idênticas.

### 8.3 Conclusão Permitida
A investigação da Issue #45 cumpriu seus objetivos experimentais, demonstrando de forma reproduzível que ajustes de taxa de aprendizado e ordenação de passos sob orçamento equivalente não justificam expansão adicional, pois a interferência severa decorre da indistinguibilidade representacional das entradas na parametrização atual. A continuidade da pesquisa é formalmente transferida para a investigação de desambiguação contextual.

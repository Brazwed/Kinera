# Protocolo Experimental: Piloto Ampliado de Replay Controlado (Issue #42)

## 1. Identificação e Rastreabilidade

- **Projeto:** Brazwed/Kinera
- **Issue Vinculada:** [#42 — Research — Experimento ampliado de replay: A/B/C em condições controladas](https://github.com/Brazwed/Kinera/issues/42)
- **Issue Pai:** [#34 — Research: Learning Memory e metacognição para aprendizado contínuo](https://github.com/Brazwed/Kinera/issues/34)
- **Baseline Aprovado:** `5dba2ccd6fadd3c2406223e3ec28657cc1fb9138` (Merge do PR #41 na branch `main`)
- **Fase:** 42 — Piloto Experimental Local de Baixo Custo
- **Ambiente de Execução:** CPU Intel Xeon E5-2680 v4 (Ubuntu Linux, Python 3.12, ambiente puramente CPU determinístico)

---

## 2. Pergunta de Pesquisa e Hipóteses

### 2.1 Pergunta Central
> Em condições controladas e com orçamento equivalente de gradiente, o replay direcionado por esquecimento observado (Variante C) melhora a retenção e recuperação de competências antigas em relação ao replay aleatório uniforme (Variante B), sem comprometer desproporcionalmente a aquisição de competências novas (plasticidade)?

### 2.2 Variantes Avaliadas
- **Variante A (Baseline sem replay):** O modelo adquire a tarefa antiga e, em seguida, é treinado exclusivamente na tarefa nova ($N_{new}$ atualizações, 0 atualizações de replay).
- **Variante B (Random Replay):** O modelo adquire a tarefa antiga e, na fase de consolidação/revisão, recebe $N_{new}$ atualizações da tarefa nova intercaladas com $K_{replay}$ atualizações de exemplos antigos sorteados por amostragem pseudo-aleatória uniforme do pool de candidatos.
- **Variante C (Directed Replay):** O modelo adquire a tarefa antiga e, na fase de consolidação/revisão, recebe $N_{new}$ atualizações da tarefa nova intercaladas com $K_{replay}$ atualizações de exemplos antigos ordenados prioritariamente pelo score de esquecimento observado (`forgetting_score`).

### 2.3 Comparações e Regras de Inferência
1. **B vs. A (Benefício do Replay):** Avalia se a inclusão de replay simples atenua o esquecimento catastrófico em relação a não realizar replay.
2. **C vs. B (Superioridade do Direcionamento):** Avalia se a priorização guiada por metacognição observada traz ganho estatisticamente superior sobre o sorteio aleatório sob **orçamento estritamente idêntico**.
3. **Regra de Não-Declaração Precipitada:** Superar a Variante A não autoriza declarar a Variante C superior. A superioridade empírica de C exige vantagem positiva sobre B em recuperação/retenção, não-inferioridade em plasticidade e equiparação rigorosa de orçamento.

---

## 3. Níveis Predefinidos de Interferência

Para superar a limitação das micro-fixtures anteriores (onde o esquecimento era uniforme ou trivial), o piloto define 3 níveis operacionais de interferência baseados na estrutura ortográfica e silábica da língua portuguesa:

| Nível de Interferência | Definição Operacional | Dinâmica Representacional | Esquecimento Esperado no Baseline (A) |
|---|---|---|---|
| **Baixo (`low`)** | Tarefas antigas e novas possuem vocabulários e contextos de prefixo ortogonais (sem sobreposição de n-gramas iniciais ou caracteres críticos). | Mínima colisão nos pesos de projeção e embeddings. | Retenção naturalmente alta ($R \ge 0.85$); baixo esquecimento. |
| **Médio (`medium`)** | Tarefas antigas e novas compartilham caracteres e rimas terminais parciais, mas preservam prefixos distintos. | Perturbação moderada dos pesos compartilhados por gradientes cruzados. | Retenção moderada ($0.40 \le R < 0.85$); degradação parcial. |
| **Alto (`high`)** | Tarefas antigas e novas compartilham o **mesmo prefixo exato** (ex.: `['<START>', 'm']` ou `['<START>', 'c']`), porém competem por caracteres/sílabas de continuação divergentes. | Conflito direto de predição autoregressiva nos mesmos parâmetros de contexto. | Esquecimento catastrófico acentuado ($R < 0.40$); forte degradação dos itens colidentes. |

---

## 4. Partição Quadripartite de Dados, Matriz de Transições e Prevenção Rigorosa de Vazamento

Para garantir validade científica, eliminar artefatos de avaliação holdout e cumprir as exigências metodológicas das Issues #34 e #42, as experiências linguísticas são particionadas em conjuntos estritamente disjuntos tanto em nível lexical quanto em identificadores estruturais (`ExampleRef` com dataset artifact ID e split únicos):

1. **Treinamento Antigo ($D_{old, train}$, split `train`):**
   - Palavras: `mala`, `pato`, `sapo`, `fogo`
   - Identificador: `dataset:pilot-old-train`
   - Exemplos autoregressivos primários utilizados para a aquisição da competência antiga e admitidos na memória de trabalho (`LearningMemory`).
2. **Sondas de Controle Interno ($P_{old, internal}$, split `internal_probe`):**
   - Palavras: `malas`, `patos`, `sapos`, `fogos`
   - Identificador: `dataset:pilot-old-internal`
   - Exemplos ortográficos distintos compartilhando o mesmo contexto de transição (`['<START>', c] -> v`).
   - Utilizadas pelo observador interno do modelo para computar perda, acurácia e o `forgetting_score` dos candidatos no `LearningState`.
   - Utilizadas pela Variante C exclusivamente para priorizar o replay.
3. **Sondas Holdout de Avaliação Final ($P_{old, holdout}$, split `holdout_test`):**
   - Palavras: `maleta`, `pateta`, `sapato`, `foguete`
   - Identificador: `dataset:pilot-old-holdout`
   - Exemplos out-of-sample estritamente isolados da memória e do pipeline de treinamento.
   - **NUNCA** observados durante o treino, **NUNCA** admitidos na memória e **NUNCA** utilizados pela Variante C para estimar esquecimento ou ordenar candidatos.
4. **Treinamento Novo ($D_{new, train}$, split `train`):**
   - Palavras por condição:
     - Baixa: `zero`, `luxo`
     - Média: `medo`, `dado`
     - Alta: `medo`, `povo`
   - Identificador: `dataset:pilot-new-train-{level}`
   - Exemplos utilizados para a aprendizagem da tarefa nova e para o plano intercalado de consolidação.
5. **Sondas Holdout da Nova Tarefa ($P_{new, holdout}$, split `holdout_test`):**
   - Palavras por condição:
     - Baixa: `zebra`, `luva`
     - Média: `meta`, `data`
     - Alta: `meta`, `pote`
   - Identificador: `dataset:pilot-new-holdout-{level}`
   - Utilizadas para avaliar a fixação da nova associação sem contaminação.

### 4.1 Matriz Completa de Transições Autoregressivas

A tabela a seguir documenta explicitamente cada transição gerada para `ex[1]` (`['<START>', onset] -> vowel`), identificando sobreposições intencionais e distinções de contexto:

| Partição | Palavras | Transição (`ex[1]`) | Contexto Autoregressivo | Alvo | Semântica e Natureza da Sobreposição |
|---|---|---|---|---|---|
| **$D_{old, train}$** | `mala`<br>`pato`<br>`sapo`<br>`fogo` | `ma`<br>`pa`<br>`sa`<br>`fo` | `['<START>', 'm']`<br>`['<START>', 'p']`<br>`['<START>', 's']`<br>`['<START>', 'f']` | `'a'`<br>`'a'`<br>`'a'`<br>`'o'` | **Treino Base Antigo:** Estabelece a competência de referência: $C_{ref} = [1.0, 1.0, 1.0, 0.0]$ (as 3 palavras de vogal `'a'` atingem $1.0$, enquanto `fogo` de vogal minoritária `'o'` permanece em $0.0$). Exemplos admitidos na `LearningMemory`. |
| **$P_{old, internal}$** | `malas`<br>`patos`<br>`sapos`<br>`fogos` | `ma`<br>`pa`<br>`sa`<br>`fo` | `['<START>', 'm']`<br>`['<START>', 'p']`<br>`['<START>', 's']`<br>`['<START>', 'f']` | `'a'`<br>`'a'`<br>`'a'`<br>`'o'` | **Sobreposição Intencional de Transição:** Compartilha a transição com $D_{old, train}$ para monitorar o estado interno de esquecimento (`forgetting_score`) sem avaliar os mesmos registros de treino ($C_{ref} = [1.0, 1.0, 1.0, 0.0]$). |
| **$P_{old, holdout}$** | `maleta`<br>`pateta`<br>`sapato`<br>`foguete` | `ma`<br>`pa`<br>`sa`<br>`fo` | `['<START>', 'm']`<br>`['<START>', 'p']`<br>`['<START>', 's']`<br>`['<START>', 'f']` | `'a'`<br>`'a'`<br>`'a'`<br>`'o'` | **Sobreposição Intencional em Registros Isolados:** Compartilha a mesma transição ($C_{ref} = [1.0, 1.0, 1.0, 0.0]$). Isola a medição da **Propriedade A** (ver 4.2) sobre as 3 sondas de referência positiva. |
| **$D_{new, train}$ (`low`)** | `zero`<br>`luxo` | `ze`<br>`lu` | `['<START>', 'z']`<br>`['<START>', 'l']` | `'e'`<br>`'u'` | **Ortogonal:** Contextos e alvos totalmente disjuntos das tarefas antigas. Mínima interferência. |
| **$P_{new, holdout}$ (`low`)** | `zebra`<br>`luva` | `ze`<br>`lu` | `['<START>', 'z']`<br>`['<START>', 'l']` | `'e'`<br>`'u'` | **Sobreposição Intencional:** Sonda holdout da tarefa nova em vocábulos isolados. |
| **$D_{new, train}$ (`medium`)** | `medo`<br>`dado` | `me`<br>`da` | `['<START>', 'm']`<br>`['<START>', 'd']` | `'e'`<br>`'a'` | **Colisão Parcial:** `medo` colide no contexto `['<START>', 'm']` com `mala` (alvo `'e'` vs `'a'`). `dado` é ortogonal. |
| **$P_{new, holdout}$ (`medium`)** | `meta`<br>`data` | `me`<br>`da` | `['<START>', 'm']`<br>`['<START>', 'd']` | `'e'`<br>`'a'` | **Sobreposição Intencional:** Sonda holdout da tarefa nova em vocábulos isolados. |
| **$D_{new, train}$ (`high`)** | `medo`<br>`povo` | `me`<br>`po` | `['<START>', 'm']`<br>`['<START>', 'p']` | `'e'`<br>`'o'` | **Colisão Direta Máxima:** Colide diretamente em `['<START>', 'm']` (vs `'a'`) e `['<START>', 'p']` (vs `'a'`). Provoca esquecimento catastrófico seletivo nos 2 itens colidentes. |
| **$P_{new, holdout}$ (`high`)** | `meta`<br>`pote` | `me`<br>`po` | `['<START>', 'm']`<br>`['<START>', 'p']` | `'e'`<br>`'o'` | **Sobreposição Intencional:** Sonda holdout da tarefa nova em vocábulos isolados. |

### 4.2 Distinção Metodológica Formal: Retenção vs. Generalização

É crucial distinguir com rigor científico o que este piloto avalia:

- **Propriedade A — Retenção de Associações Compartilhadas em Registros Isolados (Shared Transition Retention):**
  - **O que mede:** Avalia se uma associação fonético-ortográfica aprendida em um vocábulo de treino ($D_{old, train}$) é retida pelo modelo em registros lexicais mantidos estritamente fora do treino e fora da memória ($P_{old, holdout}$).
  - **Status no protocolo:** É **exatamente isso** que a métrica `holdout_retention_ratio` mede. Demonstra que o replay direcionado recupera a representação subjacente sem necessitar memorizar todos os tokens da língua.
- **Propriedade B — Generalização para Transições Inéditas (Unseen Transition Generalization):**
  - **O que mede:** Capacidade de prever transições de contexto para as quais o modelo nunca recebeu gradiente em nenhum exemplo.
  - **Status no protocolo:** **NÃO é avaliada**. Em um modelo tabula rasa treinado do zero sem pré-treino linguístico massivo, a competência em transições inéditas é nula ($C_{ref} = 0.0$), o que produziria indefinição ($0.0/0.0$) ou saturação artificial.
- **Propriedade C — Generalização Lexical Completa para Palavras Não Vistas (Full Lexical Generalization):**
  - **O que mede:** Reconhecimento, representação e geração integral de palavras desconhecidas além do prefixo de ataque.
  - **Status no protocolo:** **NÃO é avaliada**, pois o piloto foca na transição crítica de ataque (`ex[1]`), e não na sequência de todos os grafemas da palavra.

### 4.3 Rastreabilidade Estruturada via `TaskFamilySpec` e `NewTaskSpec`

Para eliminar a fragilidade de emparelhamento por listas paralelas posicionais, o protocolo formaliza contêineres tipados explícitos:
- `TaskFamilySpec(family_id, shared_prefix, train_word, internal_probe_word, holdout_probe_word)`
- `NewTaskSpec(task_id, shared_prefix, train_word, holdout_word)`
- Nomes dos grupos de sondas amarrados de forma estrita às palavras reais sob avaliação:
  - Sonda interna: `probe-int-{internal_probe_word}` (ex.: `probe-int-malas`)
  - Sonda holdout: `probe-hold-{holdout_probe_word}` (ex.: `probe-hold-maleta`)
  - Sonda nova: `probe-new-{level}-{train_word}` (ex.: `probe-new-high-medo`)
  - Sonda holdout nova: `probe-new-hold-{level}-{holdout_word}` (ex.: `probe-new-hold-high-meta`)

### 4.4 Auditoria do Seletor C, Scores e Desempates Determinísticos

O algoritmo de seleção da Variante C (`select_directed_replay`):
1. **Cálculo de `forgetting_score`:** Derivado estritamente das sondas internas de controle ($P_{old, internal}$):
   $$\text{forgetting\_score} = \max(0.0, C_{ref} - C_{pre})$$
2. **Priorização e Discriminação de Esquecimento:** Ordena candidatos em ordem estritamente decrescente de `forgetting_score`.
   - **Em Baixa e Média Interferência:** O modelo retém 100% de competência antiga; todos os 4 candidatos apresentam $\text{forgetting\_score} = 0.0$. Portanto, a seleção de C nessas condições opera estritamente via desempate determinístico FIFO (selecionando os índices 0 e 1, `malas` e `patos`), sem discriminação informada por esquecimento.
   - **Em Alta Interferência:** Os scores revelam diferenciação real: três candidatos sofrem esquecimento total ($\text{forgetting\_score} = 1.0$ para `malas`, `patos` e `sapos`), enquanto um item é preservado ($\text{forgetting\_score} = 0.0$ para `fogos`). A Variante C **exclui ativamente** o candidato de score 0.0. Entre os 3 candidatos empatados em score 1.0, seleciona os índices 0 e 1 (`malas` e `patos`) via regra estável FIFO. Em contraste, a Variante B (sorteio uniforme) seleciona `fogos` em 2 de 3 sementes (42 e 44), desperdiçando 50% de seu orçamento em uma experiência não esquecida.
3. **Desempate Determinístico:** Em caso de empate em score, o desempate é feito pela ordem original de admissão no pool (FIFO, determinístico, sem amostragem aleatória).
4. **Isolamento Causal Estrito:** O seletor C recebe apenas `candidates` e `budget`. Ele possui zero dependência e zero acesso aos resultados da avaliação holdout final ($P_{old, holdout}$).
5. **Paridade de Orçamento:** O número de itens selecionados $K_{replay}$ e o número de atualizações físicas de gradiente ($E_{replay} = 1$) são rigorosamente idênticos entre B e C.

---

## 5. Orçamento e Paridade Estrita de Atualizações Físicas de Gradiente

Para assegurar uma comparação justa e isenta de viés de capacidade computacional:
- **Épocas de Replay ($E_{replay} = 1$):** O treinamento de consolidação sobre o plano intercalado executa **exatamente 1 época**, garantindo que cada exemplo do plano gere exatamente uma atualização física de gradiente.
- **Orçamento de Replay ($K_{replay}$):** Rigorosamente igual para as Variantes B e C ($K_B = K_C = 2$). Na Variante A, $K_A = 0$.
- **Atualizações de Tarefa Nova ($N_{new}$):** Rigorosamente igual entre as três variantes ($N_{new} = 2$).
- **Total de Atualizações de Gradiente ($N_{total}$):**
  - Variante A: $N_{total} = N_{new} = 2$ atualizações físicas de gradiente.
  - Variante B: $N_{total} = N_{new} + K_B = 4$ atualizações físicas de gradiente.
  - Variante C: $N_{total} = N_{new} + K_C = 4$ atualizações físicas de gradiente.
- **Rastreabilidade e Auditoria Física em Tempo Real:** Um `step_observer` acoplado ao `train_multitask_examples` inspeciona cada passo físico de treinamento e categoriza a fonte (`new` vs `replay`). O executor valida por meio de asserções estritas que os contadores físicos reais coincidem exatamente com o `ReplayAccounting`:
  $$\text{actual\_new} \equiv N_{new}, \quad \text{actual\_replay} \equiv K_{replay}, \quad \text{actual\_total} \equiv N_{total}$$
- **Check de Conformidade:** Se $N_{total}(B) \ne N_{total}(C)$ ou $K_B \ne K_C$, o protocolo invalida a comparação com o status `INVALID_UNFAIR_BUDGET`.

---

## 6. Múltiplas Seeds e Análise de Dispersão

- **Seeds Avaliadas no Piloto Exploratório:** `[42, 43, 44]` (3 repetições determinísticas, `9` unidades `condição × seed`, artefato `experiments/phase42_pilot_results.json`).
- **Seeds Avaliadas no Experimento Ampliado (T4 CUDA):** `[42, 43, ..., 71]` (30 repetições determinísticas consecutivas, `90` unidades `condição × seed`, artefato `experiments/phase42_30seeds_t4_results.json`).
- **Escopo das Seeds:**
  - Inicialização de pesos do `AutoregressiveModel`.
  - Amostragem pseudo-aleatória uniforme na Variante B (`replay_seed`).
- **Métricas Agregadas:**
  - Média ($\mu$).
  - Desvio padrão ($\sigma$).
  - Mínimo ($\min$) e Máximo ($\max$).
- **Natureza Exploratória vs. Ampliada:** As 3 repetições iniciais serviram para validação metodológica e calibração do protocolo; a execução ampliada de 30 seeds ($N = 30 \ge 10$) avalia a estabilidade estatística das estimativas sob o mesmo vocabulário e orçamento físico.

---

## 7. Métricas Primárias, Partição de Sondas por Competência de Referência e Fórmulas

Todas as métricas são limitadas, finitas e puramente CPU:
1. **Partição Estrita de Sondas: Competência de Referência Positiva ($C_{ref} > 0$) vs. Competência de Referência Nula ($C_{ref} \le 0$):**
   - Após o treino inicial ($E_{old} = 20$), as 3 famílias que compartilham a vogal alvo `'a'` atingem competência de referência $C_{ref} = 1.0$, enquanto a 4ª família (`fogo` $\to$ `'o'`) apresenta competência de referência $C_{ref} = 0.0$ ($C_{ref} = [1.0, 1.0, 1.0, 0.0]$).
   - **Regra Metodológica:** Não se pode "reter" uma competência que nunca foi adquirida ($C_{ref} = 0.0$). Caso uma sonda com $C_{ref} = 0.0$ recebesse razão de retenção $1.0$ sempre que $C_{post} \ge 0.0$, um modelo que perdesse 100% das 3 competências aprendidas ($1.0 \to 0.0$) e adquirisse a 4ª por interferência ($0.0 \to 1.0$) aparentaria reter $0.25$ de seu conhecimento original.
   - Portanto, as sondas de tarefas antigas são particionadas em dois subconjuntos disjuntos:
     - **Subconjunto de Referência Positiva ($S_{ref > 0}$, 3 sondas) — Base exclusiva para Retenção:**
       - Sondas Internas: `["probe-int-malas", "probe-int-patos", "probe-int-sapos"]` ($C_{ref} = [1.0, 1.0, 1.0]$)
       - Sondas Holdout: `["probe-hold-maleta", "probe-hold-pateta", "probe-hold-sapato"]` ($C_{ref} = [1.0, 1.0, 1.0]$)
       - Taxa de Retenção ($R \in [0.0, 1.0]$) calculada **exclusivamente** sobre $i \in S_{ref > 0}$:
         $$R = \frac{1}{|S_{ref > 0}|} \sum_{i \in S_{ref > 0}} \min\left(\max\left(0.0, \frac{C_{post, i}}{C_{ref, i}}\right), 1.0\right)$$
     - **Subconjunto de Referência Nula ($S_{ref \le 0}$, 1 sonda) — Aquisição em Sondas Não Aprendidas Inicialmente:**
       - Sonda Interna: `["probe-int-fogos"]` ($C_{ref} = [0.0]$)
       - Sonda Holdout: `["probe-hold-foguete"]` ($C_{ref} = [0.0]$)
       - Ganho de Aquisição em Referência Nula (`zero_ref_acquisition` / `holdout_zero_ref_acquisition`) calculado **separadamente** sobre $j \in S_{ref \le 0}$:
         $$A_{ref \le 0} = \frac{1}{|S_{ref \le 0}|} \sum_{j \in S_{ref \le 0}} (C_{post, j} - C_{ref, j})$$
2. **Plasticidade Adquirida ($P \in [0.0, 1.0]$):**
   $$P = C_{new, acquired} - C_{new, initial}$$
   - Mede a taxa de aprendizagem na tarefa nova imediatamente após o treinamento interferente e **antes** de qualquer replay.
   - Como B e C adquirem a tarefa nova sob parâmetros idênticos antes do replay, $P_C \equiv P_B$ e portanto a métrica `c_vs_b_plasticity_delta` ($P_C - P_B$) é rigorosamente **0.0** em todas as condições e sementes.
3. **Variação na Nova Tarefa durante o Replay ($\Delta_{new, replay}$):**
   $$\Delta_{new, replay} = C_{new, post} - C_{new, acquired}$$
   - Mede o impacto retroativo gerado pelo replay sobre a tarefa nova recém-adquirida (*back-interference*).
   - Em alta interferência, ao atualizar os pesos para reter os onsets de `mala` e `pato`, a Variante C degrada os onsets de `medo` e `povo` ($\Delta_{new, replay} = -0.50$ em todas as sementes).
   - Essa penalidade explica por que a regra de superioridade estrita rejeita declarar C superior (fração de superioridade 0,0%).
4. **Recuperação de Competência Antiga ($\Delta_{rec}$) vs. Retenção em Referência Positiva ($R$):**
   $$\Delta_{rec} = \frac{1}{4} \sum_{k=1}^{4} (C_{old, post, k} - C_{old, pre, k})$$
   - Mede o ganho marginal líquido de competência obtido durante a fase de replay em relação ao estado imediato pós-interferência sobre as 4 sondas.
   - **Reconciliação Exata em Alta Interferência:**
     - Estado pré-replay após interferência ($C_{old, pre}$): `[0.0, 0.0, 0.0, 1.0]` (média $0.25$).
     - **Variante A (sem replay):** $C_{old, post} = [0.0, 0.0, 0.0, 1.0]$ em todas as sementes $\implies$ Retenção em $S_{ref > 0}$: **$0.0000$**; Aquisição em $S_{ref \le 0}$ (`fogos`/`foguete`): **$+1.0000$**; Recuperação $\Delta_{rec} = \mathbf{0.0000}$.
     - **Variante B (replay aleatório):**
       - Seeds 42 e 44: sorteia `fogos` ($C_{old, post} = [0.0, 0.0, 0.0, 1.0]$) $\implies$ perde todas as 3 competências aprendidas: Retenção $R = \mathbf{0.0000}$, Aquisição em $S_{ref \le 0} = \mathbf{+1.0000}$, Recuperação $\Delta_{rec} = \mathbf{0.0000}$, Trade-off $T = \mathbf{0.0000}$.
       - Seed 43: sorteia `malas` e `patos` ($C_{old, post} = [1.0, 1.0, 1.0, 0.0]$) $\implies$ preserva todas as 3 competências aprendidas: Retenção $R = \mathbf{1.0000}$, Aquisição em $S_{ref \le 0} = \mathbf{0.0000}$, Recuperação $\Delta_{rec} = \mathbf{+0.5000}$, Trade-off $T = \mathbf{0.6667}$.
       - Médias de B nas 3 sementes: $\overline{R}(B) = \mathbf{0.3333}$ ($\sigma = 0.4714$), $\overline{A}_{ref \le 0}(B) = \mathbf{0.6667}$ ($\sigma = 0.4714$), $\overline{\Delta_{rec}}(B) = \mathbf{0.1667}$ ($\sigma = 0.2357$), $\overline{T}(B) = \mathbf{0.2222}$ ($\sigma = 0.3143$).
     - **Variante C (replay direcionado):** exclui `fogos` (score $0.0$) e seleciona `malas` e `patos` em todas as sementes ($C_{old, post} = [1.0, 1.0, 1.0, 0.0]$) $\implies$ $\overline{R}(C) = \mathbf{1.0000}$ ($\sigma = 0.0$), $\overline{A}_{ref \le 0}(C) = \mathbf{0.0000}$ ($\sigma = 0.0$), $\overline{\Delta_{rec}}(C) = \mathbf{+0.5000}$ ($\sigma = 0.0$), $\overline{T}(C) = \mathbf{0.6667}$ ($\sigma = 0.0$).
     - **Comparações C vs. B em Alta Interferência:**
       - **Vantagem Média de Recuperação (`c_vs_b_recovery_advantage`):** $\overline{\Delta_{rec}}(C) - \overline{\Delta_{rec}}(B) = 0.5000 - 0.1667 = \mathbf{+0.3333}$ ($\sigma = 0.2357$).
       - **Vantagem Média de Retenção em Referência Positiva (`c_vs_b_retention_advantage` e `c_vs_b_holdout_retention_advantage`):** $\overline{R}(C) - \overline{R}(B) = 1.0000 - 0.3333 = \mathbf{+0.6667}$ ($\sigma = 0.4714$).
       - **Vantagem Média de Trade-off (`c_vs_b_tradeoff_advantage`):** $\overline{T}(C) - \overline{T}(B) = 0.6667 - 0.2222 = \mathbf{+0.4444}$ ($\sigma = 0.3143$).
5. **Trade-off Estabilidade-Plasticidade ($T \in [0.0, 1.0]$):**
   $$T = \begin{cases} \frac{2 \cdot R \cdot P}{R + P}, & \text{se } R > 0 \text{ e } P > 0 \\ 0.0, & \text{caso contrário} \end{cases}$$
6. **Métricas de Recursos e Memória:**
   - Tempo de parede efetivo de treinamento/avaliação em segundos (`cpu_time_seconds` / `total_cpu_time_seconds`) medido via `time.perf_counter()`, separado do tempo de inicialização/compilação de backend (`backend_init_seconds`).
   - Pico de Heap Python em bytes (`peak_ram_bytes` / `peak_ram_mb`) medido via `tracemalloc` **apenas quando o modo de profiling de heap é solicitado explicitamente (`--profile-heap`)**, permanecendo desativado por padrão (`profile_heap = False`) nas execuções normais para eliminar o overhead de ~`11,3x` imposto pelo rastreamento de alocações sobre listas Python no `CPUBackend`.
   - Pico de Resident Set Size (RSS real do processo no SO) em bytes (`peak_rss_bytes` / `peak_rss_mb`) medido sempre via `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024` no Linux (custo computacional nulo).
7. **Fórmula de Projeção Paramétrica de Escala:**
   $$\text{Scale Factor} = \left(\frac{N_{\text{seeds, target}}}{N_{\text{seeds, pilot}}}\right) \times \left(\frac{V_{\text{words, target}}}{V_{\text{words, pilot}}}\right) \times \left(\frac{E_{\text{epochs, target}}}{E_{\text{epochs, pilot}}}\right) = \frac{10}{3} \times \frac{500}{6} \times \frac{50}{36} \approx 385{,}8025\times$$
   $$\text{Projected Hours (Single Core)} = \frac{\text{Pilot CPU Seconds} \times \text{Scale Factor}}{3600}$$
   $$\text{Estimated Parallel Time} = \frac{\text{Projected Hours}}{N_{\text{parallel cores}}} \times 60 \text{ minutos}$$

---

## 8. Gates de Decisão, Medições Reais vs. Projeções e Infraestrutura Experimental

O piloto experimental serve como instrumento objetivo de decisão e separa rigorosamente **medições empíricas reais** de **extrapolações teóricas**:

- **Gate 1 (Validação Técnica):** O protocolo executa com sucesso, sem vazamento de dados, com paridade estrita de orçamento físico de gradientes, isolamento causal de C, separação entre retenção em referência positiva ($S_{\text{ref} > 0}$) e aquisição em referência zero ($S_{\text{ref} \le 0}$), e métricas determinísticas e reproduzíveis por seed e no agregado. O protocolo é considerado **TECNICAMENTE VÁLIDO**.
- **Gate 2 (Medições Reais vs. Projeções de Custo Computacional):**
  1. **Medições Reais no Piloto de 3 Seeds (3 condições $\times$ 3 seeds = 9 unidades = 27 treinamentos A/B/C, $V_{\text{syl}} = 27$, `2.772` passos físicos totais por backend):**
     - **Com `tracemalloc` ativo (`--profile-heap`, artefato histórico do PR #43):** `166,87 s` totais (~`18,54 s` por unidade condição-seed), pico de heap Python `0,05 MB` (`51.995 bytes`) e pico de RSS real do SO `18,62 MB`.
     - **Sem `tracemalloc` em CPU (padrão nas execuções normais, `profile_heap=False`):**
       - **PC Local (CPU):** `14,7736 s` totais (`1,6415 s` por unidade; `5,330 ms/passo`; RSS `24,87 MB`).
       - **Google Colab VM (`Intel(R) Xeon(R) CPU @ 2.00GHz`, 2 vCPUs):** `14,9429 s` (cold run, `5,391 ms/passo`) e `14,8852 s` (warm run, `5,370 ms/passo`; RSS `104,25 MB`), confirmando que o hook `tracemalloc` inflava o tempo de CPU em **~`11,3x`**.
     - **Medição Real no Google Colab com GPU `NVIDIA Tesla T4` (Compute Capability `7.5`, `15360 MiB`, Driver `580.82.07`, CUDA `13.0` / `nvcc V13.0.88`, commit `2f29fe1880c5881a9be21bb9d99c1ed11472e214`):**
       - **Cold Start (inclui compilação `nvcc` da extensão nativa CUDA + warmup):** `backend_init_seconds = 14,5790 s`, `training_wall_time_seconds = 11,3312 s` (`4,088 ms/passo`), `total_wall_time_seconds = 25,9102 s` (speedup de treino vs. CPU Colab: **`1,32x`**; speedup total incluindo compilação: **`0,58x`**).
       - **Warm Start (biblioteca `.so` já compilada em cache):** `backend_init_seconds = 0,2925 s` (`0,2296 s` no runner completo do piloto), `training_wall_time_seconds = 11,3560 s` (`4,097 ms/passo`), `total_wall_time_seconds = 11,6484 s` (speedup de treino vs. CPU Colab: **`1,31x`**; speedup total c/ init: **`1,28x`**).
       - **Equivalência Científica e Numérica CPU vs. CUDA:** `scientific_equivalence_verified = True` nas `9/9` unidades e nas agregações por condição, com erro absoluto máximo **`max_absolute_error = 0.0` (`0.000e+00`)** (`atol = 1e-12`).
       - **Auditoria de Recursos e Buffers CUDA:** `175.086` alocações e `175.086` liberações por execução do piloto de 3 seeds, com **`live_allocations_after_close = 0`** e **`unmatched_frees = 0`**.
     - **Medição Real do Custo por Passo vs. Vocabulário Silábico ($V_{\text{syl}}$) no `CPUBackend` (sem `tracemalloc`):**
       - $V_{\text{syl}} = 27$ (piloto atual): `5,18 ms/passo`
       - $V_{\text{syl}} = 150$: `14,13 ms/passo` (`2,73x`)
       - $V_{\text{syl}} = 300$ (~500 palavras): `25,50 ms/passo` (`4,92x`)
       - *(Ressalva metodológica: o ganho de `1,32x` da T4 sobre a CPU foi medido para o vocabulário atual $V_{\text{syl}}=27$; eventual ganho adicional da GPU em vocabulários ampliados permanece uma hipótese a ser medida empiricamente quando o léxico for expandido).*
  2. **Projeções Paramétricas Extrapoladas (10 seeds, 500 palavras, 50 épocas; fator linear $385{,}8025\times$):**
     - **Extrapolação linear a partir do piloto com `tracemalloc` (`166,87 s`):** $\frac{166{,}87 \times 385{,}8025}{3600} = \mathbf{17{,}88\text{ horas}}$ de CPU single-core (~`134,1 min` em 8 cores). *(Nota histórica: uma estimativa anterior de `~3,8 horas` / `~20–30 min` havia aplicado o fator de escala sobre ~`35,5 s` de 2 seeds de uma única condição em vez das 9 unidades do piloto completo).*
     - **Extrapolação linear a partir do piloto sem `tracemalloc` (~`15,0 s` em CPU / ~`11,3 s` em T4, assumindo custo por passo fixo em $V_{\text{syl}}=27$):** $\mathbf{1{,}61\text{ horas}}$ em CPU single-core (~`12,1 min` em 8 cores) e $\mathbf{1{,}21\text{ horas}}$ em GPU T4.
     - **Projeção calibrada sem `tracemalloc` incorporando o crescimento medido de $V_{\text{syl}} \approx 300$ (`4,92x` no `CPUBackend`):** $\mathbf{\sim 7{,}8\text{ a }8{,}0\text{ horas}}$ single-core (~`60 min` distribuídos em 8 cores locais).
  3. **Suporte a Execução Reproduzível, Checkpoint/Resume e Microbenchmark CPU vs. CUDA (T4):**
     - O runner (`scripts/run_expanded_replay_pilot.py`) suporta `--backend cpu|cuda` (`cpu` por padrão), `--profile-heap` opcional, `--seeds` parametrizáveis, checkpoint atômico incremental a cada par `(condition, seed)` e retomada validada via `--resume` (rejeitando configurações ou backends incompatíveis sem duplicar resultados).
     - Os resultados consolidados do microbenchmark e os logs brutos de execução na T4 estão versionados em `experiments/phase42_cpu_vs_t4_benchmark.json` e `experiments/phase42_cpu_vs_t4_benchmark.log`.

---

## 9. Execução Ampliada de 30 Seeds na NVIDIA Tesla T4 (`experiments/phase42_30seeds_t4_results.json`)

No commit validado do runner (`2f29fe1880c5881a9be21bb9d99c1ed11472e214`), o experimento foi ampliado de 3 para **30 seeds determinísticas consecutivas (`42` a `71`)** nas 3 condições (`low`, `medium`, `high`), totalizando **90 unidades `(condição, seed)`**, **270 modelos A/B/C treinados** e **`27.720` passos físicos de gradiente**.

### 9.1 Validação Operacional: Checkpoint Incremental, Retomada (`--resume`) e Determinismo

1. **Execução em 2 etapas com checkpoint e `--resume` (`experiments/phase42_30seeds_t4_results.json`):**
   - **Pass 1 (15 seeds: `42..56`, 45 unidades):** executado em `--backend cuda` em **`57,51 s`** de treino (`backend_init_seconds = 0,2930 s`), persistindo o checkpoint com `45` unidades concluídas.
   - **Pass 2 (`--resume` para 30 seeds: `42..71`, 90 unidades totais):** retomado a partir do checkpoint sem reexecutar as `45` unidades iniciais (`resumed = True`, `resume_count = 1`, `resumed_completed_units = 45`) e concluindo as `45` unidades restantes em **`56,45 s`** (`backend_init_seconds = 0,2348 s`), totalizando **`113,96 s`** de tempo acumulado de treino (`116,13 s` de parede total; pico RSS `133,08 MB`).
2. **Verificação Cruzada Ininterrupta (Single-Pass de 90 Unidades) e Paridade com Referência CPU:**
   - Uma execução contínua das 30 seeds na T4 (`Resumed Run = False`, `113,66 s` de treino, `init = 0,3019 s`) foi comparada via `--verify-against` com o artefato retomado de 90 unidades: **`checked_units = 90`, `max_abs_error = 0.000e+00`**.
   - As 3 primeiras seeds (`42, 43, 44`, 9 unidades) dentro da execução de 30 seeds na T4 foram comparadas com o artefato de referência em CPU (`experiments/phase42_pilot_results.json`): **`checked_units = 9`, `max_abs_error = 0.000e+00`**.
   - Em ambas as passagens em CUDA, `_CUDALifecycleTracker` confirmou **zero alocações vivas remanescentes** após o fechamento dos 270 modelos.

### 9.2 Resultados Estatísticos Agregados: Piloto (3 Seeds) vs. Experimento Ampliado (30 Seeds)

| Condição | Métrica | Piloto 3 Seeds (`42..44`) $\mu \pm \sigma$ | Ampliado 30 Seeds (`42..71`) $\mu \pm \sigma$ | Intervalo `[min, max]` (30 Seeds) | Interpretação Científica na Escala de 30 Seeds |
|---|---|---:|---:|---:|---|
| **`low`** | `retention_ratio` (A / B / C) | `1.0000` / `1.0000` / `1.0000` | `1.0000` / `1.0000` / `1.0000` ($\sigma=0.0$) | `[1.0, 1.0]` | Retenção perfeita das 3 sondas de referência positiva em 30/30 seeds (internas e holdout). |
| **`low`** | `c_vs_b_retention_advantage` | `0.0000` $\pm 0.0000$ | **`0.0000` $\pm 0.0000$** | `[0.0, 0.0]` | Nenhuma diferença de retenção em referências positivas. |
| **`low`** | `zero_ref_acquisition` (A / B / C) | `0.0000` / `0.0000` / `0.0000` | `0.0000` / **`0.0333` $\pm 0.1795$** / `0.0000` | B: `[0.0, 1.0]` | Na `seed=50` (1/30), B sorteou `fogos` e adquiriu a 4ª sonda ($0 \to 1$). A partição $S_{\text{ref}>0}$ impediu distorção em `retention_ratio`, enquanto $\Delta_{\text{rec}}(C)-\Delta_{\text{rec}}(B)$ registrou `-0.0083` $\pm 0.0449$. |
| **`medium`** | `retention_ratio` (A / B / C) | `1.0000` / `1.0000` / `1.0000` | `1.0000` / `1.0000` / `1.0000` ($\sigma=0.0$) | `[1.0, 1.0]` | Retenção perfeita e zero esquecimento em 30/30 seeds; todas as vantagens C vs. B iguais a `0.0000`. |
| **`high`** | `retention_ratio` — Variante A | `0.0000` $\pm 0.0000$ | **`0.0000` $\pm 0.0000$** | `[0.0, 0.0]` | Esquecimento catastrófico total das 3 competências positivas em 30/30 seeds sem replay. |
| **`high`** | `retention_ratio` — Variante B | `0.3333` $\pm 0.4714$ | **`0.5000` $\pm 0.5000$** | `[0.0, 1.0]` | Converge de `1/3` (`0.3333`) para **`15/30` (`0.5000`)**: B sorteia `fogos` (score `0.0`) em 14/30 seeds e `['patos','sapos']` na `seed=51`, perdendo as 3 competências positivas em 15/30 seeds e recuperando-as nas outras 15/30 seeds. |
| **`high`** | `retention_ratio` — Variante C | `1.0000` $\pm 0.0000$ | **`0.9778` $\pm 0.1197$** | `[0.3333, 1.0]` | C exclui `fogos` em 30/30 seeds: recupera `1.0000` (3/3 sondas) em **29/30 seeds** e `0.3333` (1/3 sonda, `malas`) na `seed=51` (`88/90 = 0.9778`), com paridade exata no holdout. |
| **`high`** | `c_vs_b_retention_advantage` (interno e holdout) | `+0.6667` $\pm 0.4714$ | **`+0.4778` $\pm 0.4920$** | `[0.0, +1.0]` | Vantagem positiva consistente de C sobre B em retenção de referências positivas ($0.9778 - 0.5000 = +0.4778$). |
| **`high`** | `c_vs_b_recovery_advantage` | `+0.3333` $\pm 0.2357$ | **`+0.2417` $\pm 0.2457$** | `[0.0, +0.50]` | Vantagem média de recuperação marginal sobre as 4 sondas ($0.4917 - 0.2500 = +0.2417$). |
| **`high`** | `c_vs_b_tradeoff_advantage` | `+0.4444` $\pm 0.3143$ | **`+0.3244` $\pm 0.3278$** | `[0.0, +0.6667]` | Vantagem média no índice harmônico estabilidade-plasticidade. |
| **`high`** | `zero_ref_acquisition` (A / B / C) | `1.0000` / `0.6667` / `0.0000` | `1.0000` / **`0.5000` $\pm 0.5000$** / **`0.0333` $\pm 0.1795$** | B: `[0.0, 1.0]`<br>C: `[0.0, 1.0]` | Ganho na 4ª sonda (`fogos`/`foguete`) permanece em `1.0` apenas quando o replay não reverte o viés vocálico para `'a'` (15/30 seeds em B; 1/30 seed em C na `seed=51`). |
| **`high`** | `new_competence_delta_during_replay` (B / C) | `-0.1667` / `-0.5000` | **`-0.2500` $\pm 0.2500$** / **`-0.4833` $\pm 0.0898$** | B: `[-0.5, 0.0]`<br>C: `[-0.5, 0.0]` | Sempre que o replay recupera as 3 palavras de vogal `'a'` (29/30 seeds em C; 15/30 em B), ocorre perda retroativa de `-0.50` na tarefa nova (*back-interference*). |
| **`high`** | `superiority_fraction` (`is_directed_superior`) | `0.0%` (`0/3`) | **`3.3%` (`1/30`, `seed=51`)** | — | Na `seed=51`, C recupera parcialmente a tarefa antiga (`R=0.3333` vs. `0.0000` de B) sem sofrer perda retroativa na tarefa nova ($\Delta_{\text{new, replay}} = 0.0$), satisfazendo todos os critérios estritos de superioridade nessa seed. Nas demais 29 seeds, a perda retroativa de `-0.50` na tarefa nova impede declarar superioridade estrita. |

*(Nota de auditoria sobre `conclusions.summary_claim`: no código congelado do commit `2f29fe1880c5881a9be21bb9d99c1ed11472e214`, os campos numéricos de `conclusions` e `condition_aggregates` são calculados dinamicamente sobre as 30 seeds, mas a string descritiva `summary_claim` mantém alguns trechos literais da redação de 3 seeds — como `with C retaining 1.0000 vs B retaining 0.3333` e `in 2 of 3 seeds` — ao lado dos valores interpolados de 30 seeds `+0.2417`, `+0.4778`, `0.5000` e `0.0333`. A tabela acima registra a reconciliação exata das 30 seeds).*

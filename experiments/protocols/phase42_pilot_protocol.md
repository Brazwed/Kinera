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

- **Seeds Avaliadas no Piloto:** `[42, 43, 44]` (3 repetições determinísticas).
- **Escopo das Seeds:**
  - Inicialização de pesos do `AutoregressiveModel`.
  - Amostragem pseudo-aleatória uniforme na Variante B (`replay_seed`).
- **Métricas Agregadas:**
  - Média ($\mu$).
  - Desvio padrão ($\sigma$).
  - Mínimo ($\min$) e Máximo ($\max$).
- **Natureza Exploratória:** As 3 repetições do piloto têm propósito estritamente metodológico e de calibração; conclusões definitivas sobre superioridade empírica requerem replicação ampliada ($N \ge 10$ seeds).

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
   - Tempo de CPU em segundos (`cpu_time_seconds`) medido via `time.perf_counter()`.
   - Pico de Heap Python em bytes (`peak_ram_bytes` / `peak_ram_mb`) medido via `tracemalloc`.
   - Pico de Resident Set Size (RSS real do processo no SO) em bytes (`peak_rss_bytes` / `peak_rss_mb`) medido via `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024` no Linux.
7. **Fórmula de Projeção Paramétrica de Escala:**
   $$\text{Scale Factor} = \left(\frac{N_{\text{seeds, target}}}{N_{\text{seeds, pilot}}}\right) \times \left(\frac{V_{\text{words, target}}}{V_{\text{words, pilot}}}\right) \times \left(\frac{E_{\text{epochs, target}}}{E_{\text{epochs, pilot}}}\right)$$
   $$\text{Projected Hours (Single Core)} = \frac{\text{Pilot CPU Seconds} \times \text{Scale Factor}}{3600}$$
   $$\text{Estimated Parallel Time} = \frac{\text{Projected Hours}}{N_{\text{parallel cores}}} \times 60 \text{ minutos}$$

---

## 8. Gates de Decisão e Recomendação de Execução Local

O piloto experimental local serve como instrumento objetivo de decisão para o escalonamento da pesquisa:
- **Gate 1 (Validação Técnica):** O protocolo executa com sucesso, sem vazamento de dados, com paridade estrita de orçamento físico de gradientes, isolamento causal de C e métricas determinísticas e reproduzíveis. O protocolo é considerado **TECNICAMENTE VÁLIDO**.
- **Gate 2 (Custo, Projeção e Viabilidade Local):**
  - O piloto executa na máquina local em ~18-20 segundos por seed sob alta interferência com consumo irrisório de memória (heap Python $< 0.1$ MB, RSS real $< 120$ MB).
  - A projeção paramétrica extrapolada para um experimento ampliado (10 seeds, 500 palavras, 50 épocas) totaliza ~3.8 horas de tempo de CPU em core único.
  - Como a execução por seed é embaraçosamente paralela (*embarrassingly parallel*), a distribuição em 8-10 cores na CPU local (28 vCPUs disponíveis) conclui a execução em **~20 a 30 minutos**.
  - **Recomendação Objetiva:** **Execução Local em Multi-Core CPU (`local_cpu`)**. Não há necessidade nem justificativa técnica medida para migração para Google Colab Pro ou GPUs pagas nesta etapa, pois o overhead de kernel de GPU para modelos autoregressivos compactos não foi benchmarkado e os recursos locais atendem plenamente ao estudo.

# Protocolo Experimental: Piloto Ampliado de Replay Controlado (Issue #42)

## 1. Identificação e Rastreabilidade

- **Projeto:** Brazwed/Kinera
- **Issue Vinculada:** [#42 — Research — Experimento ampliado de replay: A/B/C em condições controladas](https://github.com/Brazwed/Kinera/issues/42)
- **Issue Pai:** [#34 — Research: Learning Memory e metacognição para aprendizado contínuo](https://github.com/Brazwed/Kinera/issues/34)
- **Baseline Aprovado:** `5dba2cc` (Merge do PR #41 na branch `main`)
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

## 4. Partição Quadripartite de Dados e Prevenção Rigorosa de Vazamento (No Data Leakage)

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
   - Exemplos out-of-sample estritamente isolados.
   - **NUNCA** observados durante o treino, **NUNCA** admitidos na memória e **NUNCA** utilizados pela Variante C para estimar esquecimento ou ordenar candidatos.
   - Como compartilham a transição fonético-ortográfica inicial com os itens antigos, o modelo adquire competência de referência real ($C_{ref} > 0.0$) após o treino antigo. Sob interferência, essa competência degrada e, com o replay adequado, recupera-se. Isso elimina definitivamente o artefato de saturação artificial ($0.0/0.0 \to 1.0$).
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
   - Utilizadas para avaliar a generalização genuína da competência recém-adquirida sem contaminação.

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

## 7. Métricas Primárias, Fórmulas e Projeção de Recursos

Todas as métricas são limitadas, finitas e puramente CPU:
1. **Taxa de Retenção ($R \in [0.0, 1.0]$):**
   $$\text{Retention Ratio} = \min\left(\max\left(0.0, \frac{C_{post}}{C_{ref}}\right), 1.0\right)$$
2. **Plasticidade Adquirida ($P \in [0.0, 1.0]$):**
   $$P = C_{new, acquired} - C_{new, initial}$$
3. **Variação na Nova Tarefa durante o Replay ($\Delta_{new, replay}$):**
   $$\Delta_{new, replay} = C_{new, post} - C_{new, acquired}$$
4. **Recuperação de Competência Antiga ($\Delta_{rec}$):**
   $$\Delta_{rec} = C_{old, post} - C_{old, pre}$$
5. **Trade-off Estabilidade-Plasticidade ($T \in [0.0, 1.0]$):**
   $$T = \begin{cases} \frac{2 \cdot R \cdot P}{R + P}, & \text{se } R > 0 \text{ e } P > 0 \\ 0.0, & \text{caso contrário} \end{cases}$$
6. **Métricas de Recursos e Memória:**
   - Tempo de CPU em segundos (`cpu_time_seconds`) medido via `time.perf_counter()`.
   - Pico de Heap Python em bytes (`peak_ram_bytes` / `peak_ram_mb`) medido via `tracemalloc`.
   - Pico de Resident Set Size (RSS real do processo no SO) em bytes (`peak_rss_bytes` / `peak_rss_mb`) medido via `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024` no Linux.
7. **Fórmula de Projeção de Escala:**
   $$\text{Scale Factor} = \left(\frac{N_{\text{seeds, target}}}{N_{\text{seeds, pilot}}}\right) \times \left(\frac{V_{\text{words, target}}}{V_{\text{words, pilot}}}\right) \times \left(\frac{E_{\text{epochs, target}}}{E_{\text{epochs, pilot}}}\right)$$
   $$\text{Projected Hours} = \frac{\text{Pilot CPU Seconds} \times \text{Scale Factor}}{3600}$$

---

## 8. Gates de Decisão: Piloto Local vs. Google Colab Pro

O piloto experimental local serve como instrumento objetivo de decisão para o escalonamento da pesquisa:
- **Gate 1 (Validação Técnica):** Se o protocolo executar com sucesso, sem vazamento, com paridade estrita de orçamento e produzir métricas determinísticas e reproduzíveis, o protocolo é considerado **TECNICAMENTE VÁLIDO**.
- **Gate 2 (Custo e Projeção):**
  - Se o piloto rodar na máquina local em tempo aceitável ($< 30$ segundos) e com consumo insignificante de RAM ($< 100$ MB), calcularemos a projeção para um experimento em grande escala (ex.: 10 seeds, vocabulário completo de milhares de palavras).
  - Caso a projeção em grande escala caiba nos 28 vCPUs e 16 GB de RAM da máquina local em menos de 10-15 minutos, recomenda-se **Execução Local** para menor atrito e reprodutibilidade imediata.
  - Caso a projeção demande aceleração massiva por GPU (CUDA) ou dezenas de horas em CPU, recomenda-se formalmente o chaveamento para o **Google Colab Pro** (utilizando o MCP `colab-mcp`).

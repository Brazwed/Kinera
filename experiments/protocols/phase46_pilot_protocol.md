# Protocolo Experimental: Desambiguação Contextual de Entradas com Alvos Conflitantes (Issue #46 — Fase 1)

## 1. Identificação e Rastreabilidade

- **Projeto:** Brazwed/Kinera
- **Issue Vinculada:** [#46 — Research — Desambiguação contextual de entradas com alvos conflitantes](https://github.com/Brazwed/Kinera/issues/46)
- **Issues Antecessoras:** [#42 (Etapa A)](https://github.com/Brazwed/Kinera/issues/42) e [#45 (Etapa B)](https://github.com/Brazwed/Kinera/issues/45)
- **Commit Base da Investigação:** `b0502c01ab7d4ce6256fe54e9305d398f10a8702` (Merge do PR #47 na branch `main`)
- **Branch Isolada:** `research/issue-46-contextual-disambiguation`
- **Ambiente de Execução:** Local CPU (Python 3.12, determinístico, custo zero de nuvem/GPU)

---

## 2. Pergunta de Pesquisa e Hipóteses

### 2.1 Pergunta Central
> Qual informação adicional, se houver, permite desambiguar entradas com prefixos idênticos e alvos diferentes (`mala/medo`, `pato/povo`) sem vazamento causal, memorização ad hoc ou mudança não declarada da tarefa de inferência?

### 2.2 Diagnóstico da Causa Raiz (Etapas A e B)
Na transição `[1]` (`context_size=2`), os campos `context_ids`, `context_syllable_ids` e `context_feature_vectors` são 100% idênticos entre `mala` e `medo` (`['<START>', 'm']`), e entre `pato` e `povo` (`['<START>', 'p']`). Sob pooling linear determinístico e predição por $\text{argmax}$, o modelo é estruturalmente forçado a emitir uma predição única. A convivência funcional é matematicamente impossível sem informação discriminativa na representação.

### 2.3 Hipóteses Investigadas

1. **Hipótese 0 ($H_0$ — Controle Negativo de Causalidade Autoregressiva):**  
   No regime autoregressivo puro não-condicionado, nenhum atributo subléxico intra-palavra causalmente legítimo gera $d_{\text{cos}} > 0{,}00$ no prefixo inicial. Qualquer sinal silábico que desambigue a primeira vogal antes de sua emissão constitui vazamento causal direto do alvo.
2. **Hipótese 1 ($H_{2b}$ — Desambiguação Condicionada por Lema / Contexto Lexical):**  
   No regime de geração dirigida por objetivo lexical, a injeção de uma representação de lema $\vec{e}_{\text{lex}}$ discriminativa:
   - Produz separabilidade representacional pré-cabeça: $d_{\text{cos}}(\vec{h}_{\text{mala}}, \vec{h}_{\text{medo}}) \ge 0{,}10$.
   - Permite coexistência funcional no modelo: $R_{\text{pos}} = 1{,}0000 \land \Delta_{\text{new}} \ge 0{,}0000$ por seed sob predição determinística por $\text{argmax}$.
3. **Hipótese 2 ($H_3$ — Limites de Generalização e Zero-Shot):**  
   O condicionamento por ID de lema memoriza palavras conhecidas, mas falha em transferir para a partição independente não-vista, demonstrando que a generalização ampla exige atributos morfológicos/composicionais e não apenas chaves categóricas de lema.

---

## 3. Regimes de Inferência e Validade Causal

O protocolo distingue explicitamente dois regimes de inferência que não devem ser confundidos:

### 3.1 Regime 1: Predição Autoregressiva Pura Não-Condicionada
- **Definição:** Entrada consiste unicamente no fluxo temporal de caracteres à esquerda ($c_{<t}$).
- **Natureza do Problema:** Distribuição estocástica multimodal ($P(\text{'a'} \mid m) > 0, P(\text{'e'} \mid m) > 0$).
- **Status Causal:** Nenhum sinal subléxico intra-palavra pode antecipar a vogal sem violar a barreira de causalidade temporal.

### 3.2 Regime 2: Geração Condicionada por Lema / Intenção Lexical
- **Definição:** O modelo recebe a intenção de soletrar um lema específico $\mathbf{c}_{\text{lemma}}$ e gera os caracteres autoregressivamente: $P(c_t \mid c_{<t}, \mathbf{c}_{\text{lemma}})$.
- **Status Causal:** Perfeitamente válido para a tarefa de produção lexical dirigida.
- **Exigência:** O sinal de condicionamento deve ser explicitamente declarado como entrada, sem alegar equivalência com o Regime 1.

---

## 4. Partições de Dados e Controle de Holdout

1. **Treino Antigo ($D_{\text{old, train}}$):** `mala`, `pato`, `sapo`, `fogo`.
2. **Treino Novo ($D_{\text{new, train}}$):** `medo`, `povo`.
3. **Sondas de Controle Interno ($P_{\text{old, internal}}$):** `malas`, `patos`, `sapos`, `fogos`.
4. **Sondas de Diagnóstico Conhecidas (Holdout Issue #42):**
   - Antigas: `maleta`, `pateta`, `sapato`, `foguete`
   - Novas: `meta`, `pote`
5. **Partição Independente de Generalização (Palavras Não-Vistas):**
   - Novos pares colidentes em `m`: `mola` (antigo), `muro` (novo)
   - Novos pares colidentes em `p`: `pala` (antigo), `pipa` (novo)
   - Controles em outras consoantes: `selo`, `toca`, `vela`

---

## 5. Níveis de Avaliação e Critérios de Decisão Predefinidos

| Nível | Métrica | Denominador / Fórmula | Limiar de Decisão |
|---|---|---|---|
| **A. Representacional** | Distância de Cosseno ($d_{\text{cos}}$) | $1 - \frac{\vec{h}_a \cdot \vec{h}_b}{\|\vec{h}_a\|_2 \|\vec{h}_b\|_2}$ | $d_{\text{cos}} \ge 0{,}10$ |
| **A. Representacional** | Distância Euclidiana ($d_{\text{euc}}$) | $\|\vec{h}_a - \vec{h}_b\|_2$ | $d_{\text{euc}} > 0{,}0$ |
| **B. Coexistência Funcional** | Retenção Positiva ($R_{\text{pos}}$) | Sondas internas positivas ($N_{\text{pos}}=3$) | $R_{\text{pos}} = 3/3 = 1{,}0000$ |
| **B. Coexistência Funcional** | Delta da Tarefa Nova ($\Delta_{\text{new}}$) | $C_{\text{new, post}} - C_{\text{new, acq}}$ ($N_{\text{new}}=2$) | $\Delta_{\text{new}} \ge 0{,}0000$ |
| **B. Coexistência Funcional** | Log-Loss Conjunto | Média de perda nos pares conflitantes | $\le 1{,}0000$ |
| **C. Generalização** | Acurácia Holdout Diagnóstico | Média nas 6 sondas conhecidas da Etapa A | $\ge 0{,}8000$ |
| **C. Generalização** | Acurácia Holdout Independente | Média nas palavras não-vistas | Avaliação comparativa |
| **D. Regime de Inferência** | Taxonomia | Classificação obrigatória | Sem confusão metodológica |

---

## 6. Recursos, Restrições e Governança

- **Hardware e Custos:** 100% local em CPU, custo zero de nuvem.
- **Isolamento:** Nenhuma modificação em `src/kinera/` (núcleo, backends, CUDA, KEMA).
- **Aprovação Prévia:** O protocolo deve ser revisado e aprovado antes de qualquer execução de runners em larga escala.

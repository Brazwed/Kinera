# Arquitetura interna — Fase 8

## Escopo

A Fase 8 reorganiza o Kinera em um pacote instalável e prepara a fronteira de execução CPU/CUDA. A matemática, as dimensões do modelo, a causalidade, os pesos de loss, a inicialização e o protocolo experimental permanecem inalterados. A CPU continua sendo a implementação de referência; esta fase não implementa nem compila kernels CUDA de produção.

## Responsabilidades

- `src/kinera/language/`: vocabulário, ortografia, sílabas, alinhamento, datasets e adaptação de registros para entradas do modelo.
- `src/kinera/nn/`: layers, heads, losses e composição do modelo; consome o contrato `Backend`, sem importar implementação CPU/CUDA concreta.
- `src/kinera/backends/`: contrato, seleção lazy e operações concretas. `CPUBackend` delega às operações Python da implementação de referência.
- `src/kinera/training/`: treino, avaliação, checkpoint e configuração. `loop.py` e `autoregressive.py` são superfícies de compatibilidade para imports históricos.
- `src/kinera/experiments/`: comparação A/B/C/D, variantes, execução multi-seed, checkpoint/resume experimental e protocolo de comparação.
- `src/kinera/diagnostics/`: verificações numéricas de gradiente e análises silábicas/ortográficas.
- `scripts/`: entrypoints atuais de treino, comparação e preparação de dataset; `scripts/legacy/` contém runners experimentais preservados.

## Contrato de backend

`kinera.backends.base.Backend` define o contrato estável de operações usado por layers, ativações e losses. `resolve_backend(None)` seleciona `CPUBackend`; um backend injetado é validado pelo protocolo em runtime. As operações CUDA específicas das Phases 11A/11B — concatenação híbrida, mean pooling e argmax, losses/gradientes residentes, agregação determinística das heads e backward da representação — são capacidades opcionais e não ampliam o protocolo obrigatório, preservando backends CPU injetados existentes. A criação direta de buffers zero segue o mesmo padrão: `SupportsBufferZeros` é uma capability opcional; `create_zero_buffer()` usa o staging contíguo quando disponível e, para backends legados, preserva `buffer_from_host` com listas host aninhadas. Se um backend legado produzir listas, o modelo mantém a execução compatível em host; se produzir buffers residentes sem a capacidade correspondente, a operação falha explicitamente em vez de transferir dados implicitamente. `get_backend()` importa implementações sob demanda. As classes em `kinera.nn` recebem uma instância de backend e não espalham condicionais por hardware.

`CPUBackend` é a oracle Python sem dependências numéricas externas. Suas operações ficam em `backends/cpu/ops/` — incluindo a aritmética de neurônio em `ops/neuron.py`, delegada por `CPUBackend.neuron_forward`, `neuron_backward` e `neuron_step` — e preservam a implementação de referência de `Linear`, `Embedding`, `Neuron`, ativações e probabilidades.

A Phase 8 criou os pontos de extensão sem carregar biblioteca nativa nem alocar memória CUDA. A Issue #12 implementou runtime, ABI e buffers de forma opcional; o contrato está documentado em [`cuda-backend.md`](cuda-backend.md). `Linear.forward`, `Linear.backward` e `Linear.step/update` executam em CUDA e passaram seus gates CPU↔CUDA; backward também passou diferenças finitas, e step altera parâmetros em buffers residentes sem round-trip implícito. A fixture integrada `forward → backward → step` passou na Tesla T4 com erro absoluto máximo `0.0`, e a Phase 9 foi squash-merged em `main` (`a02379ca2b6bd2d473740c402b71d4e5c44f5b1b`) com CI pós-merge verde; Issue #12 está fechada. Nas Phases 10.0–10.3, `Embedding` recebeu dimensões explícitas, IDs `int32`, forward, backward determinístico sem atomics e step in-place; os gates T4 passaram com erro CPU↔CUDA máximo `0.0`, incluindo updates após 1 e 8 passos e `learning_rate=0` exato. Na Phase 10.4, a fixture integrada `forward → backward → step` passou na T4 com saída, tabela de gradientes e pesos finais CPU↔CUDA com erro máximo `0.0`, lifetime/use-after-release e fail-closed validados. Na Phase 10.5, o wheel CPU-only foi compilado/instalado e um smoke `forward → backward → step` passou sem importar módulos CUDA ou resolver `nvcc`; a suíte CPU no head `c21aca55211fb3d14299e2e3007be78f7b5a5569` tem 347 testes, OK (15 skips CUDA-only). A Phase 10.6 validou limites de IDs residentes em ABI 5 na T4 e reconstruiu o wheel CPU-only desse head; detalhes e hashes estão em [`../experiments/phase10_6_embedding_id_bounds_validation.json`](../experiments/phase10_6_embedding_id_bounds_validation.json) e [`../experiments/phase10_6_cpu_wheel_validation.json`](../experiments/phase10_6_cpu_wheel_validation.json). Evidências: [`../experiments/phase10_0_embedding_storage_validation.json`](../experiments/phase10_0_embedding_storage_validation.json), [`../experiments/phase10_1_embedding_forward_validation.json`](../experiments/phase10_1_embedding_forward_validation.json), [`../experiments/phase10_2_embedding_backward_validation.json`](../experiments/phase10_2_embedding_backward_validation.json), [`../experiments/phase10_3_embedding_step_validation.json`](../experiments/phase10_3_embedding_step_validation.json) e [`../experiments/phase10_4_embedding_integrated_validation.json`](../experiments/phase10_4_embedding_integrated_validation.json); wheel: [`../experiments/phase10_5_cpu_wheel_validation.json`](../experiments/phase10_5_cpu_wheel_validation.json). ABI 5. Nenhum benchmark ou treino ampliado foi executado.


## Treino multitarefa residente — Phase 11B

`AutoregressiveModel.multitask_train_step` mantém logits, gradientes das três heads, agregação, pooling/backward da representação e gradientes das embeddings em buffers CUDA até `Linear.step`/`Embedding.step`. As operações novas são capacidades opcionais: listas continuam usando a rota CPU, enquanto buffers residentes sem a capacidade requerida falham explicitamente. Somente losses escalares e status numéricos são copiados ao host; a ordem de agregação permanece `char → syllable → boundary`.

A fixture determinística de um único passo passou CPU↔Tesla T4 para logits, losses, gradientes e parâmetros finais em `float64`, `atol=rtol=1e-12`; erro absoluto máximo observado `1.3877787807814457e-17`. A matriz CUDA executou 96 testes sem falhas, erros ou skips; a suíte CPU executou 393 testes com 24 skips CUDA-only. Relatório e manifestos: [`../experiments/phase11b_training_validation.json`](../experiments/phase11b_training_validation.json). Estes resultados validam correção funcional, não performance; benchmark e treino ampliado continuam fora de escopo.

## Migração de caminhos

| Caminho anterior | Caminho atual |
| --- | --- |
| `nn/linear.py`, `nn/embedding.py`, `nn/neuron.py`, `nn/hybrid.py` | `kinera.nn.layers/` |
| `nn/heads.py` | `kinera.nn.heads/` |
| `nn/losses.py`, `nn/probability.py` | `kinera.nn.losses/` |
| composição autoregressiva em `training/autoregressive.py` | `kinera.nn.models.hybrid` |
| `language/features.py`, `language/alignment.py` | `kinera.language.orthography/` |
| `language/syllables.py`, `language/partial_syllable.py` | `kinera.language.syllables/` |
| `language/vocabulary.py` | `kinera.language.vocabulary/` |
| `language/dataset.py`, `language/records.py`, `language/hybrid_inputs.py` | `kinera.language.datasets/` |
| comparação e variantes em `training/` | `kinera.experiments/` |
| dados sintéticos de runners em `training/data.py` | `kinera.experiments.synthetic_data` |
| análises silábicas/ortográficas em `training/` | `kinera.diagnostics/` |

Imports antigos selecionados permanecem como reexports temporários para compatibilidade; a implementação principal vive sob `src/kinera/`.

## Entrypoints e legado

- `python -m scripts.train`: treino autoregressivo.
- `python -m scripts.compare`: comparação controlada (A/C/D por padrão; A/B/C/D explícito).
- `python -m scripts.prepare_dataset`: construção streaming do dataset de caracteres.
- Runners históricos foram movidos para `scripts/legacy/`, incluindo `phase5_3.py`, `phase5_4.py`, `phase5_4_expanded.py`, `phase5_5.py`, `phase6_validation.py`, `phase7_1_performance_baseline.py`, `run_experiment.py`, `run_syllable_analysis.py` e os testes de cinco minutos. Os nomes antigos na raiz que precisaram permanecer são wrappers de compatibilidade; a implementação fica em `scripts/legacy/`.

## Referência e validação

`experiments/phase8_0_refactor_baseline.json` registra o commit base `a6c278a71d94790b4f73122cc844f20292c3c87c`, o resultado da suíte inicial e os hashes das fixtures/invariantes CPU. `tests/equivalence/test_cpu_reference_equivalence.py` verifica os hashes congelados e repete a fixture A/B/C/D de um exemplo, comparando losses e contagens de parâmetros com a referência. Os testes de gradiente ficam em `tests/gradients/`; os checks do contrato e do scaffold CUDA permanecem na suíte geral.

Instalação/teste:

```bash
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
```

A baseline da Issue #8 e seus relatórios/checkpoints 5k/10k são artefatos externos e imutáveis; não foram regenerados ou incluídos no refactor. Não foi executado treino em escala, benchmark de performance ou CUDA.

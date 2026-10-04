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

`kinera.backends.base.Backend` define as operações usadas por layers, ativações e losses. `resolve_backend(None)` seleciona `CPUBackend`; um backend injetado é validado pelo protocolo em runtime. `get_backend()` importa as implementações sob demanda. As classes em `kinera.nn` recebem uma instância de backend e não espalham condicionais por hardware.

`CPUBackend` é a oracle Python sem dependências numéricas externas. Suas operações ficam em `backends/cpu/ops/` e preservam a aritmética existente de `Linear`, `Embedding`, `Neuron`, ativações e probabilidades.

A Phase 8 criou os pontos de extensão sem carregar biblioteca nativa nem alocar memória CUDA. A Issue #12 implementa runtime, ABI e buffers de forma opcional; o contrato está documentado em [`cuda-backend.md`](cuda-backend.md). A `CUDABackend` executa `Linear.forward`; o wrapper/kernel de `Linear.backward` está no código da Phase 9.6, mas sua equivalência CUDA ainda precisa de validação GPU. `Linear.step/update` e as demais operações continuam fail-closed. A equivalência numérica do forward passou; o check remoto de CI CPU também passou. Nenhum treinamento 5k/10k ou benchmark foi executado.


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

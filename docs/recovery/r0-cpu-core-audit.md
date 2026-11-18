# Gate R0 — auditoria do núcleo CPU

## Estado e escopo

**Parecer: PARTIAL.** Auditoria executada sobre `74f06ff498de65efacf2b803d526b37acf7b755f` (merge #81, 2026-11-06), worktree `Worktrees/R0-baseline`, Python 3.15.2. O `main` local diverge desse baseline; este relatório não o substitui. O trabalho foi somente leitura e execução de testes. Nenhuma alteração em matemática, backends ou artefatos históricos.

## Inventário do núcleo

| Componente | Implementação e papel | Dependências/API e acoplamento | Testes/estado |
|---|---|---|---|
| Embedding | `src/kinera/nn/layers/embedding.py`: tabela treinável de IDs, forward/backward/step | `Backend`, `BackendBuffer`, `resolve_backend`; construtor aceita backend opcional e usa backend resolvido; valida IDs/rank/dimensão | `test_embedding.py`, `test_embedding_buffers.py`; CPU verificado; residentes em device são contratos separados |
| Linear | `src/kinera/nn/layers/linear.py`: transformação afim `W x + b`, gradientes e atualização | Backend genérico via `kinera.backends`; estado de gradiente depende de array host ou buffer | `test_linear.py`, `test_linear_learning.py`; CPU verificado |
| HybridRepresentation | `src/kinera/nn/layers/hybrid.py`: concatena embeddings de caractere e sílaba com features | Usa as duas Embedding e o backend da embedding de caractere para buffers/operação híbrida; fallback de listas Python no host | `test_hybrid.py`, `test_hybrid_integration.py`; caminho CPU coberto |
| Mean pooling | Método `_pool_representation_backend` de `src/kinera/nn/models/hybrid.py`; média da sequência/contexto antes dos heads | Tenta `mean_pool_forward` no backend; exige suporte se recebe buffer residente; backward correspondente no treino | `test_autoregressive_model.py`, `test_hybrid_integration.py`; CPU coberto via fluxos de modelo |
| Heads | `src/kinera/nn/heads/{character,syllable,boundary,multitask}.py`; três saídas char/syllable/boundary | Heads lineares; composição no `MultiTaskHeads`/`AutoregressiveModel` | `test_heads.py`, `test_multitask_training.py`; CPU verificado |
| Losses | `src/kinera/nn/losses/{cross_entropy,binary_cross_entropy,multitask}.py`; perdas por head e agregação | backend opcional nas operações; `MultiTaskLoss` consumida por treino/eval multitarefa | `test_multitask_loss.py`, `test_multitask_training.py`; CPU verificado |
| Treino | `src/kinera/training/trainer.py`; `train_examples`, `train_multitask_examples`, fit/épocas | Modelo, adapter de dataset, vocabulário, config, evaluator; shuffle de `train_epoch` usa `random.Random(seed)` fornecido pela configuração | `test_training_loop.py`, `test_training_data.py`, `test_autoregressive_workflow.py`; determinismo de loop simples coberto, veja limites abaixo |
| Geração | `src/kinera/training/evaluator.py::generate_text`; CLI `scripts/train.py` | `AutoregressiveModel`, vocabulário, inputs de contexto; CLI constrói modelo com `--seed` | `test_autoregressive.py`, `test_autoregressive_workflow.py`; geração e validações cobertas na seleção executada |
| Checkpoints | `src/kinera/training/checkpoint.py`; JSON versionado v1 para `Neuron` e `Linear` | Tipos serializáveis host; não há dependência explícita de backend no formato | `test_checkpoint.py`, `test_checkpoint_runner.py`; round-trip e erros cobertos; modelo híbrido não suportado |

### Caminhos públicos, compatibilidade e experimentos

- API moderna de entrada: `kinera.nn.models.hybrid.AutoregressiveModel`; `scripts/train.py` é a CLI oficial indicada no baseline.
- Implementações reutilizáveis de camada em `kinera.nn.layers`; `kinera.nn.embedding` e `kinera.nn.linear` são superfícies de conveniência/reexport. `kinera.training.autoregressive` explicitamente declara ser superfície compatível e reexporta modelo, avaliação, geração e treino. Não há evidência para remover esses caminhos.
- `kinera.backends.cpu.CPUBackend` é a implementação de referência; backend não fornecido é resolvido pela camada de backend. `kinera.backends.cuda` e `kinera.backends.opencl` são implementações opcionais distintas; seus testes não foram usados como evidência do gate CPU.
- `src/kinera/experiments/*`, `src/kinera/scripts/benchmark_*` e `scripts/legacy/*` são runners/experimentos auxiliares ou legados, não a entrada oficial de treino. Manter como compatibilidade até mapear consumidores.
- `mean_pool_forward` integra-se ao contrato backend; o modelo não substitui silenciosamente operação ausente em buffers de device. No caminho CPU em listas, o backend CPU fornece operações correspondentes.

## Matriz de validação

Comando executado no baseline:

```text
PYTHONPATH=src python -m pytest -q tests/test_autoregressive.py tests/test_autoregressive_model.py tests/test_autoregressive_workflow.py tests/test_embedding.py tests/test_linear.py tests/test_hybrid.py tests/test_heads.py tests/test_multitask_loss.py tests/test_training_loop.py tests/test_training_data.py tests/test_checkpoint.py tests/test_checkpoint_runner.py tests/test_runner_resume_integration.py tests/test_legacy_checkpoint_identity.py tests/test_train_5min.py
```

**Resultado: 124 passed, 97 subtests passed, 0 failed (1.84 s).** A execução anterior de seleção menor registrou 78 testes e 75 subtests aprovados; está contida na execução complementar, não deve ser somada como casos únicos.

| Propriedade | Evidência | Resultado |
|---|---|---|
| Inicialização e treino com seed fixa | `test_training_loop.py`, `test_embedding.py`, `test_training_data.py`; execução acima | PASS nos casos existentes: ordem embaralhada, pesos iniciais e dados repetem para seed igual |
| Geração autoregressiva | `test_autoregressive.py`, `test_autoregressive_model.py`, `test_autoregressive_workflow.py` | PASS nos cenários definidos pelos testes |
| IDs e argumentos inválidos | `test_embedding.py`, `test_autoregressive_model.py`, `test_training_data.py` | PASS nos casos definidos pelos testes |
| Shapes e contratos de camadas/modelo | `test_embedding.py`, `test_linear.py`, `test_hybrid.py`, `test_heads.py`, `test_multitask_loss.py` | PASS nos casos definidos pelos testes |
| Checkpoint/resume de runners | `test_checkpoint_runner.py`, `test_runner_resume_integration.py`, `test_legacy_checkpoint_identity.py` | PASS para contratos cobertos de persistência e retomada de experimentos |
| Continuidade exata do treinamento do modelo híbrido após salvar/carregar | Checkpoint v1 + API atual | **UNVERIFIED / não suportado pelo formato atual** |
| Reprodutibilidade exata de fluxo completo CPU via checkpoint | Sem persistência de modelo híbrido, RNG e estado de treino no formato atual | **UNVERIFIED** |

## Auditoria de persistência

`checkpoint.py` declara `CHECKPOINT_FORMAT = "kinera.checkpoint"`, versão 1, payload JSON com `format`, `version`, `model`, `training` e `metadata`. `save_neuron/load_neuron` guardam arquitetura simples, pesos e bias; `save_linear/load_linear` guardam dimensões, matriz e bias. Escrita usa arquivo temporário no diretório de destino antes da substituição. Carregadores validam estrutura/tipo e convertem valores; há testes de round-trip e arquivos inválidos/incompatíveis.

Limite comprovado no código: `Model = Neuron | Linear`, e `save_checkpoint` rejeita qualquer outro tipo. Logo, `AutoregressiveModel` (embeddings, representação, pooling, três heads) não tem persistência nativa via esse módulo. O argumento `training_state` é dicionário JSON opcional e opaco; não captura automaticamente RNG, posição do dataset, shuffle, estado/gradientes do modelo ou otimizador. As rotinas observadas atualizam parâmetros por `step`/SGD; não foi encontrado estado de otimizador separado a salvar. Compatibilidade documentada apenas como versão 1 e testes de formatos suportados; política de migração futura permanece UNVERIFIED. Não houve mutação do formato.

## Problemas e riscos

| Prioridade | Classificação | Evidência / impacto |
|---|---|---|
| P0 | Nenhum comprovado | Os fluxos CPU selecionados de forward, backward, treino, geração, shapes e entradas inválidas passaram. Isso não prova ausência global de defeitos. |
| P1 | Limitação funcional comprovada | Checkpoints nativos não serializam `AutoregressiveModel`; treino completo não pode ser restaurado por `save_checkpoint`/`load_checkpoint`. Limita retomada/reprodutibilidade do modelo híbrido. |
| P1 | Risco de reprodução comprovado pelo contrato | `training_state` não captura automaticamente estado aleatório/posição de dados; checkpoints de camadas não bastam para retomar um processo de treino exato. |
| P2 | Lacuna de evidência | Não rodado o conjunto inteiro de testes CPU-only nem matriz de ambientes Python suportados. Resultados são da seleção citada em Python 3.15.2. |
| P2 | Lacuna de evidência | Não foi feita verificação independente por execução CLI com dados reais; o workflow/geração foi validado pelos testes existentes. |

Não foram reproduzidos defeitos matemáticos, alteração de pesos inesperada ou violação de contrato no escopo executado. Não são classificados como defeitos os caminhos CUDA/OpenCL ou benchmarks; estão fora deste gate.

## Próximo gate recomendado

R0 não autoriza as fases R1–R4. Após revisão deste inventário, abrir gate separado para R1: tratar somente defeitos CPU reproduzidos e lacunas essenciais aceitas, preservando aliases e contratos. Planejar checkpoint híbrido em gate próprio (R2), com arquitetura, identidade/vocabulário, seed e retomada testados; não antecipar implementação aqui. Não ampliar treino/pipeline nem habilitar aceleração durante R0.

## Validação final e fechamento

### Suíte CPU

Comando no baseline:

```text
PYTHONPATH=src python -m pytest -q --ignore-glob='tests/test_cuda_*.py' --ignore-glob='tests/test_opencl_*.py' --ignore=tests/test_expanded_replay_pilot.py
```

**596 passed, 443 subtests passed, 0 failed (12.38 s).** `pytest --collect-only` confirmou 596 testes coletados sob esses filtros. Não houve skips registrados dentro da seleção. Foram excluídos do comando **29 módulos** nomeados `test_cuda_*.py` ou `test_opencl_*.py`, pois os módulos são gates específicos de backend/hardware, com opt-in ou dependência dos runtimes/dispositivos. Não se declara que todos exigiriam hardware: os testes puramente unitários de contratos/backend nesses módulos também ficaram fora para evitar misturar evidência de backend com o gate CPU. `test_expanded_replay_pilot.py` foi excluído por ser piloto experimental de replay/CPU-CUDA, fora do escopo R0. Não foram executados benchmarks de desempenho.

### CLI oficial e repetibilidade

Foi criado fixture JSONL isolado em `/tmp/tmp.Zfk91PR53E`, sem tocar nos dados históricos: vocabulário versionado com tokens especiais e `a,b,c`; quatro exemplos contextuais, incluindo `<END>`. Comando executado duas vezes:

```text
PYTHONPATH=src python scripts/train.py --vocabulary /tmp/tmp.Zfk91PR53E/vocab.json --train /tmp/tmp.Zfk91PR53E/train.jsonl --epochs 3 --learning-rate 0.1 --max-examples 4 --context-size 2 --max-tokens 5 --seed 7
```

As duas execuções encerraram com código 0 e JSON válido; `diff -u run1.json run2.json` não encontrou diferença. Resultado: 3 épocas, 4 exemplos, loss `1.789222179258471`, `1.680384554552278`, `1.5927165395017366`; acurácia `0.25`, `0.5`, `0.5`; texto gerado `acc`. Isso verifica inicialização, carregamento JSONL, treino no backend padrão CPU, cálculo de loss e geração nesta configuração pequena.

**Determinismo contínuo:** PASS para duas execuções completas independentes com mesma seed e configuração, incluindo igualdade byte a byte do JSON de saída observado. **Determinismo após checkpoint/resume:** não suportado/verificado para o modelo híbrido; não inferir do teste CLI. O checkpoint da camada não persiste o modelo completo, RNG nem cursor do conjunto.

### Pendências e encaminhamento

- Não há falha CPU reproduzida na suíte aplicável; os fluxos centrais funcionam no baseline e nenhuma pendência observada bloqueia a recuperação funcional do núcleo CPU.
- A retomada completa e bit a bit do treino híbrido continua sem suporte nativo. É limitação real para reprodutibilidade/resume; encaminhar a gate próprio R2, sem implementar nesta auditoria.
- A suíte filtrada não valida os 29 módulos de integração CUDA/OpenCL nem o piloto experimental. Esses itens são gates de backend/experimento e não impedem conclusão do diagnóstico CPU-only.
- Não foi feita matriz em múltiplas versões Python; resultado limita-se a Python 3.15.2. CLI foi demonstrada em fixture sintético pequeno, não em corpus real.
- R1 pode tratar posteriormente apenas defeitos CPU comprovados ou gaps essenciais aprovados; nenhum defeito comprovado exige correção para concluir este diagnóstico. R2 trata checkpoint/retomada híbrida. Nenhum desses gates foi iniciado.

## Parecer final do Gate R0

**PARTIAL.** A suíte CPU aplicável passou (596 testes, 443 subtests, zero falhas), a CLI oficial concluiu e duas execuções com seed fixa produziram saída idêntica. Isso é evidência suficiente de funcionamento do caminho CPU avaliado, sem bloqueador reproduzido para recuperação futura. O gate não recebe PASS porque a retomada determinística após checkpoint do modelo híbrido não é suportada nem demonstrada, e a validação limita-se ao ambiente Python 3.15.2 e fixture sintético. Essas limitações são explícitas e podem seguir para R2/matriz de ambientes; não comprometem o baseline funcional CPU nem justificam iniciar recuperação agora.

# Issue #26 — auditoria final B005 e composição M2.2

Data da auditoria: 2026-11-06. Esta auditoria foi feita na branch
`feature/issue-26-metrics-b005`, exclusivamente na worktree
`Worktrees/Kinera_B005`.

## Parecer executivo

- **Issue #26 — `PARTIALLY_COMPLETE`.** M0, M1.1–M1.4 e M2.1 têm implementação
  e evidências nos PRs integrados. B005 fecha a lacuna estrutural de M2.2 no
  runner comparativo: publica uma seção `metrics` versionada, mantendo aliases
  legados e sem copiar os resultados por execução. A Issue não está pronta para
  aceite: a cadência opcional de validação por época continua no critério
  original, e os runners retomáveis das Fases 5.3/5.5 ainda podem reaproveitar
  resultados sem identidade verificável de conteúdo.
- **Issue #27 — aberta e parcial.** T0/T1, T2a e a integração controlada (C)
  estão integradas. T2b permanece explicitamente adiada; T3 e partes de T4 não
  foram implementadas. B005 não altera esses escopos.
- **Issue #25 — aberta.** O relatório conjunto passa a ter seção explícita de
  metrics e mantém telemetry independente, mas os gates de fechamento exigem
  que #26 e #27 concluam seus próprios critérios e que a integração publicada
  seja validada.

Não houve alteração em modelos, losses, gradientes, optimizer, RNG, dados,
ordem de updates, protocolos científicos ou backends.

## Referências e estado Git

`origin/main` foi atualizado antes da criação da branch e reconfirmado antes
da implementação:

- SHA inicial de `main`: `ed31cf90df257a18bfd02536a99470c46717d025`.
- Merge-base inicial B005: `ed31cf90df257a18bfd02536a99470c46717d025`.
- Commit de implementação B005: `374a78e` — `feat(metrics): add versioned report
  section (Kinera_B005)`.
- Branch B005 começou limpa, sem commits divergentes da `main`. Nenhuma outra
  worktree foi alterada.

Durante a preparação da publicação, a `main` avançou pelo PR #67
(`1cc06fc1d143426dfcc77ea9df295b7447940c99`), limitado a OpenCL. A worktree
local de validação incorporou a atualização por merge normal, sem editar os
arquivos do PR; a branch remota foi criada diretamente a partir dessa `main`
atualizada:

- Merge de atualização da base: `5970eb90ac54feac6d5c65e36fcec17de8fb69f6`.
- SHA final de `origin/main` conferido:
  `1cc06fc1d143426dfcc77ea9df295b7447940c99`.
- Merge-base B005 após a atualização:
  `1cc06fc1d143426dfcc77ea9df295b7447940c99`.
- A branch remota foi criada sobre o mesmo SHA de base; o merge
  local foi usado para repetir a CI completa com a atualização presente.
- O diff B005 contra essa base contém apenas README, documentação B005,
  `experiments/comparison.py` e seus testes; nenhum backend/kernel foi alterado.

Tranches de metrics observadas na linha de primeiro pai de `main`:

| Tranche | PR | Commit integrado |
| --- | --- | --- |
| M0 | #35 | `dafd0505dabe21b9b7c5add2e79a86c515a37d9a` |
| M1.1 | #50 | `5ab05239ddb40efe1862e4efac6a499373ce9121` |
| M1.2 | #51 | `373c9a56e3a3ffb698d7f820f6a170e17304573c` |
| M1.3 | #62 | `c72502db84f7e638dc58bbf2626f57e61636fbaa` |
| M1.4 | #64 | `3a9cd4df5a67557e5f8b2de0926f4b9f32b35106` |
| M2.1 | #66 | `ed31cf90df257a18bfd02536a99470c46717d025` |

Tranches de telemetry relevantes para a composição:

| Entrega | PR | Commit integrado |
| --- | --- | --- |
| T0/T1 | #36 | `e7da53faad121531d012679e514de65103e6d588` |
| T2a | #37 | `e28efec4a02d39877f7a79dcc3a3b3abae3b3eb7` |
| Integração controlada (C) | #49 | `bc54a306cf93fef6e60ea2a33fdbd21f021e381a` |

As Issues #25, #26 e #27 foram lidas integralmente, com os comentários de
integração disponíveis no GitHub. Todas foram observadas como abertas.

## Matriz final de rastreabilidade — Issue #26

| Requisito | Evidência no código e testes | Status | Lacuna, risco e ação recomendada |
| --- | --- | --- | --- |
| Fonte única para aritmética compartilhada, preservando ownership | `monitor/metrics/core.py` e `aggregation.py`; consumidores de treino, comparação, parcial, tokenizer e diagnósticos delegam números. Histórico B004 congela valores e ordem de soma. | **Concluído** | Loss diferenciável permanece em `nn/losses`/backend; grupos linguísticos e seleção permanecem em `diagnostics`; status/job control permanecem em `experiments`. Essas exceções são intencionais. |
| Confusão, suportes, baselines e interpretação binária | `monitor/metrics/confusion.py`, `baselines.py`, `training/evaluator.py`; testes em `test_monitor_metrics_m1.py`, `test_multitask_training.py` e `test_comparison.py`. O baseline majoritário usa apenas `train_boundary_distribution`, calculada de `train_examples`. | **Concluído** | Não foi encontrada leitura de distribuição de validação para formar o baseline majoritário. Nenhum comportamento de threshold foi alterado. |
| Segmentação por offsets, denominadores e aliases | `monitor/metrics/segmentation.py` e `language/syllables/tokenizer.py`; `test_syllables.py` cobre suporte vazio e fronteiras. `boundary_accuracy` histórico permanece alias. | **Concluído** | O relatório distingue métricas de offsets da head binária. Zero de F1 com suporte positivo previsto/gold continua distinto de F1 não aplicável sem qualquer suporte. |
| Schema v1, disponibilidade, motivos e suporte | `monitor/metrics/schema.py`; avaliador, tokenizer e diagnósticos publicam `metric_results`. Testes de schema validam valores finitos, contagens e estados. Distribuições de classe e estados silábicos known/unknown/end ficam no avaliador. | **Concluído** | A versão do envelope identifica o formato emitido; não transforma registros legados internos em registros v1. Versões de metrics, telemetry e fingerprint são independentes. |
| Agregação por seed, missing/invalid e conservação | `aggregation.py`, `comparison._build_aggregates()` e `test_comparison.py`; cada observação concluída classifica-se em value/missing/invalid, com soma igual ao total avaliado. | **Concluído** | Agregados são sobre seeds e carregam IDs usados; não são suporte por exemplo nem média simples entre grupos. Falha de execução não é convertida em zero. |
| Deltas pareados, unidade, orientação e elegibilidade | `summarize_paired_deltas()` e `_build_paired_deltas()`; testes de unidades, orientações, seeds comuns, missing e invalid. O runner comparativo só compara execuções concluídas nas variantes ativas. | **Concluído no runner controlado** | A comparabilidade é limitada à identidade que o fingerprint efetivamente cobre. Não interpretar “mesma seed” como evidência universal de comparabilidade fora desse runner. |
| Identidade de dados/vocabulário/configuração no resume M1.4 | `experiments/fingerprint.py`, `_build_experiment_fingerprint()` e `_validate_experiment_fingerprint()`; `test_experiment_fingerprint.py`, `test_comparison.py` e `test_telemetry_checkpoint.py`. Checkpoint do runner comparativo falha fechado sem fingerprint, valida configuração e permite expansão de seeds/variantes compatíveis. | **Parcial** | `scripts/legacy/phase5_3.py` e `phase5_5.py` usam `run_seed_jobs()`. `load_seed_checkpoint()` compara `experiment_config` e seeds, mas não recebe hashes de dados/vocabulários; caminhos fornecidos pelo `job_factory` não são identidade. Não reutilizar checkpoint de identidade desconhecida para novas alegações. Abrir/atribuir tranche específica de integridade para esses fluxos ou formalizar que são somente históricos e tornar o resume não elegível. |
| Adoção M2.1 nos diagnósticos | `diagnostics/syllables.py` e `orthography.py` usam aritmética/schema compartilhados; `docs/metrics-issue-26-m2-1.md` fixa sinais, suportes, ordem e aliases. Testes determinísticos de diagnósticos e métricas passam. `diagnostics/gradients.py` mantém diferenças finitas. | **Concluído** | Nenhum forward adicional foi introduzido pela integração. Semântica dos buckets, classificação, frequência, posição, pares e ranking seguem nos diagnósticos. |
| Seção versionada conjunta M2.2 | B005 adiciona `report.metrics = {metrics_schema_version, aggregates, paired_deltas}` em `_checkpoint_report()`. `runs[*].final_validation` mantém descritores por execução; top-level `metrics_schema_version`, `aggregates` e `paired_deltas` continuam aliases iguais. Telemetry continua opcional em `report.telemetry`. | **Implementado em B005; aguardando PR/CI** | A seção acrescenta somente os resumos já existentes; não duplica resultados completos por run. O fingerprint continua top-level e fora dos dois schemas. |
| Validação opcional por época/cadência | `train_multitask_examples()` avalia o conjunto de treino após cada época; o runner controlado avalia validação no início e no final. Não há opção para cadência de avaliação de validação. | **Pendente** | Critério original M2 ainda pede opção/cadência explícita. Não foi implementada: adicionaria forwards e custo de avaliação, e sua necessidade científica não foi demonstrada. Reconciliar formalmente o critério ou planejar uma tranche pequena, sem mudar o default. |
| Exemplos JSON de contrato, incluindo aliases | README contém prosa e nomes de estados; este relatório inclui exemplos sintéticos abaixo para valor disponível, suporte vazio, baseline indisponível e agregação com seed ausente. | **Concluído na documentação B005** | Os exemplos são fixtures de contrato, não medições nem resultados de treinamento. |
| Compatibilidade Python e imports CPU-only | `pyproject.toml` declara `>=3.10`; workflow de CI usa Python 3.14. Suite local e teste de package layout/imports passaram em Python 3.15; imports verificados sem módulos CUDA/OpenCL. | **Parcialmente verificado** | Python 3.10 não está instalado neste ambiente. Não há sintaxe nova além da versão mínima declarada; a execução direta no mínimo continua pendente de CI em matriz própria. |

### Exemplos sintéticos do schema de métricas

Os exemplos abaixo usam valores fictícios. O descritor conserva disponibilidade e
suporte; os aliases escalares antigos continuam ao lado do resultado canônico.

Observação válida com suporte:

```json
{
  "metrics_schema_version": 1,
  "char_accuracy": 0.75,
  "metric_results": {
    "char_accuracy": {
      "value": 0.75,
      "availability": "available",
      "reason": null,
      "support": {"examples": 4}
    }
  }
}
```

Suporte known vazio: o alias legado continua `0.0`; o descritor v1 não
confunde ausência de denominador com erro observado igual a zero.

```json
{
  "metrics_schema_version": 1,
  "syllable_accuracy_known": 0.0,
  "metric_results": {
    "syllable_accuracy_known": {
      "value": null,
      "availability": "not_applicable",
      "reason": "no_known_targets",
      "support": {"known_targets": 0}
    }
  }
}
```

Baseline majoritário sem distribuição de treino: o alias continua `null` e a
causa fica explícita no resultado canônico.

```json
{
  "metrics_schema_version": 1,
  "boundary_baseline_majority": null,
  "metric_results": {
    "boundary_baseline_majority": {
      "value": null,
      "availability": "unavailable",
      "reason": "training_distribution_missing",
      "support": {"examples": 4}
    }
  }
}
```

Agregação por seed com um resultado válido e uma observação ausente. O campo
top-level `aggregates` é o alias legado do payload em `metrics.aggregates`;
registros `runs` antigos sem descriptors continuam sem eles.

```json
{
  "metrics": {
    "metrics_schema_version": 1,
    "aggregates": {
      "A": {
        "mean": {"char_accuracy": 0.75},
        "std": {"char_accuracy": 0.0},
        "n_values": {"char_accuracy": 1},
        "n_missing": {"char_accuracy": 1},
        "n_invalid": {"char_accuracy": 0},
        "seeds": {"char_accuracy": [7]}
      }
    },
    "paired_deltas": []
  },
  "metrics_schema_version": 1,
  "aggregates": {
    "A": {
      "mean": {"char_accuracy": 0.75},
      "std": {"char_accuracy": 0.0},
      "n_values": {"char_accuracy": 1},
      "n_missing": {"char_accuracy": 1},
      "n_invalid": {"char_accuracy": 0},
      "seeds": {"char_accuracy": [7]}
    }
  },
  "paired_deltas": []
}
```

## Contrato conjunto metrics + telemetry

O runner controlado compõe contratos existentes; não há envelope universal
novo:

| Informação | Local | Versão/escopo | Unidade, denominador ou método |
| --- | --- | --- | --- |
| Observações científicas por execução | `runs[variant][run].final_validation.metric_results` | `metrics_schema_version` no resultado | `support` declarado por métrica, usualmente exemplos ou contagens TP/FP/FN; valores de loss/acurácia são recebidos do avaliador. |
| Resumo científico por variante | `metrics.aggregates` | `metrics.metrics_schema_version` | Média/desvio populacional por seed válida; `n_values`, `n_missing`, `n_invalid` e seeds usadas explicitam o denominador. |
| Contraste entre variantes | `metrics.paired_deltas` | mesmo schema da seção metrics | Apenas seeds comuns concluídas; orientation, unit, `n_pairs`, ausentes/inválidos e seeds usadas são persistidos. Delta de loss é base − candidato; demais deltas de qualidade são candidato − base. |
| Observação operacional por execução | `runs[variant][run].telemetry` | `telemetry_schema_version` do run | Tentativas e segmentos append-only; durações em segundos e snapshots de memória em bytes, com método/fonte/ausência nos records. |
| Observação operacional da sessão | `telemetry` | `telemetry_schema_version` do envelope | Snapshot de ambiente/recursos e início/fim de sessão; a coleta é opt-in e fail-soft. |
| Identidade científica | `experiment_fingerprint` no objeto do experimento | `fingerprint_schema_version` e algoritmo SHA-256 | Dados, vocabulários, configuração e inputs preparados por variante; sementes/telemetry volátil não compõem a identidade comum. |
| Estado de execução | `status`, `completed_runs`, `failed_runs` e estados por run | domínio de experiments | Não é métrica de qualidade nem amostra de telemetry; falha operacional não é tratada como desempenho zero. |

O resume do runner controlado valida o fingerprint antes de reaproveitar ou
persistir dados. Checkpoints antigos podem ser lidos para inspeção; a política
fail-closed impede resume comparativo sem fingerprint. Escritas B005 emitem a
seção `metrics`; leitores e aliases antigos continuam disponíveis. Registros
internos históricos permanecem sem schema e sem metadata que não foi gravada.

Os diagnósticos silábico/ortográfico são chamadas diretas sem checkpoint/resume
próprio e sem coleta de telemetry. Relatórios `partial` também continuam sendo
saídas próprias; não são convertidos em uma composição comum nesta tranche.

## Lacuna de identidade nos runners históricos

Os entrypoints `scripts/legacy/phase5_3.py` e `phase5_5.py` carregam datasets e
vocabulários a partir de caminhos incluídos no `job_factory`, mas
`run_seed_jobs()` não persiste nem compara esses payloads. Em resume,
`load_seed_checkpoint()` compara `experiment_config` e permite expansão do
conjunto de seeds; uma seed já `completed` não é executada novamente. Alterar o
conteúdo de um arquivo no mesmo caminho pode portanto reutilizar resultados
antigos, embora a configuração aparente coincidir. O mesmo risco não é coberto
pelo fingerprint de `run_controlled_experiments()`.

Esta lacuna é de identidade/reprodutibilidade experimental, não de telemetry.
Para o aceite da #26, ainda é preciso decidir se esses entrypoints permanecem
históricos e não elegíveis para novos claims, ou se receberão uma tranche
específica de fingerprint/manifest. Até essa decisão/implementação, não retomar
seus checkpoints de identidade desconhecida para produzir novos resultados
científicos.

## Inventário de consumidores

| Consumidor | Metrics | Telemetry / persistência | Classificação e exceção |
| --- | --- | --- | --- |
| `training/evaluator.py` | Confusão, suportes, schema, métricas por avaliação | Não coleta | Integrado; o avaliador continua dono dos exemplos e forwards. |
| `training/trainer.py` | Métricas de `train_epoch` e avaliação multitarefa delegam para APIs compartilhadas | `TimingSession` opcional; registra `epoch_sgd` e `epoch_evaluation` | Integrado. Métrica online de `train_epoch` (predição antes do update) não é equiparada à avaliação multitarefa pós-época. |
| `experiments/comparison.py` | Agregados, pares, missing/invalid, schema e B005 section | Telemetry opt-in, envelope e run records; checkpoint/resume com fingerprint | Integrado no runner controlado. |
| `experiments/partial.py` | Agregação comum de valores por seed; avaliador emite descriptors por execução | Ainda usa timer local, sem seção de telemetry | Metrics integrado; telemetry pertence à adoção T4/M2.2 posterior fora do relatório controlado. |
| `experiments/config.py` | `mean_std` reexporta implementação comum; status por run continua local | `RunRecord` faz round-trip de telemetry opcional | Integrado; tipos e persistência não calculam qualidade. |
| `experiments/runner_common.py` | Nenhuma fórmula científica | Checkpoint JSON e estado/status de seed | Exceção legítima de orquestração; valida apenas config/seeds no runner genérico. |
| `experiments/multi_seed.py` | Usa callback de agregação; não define métrica | Preserva envelope/run telemetry e histórico de retry | Suporte operacional integrado; não possui fingerprint científico. |
| `diagnostics/syllables.py` | Aritmética/schema compartilhados | API direta, sem resume/coleta | M2.1 integrado; buckets, pares, frequência, posição e ranking permanecem locais. |
| `diagnostics/orthography.py` | Aritmética/schema compartilhados | API direta, sem resume/coleta | M2.1 integrado; classes e pareamento A/C/B/D permanecem locais. |
| `diagnostics/gradients.py` | Não move diferenças finitas para metrics | Não coleta | Exceção de domínio legítima. |
| `language/syllables/tokenizer.py` | Métricas de offsets via core/schema | Nenhuma | Integrado; regras de segmentação continuam no domínio linguístico. |
| `scripts/compare.py` | Emite relatório do runner controlado e aliases | CLI `--collect-telemetry`, desativada por padrão | Entry point ativo integrado. |
| `scripts/train.py` e avaliação simples | Consome trainer/evaluator; não cria fórmulas paralelas | Não compõe envelope do runner controlado | Consumer de treinamento, não produtor de relatório conjunto. |
| `scripts/legacy/phase5_3.py`, `phase5_5.py` | Usam descriptors M2.1; phase5_3 resume média por seed | Generic runner; sem coleta ativa na configuração atual | Legado executável; resume sem fingerprint continua pendente de reconciliação. |
| `scripts/legacy/phase5_4_expanded.py` e relatórios congelados | Protocolo/forma próprios | Checkpoints históricos | Histórico preservado; não migrado em massa. |
| ETL, losses, gradient checks, relógios de deadline e harnesses de benchmark | Fórmulas/contadores não são duplicação de qualidade migrável | Relógios controlam protocolos locais | Exceções com owners próprios; não alteradas. |

## Validação executada

Ambiente local: Python 3.15.0; `pyproject.toml` declara Python `>=3.10` e a CI
configurada roda Python 3.14. Python 3.10 não está instalado neste ambiente.

- Testes direcionados: `133 passed`, `32 subtests passed`, 1 aviso de
  depreciação do plugin AnyIO; incluíram comparison, fingerprints, telemetry
  checkpoint/records, metrics schema/core, tokenizer e diagnósticos.
- Seleção completa idêntica ao workflow `.github/workflows/ci.yml`, repetida
  após a atualização para PR #67:
  `734 passed`, `52 skipped`, `5 deselected`, `534 subtests passed` em 40.46 s;
  1 aviso de depreciação do AnyIO. Os cinco deselects são os casos de profiling,
  execução ampliada de piloto e benchmark explicitamente excluídos pela CI.
- Os testes cobriram relatórios com/sem telemetry, seção metrics e aliases,
  round-trip de checkpoint, resume fingerprint compatível/incompatível,
  checkpoints históricos, retry/attempt append-only e imports CPU-only.
- `python -m compileall -q src tests scripts`: aprovado.
- `git diff --check`: aprovado antes dos commits B005 e novamente após atualizar
  a base; exemplos JSON do relatório também foram parseados com `json.loads`.
- Import explícito de metrics, telemetry, comparison e diagnostics sem módulos
  Kinera CUDA/OpenCL: aprovado.
- Nenhum treinamento ampliado, benchmark ou execução GPU foi realizado.
- Duas tentativas iniciais de teste não coletaram testes: uma apontava para o
  arquivo inexistente `tests/test_tokenizer.py`; outra não configurava o
  `src/` no `PYTHONPATH`. A suíte foi repetida com os caminhos reais e
  `PYTHONPATH=src`; o resultado válido é o acima.

Os testes de round-trip/resume são fixtures pequenas determinísticas. Eles
demonstram persistência e contrato, não equivalência científica universal dos
fingerprints nem equivalência CPU/CUDA. A CI hospedada do PR B005 deve ser
verificada após abrir o PR; a validação local acima não a substitui.

## Pendências, donos e recomendação

| Pendência | Responsabilidade sugerida | Critério de saída |
| --- | --- | --- |
| Decidir ou implementar cadência opcional de validação por época | Issue #26 / owners de training | Decisão de escopo registrada ou opção explícita, default inalterado, testes demonstrando que não muda parâmetros/RNG/ordem de updates; se implementada, não alegar equivalência científica só por serialização. |
| Fingerprint/manifest ou bloqueio formal de resume nas Fases 5.3 e 5.5 | Issue específica de integridade experimental, coordenada com #26 | Mudança de conteúdo de dados/vocabulário no mesmo caminho deve impedir reuso; expansão legítima de seeds deve seguir válida; checkpoint legado sem identidade não pode virar nova alegação. |
| Concluir T2b/T3/T4 de telemetry quando critérios justificarem | Issue #27 | Manter T2b explicitamente adiada; não alegar sampler/probes GPU nem adoção ampla antes dos respectivos testes e contratos. |
| Validar Python mínimo em ambiente compatível | CI/package maintainers | Rodar import/teste de distribuição pelo menos em Python 3.10, ou adicionar matriz de CI. |

**Recomendação:** integrar o PR B005 depois da revisão e CI verde, pois a mudança
é aditiva, mantém os campos existentes idênticos e cobre a seção conjunta
exigida por #25 sem mexer em protocolos. Manter a Issue #26 aberta e
`PARTIALLY_COMPLETE`; não fechar #25 até #26 e #27 cumprirem seus gates. Não
usar checkpoints antigos das Fases 5.3/5.5 para alegações novas até existir
fingerprint verificável ou decisão formal de não elegibilidade histórica.

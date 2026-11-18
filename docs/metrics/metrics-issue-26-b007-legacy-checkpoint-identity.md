# B007 — Integridade de checkpoints dos runners históricos

## Referências

- Base usada: `a3c0dd2da678ecab29852dd8ccaefc8a4be10398` (`main`, após PR #70).
- Merge-base: `a3c0dd2da678ecab29852dd8ccaefc8a4be10398`.
- PR #70: integrado; HEAD de origem reportada `a9f710190e59b5510d0f89f7d162423438376b3d`.
- PRs anteriores: #35, #50, #51, #62, #64, #66, #68 e #70.

## Auditoria do fluxo anterior

As Fases 5.3 e 5.5 montavam um `ExperimentConfig` com fase, variantes,
épocas, learning rate, `max_examples` e, na 5.3, `top_examples`. O runner
genérico compara cada chave dessa configuração ao carregar um checkpoint,
exceto `seeds`; exige que as seeds antigas sejam subconjunto das solicitadas.
`workers`, caminhos, timestamps e telemetry ficam fora dessa identidade.

O runner carrega e valida o checkpoint antes de escrever. A primeira escrita
ocorre após `load_seed_checkpoint()` e `prepare_seed_records()`, marcando o
relatório como `running`; em seguida calcula as seeds pendentes e reutiliza
registros `completed`. Antes desta tranche, a configuração não continha hashes
dos dados ou vocabulários. Portanto, arquivos com o mesmo caminho e conteúdo
alterado passavam pela validação e seeds concluídas podiam ser reutilizadas.

Os dois scripts carregam JSONL e vocabulários dentro do worker. O carregador
ignora linhas em branco e seleciona os primeiros `max_examples` objetos JSONL,
preservando a ordem. Os diagnósticos recebem essa lista e aplicam novamente o
mesmo prefixo. `prepare_variant_examples()` copia os exemplos e mascara os
canais previstos para A/B/C/D sem reordenar registros.

Na Fase 5.3, preparação e treino consomem `context_ids`,
`context_syllable_ids`, `context_feature_vectors`, `target_char_id`,
`target_syllable_id` e `target_is_syllable_end`. O diagnóstico também lê
`word_key` e `target_char` para posições, buckets e efeitos por caractere.
Na Fase 5.5, o modelo recebe os mesmos campos de contexto e os alvos; a classe
ortográfica é calculada a partir de `target_char`. Os registros já chegam como
exemplos de modelo serializados; não há nova divisão de dataset nestes scripts.
O fingerprint inclui cada registro selecionado por inteiro, cobrindo esses
campos, metadados usados pelos diagnósticos e possíveis campos futuros.

| Consumidor de `run_seed_jobs()` | Identidade B007 | Risco/decisão |
| --- | --- | --- |
| `scripts/legacy/phase5_3.py` | Manifesto científico v1, obrigatório em novos checkpoints e resumes | Migrado; conteúdo e vocabulários são revalidados no worker. |
| `scripts/legacy/phase5_5.py` | Mesmo contrato v1 | Migrado; conteúdo e vocabulários são revalidados no worker. |
| `scripts/legacy/phase5_4_expanded.py` | Nenhuma mudança nesta tranche | Continua fora da garantia B007; seu checkpoint não deve ser presumido compatível por este contrato. |
| `src/kinera/experiments/comparison.py` | Fingerprint M1.4 já existente | Inalterado; não usa o caminho de identidade B007. |

Outros usos de `run_seed_jobs()` encontrados no repositório são a própria
implementação do runner e testes. A API e o comportamento genérico do runner
não foram alterados. A Fase 5.4 usa o mesmo runner, mas não é migrada para
evitar ampliar a tranche e alterar um protocolo histórico independente.

## Componentes da identidade

As fases usam o envelope `fingerprint_schema_version=1`, algoritmo SHA-256 e
`canonical_sha256()` existente em `kinera.experiments.fingerprint`. O digest
composto segue a mesma codificação tipada e o mesmo cálculo de digest de
componentes do contrato M1.4. O escopo dos componentes difere porque os dados
de entrada dos runners históricos não são os exemplos preparados do runner
comparativo.

O manifesto contém:

- `training_examples_sha256`: todos os campos dos registros efetivamente
  selecionados de treino, na ordem consumida.
- `validation_examples_sha256`: todos os campos dos registros efetivamente
  selecionados de validação, na ordem consumida.
- `character_vocabulary_sha256`: formato versionado, tokens ordenados e IDs
  especiais.
- `syllable_vocabulary_sha256`: formato versionado, tokens ordenados e ID
  desconhecido.
- `scientific_configuration_sha256`: fase, versão explícita do protocolo,
  parâmetros científicos serializados, variantes, tamanhos de vocabulário,
  dimensões do modelo e pesos da loss.
- `digest`: digest SHA-256 do envelope versionado e dos digests acima.

Os registros selecionados são hasheados integralmente em vez de manter uma
lista de campos considerada completa. Assim, alterações em metadados não
consumidos também podem invalidar o resume. Isso é conservador e evita que
campos usados por diagnósticos sejam omitidos. O fingerprint não contém o
dataset em si.

Seeds não fazem parte da identidade comum, de modo que a expansão compatível
continua permitida. Caminhos, diretório de execução, número de workers,
timestamps e telemetry operacional também não fazem parte dela. Trocar um
caminho por outro com conteúdo efetivo igual mantém o fingerprint.

`phase_protocol_version` começa em 1 e precisa ser incrementada em uma
alteração futura que mude o protocolo científico sem ser refletida nos dados
ou parâmetros já identificados. O fingerprint não detecta automaticamente
alterações científicas no código; toda alteração desse tipo requer decisão
explícita de versão.

## Elegibilidade de resume e integridade do arquivo

Antes de chamar `run_seed_jobs()`, cada entrypoint carrega os prefixos efetivos,
os vocabulários e constrói a identidade atual. Em `--resume`, valida o manifesto
persistido de forma estrita: versão, algoritmo, fase, conjunto e formato dos
componentes, digest interno e igualdade com a identidade atual. Checkpoints
legados sem manifesto, versões desconhecidas, digests inválidos ou componentes
divergentes são rejeitados. Essa validação é somente de leitura; o arquivo não
é reescrito em caso de incompatibilidade.

Depois do preflight, `load_seed_checkpoint()` ainda aplica as verificações
genéricas de configuração e expansão de seeds. Essa segunda comparação ocorre
antes da primeira escrita de checkpoint e antes da seleção das seeds
concluídas/pendentes.

Como os workers reabrem os arquivos, cada worker carrega seus próprios dados e
vocabulários, recalcula o manifesto com os mesmos parâmetros e compara com o
esperado antes de chamar o diagnóstico. Isso funciona tanto sequencialmente
quanto via `ProcessPoolExecutor`. Uma alteração externa entre o preflight e a
leitura do worker faz a seed falhar antes do treino; uma alteração antes do
preflight faz o próprio resume falhar sem tocar no checkpoint. Após a leitura e
validação no worker, os objetos em memória usados pelo diagnóstico são os
mesmos que foram hasheados.

Registros concluídos em checkpoint legado continuam podendo ser lidos como
JSON para inspeção histórica, mas não podem ser reutilizados automaticamente
pelas Fases 5.3/5.5. Não há migração retroativa de identidades.

## Limites e riscos residuais

SHA-256 é um hash de integridade de conteúdo para detectar divergências; não é
assinatura de autenticidade nem prova universal de equivalência científica.
O contrato cobre apenas os dados, vocabulários e parâmetros explicitamente
incluídos. O código do protocolo é representado por uma versão manual, não por
uma identidade automática do commit. Este B007 não protege a Fase 5.4 nem os
demais runners históricos que não foram migrados.

Não houve mudança em seleção/ordem de exemplos, treinamento, modelo, loss,
optimizer, RNG, seeds, diagnósticos, backends ou telemetry. A única nova leitura
de dados fora dos workers é o preflight necessário para calcular a identidade.

## Evidências de teste

Os testes determinísticos cobrem estabilidade, tipos e ordem do conteúdo,
alterações de targets/contextos/metadados, split de validação, mapeamentos de
vocabulário, configuração científica, seleção por prefixo, rejeição de
manifestos antigos ou inválidos, preservação byte a byte do checkpoint,
checagem no worker, expansão de seeds e mudança de caminho/worker com conteúdo
igual. O teste de integração da Fase 5.4 continua cobrindo o consumidor
genérico não migrado.

Validação local na base B007: `PYTHONPATH=src python -m pytest -q` aprovou
772 testes, com 54 skips e 740 subtestes em 118,60 s. O pytest emitiu um aviso
de depreciação do plugin AnyIO; não houve falhas. `compileall` de `src`,
`scripts` e `tests`, `git diff --check` e imports CPU-only dos módulos de
comparação e dos dois entrypoints passaram. O ambiente disponível é Python
3.15.0; Python 3.10 não está instalado, portanto a versão mínima não foi
executada localmente. CI da branch deve ser verificada após publicação.

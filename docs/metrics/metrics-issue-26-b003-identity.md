# Issue #26 — M1.4: identidade e resume de checkpoints

## Auditoria do fluxo anterior

`scripts/compare.py` carregava os vocabulários e os splits, limitando cada JSONL
com `--max-examples`. `run_controlled_experiments()` preparava cada variante,
preservava a ordem dos exemplos e aplicava novamente o limite. O runner então
criava um modelo por variante/seed, avaliava, treinava e escrevia o checkpoint
após cada resultado. Ao retomar, registros `completed` eram pulados por
variante e seed; registros incompletos eram repetidos e os agregados/deltas
eram recompostos a partir dos registros carregados.

Antes de M1.4, o resume conferia os parâmetros declarados e o backend. Também
permitia adicionar seeds, mas comparava a lista de variantes integralmente.
Não conferia conteúdo dos exemplos nem os mapeamentos de vocabulário. Assim,
um checkpoint podia reutilizar métricas e deltas de outro conteúdo com a mesma
configuração nominal. As variantes do runner compartilham os mesmos dados de
origem e diferem pelas máscaras de entrada A/B/C/D.

## Contrato de fingerprint v1

O checkpoint grava `experiment_fingerprint`, separado de
`metrics_schema_version` e `telemetry_schema_version`. O contrato usa SHA-256
de uma codificação binária tipada e com comprimentos explícitos:

- dicionários exigem chaves string e são ordenados por chave;
- listas preservam sua ordem;
- `None`, booleanos, inteiros, floats e strings têm tags distintos;
- floats finitos usam `float.hex()`; NaN e infinitos são rejeitados;
- outros tipos, como tuplas e objetos arbitrários, são rejeitados sem `repr()`.

O objeto contém um digest composto para componentes comuns, digests separados
para cada variante e a versão do contrato. Nenhum dataset é copiado para o
checkpoint.

| Componente | Conteúdo incluído |
| --- | --- |
| Treino e validação | Splits separados, ordem original e prefixo efetivamente selecionado; para cada exemplo: `context_ids`, `context_syllable_ids`, `context_feature_vectors`, `target_char_id`, `target_syllable_id` e `target_is_syllable_end` |
| Vocabulário de caracteres | Mapeamento serializado completo, versão e IDs de START/END/UNK |
| Vocabulário silábico | Mapeamento serializado completo, versão e ID de UNK |
| Configuração científica | Épocas, learning rate, truncamento, tamanho/ID desconhecido silábico, pesos das losses, família e dimensões do modelo, versão do protocolo |
| Entradas por variante | Entradas efetivamente preparadas para treino e validação, ordenadas, junto da versão de preparação |

Campos auxiliares que não chegam ao adaptador, treino ou avaliação — por
exemplo texto de origem, caminhos, timestamps e IDs de linha — não entram no
hash. `max_examples` e os splits são representados separadamente. Seeds não
fazem parte do digest comum; o checkpoint exige que as seeds antigas sejam
subconjunto das solicitadas. Variantes antigas também devem ser subconjunto da
seleção atual e seus fingerprints individuais devem coincidir. Isso permite
adicionar seeds/variantes sem invalidar resultados válidos já concluídos. A
adição de uma variante cria runs próprias e não altera os digests das variantes
anteriores.

O backend permanece sujeito à checagem existente de igualdade CPU/CUDA porque
diferenças numéricas de computação podem afetar resultados. Workers e coleta de
telemetria são operacionais e não entram no fingerprint científico. O envelope
de relatório com `metrics_schema_version=1` identifica o formato do envelope;
não afirma que cada registro histórico interno tenha sido produzido com
descritores v1. Relatórios históricos continuam legíveis pelos consumidores
compatíveis com os formatos anteriores.

## Política de resume e limites

O runner calcula e valida a identidade antes de instanciar backend/modelo,
reutilizar qualquer execução ou escrever no checkpoint. Diferença em split,
vocabulário, configuração ou entrada preparada de variante rejeita o resume,
indicando o componente sem imprimir dados. A rejeição preserva o checkpoint.
Uma versão de fingerprint desconhecida, digest interno inválido ou checkpoint
legado sem identidade também falha fechado. Checkpoints legados podem ser lidos
e inspecionados como relatórios; não há migração ou bypass automático. Para
criar checkpoint, o chamador precisa fornecer `SyllableVocabulary`, além do
tamanho e ID desconhecido já usados pelo runner.

SHA-256 demonstra igualdade dos componentes cobertos por este contrato; não é
assinatura contra adulteração deliberada e não prova equivalência científica
universal. Alterações futuras na semântica do modelo, treino, losses, adaptação
ou preparação de variantes devem atualizar `EXPERIMENT_PROTOCOL_VERSION` ou
`VARIANT_PREPARATION_VERSION`, conforme aplicável. Código, ambiente Python,
dependências e revisão Git não são hashados. A checagem CPU/CUDA existente e os
metadados de execução continuam separados da identidade do conteúdo.

## Persistência, telemetry e comportamento científico

O fingerprint é persistido no nível do relatório/checkpoint; todos os registros
e agregados naquele arquivo herdam essa identidade única. No resume, identidade
é validada antes de os registros serem carregados para agregação. As métricas,
orientação dos deltas, schema de telemetry, campos legados e lógica de
`_preserve_run_telemetry()` permanecem inalterados. A implementação não altera
seeds, RNG, inicialização, modelo, losses, gradientes, optimizer, seleção ou
ordem de updates, nem protocolo/backends científicos. Nenhum treinamento
ampliado, benchmark ou execução GPU faz parte desta tranche.

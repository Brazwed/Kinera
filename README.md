# Kinera — núcleo do neurônio

Este diretório contém o primeiro marco matemático do projeto Kinera.

O objetivo desta etapa é implementar um neurônio treinável usando somente Python e a biblioteca padrão. Não há dependência de NumPy, PyTorch, TensorFlow, Keras, JAX, scikit-learn ou bibliotecas prontas de machine learning.

A implementação começa com:

- ativações ReLU, LeakyReLU e linear;
- `forward`, `backward` e `step` manuais;
- gradient checking por diferenças finitas;
- teste de aprendizado da função `y = 2x`.

Embeddings, sílabas e atenção ficam para marcos posteriores.

## Contrato do `Neuron`

O neurônio é criado com o número de entradas e uma seed opcional:

```python
from nn.neuron import Neuron

neuron = Neuron(input_size=2, seed=7)
output = neuron.forward([0.5, 1.0])
neuron.backward(1.0)
neuron.step(learning_rate=0.01)
```

A interface deste marco é:

- `forward(inputs)`: calcula `z = soma(inputs * weights) + bias` e aplica ReLU;
- `backward(grad_output)`: calcula manualmente os gradientes dos pesos, bias e entradas;
- `step(learning_rate)`: aplica gradient descent uma única vez sobre o backward mais recente.

O ciclo esperado é `forward → backward → step`. Um novo `step` exige um novo `backward`, evitando aplicar duas vezes o mesmo gradiente. Entradas, gradientes e learning rate devem ser numéricos, finitos e não booleanos.

## Ativações e neurônio morto

O `Neuron` mantém `relu` como ativação padrão para preservar o baseline original, mas aceita:

```python
relu_neuron = Neuron(input_size=2, seed=7, activation="relu")
leaky_neuron = Neuron(
    input_size=2,
    seed=7,
    activation="leaky_relu",
    negative_slope=0.01,
)
linear_neuron = Neuron(input_size=2, seed=7, activation="linear")
```

Com a ReLU, uma inicialização que produz `z <= 0` para todas as amostras gera derivada zero em todos os parâmetros. O treino corrigido usa LeakyReLU para manter um gradiente pequeno no lado negativo. A ativação linear existe como controle afim e como preparação para futuras cabeças de saída.

## Executar os testes

Na pasta `main`:

```bash
python -m unittest discover -s tests -v
```

Os testes cobrem as funções ReLU, inicialização determinística, forward, backward, atualização de parâmetros, gradient checking e aprendizado de `y = 2x`.

## Ciclo de treinamento

O módulo `training.loop` separa treinamento e avaliação:

```python
from training.loop import TrainingConfig, fit

config = TrainingConfig(
    learning_rate=0.02,
    max_epochs=100,
    patience=5,
    min_delta=1e-8,
    shuffle=True,
    seed=42,
)
result = fit(neuron, train_samples, validation_samples, config)
```

O resultado registra loss de treino e validação por época, quantidade de passos, melhor época e motivo da parada. O early stopping restaura os pesos e o bias do melhor resultado de validação. A função `evaluate` não executa `backward` ou `step` e preserva o estado observável do neurônio.

O arquivo `train_5min.py` permanece como adaptador de teste de resistência. Ele reutiliza `train_epoch` e `evaluate`, mas o ciclo normal deve preferir `fit`, com limite de épocas e early stopping.

## Dados de treino e validação

O módulo `training.data` gera dados reproduzíveis para o experimento atual:

```python
from training.data import generate_samples, split_samples

samples = generate_samples(count=100, seed=7)
train_samples, validation_samples = split_samples(
    samples,
    validation_ratio=0.2,
    seed=42,
)
```

As amostras usam a função conhecida:

```text
y = 2x₁ + 3x₂ + 1
```

`generate_samples` usa valores positivos por padrão, mantendo o experimento na região adequada para a ReLU atual. `split_samples` embaralha com seed, cria cópias profundas das entradas e garante que treino e validação não compartilhem listas mutáveis.

A validação serve apenas para medir o modelo. Ela não deve ser enviada para `backward` ou `step`.

## Experimento com múltiplas seeds

O executor `run_experiment.py` usa o mesmo conjunto de dados e a mesma configuração de treinamento para várias inicializações:

```bash
python run_experiment.py
```

Por padrão, ele executa as seeds:

```text
1, 7, 42, 123, 2026
```

Também é possível escolher seeds e parâmetros:

```bash
python run_experiment.py \
    --seeds 1 7 42 \
    --activation leaky_relu \
    --negative-slope 0.01 \
    --sample-count 100 \
    --validation-ratio 0.2 \
    --max-epochs 500 \
    --patience 20 \
    --convergence-threshold 0.000001
```

O executor usa `leaky_relu` por padrão. Para reproduzir o baseline original:

```bash
python run_experiment.py --activation relu
```

O módulo `training.experiments` registra cada execução como `converged`, `not_converged` ou `error`. Nenhuma seed é descartada. O relatório inclui ativação, atividade inicial, épocas, passos, melhor loss de validação, pesos finais e taxa de convergência.

Resultado observado com a configuração padrão em 1 de outubro de 2026:

```text
ReLU:       2/5 convergiram (40%)
LeakyReLU:  5/5 convergiram (100%)
```

As seeds `7`, `123` e `2026` começaram com atividade `0/80` na ReLU e convergiram com LeakyReLU, sem alterar a inicialização dos pesos.

Uma seed que falhar por inicialização ruim da ReLU é um resultado válido do experimento e deve ser analisada, não escondida.

## Camada `Linear`

`nn.linear.Linear` implementa a transformação afim:

```text
y = W x + b
```

A camada:

- aceita múltiplas entradas e múltiplas saídas;
- usa `list[list[float]]` para os pesos;
- implementa `forward`, `backward` e `step` manualmente;
- não aplica ReLU nem outra ativação implicitamente;
- passa por gradient checking;
- foi comparada com vários `Neuron(activation="linear")` independentes;
- aprende duas funções de saída simultaneamente.

Exemplo:

```python
from nn.linear import Linear

layer = Linear(input_size=2, output_size=2, seed=7)
output = layer.forward([1.0, 2.0])
```

A separação entre `Linear` e ativação prepara as futuras cabeças de logits do modelo linguístico, que precisarão aceitar valores negativos.

## Checkpoints de pesos

Os pesos podem ser salvos em JSON versionado e recarregados sem `pickle`:

```python
from training.checkpoint import load_neuron, save_neuron

save_neuron(
    "checkpoints/neuron.json",
    neuron,
    training_state={"epoch": 100, "steps": 8000},
    metadata={"experiment": "multi_seed", "seed": 7},
)
restored, info = load_neuron("checkpoints/neuron.json")
```

O formato preserva pesos, bias, arquitetura, ativação, `negative_slope` e metadados. Estado transitório de `forward` e gradientes pendentes não são restaurados; o modelo carregado exige um novo `forward` antes de continuar o treino.

Para salvar o modelo de uma seed durante o experimento:

```bash
python run_experiment.py \
    --seeds 7 \
    --checkpoint-seed 7 \
    --checkpoint-path checkpoints/seed-7.json
```

O executor valida imediatamente o arquivo salvo carregando-o novamente e comparando os parâmetros persistidos.

## Datasets linguísticos

O primeiro dataset silábico está em:

```text
data/raw/silabas_pt/syls.csv
```

O arquivo original é mantido sem alteração. O importador normaliza os registros para:

```text
data/processed/syllable_records.jsonl
```

Comandos:

```bash
python -m scripts.inspect_silabas_pt data/raw/silabas_pt/syls.csv

python -m scripts.import_silabas_pt \
    data/raw/silabas_pt/syls.csv \
    data/processed/syllable_records.jsonl \
    --report data/processed/source_report.json

python -m scripts.build_syllable_dataset \
    data/processed/syllable_records.jsonl \
    --grouped data/processed/grouped_syllable_records.jsonl \
    --train data/processed/splits/train.jsonl \
    --validation data/processed/splits/validation.jsonl \
    --report data/processed/dataset_report.json
```

O dataset importado possui 181.560 registros e 155.323 palavras agrupadas. A divisão atual contém 124.259 grupos de treino e 31.064 de validação. Foram identificados 9 grupos com conflitos de análise silábica; eles foram preservados e marcados, não descartados.

Exportações opcionais do Portal podem ser convertidas com `scripts.import_portal_lingua`. Esse adaptador aceita JSONL, JSON e CSV, preserva IPA/categoria/variante e não torna o runtime dependente de rede ou de pacote externo.

Toda fonte é registrada em `data/manifests/sources.json`, incluindo hash, data de obtenção, transformação e status de redistribuição.

## Segmentador silábico inicial

O módulo `language.syllables` contém um segmentador baseado em regras para português. Ele trata núcleos vocálicos, dígrafos, ditongos, encontros consonantais e marcações comuns de `qu`/`gu`.

Avaliação contra o dataset importado:

```bash
python -m scripts.evaluate_syllables \
    data/processed/syllable_records.jsonl
```

Resultado atual do segmentador inicial:

```text
acurácia exata de palavras: 81,47%
acurácia de fronteiras:     94,12%
registros avaliados:        181.530
```

Esses números são um baseline para orientar a evolução das regras. O segmentador ainda não deve ser tratado como gold label; o dataset importado continua sendo a referência de avaliação.

## Alinhamento e features ortográficas

O módulo `language.alignment` mapeia cada caractere à sílaba correspondente, preservando separadores como hífen. O módulo `language.features` adiciona, por caractere:

- vogal/consoante;
- acentuação;
- maiúscula/minúscula;
- dígito;
- espaço;
- pontuação;
- categoria Unicode;
- posição dentro da sílaba;
- início/fim de sílaba.

Para gerar o dataset anotado:

```bash
python -m scripts.annotate_syllable_dataset \
    data/processed/syllable_records.jsonl \
    data/processed/annotated_syllable_records.jsonl \
    --report data/processed/annotation_report.json
```

Resultado atual:

```text
registros anotados:       181.560
alinhamentos válidos:     181.533
alinhamentos sinalizados: 27
```

Os 27 casos inconsistentes permanecem no dataset com flags de qualidade, em vez de serem apagados.

## Vocabulário e dataset autoregressivo

O vocabulário de caracteres usa tokens especiais estáveis:

```text
<START> → 0
<END>   → 1
<UNK>   → 2
```

Os exemplos autoregressivos usam contexto fixo de 8 caracteres e incluem:

- `context_ids`;
- `target_char_id`;
- caractere-alvo;
- índice da sílaba-alvo;
- indicação de fim de sílaba;
- `word_key` para auditoria.

Para gerar os artefatos:

```bash
python -m scripts.build_character_dataset \
    --annotated data/processed/annotated_syllable_records.jsonl \
    --train-groups data/processed/splits/train.jsonl \
    --validation-groups data/processed/splits/validation.jsonl \
    --vocabulary data/processed/vocabulary.json \
    --syllable-vocabulary data/processed/syllable_vocabulary.json \
    --train-output data/processed/autoregressive_train.jsonl \
    --validation-output data/processed/autoregressive_validation.jsonl \
    --report data/processed/character_dataset_report.json \
    --context-size 8
```

Resultado atual:

```text
vocabulário:              51 tokens
registros de treino:     145.287
exemplos de treino:    1.512.240
registros de validação:  36.273
exemplos de validação:  377.894
caracteres desconhecidos na validação: 0
```

O split é feito por palavra antes da geração dos exemplos, evitando que a mesma palavra apareça nos dois conjuntos.

## Embeddings

`nn.embedding.Embedding` transforma IDs de vocabulário em vetores treináveis:

```python
import json
from pathlib import Path

from nn.embedding import Embedding
from language.vocabulary import CharacterVocabulary

vocabulary = CharacterVocabulary.from_dict(
    json.loads(Path("data/processed/vocabulary.json").read_text())
)
embedding = Embedding(
    vocab_size=len(vocabulary),
    embedding_dim=16,
    seed=7,
)
vectors = embedding.forward(vocabulary.encode("ação", add_boundaries=True))
```

A camada implementa:

- lookup de linhas por ID;
- `forward` preservando a ordem dos caracteres;
- `backward` esparso, acumulando gradientes quando um ID se repete;
- `step` apenas sobre as linhas que receberam gradiente.

O embedding não recebe texto diretamente. O fluxo correto é:

```text
texto → CharacterVocabulary → IDs → Embedding → vetores
```

## Representação híbrida

O marco seguinte combina, para cada posição do texto:

```text
embedding do caractere
+ features ortográficas
+ embedding da sílaba
```

`nn.hybrid.HybridRepresentation` concatena os três blocos e propaga o gradiente para as duas tabelas de embeddings. As features ortográficas permanecem observáveis e não treináveis nesta etapa.

As 10 features usadas, em ordem fixa, são:

```text
is_vowel, is_consonant, is_accented, is_alpha, is_digit,
is_space, is_punctuation, is_upper, is_lower, is_separator
```

O vocabulário silábico foi gerado a partir dos registros anotados:

```text
registros lidos: 181.560
sílabas distintas: 3.931
vocabulário: 3.932 tokens
token desconhecido: <UNK_SYL>
```

Artefatos:

```text
data/processed/syllable_vocabulary.json
data/processed/syllable_vocabulary_report.json
```

Para regenerar:

```bash
python -m scripts.build_syllable_vocabulary \
    --source data/processed/annotated_syllable_records.jsonl \
    --vocabulary data/processed/syllable_vocabulary.json \
    --report data/processed/syllable_vocabulary_report.json
```

`language.hybrid_inputs.record_to_hybrid_inputs` preserva o alinhamento caractere/sílaba. Separadores sem sílaba recebem `<UNK_SYL>` e continuam presentes na sequência.

## Restrições deste marco

O código usa somente a biblioteca padrão do Python. Nenhuma etapa deste diretório deve importar NumPy, PyTorch, TensorFlow, Keras, JAX, scikit-learn ou uma implementação pronta de redes neurais.

## Primeiro modelo autoregressivo

O módulo `training.autoregressive` conecta a representação híbrida a uma
cabeça de logits. O modelo usa média dos vetores do contexto como agregador
intencionalmente simples antes da futura implementação de atenção.

O executor treina uma amostra limitada, mede validação sem atualizar pesos e
faz geração gulosa:

```bash
python run_autoregressive.py \
    --vocabulary data/processed/vocabulary.json \
    --train data/processed/autoregressive_train.jsonl \
    --validation data/processed/autoregressive_validation.jsonl \
    --syllable-vocabulary data/processed/syllable_vocabulary.json \
    --epochs 3 \
    --max-examples 1000
```

A geração atual do dataset persiste `context_syllable_ids` e
`context_feature_vectors` alinhados aos oito caracteres do contexto. Exemplos
antigos continuam compatíveis: quando esses campos não existem, o adaptador
usa `<UNK_SYL>` como fallback.

Quando um vocabulário silábico é informado ao builder, cada exemplo também
persiste `target_syllable_id`. O token `<END>` e caracteres sem alinhamento
silábico usam o ID de `<UNK_SYL>`.

Os IDs silábicos do contexto são causais: uma sílaba só recebe seu embedding
depois que uma posição observada possui `is_syllable_end=true`. Enquanto a
sílaba atual está parcial, suas posições usam `<UNK_SYL>`. Isso impede que o
próximo caractere receba a forma completa de uma sílaba que ainda contém
caracteres futuros.

O builder processa o JSONL em streaming: ele faz duas passagens pelo arquivo
de anotações e escreve os exemplos diretamente nos arquivos de saída, sem
carregar o corpus completo na memória.

## Treino multiobjetivo

`training.autoregressive.train_multitask_examples` treina as três heads e
registra `char_loss`, `syllable_loss`, `boundary_loss`, `total_loss` e as três
acurácias. A avaliação também separa `syllable_accuracy_all` de
`syllable_accuracy_known` e calcula precision, recall e F1 da fronteira
silábica. `evaluate_multitask_examples` executa somente forward e não altera
os parâmetros.

O fluxo recomendado para experimentos usa uma amostra limitada do JSONL:

```python
from language.dataset import load_jsonl
from training.autoregressive import (
    AutoregressiveModel,
    evaluate_multitask_examples,
    train_multitask_examples,
)

train = load_jsonl("data/processed/autoregressive_train.jsonl", max_records=1000)
validation = load_jsonl(
    "data/processed/autoregressive_validation.jsonl",
    max_records=1000,
)
```

## Comparação controlada A/B/C/D

O runner `run_comparison.py` mantém a mesma dimensão interna, heads, loss,
split, seeds e configuração de treino. Ele apenas mascara os canais de
entrada:

```text
A = caracteres
B = caracteres + features ortográficas
C = caracteres + sílabas causais
D = caracteres + features + sílabas causais
```

As sílabas usadas em C/D são somente as sílabas já completamente observadas.
O runner registra média e desvio entre seeds, além da contagem de parâmetros.

Na Fase 7.2, a classificação é explícita: A é o baseline mínimo, B é uma
ablação opcional, e C/D são as variantes principais. `run_comparison.py` aceita
`--variants` (choices A/B/C/D) e, em novas execuções, usa A/C/D por padrão.
Por exemplo, `--variants D` executa somente D e `--variants A B C D`
reproduz a comparação controlada histórica completa. A lista escolhida fica no
relatório/checkpoint; `--resume` rejeita uma seleção diferente.

E/F são variantes experimentais históricas da Fase 5.4. Elas não são suportadas
por `run_comparison.py` nem entram nos novos treinos por padrão; a reprodução
A/B/C/D/E/F continua disponível deliberadamente em `run_phase5_4_expanded.py`.

Para a Fase 5.1 expandida:

```bash
python run_comparison.py \
  --vocabulary data/processed/vocabulary.json \
  --syllable-vocabulary data/processed/syllable_vocabulary.json \
  --train data/processed/autoregressive_train.jsonl \
  --validation data/processed/autoregressive_validation.jsonl \
  --seeds 7 42 123 --epochs 2 --learning-rate 0.05 \
  --variants A B C D \
  --max-examples 1000 \
  --checkpoint experiments/phase5_1_checkpoint.json \
  --workers 2
```

Além das métricas, o relatório registra duração por seed e o uso efetivo do
canal silábico: posições não-`<UNK_SYL>` e quantidade de IDs silábicos únicos
presentes em treino e validação.

O checkpoint é atualizado depois de cada variante/seed usando substituição
atômica. Se o processo for interrompido, a execução pode ser retomada com os
mesmos parâmetros e `--resume`; combinações concluídas não são repetidas:

```bash
python run_comparison.py \
  --vocabulary data/processed/vocabulary.json \
  --syllable-vocabulary data/processed/syllable_vocabulary.json \
  --train data/processed/autoregressive_train.jsonl \
  --validation data/processed/autoregressive_validation.jsonl \
  --seeds 7 42 123 --epochs 2 --learning-rate 0.05 \
  --variants A B C D \
  --max-examples 1000 \
  --checkpoint experiments/phase5_1_checkpoint.json \
  --workers 2 \
  --resume
```

Falhas individuais ficam registradas com traceback sem apagar os resultados
anteriores. Os agregados consideram somente execuções concluídas.

Cada job possui estado explícito no checkpoint: `pending` antes de ser enviado
ao pool, `running` enquanto aguarda o worker, `completed` após sucesso e
`failed` quando ocorre erro.

`--workers 1` mantém a execução sequencial de referência. Valores maiores
executam combinações independentes `variante × seed` em processos separados;
a escrita do checkpoint continua centralizada no processo principal.

Benchmark registrado em `experiments/phase5_2_parallel_benchmark.json`, com 50
exemplos por split, 1 seed, 1 época e 4 vCPUs:

```text
workers    tempo       speedup    pico RSS/worker    resultados iguais
1          21,06 s     1,00x      37,1 MB           referência
2          10,46 s     2,01x      36,0 MB           sim
4           5,50 s     3,83x      36,0 MB           sim
```

Smoke comparison registrada em `experiments/phase5_controlled_comparison.json`
com 50 exemplos, 2 épocas e 3 seeds:

```text
        char_loss médio    char_accuracy médio
A       3.7355             4,0%
B       3.6793            11,33%
C       3.7514            10,0%
D       3.6880            11,33%
```

B e D foram melhores neste smoke test, mas a diferença ainda não é evidência
definitiva de que sílabas causais ajudam. A comparação maior deve usar a mesma
configuração e mais exemplos antes de concluir.

### Fase 5.1 expandida

A execução expandida foi concluída com 1.000 exemplos de treino, 1.000 de
validação, 2 épocas e as seeds `7`, `42` e `123`. Cada variante manteve
139.432 parâmetros treináveis.

```text
        char_loss médio ± desvio    char_accuracy médio ± desvio
A       2.521841 ± 0.001549         30,900% ± 0,000%
B       2.507925 ± 0.002448         31,167% ± 0,189%
C       2.396040 ± 0.010389         32,167% ± 0,205%
D       2.352711 ± 0.002371         31,500% ± 0,082%
```

C e D usaram efetivamente o canal silábico causal: 3.593 posições não-`<UNK_SYL>`
e 99 IDs únicos no treino; 3.883 posições e 95 IDs únicos na validação.

Resultado: C teve a maior acurácia de caractere, enquanto D teve a menor
`char_loss`. Em relação a A, D reduziu a `char_loss` em aproximadamente 0,1691
e aumentou a acurácia em 0,6 ponto percentual. Em relação a B, D reduziu a
`char_loss` em aproximadamente 0,1552, mas a acurácia aumentou apenas 0,33
ponto percentual. O resultado é positivo, mas ainda deve ser interpretado com
cautela porque a avaliação usou apenas 1.000 exemplos e 2 épocas.

Relatório completo: `experiments/phase5_1_expanded_comparison.json`.

### Fase 5.3 — Análise do ganho silábico

O runner `run_syllable_analysis.py` diagnostica o efeito do canal silábico sem
alterar a arquitetura. Ele compara A/C e B/D por exemplo, separando posições
com informação silábica causal ativa das posições que permanecem em
`<UNK_SYL>`.

Também registra buckets por posição na palavra, proximidade de fronteira,
frequência do target silábico, efeitos por caractere e exemplos
representativos de ganho/regressão. A forma do exemplo continua baseada apenas
no prefixo causal persistido no dataset.

Execução de smoke registrada em
`experiments/phase5_3_syllable_gain_analysis.json`:

```bash
python run_syllable_analysis.py \
  --vocabulary data/processed/vocabulary.json \
  --syllable-vocabulary data/processed/syllable_vocabulary.json \
  --train data/processed/autoregressive_train.jsonl \
  --validation data/processed/autoregressive_validation.jsonl \
  --seeds 7 --epochs 1 --learning-rate 0.05 \
  --max-examples 50 --output experiments/phase5_3_syllable_gain_analysis.json
```

No smoke test corrigido, 42/50 posições tinham informação silábica causal
ativa. A/C e B/D foram avaliados por deltas de loss, acurácia e probabilidade
do caractere correto. Os buckets agora usam a metadata causal de C/D, não os
IDs mascarados de A/B. O primeiro smoke tinha essa classificação incorreta e
não deve ser usado para conclusões por bucket.

No smoke corrigido, o ganho médio de loss continuou levemente negativo:
`-0,002953` para A/C nas posições ativas e `-0,002199` para B/D. Isso é uma
amostra pequena e não altera a conclusão da Fase 5.1; a execução com 1.000
exemplos e 3 seeds ainda deve ser feita antes de concluir que o ganho é amplo.

Para a execução ampliada da Fase 5.3, use o runner retomável por seed:

```bash
python run_phase5_3_expanded.py \
  --vocabulary data/processed/vocabulary.json \
  --syllable-vocabulary data/processed/syllable_vocabulary.json \
  --train data/processed/autoregressive_train.jsonl \
  --validation data/processed/autoregressive_validation.jsonl \
  --seeds 7 42 123 --epochs 2 --learning-rate 0.05 \
  --max-examples 1000 --workers 3 \
  --checkpoint experiments/phase5_3_expanded_analysis.json \
  --resume
```

Cada seed é um job independente e o checkpoint é atualizado quando cada job
termina. A execução usa a mesma arquitetura A/B/C/D da Fase 5.1; somente a
análise e a escala dos dados mudam.

A execução ampliada foi concluída com 1.000 exemplos por split, 2 épocas e as
três seeds. A cobertura foi idêntica nas seeds: 894/1.000 posições (`89,4%`)
com informação silábica causal ativa e 106/1.000 sem essa informação.

```text
comparação   delta médio de loss    delta médio de accuracy
A vs C        +0,125801              +1,27 p.p.
B vs D        +0,155213              +0,33 p.p.
```

O delta de loss é `loss_base - loss_aumentada`; portanto, positivo favorece a
variante com canal silábico. C/D melhoraram a loss média nos três seeds. C
teve o ganho médio de acurácia mais forte. O relatório completo, incluindo
resultados por seed, buckets e exemplos, está em
`experiments/phase5_3_expanded_analysis.json`.

### Fase 5.4 — Sílaba parcial causal

A Fase 5.4 adiciona duas variantes sem substituir A/B/C/D:

```text
E = caracteres + sílaba parcial causal
F = caracteres + features + sílaba parcial causal
```

A sílaba parcial é construída somente com os caracteres observados desde a
última sílaba concluída. IDs de sílabas concluídas são preservados; prefixos
parciais usam buckets determinísticos dentro da mesma tabela de embeddings,
mantendo a contagem de parâmetros controlada.

`run_phase5_4_expanded.py` continua sendo o runner histórico para reprodução
explícita da comparação A/B/C/D/E/F. E/F não são movidas para `run_comparison.py`
e não entram no default A/C/D da Fase 7.2.

Smoke test registrado em `experiments/phase5_4_partial_syllable_smoke.json`.
O runner ampliado e retomável é:

```bash
python run_phase5_4_expanded.py \
  --vocabulary data/processed/vocabulary.json \
  --syllable-vocabulary data/processed/syllable_vocabulary.json \
  --train data/processed/autoregressive_train.jsonl \
  --validation data/processed/autoregressive_validation.jsonl \
  --seeds 7 42 123 --epochs 2 --learning-rate 0.05 \
  --max-examples 1000 --workers 3 \
  --checkpoint experiments/phase5_4_expanded_analysis.json \
  --resume
```

Os testes garantem que a representação parcial não usa caracteres futuros,
muda apenas quando novos caracteres observáveis entram e mantém o mesmo
número de parâmetros de A/B/C/D.

A execução ampliada da Fase 5.4 foi concluída com 1.000 exemplos por split,
2 épocas e seeds `7`, `42`, `123`. Todas as variantes mantiveram 139.432
parâmetros.

```text
variante   char_loss médio ± desvio    char_accuracy médio ± desvio
C          2.396040 ± 0.010389         32,167% ± 0,205%
D          2.352711 ± 0.002371         31,500% ± 0,082%
E          2.430864 ± 0.000436         31,133% ± 0,047%
F          2.394122 ± 0.002124         31,133% ± 0,047%
```

Conclusão desta rodada: a representação parcial causal não superou a
representação de última sílaba concluída. E ficou 1,03 ponto percentual abaixo
de C em acurácia e com `char_loss` 0,034824 maior. F ficou 0,37 ponto
percentual abaixo de D e com `char_loss` 0,041410 maior. Portanto, a sílaba
parcial está causalmente válida, mas não deve substituir o esquema atual com
base neste experimento.

Relatório: `experiments/phase5_4_expanded_analysis.json`.

### Fase 5.5 — Análise por classe ortográfica

Esta fase diagnostica o ganho de C/D sem alterar a arquitetura. Os targets são
separados em `vogal`, `consoante`, `vogal_acentuada`, `<END>`,
`separador_pontuacao` e `outros`. `<END>` e pontuação não são misturados com
letras na interpretação.

Runner ampliado e retomável:

```bash
python run_phase5_5_expanded.py \
  --vocabulary data/processed/vocabulary.json \
  --syllable-vocabulary data/processed/syllable_vocabulary.json \
  --train data/processed/autoregressive_train.jsonl \
  --validation data/processed/autoregressive_validation.jsonl \
  --seeds 7 42 123 --epochs 2 --learning-rate 0.05 \
  --max-examples 1000 --workers 3 \
  --checkpoint experiments/phase5_5_orthographic_class_analysis.json \
  --resume
```

Para cada classe, o relatório registra quantidade de exemplos, `char_loss`,
`char_accuracy`, delta de loss, delta de acurácia, probabilidade média do
target correto e fração de exemplos melhorados/regredidos.

A execução ampliada foi concluída com 1.000 exemplos por split, 2 épocas e as
seeds `7`, `42`, `123`. Médias entre seeds:

```text
classe              n     A vs C loss   A vs C acc    B vs D loss   B vs D acc
vogal               456   +0,242796     +2,12 p.p.    +0,222917     +1,17 p.p.
consoante           426   +0,067275      0,00 p.p.    +0,149653      0,00 p.p.
<END>                98   -0,071900     +3,06 p.p.    -0,071622     -2,04 p.p.
vogal_acentuada      13   -0,280952      0,00 p.p.    -0,200466      0,00 p.p.
separador_pontuacao   7   -0,410634      0,00 p.p.    -0,080541      0,00 p.p.
```

O canal silábico mostrou ganho consistente de `char_loss` em vogais e
consoantes, principalmente vogais. Já `<END>`, vogais acentuadas e pontuação
regrediram em loss; as classes pequenas exigem cautela, especialmente as 13
vogais acentuadas e os 7 separadores. O resultado não autoriza generalizar o
benefício para todos os targets.

Relatório: `experiments/phase5_5_orthographic_class_analysis.json`.

### Fase 6 — Validação final

O relatório `experiments/phase6_final_validation.json` registra a validação dos
critérios finais:

- todos os testes existentes passam;
- o micro-overfit reduziu `total_loss` de `2,636048` para `0,513922`;
- `char_accuracy` final do subconjunto: `66,67%`;
- avaliação não altera parâmetros;
- mesma seed produz o mesmo histórico;
- gradient checks existentes continuam válidos;
- evidência held-out das Fases 5.1/5.3 confirma redução de `char_loss` em C vs
  A e D vs B.

Isso valida o estado atual sem encerrar a Issue: a decisão de manter ou alterar
a arquitetura deve continuar baseada nos diagnósticos já registrados.

### Fase 7 — Consolidação estrutural

A Fase 7 consolidou a execução experimental sem alterar a matemática do modelo:

- `ExperimentConfig` separa configuração matemática de `ExecutionConfig`;
- `RunRecord` e `JobStatus` formalizam estados e metadados por seed;
- `training.runner_common` centraliza checkpoint atômico, resume e agregação;
- `training.multi_seed_runner` centraliza execução serial/multiprocessada,
  expansão de seeds, retry e persistência incremental;
- `workers=1` permanece a referência determinística;
- os runners 5.3, 5.4 e 5.5 são mantidos como adaptadores históricos.

As invariantes de causalidade, dimensões, loss, parâmetros, avaliação e
checkpoints estão documentadas em `docs/phase7-invariants.md`. A equivalência
matemática antes/depois da consolidação e entre `workers=1`/`workers=2` está
registrada em `experiments/phase7_deterministic_validation.json`.

### Fase 7.1 — Performance sem alterar a matemática

O baseline foi medido na variante D com:

```text
seed=7, workers=1, epochs=1, max_examples=1000, context_size=8
```

O runner `run_phase7_1_performance_baseline.py` separa preparação/features,
`forward`, loss, backward, atualização do otimizador e avaliação. Também registra
tempo total, percentual por etapa, pico de RSS e hotspots do `cProfile`.

A primeira otimização aprovada troca o generator expression do produto interno
por `sum(map(operator.mul, row, inputs))` em `Linear.forward()`. O microbenchmark
pareado confirmou igualdade exata e B mais rápida em todos os blocos. No workload
D, o tempo foi de **256,01 s para 180,31 s** (−75,70 s; **29,57%**), com todas
as métricas e a contagem de **139.432 parâmetros** exatamente iguais. Na validação
dessa mudança, os 6 gradient checks e os 225 testes passaram.

A segunda otimização aprovada aplica C ao backward: converte `grad_output` uma
vez e materializa as colunas dos pesos por chamada antes de usar
`sum(map(operator.mul, column, grad_output_float))`. Os testes permanentes comparam
`grad_inputs`, `grad_weights` e `grad_bias` exatamente com a referência generator
nos shapes `51×26` e `3932×26`. Dois pares completos separados (A1/C1 e C2/A2)
reduziram o workload D em **19,08%** e **17,33%**, respectivamente; as métricas
completas e os **139.432 parâmetros** ficaram idênticos. A suíte passou com
**226 testes**, incluindo **6 gradient checks**. O RSS aumentou **964 KiB (1,80%)**
no primeiro par e **616 KiB (1,16%)** no segundo. Os 819.624 bytes atribuídos à
transposição são apenas o tamanho superficial dos containers; o RSS de processo
é a medida prática registrada nos pares.

O reprofile pós-commit da alteração de backward levou **149,50 s**. A avaliação
voltou a ser o maior estágio (**28,01%**), seguida de backward (**21,04%**),
atualização do otimizador (**20,10%**), forward (**19,27%**) e loss (**10,95%**).

A terceira otimização aprovada hoista `float(learning_rate)` para fora dos loops
em `Linear.step()`, mantendo a reconstrução das listas. Dois pares frescos
(A1/B1 e B2/A2) reduziram o workload D de **149,10 para 138,29 s** (**7,25%**) e
de **147,77 para 137,96 s** (**6,64%**). `initial_validation`, `training.history`,
`final_validation` e os **139.432 parâmetros** ficaram exatamente iguais; RSS
caiu 420 KiB e 748 KiB nos pares. A suíte continuou em **226/226**, com **6/6
gradient checks**.

O reprofile pós-commit de B marcou **140,45 s**; `optimizer_update` ficou em
**19,67 s (14,00%)** e `Linear.step()` em **11,51 s** de tempo próprio. O estágio
`evaluation` aparecia com 31,25%, mas `evaluate_multitask_examples()` tinha apenas
**0,22 s** de tempo próprio (67,11 s cumulativos): agrega o custo das primitivas
compartilhadas, não é um hotspot isolado.

A quarta otimização aprovada altera `Embedding.forward()` sem mudar seu contrato:
cacheia `vocab_size`/`embedding_dim` localmente e usa `[0.0] * embedding_dim` para
cada linha, mantendo o reset integral de `grad_weights`, linhas independentes,
validação e cópias de saída. O microbenchmark nos shapes reais `3932×8` e `51×8`
confirmou igualdade exata; B venceu A nos 6/6 blocos, com speedups medianos de
**1,925×** e **2,355×**. C, que resetava só IDs usados, foi descartada: o teste de
múltiplos batches mostrou que `Embedding.step()` reaplicaria gradientes antigos
nas linhas omitidas.

No workload D, os pares frescos A1→B1 e B2→A2 reduziram o total de **146,03 para
139,05 s (4,78%)** e de **140,01 para 135,73 s (3,06%)**. `initial_validation`,
`training.history`, `final_validation`, workload e **139.432 parâmetros** ficaram
exatamente iguais nos quatro runs. O RSS variou **+584 KiB (+1,09%)** no primeiro
par e **−564 KiB (−1,04%)** no segundo, sem regressão consistente. A suíte passou
com **226/226 testes** e **6/6 gradient checks**. O teste permanente de forward
agora cobre dois batches e garante que gradientes antigos não alterem pesos em
linhas ausentes do batch seguinte. Implementação/teste: commit `4c700bf`.

O reprofile pós-commit de `Embedding.forward()` marcou **131,20 s**. Os estágios
foram avaliação **38,82 s (29,59%)**, backward **31,26 s (23,83%)**, forward
**23,15 s (17,64%)**, atualização **20,29 s (15,46%)** e loss **16,75 s (12,76%)**.
`evaluate_multitask_examples()` manteve tempo próprio baixo (**0,20 s**, 57,99 s
cumulativos). O profile ainda mostrava custo em `_validate_logits` (**8,90 s
próprios**), `losses._validate_number` (**8,68 s**) e `losses._validate_outputs`
(**6,49 s**). A investigação A/B separada manteve as APIs públicas validadas e
fez `MultiTaskLoss` chamar helpers aritméticos internos depois de validar outputs
e targets. O microbenchmark exato e a bateria de exceções estão em
`experiments/phase7_1_validated_loss_helpers_microbenchmark.json`.

O trial temporário D passou nos dois pares frescos: A1→B1 caiu de **134,21 para
111,93 s (−16,60%)** e B2→A2 de **136,04 para 116,88 s (−14,09%)**. Os quatro runs
preservaram exatamente `initial_validation`, `training.history`, `final_validation`,
configuração e **139.432 parâmetros**. RSS: **+196 KiB (+0,36%)** no primeiro par;
**−28 KiB (−0,05%)** no segundo. Em ambos os pares, o estágio `loss` caiu de
**16,67/16,84 s** para **11,39/11,45 s**. A implementação permanente mantém
`softmax()`, `cross_entropy()` e `cross_entropy_gradient()` públicos com validação;
`MultiTaskLoss` usa os helpers internos após seus próprios checks. Foram adicionados
testes permanentes de tipos e mensagens de erro. Suíte: **229/229**; gradient
checks: **6/6**. Implementação/testes: commit `9f3c7e7`.

O reprofile pós-commit marcou **116,61 s**, com evaluation **33,94 s (29,11%)**,
backward **25,80 s (22,13%)**, forward **24,24 s (20,79%)**, atualização
**19,91 s (17,07%)** e loss **11,76 s (10,08%)**. `_validate_logits` não aparece
entre os hotspots principais reportados; permanecem `_validate_outputs`
(**6,57 s próprios**) e `_validate_number` (**8,71 s**), ainda necessários no
contrato atual. Qualquer tentativa de reduzir esses checks exige uma investigação
separada e não está incluída nesta otimização.

Relatórios versionados:

```text
experiments/phase7_1_performance_baseline.json
experiments/phase7_1_linear_dot_microbenchmark.json
experiments/phase7_1_linear_dot_candidate.json
experiments/phase7_1_linear_dot_comparison.json
experiments/phase7_1_linear_dot_post_profile.json
experiments/phase7_1_linear_backward_microbenchmark.json
experiments/phase7_1_linear_backward_microbenchmark_round2.json
experiments/phase7_1_linear_backward_comparison.json
experiments/phase7_1_linear_backward_full_D_A1.json
experiments/phase7_1_linear_backward_full_D_C1.json
experiments/phase7_1_linear_backward_full_D_C2.json
experiments/phase7_1_linear_backward_full_D_A2.json
experiments/phase7_1_linear_backward_paired_validation.json
experiments/phase7_1_linear_backward_post_commit_profile.json
experiments/phase7_1_linear_step_microbenchmark.json
experiments/phase7_1_linear_step_microbenchmark_round2.json
experiments/phase7_1_linear_step_comparison.json
experiments/phase7_1_linear_step_full_D_A1.json
experiments/phase7_1_linear_step_full_D_B1.json
experiments/phase7_1_linear_step_full_D_B2.json
experiments/phase7_1_linear_step_full_D_A2.json
experiments/phase7_1_linear_step_paired_validation.json
experiments/phase7_1_linear_step_post_commit_profile.json
experiments/phase7_1_embedding_forward_microbenchmark.json
experiments/phase7_1_embedding_forward_comparison.json
experiments/phase7_1_embedding_forward_full_D_A1.json
experiments/phase7_1_embedding_forward_full_D_B1.json
experiments/phase7_1_embedding_forward_full_D_B2.json
experiments/phase7_1_embedding_forward_full_D_A2.json
experiments/phase7_1_embedding_forward_paired_validation.json
experiments/phase7_1_embedding_forward_post_commit_profile.json
experiments/phase7_1_validated_loss_helpers_microbenchmark.json
experiments/phase7_1_validated_loss_helpers_full_D_A1.json
experiments/phase7_1_validated_loss_helpers_full_D_B1.json
experiments/phase7_1_validated_loss_helpers_full_D_B2.json
experiments/phase7_1_validated_loss_helpers_full_D_A2.json
experiments/phase7_1_validated_loss_helpers_paired_full_D.json
experiments/phase7_1_validated_loss_helpers_post_commit_profile.json
```

**Fase 7.1 encerrada.** No workload D de referência, o baseline de **256,01 s** foi a **116,61 s** (**−54,45%**), com matemática e **139.432 parâmetros** preservados, suítes de testes e gradient checks aprovados e as cinco otimizações sustentadas por comparação pareada. Novas otimizações só serão retomadas após profiling em workloads maiores; os ganhos restantes não justificam aumentar complexidade sem nova evidência.

### Fase 7.2 — Consolidação dos experimentos principais

A seleção do runner controlado é explícita: A é o baseline mínimo, B é uma
ablação opcional, e C/D são as variantes principais. `run_comparison.py` executa
A/C/D por padrão; `--variants A B C D` reproduz a comparação controlada
histórica completa. A lista escolhida é registrada nos relatórios/checkpoints e
uma retomada com seleção incompatível é rejeitada.

E/F permanecem como variantes experimentais históricas da Fase 5.4. Não entram
em novos treinos do runner controlado; a reprodução A/B/C/D/E/F continua no
`run_phase5_4_expanded.py`.

A validação autocontida em `experiments/phase7_2_final_validation.json` compara o
runner pré-7.2 (default A/B/C/D) com a seleção pós-7.2 explícita, usando a mesma
seed/configuração e um exemplo determinístico. `initial_validation`,
histórico de treino, `final_validation` e contagens de parâmetros foram
idênticos. O mesmo relatório confirma o default A/C/D, a seleção D-only e a
preservação histórica A/B/C/D/E/F. É uma equivalência funcional em fixture
pequena, não uma conclusão de qualidade ou performance no dataset completo.

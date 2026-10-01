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

Para a Fase 5.1 expandida:

```bash
python run_comparison.py \
  --vocabulary data/processed/vocabulary.json \
  --syllable-vocabulary data/processed/syllable_vocabulary.json \
  --train data/processed/autoregressive_train.jsonl \
  --validation data/processed/autoregressive_validation.jsonl \
  --seeds 7 42 123 --epochs 2 --learning-rate 0.05 \
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

No smoke test, 42/50 posições tinham informação silábica causal ativa. A/C e
B/D foram avaliados por deltas de loss, acurácia e probabilidade do caractere
correto. A execução com 1.000 exemplos e 3 seeds ainda deve ser feita antes de
concluir que o ganho é amplo ou generalizado.

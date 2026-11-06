# F2.0 — Auditoria funcional de representação e perdas OpenCL

## Resumo

Esta auditoria mapeia os caminhos necessários para levar o backend OpenCL além dos gates individuais de Linear e Embedding. O foco é a composição residente de HybridRepresentation, pooling, heads e MultiTaskLoss, além dos consumidores em AutoregressiveModel, trainer, evaluator e runners.

**Conclusão:** no baseline funcional auditado, CPU e CUDA possuem os caminhos de representação, pooling e multitarefa descritos neste relatório. OpenCL ainda não oferece fusão de HybridRepresentation, mean pooling, operações residentes de loss, agregação de gradientes das heads, backward do pooling, divisão residente da representação ou argmax. Portanto, o modelo AutoregressiveModel não está funcionalmente validado em OpenCL. F1a–F1d comprovam operações individuais e o ciclo integrado da classe Embedding; não comprovam o ciclo do modelo.

O primeiro gate funcional recomendado é F2.1, fusão residente do forward de HybridRepresentation. A validação de entradas residentes precisa detectar NaN e ±Inf sem baixar matrizes completas. Os gates F2.8 e F2.9 — mean_pool_backward e HybridRepresentation.backward — são implementáveis e testáveis separadamente; somente a cadeia do modelo depende dos dois.

Este documento registra o plano. Não autoriza implementação F2 nem altera a checklist da Issue #61.

## Proveniência e escopo da auditoria

- **Baseline funcional auditado:** commit 36f7fb9e5433e295b8556328db4e89ab669245af, merge do PR #69 e conclusão de F1d.
- **Main confirmada para esta publicação:** a3c0dd2da678ecab29852dd8ccaefc8a4be10398, merge do PR #70.
- O delta de 36f7fb9 até a3c0dd2 adiciona validação opcional por época ao runner de comparação e ao trainer, sem adicionar operações OpenCL ou mudar os contratos de representação/perdas aqui auditados. O runner continua selecionando somente CPU/CUDA via ExecutionConfig. Essa diferença foi inspecionada para manter a análise de consumidores atualizada; os gates experimentais F1 não foram refeitos.
- A Issue #61 confirma que F1a–F1d foram integrados e que os quatro itens F0 permanecem desmarcados até revisão/publicação documental. As alterações de revisão registradas na própria Issue estão incorporadas abaixo.
- **Nenhum teste, benchmark ou experimento foi executado nesta entrega.** A cobertura citada vem da inspeção do código e dos testes existentes.

## Como ler a matriz

Os status são separados para não confundir existência de método com suporte de fluxo:

- **Impl.**: operação implementada no backend.
- **Integr.**: operação efetivamente chamada pelo consumidor descrito.
- **Valid.**: cobertura existente indicada; unit significa testes CPU/software, fake significa fake OpenCL API, GPU opt-in significa gate de hardware que exige habilitação.
- **Indisp.**: chamada falha explicitamente ou a capacidade não está implementada.

Uma célula CUDA/OpenCL descrita como residente trata de buffers no dispositivo. Ela não implica que as mesmas operações estejam disponíveis através da API pública que recebe listas Python.

## Matriz operação × consumidor × backend

| Operação | Consumidor e contrato observado | CPU | CUDA | OpenCL |
|---|---|---|---|---|
| Transferência host/device | Inicialização de parâmetros e entrada de features; cópia explícita e dtype/shape declarados | Impl. em listas; integrada nos layers; unit | Impl. em buffers; integrada; testes de backend | Impl. em buffers; integrada em Linear/Embedding; fake API e gates F1 GPU opt-in |
| Linear forward/backward/step | Três heads independentes; vetores de logits e parâmetros float64 | Impl./integr.; unit | Impl. residente/integr.; caminho multitarefa GPU opt-in | Impl./integr. em Linear; validado nos gates Linear e na camada Linear, não em AutoregressiveModel |
| Embedding forward/backward/step | Lookup, acumulação por IDs e atualização da tabela; pesos/gradientes float64, IDs int32 | Impl./integr.; unit | Impl./integr.; testes aplicáveis | Impl./integr. em Embedding; gates F1a–F1d opt-in na RX 550; teste integrado limitado à classe Embedding |
| Fusão de HybridRepresentation.forward | Concatenação por linha na ordem caractere → features → sílaba | Impl. em listas; camada integrada; unit | Impl. residente; camada integrada; cobertura CUDA existente | Indisp.; camada falha explicitamente ao receber buffers quando falta a capacidade |
| Mean pooling forward | Reduz linhas por coluna a um vetor da dimensão da representação | Impl. em listas; integrado; unit com soma/cancelamento | Impl. residente; integrado no modelo | Indisp.; impede o próximo passo do forward residente |
| Forward das heads | MultiTaskHeads chama três camadas Linear: char, syllable e boundary | Impl./integr.; unit | Impl./integr.; resident path disponível | Linear implementado, mas a entrada pooled não é produzida pelo OpenCL atual; fluxo integrado indisponível |
| Cross-entropy host/lista | Softmax, CE e gradiente usados por APIs públicas e pelo caminho legado de treino/eval | Impl./integr.; unit | Métodos públicos de lista falham explicitamente | Métodos públicos de lista falham explicitamente |
| Cross-entropy residente | MultiTaskLoss pede CE escalar e gradiente por logits, com target inteiro de vocabulário | Caminho equivalente em listas; sem buffer-resident CPU | Impl./integr.; escala do gradiente e cópia de scalar/status pequenos | Indisp.; capacidades residentes ausentes |
| BCE-with-logits host/lista | Loss e gradiente do logit de fronteira | Impl./integr.; unit, incluindo extremos | Métodos públicos de lista falham explicitamente | Métodos públicos de lista falham explicitamente |
| BCE-with-logits residente | MultiTaskLoss pede loss escalar e gradiente para boundary, cujo shape é (1,) | Caminho equivalente em listas; sem buffer-resident CPU | Impl./integr.; scalar/status pequenos ao host | Indisp.; capacidades residentes ausentes |
| Composição MultiTaskLoss | Valida as três heads, calcula losses e aplica pesos char/syllable/boundary | Caminho de listas integrado; unit e gradiente por diferenças finitas | Caminho residente integrado por capacidades opcionais; fake/model tests e CUDA opt-in | A camada tem despacho por capacidade, mas OpenCL falha explicitamente por ausência das operações |
| Argmax | AutoregressiveModel.predict e multitask_eval_step; empate escolhe primeiro índice | Impl. em listas; integrado | Impl. residente; exporta índice pequeno; integrado | Indisp.; impede predict e avaliação residente |
| Soma dos gradientes das heads | multitask_train_step agrega char → syllable → boundary em cada elemento | Soma de listas no modelo | Impl. residente; integração CUDA | Indisp. |
| Mean-pool backward | Repete gradiente do vetor pooled por linha e divide pelo contexto | Operação de listas no modelo | Impl. residente | Indisp. |
| HybridRepresentation.backward residente | Divide gradiente de matriz em gradientes char/features/syllable; chama backward das duas Embeddings | Split em listas; integrado | Split residente; integrado | Indisp.; Embedding.backward existe, mas falta a operação de split |
| Neuron e ativações | API Backend geral; não é consumida pelas três heads multitarefa, que são Linear | Impl./unit | Métodos públicos correspondentes indisponíveis | Métodos indisponíveis |
| AutoregressiveModel e workflows | forward, predict, eval, train_step, multitask_train_step; trainer/evaluator | Impl./integr.; cobertura software CPU | Caminho multitarefa residente com teste CUDA opt-in | Não validado e não funcional hoje; faltam capacidades acima e ExecutionConfig exclui OpenCL |

### Limites de evidência

Os testes residentes de MultiTaskLoss e HybridRepresentation usam backends/buffers fake. Eles verificam o despacho e os contratos da camada, não executam kernels OpenCL. Os gates OpenCL existentes exercitam Linear e Embedding, não representação, pooling ou losses. Os testes GPU CUDA multitarefa são opt-in. Não foi executada uma suíte nesta auditoria.

Referências de cobertura: [test_hybrid.py](https://github.com/Brazwed/Kinera/blob/36f7fb9e5433e295b8556328db4e89ab669245af/tests/test_hybrid.py#L121-L309), [test_multitask_loss.py](https://github.com/Brazwed/Kinera/blob/36f7fb9e5433e295b8556328db4e89ab669245af/tests/test_multitask_loss.py#L12-L170), [testes residentes fake de MultiTaskLoss](https://github.com/Brazwed/Kinera/blob/36f7fb9e5433e295b8556328db4e89ab669245af/tests/test_multitask_loss.py#L310-L441), [gate CUDA multitarefa opt-in](https://github.com/Brazwed/Kinera/blob/36f7fb9e5433e295b8556328db4e89ab669245af/tests/test_cuda_multitask_training.py#L29-L50) e [gate F1d OpenCL da classe Embedding](https://github.com/Brazwed/Kinera/blob/36f7fb9e5433e295b8556328db4e89ab669245af/tests/test_opencl_embedding_integration.py#L1-L40).

## Caminhos e contratos de dados

### HybridRepresentation.forward

HybridRepresentation recebe listas paralelas de IDs de caractere, IDs de sílaba e matrizes de features. As três entradas precisam ter o mesmo número de linhas; cada linha de features tem dimensão feature_dim e valores numéricos finitos. Embedding produz as matrizes de lookup. As features são enviadas explicitamente ao backend em float64; o backend residente deve receber três matrizes de rank 2 com layout row-major e igual tamanho de contexto.

A saída tem shape (context_size, char_dim + feature_dim + syllable_dim), dtype float64 e ordem de colunas caractere → features → sílaba. CPU concatena linhas sem operação aritmética. O caminho da camada libera o upload temporário de features e os buffers de saída intermediários das duas embeddings, retendo a matriz fusionada como estado da representação.

**F2.1 deve validar NaN, +Inf e -Inf em todas as matrizes residentes de entrada antes de publicar o resultado.** A validação precisa ocorrer no dispositivo ou por protocolo equivalente, trazendo ao host somente status pequeno; não baixar as matrizes completas. A validação de features na camada cobre a lista host original, mas não substitui a checagem dos buffers residentes: chamadas diretas ao backend podem contornar essa camada e entradas/estado residente podem conter valores não finitos. A fusão não deve retornar uma saída quando o preflight falhar.

O caso de contexto vazio é [] no contrato host da camada; não se deve criar buffer de dimensão zero. AutoregressiveModel rejeita contexto vazio. A forma de erro para chamada residente direta deve ser explícita e coberta, preservando os contratos existentes.

### Falhas e substituição de estado em F2.1

O método atual valida os tipos dos contêineres, comprimentos paralelos e shape/tipo/finitude das features antes de liberar o output anterior. Erros detectados nessas verificações ocorrem antes da substituição e preservam o estado anterior. A validade dos elementos dos IDs é verificada por cada Embedding.forward depois que HybridRepresentation já liberou seu output anterior e iniciou a substituição; um ID inválido, portanto, pertence ao segundo grupo de falhas.

Depois dessa validação, o forward libera o output anterior, limpa metadados de contexto e inicia novos forwards de Embedding. Uma falha posterior de alocação, build, preflight, enqueue ou limpeza já ocorre durante a substituição. O caminho existente descarta os estados de forward parciais das embeddings; não se deve prometer restaurar o output antigo. O critério de segurança é não devolver saída parcial, liberar temporários adquiridos, deixar a camada em estado consistente e reutilizável após falha, e preservar a exceção primária quando a limpeza também falhar. Os testes devem verificar separadamente as falhas anteriores à substituição e as ocorridas durante a substituição. Nenhuma garantia de rollback de operações assíncronas pós-submissão é presumida.

### Pooling e semântica numérica

Mean pooling recebe uma matriz float64 de shape (context_size, representation_dim) e devolve vetor (representation_dim,). A referência CPU soma cada coluna com sum(...) na ordem das linhas e divide por context_size. Casos de cancelamento são necessários para comparar a semântica efetiva do Python da versão sob teste. Não inferir comportamento numérico apenas pela aparência algébrica.

Mean-pool backward recebe o vetor de gradientes (representation_dim,) e o tamanho do contexto, e produz matriz (context_size, representation_dim) em que cada linha recebe o gradiente dividido por context_size. Requer contexto positivo. Sua implementação e seus testes não dependem da implementação de HybridRepresentation.backward.

### Heads e perdas

MultiTaskHeads aplica três camadas Linear a uma mesma representação residente. As saídas têm vetores de logits float64 com tamanhos dos vocabulários de caractere e sílaba e um vetor boundary de shape (1,).

MultiTaskLoss exige exatamente as chaves char, syllable e boundary; os valores devem ser todos listas ou todos BackendBuffer, sem mistura. Para buffers residentes, exige buffer aberto, dtype float64, vetor não vazio e mesmo device/device_id; boundary deve ter shape (1,). Targets char/syllable são inteiros no intervalo dos respectivos vocabulários. Target boundary é numérico, finito e entre 0 e 1. A validação genérica não consegue inspecionar os valores dentro dos buffers e não compara identidade de manager/contexto; o backend da operação deve validar propriedade e conteúdo antes de usar os dados.

Na rota host, a CPU calcula softmax com subtração do máximo; cross-entropy usa log-sum-exp; gradiente CE é softmax menos one-hot. BCE com logits usa forma numericamente estável e seu gradiente usa ramos de sigmoid para sinais positivo e negativo. Na rota residente, CUDA exporta losses escalares e status pequenos, mas mantém vetores de logits e gradientes no dispositivo. OpenCL deve preservar precisão float64 e tolerância atol=rtol=1e-12; não substituir essas fórmulas por aproximações sem validação.

Pesos padrão da MultiTaskLoss: char 1.0, syllable 0.7, boundary 0.3; gradientes residentes recebem a escala da respectiva tarefa. A loss total é agregada como escalares no host. Isso não autoriza transferir logits completos.

### Backward das heads e representação

Cada Linear.backward devolve gradientes em relação à entrada compartilhada. O modelo agrega os três vetores na ordem char → syllable → boundary e divide por context_size para propagar através do mean pooling. O gradiente matricial resultante é então dividido nas fatias char/features/syllable. Os dois gradientes de embedding alimentam F1b; o gradiente de features é devolvido ao chamador.

**HybridRepresentation.backward e mean_pool_backward são capacidades independentes para implementação e validação isoladas.** A primeira pode ser testada com uma matriz de gradiente sintética (context_size, output_dim) após um forward válido; a segunda pode ser testada com um vetor pooled sintético e context_size. A cadeia de treinamento do modelo exige ambas, com sum_head_input_gradients entre as heads e mean_pool_backward antes de HybridRepresentation.backward.

O split de gradientes é cópia sem redução; igualdade exata é o critério natural para a divisão de colunas. A acumulação subsequente de Embedding preserva F1b: IDs repetidos acumulam em ordem determinística e o CPUBackend continua sendo oracle. A implementação atual chama backward da embedding de caractere e depois da embedding de sílaba; se a segunda falhar depois de a primeira já modificar gradientes, não há rollback entre elas demonstrado. O gate deve separar falhas de validação/pré-dispatch de falhas posteriores à submissão e não reivindicar atomicidade sem estratégia e evidência próprias.

## Buffers, propriedade e lifecycle

BackendBuffer expõe device, device_id, shape, dtype, nbytes, layout, owns_memory e closed. Não expõe manager/contexto proprietário. release_backend_buffers fecha apenas buffers owned, abertos, e evita repetir o close para a mesma identidade durante uma chamada. As operações OpenCL diretas devem ainda validar que cada buffer pertence ao OpenCLMemoryManager/contexto do backend que executa o kernel.

Para F2.1, o output pertence ao mesmo manager/contexto OpenCL e fica sob ownership da camada HybridRepresentation até release/substituição/close. O chamador não pode liberar inputs emprestados; os buffers intermediários de embedding e o upload explícito de features devem ser liberados depois do consumo. Para losses, os logits permanecem residentes; outputs de gradiente pertencem à loss/model path até consumo/cleanup. Scalars e status pequenos podem cruzar para o host onde o contrato exige.

Não existe fallback silencioso de dados residentes para CPU. get_backend("opencl") registra/constrói backend, mas isso não significa que um consumidor é compatível. O contrato Backend não lista todas as capacidades residentes opcionais; MultiTaskLoss e AutoregressiveModel as descobrem por métodos específicos e falham explicitamente quando faltam. Prefere-se continuar com capacidades explícitas e validação local por operação; esta auditoria não recomenda um registro global de capacidades.

multitask_eval_step limpa forward state em finally. multitask_train_step limpa esse estado automaticamente apenas quando backend.name == "cuda"; a limpeza de temporários é separada. Isso pode manter buffers OpenCL vivos até próximo forward ou close, mas o código auditado não demonstra vazamento persistente. A política deve ser revista antes de qualquer integração de treino OpenCL. ExecutionConfig permanece restrito a cpu/cuda e não deve incluir OpenCL com base apenas em capacidades parciais.

## Dependências de #59

1. **Guarda de Embedding.backward:** F1b já acrescentou um caminho OpenCL específico que exige OpenCLBackend/OpenCLBuffer concretos e o mesmo manager, preservando o caminho CUDA. Nenhuma generalização adicional é necessária para iniciar F2.1.
2. **Lifecycle de multitask_train_step:** a condição baseada em backend.name continua limitando a limpeza automática ao CUDA. Registrar como dependência de integração futura, sem alterar nesta auditoria.
3. **ExecutionConfig e runners:** get_backend conhece OpenCL, mas ExecutionConfig aceita apenas cpu/cuda. Essa separação permanece intencional até completar e validar as operações exigidas pelo workflow alvo.
4. **Propriedade de buffers:** BackendBuffer não formaliza identidade do manager; use validação específica do backend em operações residentes. Não mudar o protocolo geral preventivamente.

Referências da arquitetura #59: [Issue #59](https://github.com/Brazwed/Kinera/issues/59). Nenhuma correção da #59 foi implementada ou proposta como pré-requisito de F2.1 além da validação local necessária.

## Plano dos gates F2.1–F2.9

### Critérios transversais de GO/NO-GO

Cada gate deve ter fixtures determinísticas CPU-only com fake OpenCL API e um gate opt-in real na RX 550/Mesa Rusticl com suporte FP64, quando a operação exigir dispositivo. Usar CPUBackend como oracle, float64 e atol=rtol=1e-12 para cálculos; operações de cópia/split devem preservar valores exatamente. D2H explícito só nos checkpoints de comparação ou para scalars/status/índices previstos no contrato. Sem benchmarks ou treinamento nesta fase.

**GO** exige equivalência dentro do critério do gate, validação de shape/dtype/closed/device/manager/layout e limites de dispatch, cleanup em falhas pré-submissão, ausência de fallback silencioso, e teste em hardware quando aplicável. **NO-GO** inclui qualquer divergência fora da tolerância, NaN/Inf aceitos quando proibidos, mutação antes de preflight válido, recurso órfão demonstrado ou estado parcial publicado.

Falhas anteriores ao enqueue devem preservar estado de entrada e parâmetros. Depois da submissão de operação in-place, estado potencialmente mutado e falta de rollback permanecem explicitamente documentados; não prometer retry seguro.

| Gate | Operação e dependências | Testes CPU-only | Gate RX 550 e aprovação objetiva |
|---|---|---|---|
| **F2.1 — HybridRepresentation.forward residente** | Concatena embeddings char, features e embeddings syllable; depende de F1a e upload explícito das features | Ordem de colunas; shapes e dimensões; entrada vazia sem buffer zero; input NaN/±Inf em cada buffer residente; falha do preflight sem output; propriedade/contexto incorreto; falhas de alloc/build/enqueue/release; validar que erro pré-substituição preserva estado, enquanto IDs inválidos e falhas durante a substituição limpam o estado parcial sem prometer restaurar o anterior | Validar as mesmas entradas em Rusticl FP64. Concatenar exatamente, erro máximo 0.0; status pequeno pode ser lido no host, sem D2H integral; saída no manager correto e intermediários liberados |
| **F2.2 — mean_pool_forward** | Reduz matriz contextual; depende do output F2.1 no fluxo, mas também admite teste direto | Contextos 1 e maiores; positivos/negativos/zeros; redução com cancelamento; shapes inválidos, dimensões vazias e falhas; ordem CPU observada | Comparar vetor inteiro com CPU a 1e-12; nenhum download da matriz; documentar eventuais diferenças de arredondamento |
| **F2.3a — cross-entropy residente forward** | Recebe logits 1D float64 e target inteiro válido; retorna scalar | Classes unitárias e múltiplas; logits extremos, iguais e cancelamento; target bool/fora do vocabulário; NaN/±Inf; falhas/status/cleanup | Comparar loss ao CPU a 1e-12; somente scalar e status pequenos ao host |
| **F2.3b — cross-entropy residente gradient** | Mesmo logits/target; retorna vetor de gradiente float64, com escala de tarefa | Comparar vetor completo; pesos zero/positivos; alvos nos extremos; validação de finitude e escala; falhas sem buffer órfão | Comparar cada componente ao CPU a 1e-12; logits permanecem residentes |
| **F2.4a — BCE-with-logits residente forward** | Vetor de logit shape (1,); target em [0,1]; retorna scalar | Logits grandes positivos/negativos e próximos de zero; targets 0, 1 e intermediários; valores não finitos; shape/target inválidos e falhas | Comparar loss estável ao CPU a 1e-12; somente scalar/status pequenos ao host |
| **F2.4b — BCE-with-logits residente gradient** | Mesmo logit/target; vetor de gradiente shape (1,), ponderado | Comparar valor e escala; extremos, targets limites, escala zero; falhas/cleanup | Comparar ao CPU a 1e-12; sem D2H do logit |
| **F2.5 — composição residente MultiTaskLoss** | Requer F2.3a/b e F2.4a/b | Pesos padrão/customizados; losses individuais/total; três gradientes; buffers mistos, targets inválidos, contexts/owners diferentes; falha numa operação após alocações anteriores e liberação dos resultados parciais | Comparar scalar losses e todos gradientes com CPU a 1e-12; logits completos sempre residentes |
| **F2.6 — argmax residente** | Necessário para predict e multitask_eval_step, não para calcular loss | Vetor unitário; máximos únicos e empates (primeiro índice); vazio; NaN/±Inf; índice inválido; cleanup | Igualdade exata com CPU; transferir somente índice pequeno e status se necessário |
| **F2.7 — sum_head_input_gradients** | Agrega gradientes char → syllable → boundary de shape idêntico; necessário ao backward multitarefa | Valores assinados, zeros, cancelamento e buffers com owner/device inválidos; ordem declarada e falhas | Comparar vetor inteiro com referência CPU a 1e-12; nenhum D2H dos gradientes |
| **F2.8 — mean_pool_backward** | Recebe pooled gradient 1D e contexto positivo; produz matriz replicada/dividida | Contexto 1 e múltiplo; valores assinados/extremos; dtype/shape/contexto inválidos; falhas de alocação/enqueue/release | Comparar matriz completa ao CPU a 1e-12; pode ser desenvolvido/testado sem HybridRepresentation.backward |
| **F2.9 — HybridRepresentation.backward residente** | Split de uma matriz sintética ou gradiente real (context, output_dim); depende de F1b para acumulação das embeddings, **não depende de F2.8 para implementação/teste isolado** | Verificar fatias exatas char/features/syllable, IDs/contexto/cache, manager, dtype/shape; IDs repetidos via F1b; chamadas consecutivas; falhas antes e depois de submissão; distinguir acumulação parcial sem rollback | Split exato; comparar gradientes das duas tabelas e feature gradients ao CPU a 1e-12; testar no hardware com Rusticl FP64 |

F2.7, F2.8 e F2.9 formam dependências do caminho de treinamento residente, mas F2.8 e F2.9 podem ser implementados e aprovados isoladamente. A integração entre eles só pode ser alegada quando um gate posterior exercitar a cadeia completa.

## F0 — reconciliação proposta

| Item F0 da Issue #61 | Evidência versionada neste documento | Status proposto após revisão |
|---|---|---|
| Mapear Backend, OpenCLBackend, Embedding, HybridRepresentation, MultiTaskLoss, AutoregressiveModel, trainer/evaluator/runners e métodos consumidos | Matriz, contratos e caminhos deste relatório | Satisfeito como auditoria F0 |
| Matriz operação × consumidor × CPU/CUDA/OpenCL, incluindo obrigatório/opcional e estado funcional | Matriz acima separa implementação, integração, validação e indisponibilidade | Satisfeito como auditoria F0 |
| Mapear buffers, shape/dtype/device, ownership e release, relacionado à #59 | Seções de contratos, lifecycle e dependências #59; limitação de identidade de manager em BackendBuffer registrada | Satisfeito com limitação documentada; não requer alteração global de protocolo para F0 |
| Definir gates e riscos/testes CPU-only antes da implementação | Plano F2.1–F2.9, testes RX 550 e critérios GO/NO-GO | Satisfeito como planejamento; não autoriza implementação |

Os quatro checkboxes F0 permanecem desmarcados na Issue #61 até que a revisão documental seja aceita e a reconciliação seja aplicada separadamente. Nenhuma capacidade F2 é marcada como implementada ou validada.

## Fluxos além do escopo da auditoria

- **Treinamento completo:** requer integrar todas as capacidades de forward e backward, validar estado/lifecycle e comparar parâmetros após atualização. O ciclo da classe Embedding não substitui essa evidência.
- **Trainer, evaluator e runners:** continuam fora do gate F2 funcional. PR #70 acrescentou cadência opcional de validação por época ao runner; ele usa os mesmos consumers e não habilita OpenCL. ExecutionConfig permanece cpu/cuda.
- **API geral Backend:** Neuron, ativações, softmax público, CE/BCE de lista e outras operações fora da cadeia multitarefa podem exigir gates próprios antes de alegar suporte amplo ao protocolo.
- **#59:** permanece investigação arquitetural independente; esta auditoria registra acoplamentos relevantes, sem refatoração.
- Scheduler CPU+GPU, Rust, otimização de desempenho e benchmarks não fazem parte dos gates listados.

## Referências de código

As referências que seguem estão fixadas no baseline funcional auditado 36f7fb9e5433e295b8556328db4e89ab669245af, salvo as referências do delta de runner indicadas como main atual.

- [BackendBuffer, Backend e release_backend_buffers](https://github.com/Brazwed/Kinera/blob/36f7fb9e5433e295b8556328db4e89ab669245af/src/kinera/backends/base.py#L16-L175)
- [Descoberta lazy de backends](https://github.com/Brazwed/Kinera/blob/36f7fb9e5433e295b8556328db4e89ab669245af/src/kinera/backends/__init__.py#L14-L40)
- [CPUBackend: fusão, pooling, argmax e losses](https://github.com/Brazwed/Kinera/blob/36f7fb9e5433e295b8556328db4e89ab669245af/src/kinera/backends/cpu/backend.py#L139-L225)
- [Aritmética CPU de softmax, CE e BCE](https://github.com/Brazwed/Kinera/blob/36f7fb9e5433e295b8556328db4e89ab669245af/src/kinera/backends/cpu/ops/probability.py#L6-L38)
- [CUDABackend: fusão, pooling, losses residentes, redução e backward](https://github.com/Brazwed/Kinera/blob/36f7fb9e5433e295b8556328db4e89ab669245af/src/kinera/backends/cuda/backend.py#L266-L765)
- [OpenCL: chamadas indisponíveis, allowlist de kernels e erro fail-closed](https://github.com/Brazwed/Kinera/blob/36f7fb9e5433e295b8556328db4e89ab669245af/src/kinera/backends/opencl/backend.py#L728-L883)
- [HybridRepresentation.forward, backward, validações e estado](https://github.com/Brazwed/Kinera/blob/36f7fb9e5433e295b8556328db4e89ab669245af/src/kinera/nn/layers/hybrid.py#L50-L280)
- [MultiTaskLoss: losses, gradientes e validações resident](https://github.com/Brazwed/Kinera/blob/36f7fb9e5433e295b8556328db4e89ab669245af/src/kinera/nn/losses/multitask.py#L48-L272)
- [MultiTaskHeads: três Linear e ordem de backward](https://github.com/Brazwed/Kinera/blob/36f7fb9e5433e295b8556328db4e89ab669245af/src/kinera/nn/heads/multitask.py#L52-L98)
- [Modelo: forward/eval, pooling, estado e multitask_train_step](https://github.com/Brazwed/Kinera/blob/36f7fb9e5433e295b8556328db4e89ab669245af/src/kinera/nn/models/hybrid.py#L92-L292)
- [Modelo: treino, agregação residente e argmax](https://github.com/Brazwed/Kinera/blob/36f7fb9e5433e295b8556328db4e89ab669245af/src/kinera/nn/models/hybrid.py#L367-L515)
- [ExecutionConfig restrito a CPU/CUDA](https://github.com/Brazwed/Kinera/blob/36f7fb9e5433e295b8556328db4e89ab669245af/src/kinera/experiments/config.py#L75-L97)
- [Runner comparison e construção do backend — main atual](https://github.com/Brazwed/Kinera/blob/a3c0dd2da678ecab29852dd8ccaefc8a4be10398/src/kinera/experiments/comparison.py#L362-L366)
- [Trainer com observer de época — main atual](https://github.com/Brazwed/Kinera/blob/a3c0dd2da678ecab29852dd8ccaefc8a4be10398/src/kinera/training/trainer.py#L171-L254)

## Estado desta publicação

Documento preparado contra a main a3c0dd2da678ecab29852dd8ccaefc8a4be10398, mantendo 36f7fb9e5433e295b8556328db4e89ab669245af como baseline funcional da auditoria. É exclusivamente documental. Não altera kernels, código, testes, dados experimentais ou checklist F0/F2.

# Kinera Issue #52 — Avaliação científica final P3

**Escopo:** auditoria e interpretação dos resultados P3, P3b, P3c e P3e já publicados. Nenhum benchmark foi repetido e nenhum resultado histórico foi alterado.

**Revisão feita em:** `main` no commit `9a79fef140d8923f6bf8338af2ef930d465dc645` (merge do PR #58).

## Resumo executivo

Os dados sustentam manter o OpenCL como backend **opcional e de escopo explicitamente limitado** na combinação testada: AMD Radeon RX 550, Mesa Rusticl, suporte FP64 ativo e operações `Linear` verificadas. CPU↔OpenCL teve erro máximo observado `0.0` com `float64` e `atol=rtol=1e-12`; os testes de contrato/lifecycle e o profiling de eventos apoiam essa decisão.

Nos ciclos end-to-end maiores, OpenCL foi cerca de `1.80×` mais rápido para `3932×26` e `1.67–1.68×` para `26×3932`, comparado ao `CPUBackend` atual, que usa listas e loops Python. Isso não compara contra CPU vetorizada/nativa e não demonstra vantagem geral. `1×1` favoreceu CPU; `51×26` ficou próximo da paridade end-to-end. O `forward` residente de `26×3932` teve dispersão alta.

A auditoria de memória fecha a **interpretação** dos registros existentes, mas não transforma snapshots globais em medida de VRAM do processo. O RSS do processo foi medido como high-water; seu delta não é consumo incremental isolado da operação. A RX 550 reportou 4 GiB de memória global e limite de 1 GiB por alocação, mas o maior buffer de pesos efetivamente usado foi 817.856 bytes. Não houve teste de saturação nem medida de pico VRAM atribuível ao Kinera.

**Recomendação:** alternativa **B — manter com restrições explícitas**. Tratar o backend como opcional para os gates `Linear` verificados nesta combinação de dispositivo/runtime/FP64. Não anunciar suporte geral ao modelo, Embedding, treinamento integrado ou workloads próximos de 4 GiB. Não há evidência impeditiva que justifique descartar o backend na RX 550; tampouco há evidência para prometer suporte mais amplo.

## Proveniência e integridade

Os quatro conjuntos de resultados são coortes diferentes. Seus dados não foram agregados em uma só população.

| Etapa | Base/código de coleta registrados | Protocolo relevante | Resultado estruturado |
| --- | --- | --- | --- |
| P3 | Base Kinera `781f02386868a2e7d2001c001e6727433d88fb1d`; fingerprint do benchmark/fontes `a08a863613b13222dc591929d749ca29e90721a7c9c7d0c2390f5ccb9060446f` | Quatro shapes, `float64`, 3 warmups e 15 repetições pareadas; tempos host | `opencl-rx550-p3-benchmark.json`, SHA-256 `6fa50759b3146d9455ccf05cb26ea29a683438f3d7df94636dc46d41b6079da0` |
| P3b | Mesma base e mesmo benchmark P3; harness de decomposição revisado | Quatro processos independentes para estabilidade; análise de custos instrumentada | Original P3b SHA-256 `2057cdec6a561e3dee3c733778e079c2b708143229e6c8f4cdb0d8af587e53fd`; diagnóstico corrigido `c61235b04424dd5f0fe51a91062c1dffda6b644e2d562c1b37056079e7614255` |
| P3c | Base `434745790f984f718788bcb22b0a945df50b71a5` (PR #54); harness final SHA-256 `bf11fef67ee39b1f8432bef90bfc4619a990b5fab9d3871ca0d14ae479fe2856` | 32 processos independentes; 4 processos por backend/shape, 2 warmups e 8 repetições; conversões e RSS | `opencl-rx550-p3c-host-conversions.json`, SHA-256 `d8d0b5932037320a27f585ebd6c74d7cbfd22ab81c67ac42ee8abbeb93e4adc8` |
| P3e | Baseline de coleta `1759bd196dee268e4a9e0e80f6820e0a37db36ed`; fingerprint de fontes medido `a50ee80ae65123504e607ccb0fc615c9fd76a38a6a40bdc86fb036972bc4b868` | 3 processos sequenciais; por operação, 5 warmups e 30 repetições; fila com profiling de eventos | `opencl-rx550-p3e-device-profiling.json`, SHA-256 `3abcf58ab4fa4cf9e468665f623789d18b367e1d3a1c654ab4a9c6941a9af5ff` |

P3b preservou os resultados originais P3 e P3b. Sua correção passou a subtrair spans filhos dos inclusivos antes de compor resíduos; as identidades são verificadas amostra a amostra. O relatório de custos mantém uma tabela OpenCL histórica explicitamente marcada como supersedida; a tabela corrigida e o diagnóstico são os usados aqui.

Os hashes atuais dos JSONs P3, P3b original, P3b diagnóstico, P3c e P3e correspondem aos valores registrados nos testes/relatórios de integridade. P3c também fixa os hashes históricos de P3/P3b. A impressão digital de `memory.py` no relatório P3c é a do código **na coleta**; sua diferença para o arquivo atual é esperada porque a correção de lifecycle foi integrada depois pelo PR #57, sem recoletar P3c. O PR #57 corrigiu a liberação em falhas de staging host, issue #56, e não alterou as amostras P3c.

P3e contou diretamente no JSON **2.160 eventos de kernel** e **720 eventos de transferência** (360 H2D e 360 D2H). Todos tiveram timestamps disponíveis, ordem válida e uma liberação bem-sucedida. A serialização compacta do JSON P3e preservou os valores JSON, conforme registrado no relatório. Não encontrei indicação de alteração dos dados históricos.

Os testes de integridade P3b/P3c/P3e verificam estatísticas, componentes, roundtrips, lifecycle de eventos e fingerprints com mocks/biblioteca padrão. Eles foram inspecionados, mas não executados nesta auditoria; tampouco foram repetidas as regressões de hardware ou as coletas, respeitando o escopo somente de leitura.

## Comparação científica de desempenho

P3 e P3b usam fixtures determinísticas idênticas entre CPU e OpenCL, `float64`, learning rate `0.03125` e `atol=rtol=1e-12`. O erro absoluto máximo registrado foi `0.0` nos quatro shapes. A referência CPU é o backend Python puro atual.

| Shape | P3 CPU end-to-end (ms) | P3 OpenCL end-to-end (ms) | Razão pareada CPU/OpenCL | Leitura |
| --- | ---: | ---: | ---: | --- |
| `1×1` | 0.198 | 1.276 | 0.14× | CPU vence claramente |
| `51×26` | 5.384 | 5.108 | 1.05× | Próximo da paridade |
| `3932×26` | 399.208 | 222.567 | 1.80× | OpenCL favorecido nesta fixture |
| `26×3932` | 364.532 | 219.254 | 1.67× | OpenCL favorecido end-to-end; dispersão residente |

P3b repetiu os ciclos end-to-end em quatro processos independentes: `1.80×` em `3932×26` e `1.68×` em `26×3932`, próximos ao P3. Essa concordância contextualiza a estabilidade end-to-end observada; não torna os processos P3 e P3b uma só população. P3b observou alta variação entre processos em operações residentes, sobretudo `forward` `26×3932` (range relativo OpenCL de `47.1%` na tabela corrigida), apesar de o ciclo end-to-end ser mais estável.

P3e distingue execução de comando no dispositivo de tempo host. No `END−START`, as medianas foram `312.80 µs` para forward e `4,423.36 µs` para backward em `3932×26`; em `26×3932`, forward foi `4,579.12 µs` e backward `334.72 µs`. O IQR do forward `26×3932` foi aproximadamente `1.427 ms`, e as medianas dos três processos variaram de `2.685` a `4.582 ms`. O evento mede intervalo de execução do comando OpenCL no relógio do dispositivo; não mede ciclos de hardware nem ocupação de ALUs.

Não se deve somar evento do dispositivo a `clFinish`, launch ou tempo público como se fossem parcelas com a mesma origem de relógio. P3e registra as parcelas host separadamente e compara fila instrumentada com fila normal, mas os deltas de instrumentação têm dispersão e não formam um fator fixo de correção. P3/P3b continuam sendo as fontes para end-to-end não instrumentado; P3e esclarece a execução do kernel, sem substituir esses tempos. P3c é um diagnóstico de conversões, com probes isolados que não são aditivos exatos.

Em P3c, para os shapes grandes, o caminho público host envolve validação/flatten combinados de `90.989–100.264 ms`, packing `ctypes` de cerca de `23 ms`, H2D bloqueante abaixo de `0.40 ms`, materialização de lista de cerca de `18.1 ms` e reshape de `0.313–2.448 ms`. Esses tempos identificam passagens caras nesta implementação; não provam ganhos realizáveis por remover validação/cópias.

Portanto, há utilidade observada para ciclos de `Linear` maiores frente a **este** CPUBackend Python. Não se conclui que OpenCL vença CPU nativa/vetorizada, todas as shapes, outras RX 550, outros drivers ou aplicações completas. O aviso `Patched Mesa libclc not detected` foi registrado em Rusticl; a equivalência observada não prova correção para inputs/kernels não testados.

## Auditoria de RAM e VRAM

### RAM do processo

P3c consultou `resource.getrusage(RUSAGE_SELF).ru_maxrss` antes e depois do trabalho em cada processo. No Linux, os valores reportados estão em KiB. `ru_maxrss` é o máximo de RSS atingido pelo processo até aquele instante, e não uma amostra de memória corrente nem um contador da operação atual. O baseline já inclui imports, criação dos dados e, no processo OpenCL, runtime/contexto. O delta reportado é somente a elevação do high-water acima desse baseline; se uma fase anterior já atingiu valor maior, uma operação pode ter delta zero sem consumir zero memória.

| Shape | CPU: pico / delta high-water (KiB) | OpenCL: pico / delta high-water (KiB) |
| --- | ---: | ---: |
| `1×1` | 24.462 / 272 | 146.650 / 0 |
| `51×26` | 24.752 / 496 | 146.762 / 0 |
| `3932×26` | 45.478 / 13.588 | 162.930 / 9.186 |
| `26×3932` | 42.238 / 12.390 | 164.168 / 13.124 |

O RSS de processo medido no worker OpenCL é substancialmente maior que no worker CPU, inclusive antes da fase medida. Isso é relevante como footprint host observado deste processo e runtime, mas não permite atribuir toda a diferença ao contexto OpenCL, separar mapeamentos do driver ou converter esse valor em VRAM. As elevações de high-water nos shapes grandes são comparáveis em ordem de grandeza nos dois workers; não são estimativas isoladas do pico temporário de uma operação. O flatten de listas e o array `ctypes` criam representações host intermediárias; a lista materializada e o reshape também alocam host. Seus bytes simultaneamente vivos não foram medidos por tipo de objeto, e o overhead de objetos/allocator não foi estimado numericamente.

### VRAM e buffers

| Aspecto | Classificação | Evidência e interpretação |
| --- | --- | --- |
| Memória global `4 GiB` e limite de alocação individual `1 GiB` | Medido diretamente como capability reportada pelo runtime | Valores consultados no device RX 550/Rusticl. São limites anunciados para este runtime, não teste de capacidade útil de um workload de 4 GiB. |
| Maior buffer individual de pesos | Medido no protocolo; tamanho também derivável | `3932×26×8 = 817.856` bytes. Nenhuma alocação individual aproximou o limite de 1 GiB. |
| H2D/D2H por ciclo | Medido como bytes transferidos | P3 registra, por exemplo, `880.976 / 1.730.288` bytes em `3932×26` e `849.728 / 1.667.792` em `26×3932`. Tráfego transferido não é memória simultaneamente residente. |
| Payload de buffers OpenCL explícitos simultâneos num ciclo `Linear` | Derivado do shape e lifetime do harness/código; não medido por contador de VRAM | Os buffers vivos durante backward incluem parâmetros, input, output, gradiente de saída e três gradientes. Para `3932×26`, esse payload é `1.761.952` bytes. Na passagem para o ciclo seguinte há breve sobreposição do output anterior/novo e dos gradientes anteriores, chegando a `1.793.200` bytes de payload explícito segundo o fluxo residente P3. Para `26×3932`, o pico calculado desse conjunto explícito é `1.699.456` bytes. Esses totais excluem overhead do contexto, driver, allocator e quaisquer clientes externos; não são consumo físico de VRAM medido. |
| Snapshots de uso global da VRAM | Medidos diretamente, sem atribuição ao processo | P3e observou cerca de `858–882 MiB` antes e `894–930 MiB` depois de workers. `/sys/.../mem_info_vram_used` é global ao dispositivo; o delta pode incluir outros clientes, caches ou alterações do driver. |
| Pico de VRAM exclusivo do processo / commitment físico de cada buffer | Não demonstrado | Não existe contador por processo nos artefatos P3e. `clCreateBuffer` mede a chamada API, não quando/quanto o driver compromete fisicamente. |
| Saturação, fragmentação, limite útil agregado de 4 GiB e pico em treinamento completo | Não demonstrado | Nenhum teste aproximou o limite individual ou a capacidade global; nenhum teste de saturação foi executado. |

O ciclo foi pequeno frente à capacidade reportada, sem sinal de pressão de memória nos casos medidos. Isso basta para dizer que **os shapes de uma única camada Linear testados não se aproximaram dos limites**. Não basta para afirmar capacidade de 4 GiB para um modelo, somar capacidades de várias camadas como VRAM disponível, nem garantir recuperação física de VRAM após `close()`.

### Lifecycle e liberação

Os buffers OpenCL têm tamanho/shape/dtype/device e ownership registrados; o backend usa alocações explícitas e `close`/`release`, e os gates P2 verificaram estado fechado, ownership, substituição de estado e liberação. O PR #57 acrescentou testes determinísticos para falhas de staging host e de escrita, incluindo propagação de erro e liberação controlada. P3e registrou uma liberação bem-sucedida para cada evento. Isso é evidência de chamadas e lifecycle lógico em sucesso/erro nos casos cobertos; não é medição do decaimento de `mem_info_vram_used` nem prova de ausência de qualquer cache interno do driver.

## Alcance funcional e arquitetura

O P0 detectou a GPU `0000:03:00.0` via Mesa Rusticl; `cl_khr_fp64` e `CL_DEVICE_DOUBLE_FP_CONFIG=63` estavam disponíveis. O caminho de descoberta OpenCL é opcional/lazy e o caminho CPU-only não depende do runtime OpenCL. A compatibilidade observada é específica a Fedora/Mesa/Rusticl/driver desta coleta, com aviso `Patched Mesa libclc not detected`.

P2 validou `Linear.forward`, `Linear.backward` (`grad_weights`, `grad_bias`, `grad_inputs`), `Linear.step` e ciclos integrados pela classe `Linear`, inclusive buffers residentes e atualizações repetidas. P3/P3e ampliaram as medições para as duas orientações de shape grande. Isso não é treinamento completo do modelo Kinera.

O backend OpenCL atual não implementa `Embedding`, `Neuron`, ativações ou perdas; os caminhos fora do gate falham explicitamente como indisponíveis. Nenhum teste integrado de representação/loss/backpropagation de um modelo completo foi feito. Issue #59 (acoplamentos e seleção multibackend) continua separada: não impede a conclusão sobre o gate isolado de `Linear`, mas deve ser tratada antes de alegar integração arquitetural geral.

## Matriz de decisão

| Alternativa | Evidências favoráveis | Riscos/limitações | Parecer |
| --- | --- | --- | --- |
| **A — manter opcional** | Equivalência exata registrada; lifecycle testado; caminho CPU-only sem dependência OpenCL; vantagem end-to-end observada em dois shapes maiores. | O escopo atual já é estreito; aviso libclc; nenhum modelo completo; não comparado com CPU nativa. Sem explicitar limites, “suporte OpenCL” pode ser interpretado amplamente demais. | Viável se a descrição do suporte deixar o escopo cristalino. |
| **B — manter com restrições adicionais** | Preserva a utilidade observada e corresponde ao que foi efetivamente exercitado; operações não cobertas já falham explicitamente. Os dados de memória são suficientes para o `Linear` medido, mas não para alegar capacidade geral. | Restringe as alegações ao RX 550/Rusticl/FP64 e ao conjunto de operações/shapes testados; outros ambientes precisam de gates próprios. | **Recomendado.** Manter backend opcional e documentar o suporte como experimental/limitado ao gate `Linear`; sem promessa de treino completo ou capacidade VRAM ampla. |
| **C — descartar neste hardware** | Poderia ser considerado se houvesse incompatibilidade impeditiva ou ausência de utilidade. | Não há falha impeditiva demonstrada; Rusticl entregou timestamps válidos, gates numéricos passaram, e há vantagem de ciclo end-to-end em shapes maiores contra CPU Python. | Não sustentado pelas evidências atuais. |

### Condições para manter e ampliar

Para manter o suporte atual: preservar detecção/capability FP64, falhas explícitas sem fallback silencioso, dependência OpenCL opcional, tolerâncias numéricas congeladas e lifecycle observado. Descrever compatibilidade como verificada na combinação RX 550/Rusticl/Mesa coletada; manter o alerta de `libclc`.

Para ampliar: exigir gates separados para cada operação e caminho completo; validar arquitetura multibackend conforme Issue #59; comparar CPU vetorizada apenas em pesquisa futura autorizada; e controlar a concorrência para repetir `forward` `26×3932`. Uma alegação de capacidade VRAM do processo precisaria de método de contabilização por processo confiável neste driver. Não é necessária nova medição para a recomendação limitada acima. Se futuramente se quiser alegar uso próximo de 4 GiB, primeiro definir um experimento independente de pressão de memória e critérios; nenhum teste de saturação é autorizado por esta auditoria.

## Checklist da Issue #52 e proposta de atualização

Estado observado da Issue em `2026-11-06`: aberta. Não houve alteração na Issue.

| Grupo | Avaliação |
| --- | --- |
| P0 | Critérios de dispositivo, FP64, execução mínima e GO estão documentados e marcados completos. |
| P1 | Backend opcional/lazy, erros explícitos e contratos básicos estão documentados e completos. |
| P2 `Linear` | Forward/backward/step e fixture integrada constam como completos. A auditoria confirma o limite de escopo: não equivale a modelo/treinamento completo. |
| P2 Embedding/loss/representação | Permanecem incompletos e devem continuar desmarcados. São expansão funcional futura, não resultado demonstrado. O terceiro item de preservação de protocolo só pode ser concluído quando esses caminhos forem exercitados. |
| P3 — satisfeito | Casos em que cada backend vence; documentação Fedora/instalação; equivalência `float64` nos quatro shapes; dados por shape e custo end-to-end/residente; análise de conversões P3c; profiling P3e e lifecycle de eventos. Os respectivos itens de evidência detalhada já estão marcados `[x]` na Issue. |
| P3 — satisfeito com limitações | Protocolo e desempenho: SHA, seeds, shapes, warmups, repetições e ordem estão registrados; CPU é Python puro de uma thread e affinity/condições foram observadas. Clocks e ocupação externa não foram fixados/excluídos. RAM/VRAM: RSS high-water foi medido e snapshots VRAM são globais; os limites do runtime e payload dos buffers são reportados, mas pico exclusivo e saturação não foram medidos. Os dois critérios amplos de método/memória seguem `[ ]` no texto atual até revisão de sua redação. |
| P3 — não demonstrado | Pico VRAM atribuível ao processo, compromisso físico por buffer, fragmentação/saturação da capacidade de 4 GiB e desempenho contra CPU nativa/vetorizada. Não são resultados dos experimentos existentes. |
| Decisão/fechamento | A recomendação B está fundamentada neste relatório local; o critério de decisão e fechamento permanece `[ ]` na Issue até publicação/aceite. A Issue não foi atualizada. |

Proposta objetiva, para aplicar somente após revisão:

1. Substituir o critério de comparação controlada por: “Registrar SHA, fixtures, dtype, threads/configuração de execução, warmups, repetições, ordem e condições observáveis; declarar clocks/concorrência não controlados e limitar as conclusões à máquina medida.” Então marcá-lo concluído com a ressalva mantida.
2. Substituir o critério de memória por: “Documentar RSS/high-water, snapshots globais de VRAM, capabilities reportadas e payload dos buffers testados; declarar que pico exclusivo por processo e saturação de 4 GiB não foram validados.” Então marcar como concluída a **interpretação**, não uma medição de VRAM isolada.
3. Após aceitar este relatório, marcar como concluídos os itens de auditoria de memória e recomendação P3. Manter explícito que a recomendação B não valida consumo de VRAM de treinamento/modelo completo.
4. Manter desmarcados Embedding, losses/representação e protocolo de treino completo. Ao encerrar o escopo de viabilidade `Linear`, reclassificá-los no corpo como trabalho futuro ou movê-los para issues próprias, sem apresentá-los como implementados.
5. Não alterar a Issue #59 nem misturar scheduler/Issue #13 ao fechamento P3.

**Possibilidade de encerramento:** não fechar a Issue #52 em seu texto atual, porque há itens funcionais futuros explicitamente desmarcados, dois critérios P3 precisam de redação que reflita os controles realmente aplicados e a decisão ainda não foi publicada. A pesquisa de viabilidade do backend para `Linear` pode ser encerrada após a revisão deste relatório, ajuste da checklist para separar observações condicionais/limites do escopo futuro e aceite da recomendação. A #59 não precisa bloquear esse encerramento restrito; ela continua necessária para alegações de integração multibackend mais ampla.

## Suficiência e medições adicionais

Para decidir a continuidade do backend no escopo atual — `Linear` na RX 550/Rusticl com FP64 — a evidência é suficiente para uma decisão limitada: correção numérica, funcionamento/lifecycle e custo end-to-end estão medidos, e os dados de memória não mostram pressão para os shapes exercitados. A principal conclusão de VRAM é negativa mas útil: **não há medição atribuível ao processo**, e os snapshots globais não podem preenchê-la.

Não há experimento adicional indispensável para recomendar manter o backend opcional e restrito. Atribuição de pico VRAM e saturação só se tornam necessárias para prometer capacidade de um modelo maior/treinamento completo ou comportamento perto do limite da placa. Esse seria um novo experimento, fora desta etapa, a ser delimitado e autorizado separadamente.

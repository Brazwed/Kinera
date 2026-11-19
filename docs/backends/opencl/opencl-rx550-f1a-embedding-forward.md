# Kinera OpenCL F1a — `Embedding.forward` na RX 550

Data da validação: 2026-11-06.

## Resultado

O gate F1a passou na AMD Radeon RX 550 usando Mesa Rusticl e FP64. O OpenCL
implementa somente `Embedding.forward`; `Embedding.backward` e `Embedding.step`
continuam indisponíveis. A camada `Embedding` e os contratos CPU/CUDA não foram
alterados.

O kernel copia cada linha indicada por um ID, mantendo a ordem e as repetições.
Antes de alocar a saída ou enfileirar o lookup, um kernel de preflight percorre
os IDs residentes e escreve em um escalar `int32` a posição do primeiro ID
inválido, ou `-1` se todos forem válidos. O host lê apenas esse escalar. O
lookup só é submetido depois de o preflight terminar com sucesso; a saída
permanece em um `OpenCLBuffer` pertencente ao mesmo `OpenCLMemoryManager`.

O caso `ids=[]` valida os pesos e devolve `[]` sem alocação de dimensão zero e
sem compilar ou executar kernels. Pela camada `Embedding`, o comportamento
existente de recriar os gradientes e substituir o estado do forward é mantido.

## Proveniência e ambiente

- Baseline solicitado/auditado: `373c9a56e3a3ffb698d7f820f6a170e17304573c`.
- Durante o trabalho, `main` avançou com o PR #62 para
  `c72502db84f7e638dc58bbf2626f57e61636fbaa`; a branch F1a foi rebaseada sobre
  esse commit antes da validação final.
- Commit remoto publicado e referência reproduzível do código testado: `b11c804aeec78dcd857d00adf8966b430f7d02c1`.
- SHA local usado durante a validação antes da consolidação/publicação: `bcd134f0425a4b8187677b6ea2045b314416f3b7` (não publicado no branch remoto).
- Sistema: Fedora Linux 45 Workstation.
- Python: 3.15.0.
- Dispositivo: AMD Radeon RX 550 / 550 Series, PCI `0000:03:00.0`.
- Runtime: Mesa Rusticl, OpenCL 3.1, OpenCL C 1.2, driver `26.2.6`.
- FP64: disponível; gate executado com `RUSTICL_ENABLE=radeonsi` e
  `RUSTICL_FEATURES=fp64`.

Rusticl imprimiu `Patched Mesa libclc not detected`. O gate passou, mas o aviso
do runtime permanece uma limitação de confiança do compilador e deve continuar
visível nos registros de execução.

## Validação no hardware

Comando usado:

```bash
RUSTICL_ENABLE=radeonsi RUSTICL_FEATURES=fp64 KINERA_RUN_OPENCL_TESTS=1 \
  PYTHONPATH=src:tests python -m unittest -v \
  test_opencl_forward test_opencl_backward test_opencl_step \
  test_opencl_embedding_forward
```

Resultado: **14 testes passaram na RX 550**, incluindo 4 casos F1a e os gates
existentes de `Linear.forward`, `Linear.backward` e `Linear.step`. Os casos F1a
cobriram:

- pesos e seeds idênticos entre CPU e OpenCL, ordem original e IDs repetidos;
- vocabulário unitário e IDs vazios;
- IDs residentes `-1`, `vocab_size`, `INT32_MIN` e `INT32_MAX`, rejeitados antes
  do lookup e sem retornar saída;
- ownership do output, buffers de entrada ainda abertos e fechamento repetido;
- comparação integral CPU ↔ OpenCL dos valores copiados.

O maior erro absoluto observado na comparação F1a foi **`0.0`**. Lookup é uma
cópia de valores `float64`, por isso a comparação exigiu igualdade exata. A
materialização usada para comparar ocorreu por uma chamada explícita a
`buffer_to_host()` depois da operação.

## Testes CPU-only e falhas

O fake OpenCL API cobre IDs válidos/repetidos, IDs inválidos residentes,
validação de shape/dtype/device/owner/lifecycle e os caminhos de falha de
alocação, build, criação/configuração do kernel, enqueue, leitura do status e
release. Também verifica que o erro primário é preservado, que recursos
temporários são tentados para liberação uma vez por caminho e que buffers
caller-owned não são fechados pelo backend. A camada mantém seus buffers de
estado anteriores quando um forward falha.

Os testes CPU de `Embedding`, os testes de storage com bridge CUDA simulada, o
contrato de backend e o teste de importação CPU-only passaram. Os testes CUDA
que exigem NVIDIA foram pulados por não serem aplicáveis neste host.

O comando da suíte configurado no GitHub Actions, com os cinco casos
experimentais/custosos já excluídos pelo workflow, terminou com **696 passed,
42 skipped, 5 deselected**. Houve um aviso de compatibilidade do plugin `anyio`
com pytest 9.2.0, sem falha. Um teste P3c de fingerprint inicialmente detectou
que o código OpenCL atual mudou; sua expectativa foi atualizada para o fingerprint
da implementação F1a, preservando o fingerprint P3c como referência histórica.
O relatório e os resultados P3c não foram modificados.

## Empacotamento

O wheel `kinera-0.1.0-py3-none-any.whl` foi construído e verificado. Inclui
`kinera/backends/opencl/kernels/embedding_forward.cl`; a instalação em diretório
temporário permitiu importar `CPUBackend` sem carregar módulos OpenCL e ler o
kernel usando `importlib.resources`. O wheel tem SHA-256
`a379da9fd2b8e154864c6f9f738fa17a434225ee5ba0444a619af657f8be23f5`.
`setuptools>=68` foi usado somente numa pasta temporária de build; nenhuma
dependência foi adicionada ao projeto.

## Limites

- Somente `Embedding.forward` está incluído. Backward, step e treinamento
  integrado não foram implementados nem validados.
- A validação de IDs residentes usa uma passagem serial de preflight e uma
  leitura bloqueante de 4 bytes. Nenhuma medição de desempenho foi feita.
- O hardware e o runtime testados foram uma única RX 550 com Rusticl e FP64.
- As falhas simuladas cobrem erros síncronos de API. Não há evidência de rollback
  seguro se uma falha assíncrona for reportada depois da submissão do kernel; o
  gate não promete essa garantia.
- O aviso `Patched Mesa libclc not detected` permanece aberto como ressalva do
  ambiente.

## Fingerprints

SHA-256 dos arquivos centrais da implementação testada:

| Arquivo | SHA-256 |
| --- | --- |
| `src/kinera/backends/opencl/backend.py` | `e20423c0386701f9a7e43b9dbb912df85a352b015c5c5597ed2b4e8e6ce34227` |
| `src/kinera/backends/opencl/kernels/embedding_forward.cl` | `41e6b1512268e924b63ba11d1d51ad79a4ffba653cc439fd621cdb8fa54f14e4` |
| `tests/test_opencl_embedding_forward.py` | `024d0ccf7d6fb786e394ad1b0772d4f6ad891e3a58980e8d5bba531243ca5ebe` |

O fingerprint P3c da implementação anterior do backend OpenCL permanece
registrado no teste histórico; os dados P3c e seus relatórios não foram
regenerados nem ajustados.

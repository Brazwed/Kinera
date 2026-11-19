# Kinera — R2: checkpoint híbrido CPU e retomada por época

**ID documental:** KIN-DOC-0035  
**Tipo:** SPEC / GUIDE / AUDIT  
**Issue:** [#92 — R2](https://github.com/Brazwed/Kinera/issues/92)  
**Escopo:** `AutoregressiveModel` **CPU**, persistência integral de parâmetros para inferência e retomada determinística **somente em limite de época**.  
**Situação:** R2.1–R2.4 concluídas e PRs #93–#95 integradas; a conclusão administrativa de R2.5 será registrada na Issue #92 após CI e merge da documentação.

Este documento descreve os contratos implementados e suas evidências, **não** estabelece suporte genérico a checkpoint CUDA/OpenCL, retomada no meio de uma época ou runners científicos.

## 1. Proveniência e evidências

| Etapa | Código / evidência | Resultado |
| --- | --- | --- |
| R0 | [Issue #86](https://github.com/Brazwed/Kinera/issues/86), [relatório R0](r0-cpu-core-audit.md) | Histórico `PARTIAL`, não reclassificado retroativamente |
| R1 | [Issue #90](https://github.com/Brazwed/Kinera/issues/90), [PR #91](https://github.com/Brazwed/Kinera/pull/91) | `PASS` regressão CPU, gate independente |
| R2.1 | [Auditoria na Issue #92](https://github.com/Brazwed/Kinera/issues/92#issuecomment-7009753615) | `GO` somente para CPU |
| R2.2 | [PR #93](https://github.com/Brazwed/Kinera/pull/93), merge `b8f9c8fb7a02f1a033f7af9132fe2f958b03e280` | Round-trip do modelo; [CI #44529929856](https://github.com/Brazwed/Kinera/actions/runs/44529929856): 803 passed, 62 skipped, 5 deselected, 1107 subtests por versão |
| R2.3 | [PR #94](https://github.com/Brazwed/Kinera/pull/94), merge `a742044bfafa54ef8840abe04f3406218298d5bb` | Validação fail-closed e compatibilidade; [CI #44534224348](https://github.com/Brazwed/Kinera/actions/runs/44534224348): 813 passed, 62 skipped, 5 deselected, 1147 subtests por versão |
| R2.4 | [PR #95](https://github.com/Brazwed/Kinera/pull/95), merge `d400f8b2b15d0c5bf1b69c699cb1c69a38cadf14` | Comparações contínuo vs. retomado exatas nas condições declaradas |
| R2.5 | [CI pós-merge da #95 #44538301297](https://github.com/Brazwed/Kinera/actions/runs/44538301297) | `main` no SHA `d400f8b2b15d0c5bf1b69c699cb1c69a38cadf14`: Python **3.10 e 3.14**, **822 passed, 62 skipped, 5 deselected, 1159 subtests passed** em cada versão, zero falhas |

A matriz da CI usa `ubuntu-latest`, Python 3.10/3.14, `pip install -e .[test]` e `python -m pytest -q tests` com **cinco** exclusões explícitas do piloto ampliado (ver [workflow](../../.github/workflows/ci.yml)). Essa suíte executa também os arquivos direcionados `tests/test_checkpoint.py`, `tests/test_hybrid_checkpoint.py`, `tests/test_hybrid_checkpoint_integrity.py`, `tests/test_hybrid_resume.py`, `tests/test_autoregressive_model.py` e `tests/test_multitask_training.py`. **Não foi executado comando local separado na R2.5**; as CIs indicadas são as execuções auditáveis.

Comando direcionado para reprodução local (documentado, **não registrado como nova execução**):

~~~bash
python -m pip install -e '.[test]'
python -m pytest -q \
  tests/test_checkpoint.py \
  tests/test_hybrid_checkpoint.py \
  tests/test_hybrid_checkpoint_integrity.py \
  tests/test_hybrid_resume.py \
  tests/test_autoregressive_model.py \
  tests/test_multitask_training.py \
  tests/test_cli_cpu_determinism.py
~~~

## 2. Formatos, API e compatibilidade

Os dois envelopes são **independentes**, sem conversão/migração/downgrade automática.

| Formato / versão | API | Conteúdo |
| --- | --- | --- |
| `kinera.checkpoint` v1 | `save_neuron/load_neuron`, `save_linear/load_linear`, `save_checkpoint/load_checkpoint` | `Neuron` e `Linear` legados, inalterados no R2 |
| `kinera.hybrid.checkpoint` v1 | `save_autoregressive_checkpoint/load_autoregressive_checkpoint` | `AutoregressiveModel` completo para inferência CPU; `training` opcional |
| Bloco `training.schema_version` = 1 | `train_hybrid_epochs`, `save_hybrid_training_checkpoint`, `load_hybrid_training_checkpoint` | Retomada CPU após época completa; `training: null` significa inferência apenas |

**Envelope JSON híbrido:** campos `format`, `version`, `model`, `identity`, `training` e `metadata`. Em `model.architecture`, são obrigatórios `char_vocab_size`, `syllable_vocab_size`, `char_embedding_dim`, `syllable_embedding_dim` e `feature_dim`. `model.parameters` guarda as **duas embeddings** (`char_embedding.weights`, `syllable_embedding.weights`) e as **três heads** (`char`, `syllable`, `boundary`), com `weights` e `bias` para cada head, totalizando **oito arrays treináveis**.

`output_dim` é derivado da soma das dimensões de embedding/características; `output_layer` é alias de `heads.char`, e mean pooling não tem pesos. Não são persistidos gradientes, caches, ativações, buffers de backend, objetos executáveis ou estado transitório de forward/backward. O formato é **JSON**, não `pickle`.

`identity` inclui os dois vocabulários versionados (ordem dos tokens/IDs) e digests `sha256`, verificados contra **vocabulários esperados** passados no carregamento. Só a igualdade de tamanhos dos vocabulários não autoriza a restauração. `metadata` é auxiliar; não substitui identidade, pesos nem estado de treino.

**Estado de retomada:** quando `training` não é `null`, contém:

| Campo | Contrato |
| --- | --- |
| `schema_version` | inteiro igual a 1 |
| `mode` | `"char"` ou `"multitask"` |
| `completed_epochs` e `global_step` | não negativos; `global_step == completed_epochs * selected_examples` |
| `selected_examples` | contagem positiva dos exemplos efetivamente selecionados |
| `examples_sha256` | hash tipado da lista de exemplos selecionados **na ordem usada**, respeitando `max_examples` |
| `model_parameters_sha256` | digest dos oito arrays, vinculando pesos ao cursor de época |
| `configuration` | `learning_rate`, `max_examples`, `syllable_unknown_id` e `loss_weights` das três tarefas em modo multitask; `null` no modo char |
| `history` | uma entrada de métricas por época concluída; quantidade consistente com `completed_epochs` |

As identidades usam `kinera.experiments.fingerprint.canonical_sha256` (tipos e ordem preservados). O digest dos parâmetros detecta incompatibilidade acidental entre **registro de treino e pesos**, não prova autenticidade. Snapshots **só de inferência** continuam sem checksum próprio dos pesos.

**Validação e atomicidade:** o leitor híbrido rejeita JSON truncado, UTF-8 inválido, chaves JSON duplicadas, `NaN`/`Infinity`, formatos/versões/tipos desconhecidos, campos essenciais ausentes ou extras, dimensões/shapes incoerentes e identidade incompatível. O leitor constrói **nova instância CPU** somente após validar o payload; não modifica uma instância de destino do chamador. A gravação reutiliza arquivo temporário, `fsync` e `os.replace`; há testes de falha injetada que preservam os bytes anteriormente publicados. O leitor antigo continua aceitando as fixtures v1 de `Neuron`/`Linear` e rejeita o formato híbrido.

## 3. Exemplo: inferência após save/load

~~~python
from kinera.backends.cpu.backend import CPUBackend
from kinera.language.vocabulary import CharacterVocabulary, SyllableVocabulary
from kinera.nn.models.hybrid import AutoregressiveModel
from kinera.training.hybrid_checkpoint import (
    save_autoregressive_checkpoint,
    load_autoregressive_checkpoint,
)

chars = CharacterVocabulary(["<START>", "<END>", "<UNK>", "a", "b"])
sylls = SyllableVocabulary(["<UNK_SYL>", "ka"])
model = AutoregressiveModel(
    char_vocab_size=len(chars), syllable_vocab_size=len(sylls),
    feature_dim=10, seed=7, backend=CPUBackend(),
)
save_autoregressive_checkpoint(
    "modelo.json", model,
    character_vocabulary=chars, syllable_vocabulary=sylls,
    metadata={"purpose": "inference"},
)
restored, info = load_autoregressive_checkpoint(
    "modelo.json",
    expected_character_vocabulary=chars,
    expected_syllable_vocabulary=sylls,
)
heads = restored.forward_heads(
    [3, 4], [1, 0], [[0.0] * 10, [0.0] * 10]
)
assert set(heads) == {"char", "syllable", "boundary"}
assert info["training"] is None
~~~

A aplicação deve preservar/reconstruir **os mesmos mapeamentos** de vocabulário. `training=None` não é checkpoint retomável.

## 4. Exemplo: treino interrompido e retomado na fronteira de época

Os wrappers de R2.4 reutilizam `train_examples` e `train_multitask_examples` sem alterar matemática ou sequência SGD. `epochs` em `train_hybrid_epochs` significa **épocas adicionais**, não total desejado.

~~~python
from kinera.backends.cpu.backend import CPUBackend
from kinera.language.vocabulary import CharacterVocabulary, SyllableVocabulary
from kinera.nn.models.hybrid import AutoregressiveModel
from kinera.training.hybrid_resume import (
    train_hybrid_epochs,
    save_hybrid_training_checkpoint,
    load_hybrid_training_checkpoint,
)

chars = CharacterVocabulary(["<START>", "<END>", "<UNK>", "a", "b"])
sylls = SyllableVocabulary(["<UNK_SYL>", "ka"])
examples = [{
    "context_ids": [0, 3],
    "context_syllable_ids": [0, 1],
    "context_feature_vectors": [[0.0] * 10, [1.0] + [0.0] * 9],
    "target_char_id": 4,
    "target_syllable_id": 1,
    "target_is_syllable_end": 1,
}]
model = AutoregressiveModel(
    char_vocab_size=len(chars), syllable_vocab_size=len(sylls),
    feature_dim=10, seed=7, backend=CPUBackend(),
)
settings = dict(
    examples=examples, character_vocabulary=chars,
    syllable_vocabulary=sylls, mode="multitask", learning_rate=0.04,
)
_, state = train_hybrid_epochs(model, epochs=2, **settings)
save_hybrid_training_checkpoint("treino.json", model, state, **settings)

# Em outro processo, reconstruir os mesmos exemplos, vocabulários e configuração.
restored, state, metadata = load_hybrid_training_checkpoint("treino.json", **settings)
report, state = train_hybrid_epochs(
    restored, epochs=2, previous_state=state, **settings,
)
assert report["epochs"] == 4
assert state["global_step"] == 4 * len(examples)
~~~

Para pesos de loss customizados, reconstruir uma `MultiTaskLoss` compatível associada ao backend CPU do modelo restaurado e passá-la a todos os passos. Em `mode="char"`, `loss_function` deve ser `None`. O wrapper não introduz CLI, observers ou hooks de checkpoint nos runners científicos.

## 5. Evidências dos critérios de aceite

| Critério #92 | Evidência | Conclusão |
| --- | --- | --- |
| Modelo completo, ambas embeddings e três heads | `tests/test_hybrid_checkpoint.py` | PASS; round-trip de pesos realmente atualizados e logits iguais |
| Integridade, schemas, identidade | `tests/test_hybrid_checkpoint_integrity.py` e `tests/test_hybrid_resume.py` | PASS nos casos negativos exercitados |
| Leitura de v1 `Neuron`/`Linear` | `tests/test_checkpoint.py` e fixtures v1 de integridade | PASS sem migração |
| Continuidade de treino | `tests/test_hybrid_resume.py`: multitask 4 vs. 2+2; char 3 vs. 1+2 | `assertEqual` para pesos, logits e histórico, **tolerância 0** dentro de cada Python |
| Regressão CPU e matriz Python | [CI pós-merge #44538301297](https://github.com/Brazwed/Kinera/actions/runs/44538301297) | PASS, Python 3.10/3.14; exclusions conservadas |

## 6. Limitações, decisão e próximos gates

1. **CPU apenas.** Nenhuma equivalência ou validação física CUDA/OpenCL; os 62 skips e cinco exclusões não demonstram suporte GPU.
2. **Fronteira de época.** Não existe retomada em meio-época, cursor por exemplo ou recuperação dos updates após o último snapshot publicado.
3. **Sem integração universal.** Não há CLI de resume, suporte ao estado dos callbacks `step_observer` / `epoch_end_observer`, `timing_session`, replay ou runners científicos. Isso não é automaticamente coberto pelo wrapper.
4. **Determinismo delimitado.** Dados/configuração fixos, ordem fixa, SGD simples, sem RNG por época. Igualdade exata foi demonstrada **em cada ambiente Python separadamente**, não entre versões, CPUs ou backends.
5. **Integridade residual.** Hashes não são assinaturas autenticadas; `history` tem validação estrutural/contagem, não reexecução para autenticidade científica. Inferência-only não tem digest independente dos pesos.
6. **Não testado:** carga de grandes datasets, outros otimizadores com estado ou algoritmos novos; não projetar essa cobertura a partir dos testes mínimos.

**Parecer técnico recomendado:** **PASS do Gate R2 no escopo explícito de checkpoint do híbrido CPU e retomada após época inteira**, com limitações acima. Esse resultado não reclassifica o histórico R0 `PARTIAL` nem antecipa R3/R4. A decisão administrativa de R2.5, seus SHAs e a confirmação de CI da PR documental devem ser registrados na [Issue #92](https://github.com/Brazwed/Kinera/issues/92) após a integração.

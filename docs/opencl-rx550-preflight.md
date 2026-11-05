# Preflight OpenCL — AMD Radeon RX 550

## Decisão

**GO para o scaffold P1 e o gate inicial de `Linear.forward`.** Com os pacotes
abertos do Fedora instalados, Rusticl enumera a Radeon RX 550 pelo BDF esperado,
expõe `cl_khr_fp64` com `CL_DEVICE_DOUBLE_FP_CONFIG=0x3f`, e executa o kernel
isolado em `float64` e `int32` com erro máximo absoluto zero contra a CPU.

Esse GO abre P1 e a primeira operação isolada. O aviso do runtime sobre o
`libclc` não corrigido pela Mesa continua como risco: o smoke cobre operações
simples, não a equivalência completa de `Linear`, treino ou todos os kernels.
Mantenha as tolerâncias existentes e exija PASS dos gates de `Linear.forward`,
`Linear.backward` e `Linear.step` antes de ampliar o backend.

## Ambiente observado em 2026-11-05

| Campo | Valor |
| --- | --- |
| Sistema | Fedora Linux 45 Workstation |
| Kernel | `7.2.11-300.fc45.x86_64` |
| CPU | Intel Xeon E5-2680 v4 @ 2.40 GHz, 14 núcleos / 28 threads |
| RAM | 16 GiB instalados; aproximadamente 10 GiB disponíveis durante a coleta |
| GPU anunciada pelo OpenCL | `AMD Radeon RX 550 / 550 Series` (`radeonsi`, `polaris12`, ACO) |
| GPU PCI | `0000:03:00.0`, `1002:699f`, revision `c7` |
| Identificação PCI | AMD Lexa PRO, família Radeon 540/540X/550/550X/RX 550 |
| Driver de kernel | `amdgpu` |
| VRAM | `4294967296` bytes (4 GiB); `692867072` bytes em uso durante a coleta |
| Mesa instalada | `26.2.6-1.fc45` (`mesa-libOpenCL`, `mesa-dri-drivers`, `mesa-vulkan-drivers`, `mesa-filesystem`) |
| LLVM instalado | `23.1.4-1.fc45` (`llvm-libs`) |
| ICD OpenCL | Mesa Rusticl em `/etc/OpenCL/vendors/rusticl.icd` |
| Loader e `clinfo` | `ocl-icd` `2.3.5-1.fc45`; `clinfo` `3.1.26.09.26-1.fc45` |
| Dispositivo / OpenCL C | OpenCL 3.1 / OpenCL C 1.2; `radeonsi` driver `26.2.6` |
| FP64 | `cl_khr_fp64`; `CL_DEVICE_DOUBLE_FP_CONFIG=0x3f`; não emulado em software |
| Memória reportada pelo device | Global 4 GiB; alocação máxima 1 GiB; memória local 64 KiB |
| Limites reportados | Workgroup 1024; dimensões de work item `1024×1024×1024` |
| ICDs extras | POCL e ROCm/HIP não instalados; o teste seleciona GPU AMD, nunca CPU |
| PyOpenCL | Não instalado; não é necessário para o smoke C |
| ROCm/HIP | Não instalados; a RX 550 não aparece na matriz oficial atual de GPUs Radeon compatíveis com ROCm |

O PCI ID identifica o chip/família, mas a descrição fornecida pelo `lspci` não
separa o modelo da placa entre as variantes Radeon 540 e 550. O relatório,
portanto, registra o PCI ID e a descrição completa do sistema, sem inferir a
SKU exata só pelo nome comercial.

## Execução feita

1. `uname -a`, `/etc/os-release`, `lspci -nnk`, sysfs DRM/PCI, `/dev/dri`,
   inventário RPM e `ldconfig` confirmaram o ambiente listado acima.
2. No estado inicial, `clinfo`, `mesa-libOpenCL`, `ocl-icd` e o arquivo de ICD
   não estavam instalados. O Fedora 45 oferece esses pacotes e uma implementação
   aberta pela Mesa Rusticl. A extração exploratória dos RPMs em `/tmp` listou a
   plataforma `rusticl`, mas zero dispositivos porque a biblioteca procura
   módulos `libclc` em `/usr/lib64/clc22`, caminho criado pela instalação
   padrão. Esse resultado de extração parcial não mediu compatibilidade.
3. Depois da instalação normal dos pacotes Fedora, `clinfo --list`, com
   `RUSTICL_ENABLE=radeonsi RUSTICL_FEATURES=fp64`, enumerou exatamente um
   dispositivo GPU AMD, `AMD Radeon RX 550 / 550 Series`, OpenCL 3.1 e PCI
   `0000:03:00.0`. O device expõe `cl_khr_pci_bus_info`, e o BDF bate com o
   `lspci`. A descrição do `lspci` também informa o PCI ID `1002:699f`, Lexa
   PRO/família Radeon 540/550, revision `c7`, com `amdgpu`.
4. `clinfo` reportou FP64 IEEE com `cl_khr_fp64`, operações subnormal,
   infinito/NaN, arredondamentos ao mais próximo/zero/infinito e FMA; a aritmética
   FP64 não é emulada em software. `int32` foi verificado pelo kernel. Limites:
   4 GiB global, alocação máxima 1 GiB, memória local 64 KiB, workgroup máximo
   1024 e work item `1024×1024×1024`.
5. O runtime imprime `Patched Mesa libclc not detected`; ele próprio alerta que o
   upstream libclc pode conter bugs ou mudanças incompatíveis. A documentação
   Mesa classifica `radeonsi` como tendo suporte razoável, com possíveis bugs ou
   falhas de conformidade. Registre esse aviso junto à versão do runtime e rode
   gates por operação antes de ampliar o uso.
6. O programa isolado compilou com `cc -std=c11 -O2 -Wall -Wextra -Werror` e
   selecionou o BDF fornecido. Criou contexto e buffers, transferiu 320 bytes
   para a GPU e 192 bytes para a CPU, compilou e executou o kernel. Os 16
   resultados `float64` e `int32` coincidiram com a referência CPU; erro máximo
   `float64=0`, sob `atol=rtol=1e-12`. Resultado: `GO_P0_SMOKE_PASS`.

## Reproduzir o preflight

Em outra instalação Fedora, instale os mesmos pacotes abertos usados nesta
coleta; isso não instala ROCm, HIP nem driver proprietário:

```bash
sudo dnf install clinfo mesa-libOpenCL ocl-icd-devel opencl-headers gcc
```

O Mesa Rusticl exige ativação explícita do driver Gallium. A opção `fp64`
também é opt-in; o script pede essa capacidade para medir o contrato numérico
do Kinera. Execute:

```bash
RUSTICL_ENABLE=radeonsi RUSTICL_FEATURES=fp64 clinfo --list
RUSTICL_ENABLE=radeonsi RUSTICL_FEATURES=fp64 clinfo
cc -std=c11 -O2 -Wall -Wextra -Werror \
  experiments/opencl_rx550_preflight.c -lOpenCL \
  -o /tmp/opencl_rx550_preflight
RUSTICL_ENABLE=radeonsi RUSTICL_FEATURES=fp64 \
  /tmp/opencl_rx550_preflight --pci-bdf 0000:03:00.0
```

O programa recusa selecionar um device CPU, exige um device AMD anunciado como
RX 550 (ou o BDF esperado quando `cl_khr_pci_bus_info` estiver disponível),
registra nome, vendor, versões, extensões, `CL_DEVICE_DOUBLE_FP_CONFIG`, memória
e limites de workgroup, e exige buffers reais e execução de kernel. Se o driver
não expuser o BDF via extensão, a saída informa essa limitação e faz a
identificação pelo vendor, tipo GPU e nome do device. A comparação usa 16
valores `float64` contra referência CPU com `atol=rtol=1e-12` e verifica
`int32`, conforme a política congelada em [cuda-backend.md](cuda-backend.md).

O resultado medido nesta máquina foi `GO_P0_SMOKE_PASS`. O gate P0 libera o
scaffold P1. Cada nova operação ainda precisa passar equivalência contra CPU;
falha de enumeração, ausência de FP64, erro de execução ou divergência mantém
NO-GO para aquela etapa, sem relaxar a matemática ou as tolerâncias.

Na auditoria pré-PR, o probe foi recompilado com `gcc -std=c11 -O2 -Wall
-Wextra -Werror` e `-lOpenCL`, e executado novamente no BDF
`0000:03:00.0`; o resultado continuou `GO_P0_SMOKE_PASS`. O fingerprint final
do conjunto de fontes, kernels, testes e probe está em
[`opencl-rx550-p2-linear-integration.md`](opencl-rx550-p2-linear-integration.md).

## Referências primárias

- [AMD — Radeon RX 550](https://www.amd.com/en/support/downloads/drivers.html/graphics/radeon-600-500-400/radeon-rx-500-series/radeon-rx-550.html): especifica OpenCL 2.0 como capacidade do produto; isso não confirma um ICD/runtime funcional nesta instalação Fedora.
- [Mesa — Rusticl](https://docs.mesa3d.org/rusticl.html): implementação OpenCL Gallium e ativação explícita de devices pelo ambiente.
- [Mesa — variáveis Rusticl](https://docs.mesa3d.org/envvars.html#rusticl-environment-variables): documenta `RUSTICL_ENABLE=radeonsi`, `RUSTICL_FEATURES=fp64` e classifica `radeonsi` como suporte razoável com possíveis bugs ou falhas de conformidade.
- [AMD ROCm — requisitos e GPUs Linux suportadas](https://rocm.docs.amd.com/projects/install-on-linux/en/develop/reference/system-requirements.html): lista as GPUs oficialmente suportadas; RX 550 não aparece na lista consultada.

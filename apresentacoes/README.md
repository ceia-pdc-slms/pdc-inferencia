# Apresentações

Todo o material das reuniões de sexta, separado por **semana** e, dentro de cada semana, por **pessoa**.

## Como as pastas são nomeadas

```
apresentacoes/
└── semana-NN_AAAA-MM-DD/              uma pasta por reunião, com um README da semana
    └── <tipo>_<pessoa>_<tema>/         uma pasta por apresentação
```

| Parte | Valores | Exemplo |
|---|---|---|
| `tipo` | `principal`, `keyword`, `artigo` (discussão mensal), `atividade` (exercício da semana), `material` (guia ou apoio) | `keyword` |
| `pessoa` | primeiro nome, minúsculo, sem acento: `beatriz`, `joao`, `marcos`, `pedro`, `wevton`, `hugo` | `joao` |
| `tema` | o assunto em poucas palavras, minúsculo, sem acento, com hífens | `kernels-1-o-que-e-um-kernel` |

Dentro da pasta: os slides **em PDF** (é o que abre direto no GitHub), e se quiser o original (`.pptx`, link do Canva/Google), notebooks e código. Nomes de arquivo também sem espaço e sem acento. O nome exato de cada pasta já está no README de cada semana.

## Por semana

| # | Data | Principal | Keyword |
|---|---|---|---|
| [0](semana-00_2026-09-11/) | 11/09 | *Abertura* — [guia de SLMs da Beatriz](semana-00_2026-09-11/material_beatriz_guia-didatico-slms/guia-didatico-slms.pdf) | |
| [1](semana-01_2026-09-18/) | 18/09 | Debrief: o que deu errado ao rodar um modelo + Git na prática (oficina) · **Wevton**<br>*sem material* | Infraestrutura de inferência 1/2 — servir é diferente de treinar · **Marcos**<br>[Slides (PDF)](semana-01_2026-09-18/keyword_marcos_infraestrutura-de-inferencia-1/infraestrutura-de-inferencia-1.pdf) |
| [2](semana-02_2026-09-25/) | 25/09 | Docker na prática (oficina) · **Beatriz**<br>[Slides (PPTX)](semana-02_2026-09-25/principal_beatriz_docker-na-pratica/docker-na-pratica.pptx) | Quantização 1/3 — representação numérica · **Beatriz**<br>[Slides (PDF)](semana-02_2026-09-25/keyword_beatriz_quantizacao-1-representacao-numerica/quantizacao-representacao-numerica.pdf) · [Guia de estudo](semana-02_2026-09-25/keyword_beatriz_quantizacao-1-representacao-numerica/guia-de-estudo.md) · [Código: quantização na mão](semana-02_2026-09-25/keyword_beatriz_quantizacao-1-representacao-numerica/quantizacao.py) |
| [3](semana-03_2026-10-02/) | 02/10 | Arquitetura de computadores: CPU, GPU, TPU, NPU, LPU — e o que muda para inferência · **Pedro**<br>*sem material* | Kernels/HPC 1/2 — o que é um kernel · **João Vitor**<br>[Slides (PDF)](semana-03_2026-10-02/keyword_joao_kernels-1-o-que-e-um-kernel/kernels-o-que-e-um-kernel.pdf) · [Notebook](semana-03_2026-10-02/keyword_joao_kernels-1-o-que-e-um-kernel/kernel_hpc_01_o_que_e_um_kernel.ipynb) |
| [4](semana-04_2026-10-09/) | 09/10 | O que é um modelo de linguagem: tokens, vocabulário, embeddings · **Marcos**<br>*sem material* | Testbench 1/2 — métricas de desempenho e procedência · **Pedro**<br>*sem material* |
| [5](semana-05_2026-10-16/) | 16/10 | Transformer por dentro: atenção, pesos, funções de ativação · **João Vitor** | Compressão 1/2 — as quatro famílias · **Marcos** |
| [6](semana-06_2026-10-23/) | 23/10 | Como um modelo gera texto: prefill, decode, KV cache, sampling · **Beatriz** | Quantização 2/3 — granularidade e outliers · **Wevton** |
| [7](semana-07_2026-10-30/) | 30/10 | Memória e banda: o roofline, por que decode é lento · **João Vitor** | Infraestrutura de inferência 2/2 — KV cache e batching · **Beatriz** |
| [8](semana-08_2026-11-06/) | 06/11 | Quantização a fundo: FP16 → FP8 → INT4 → FP4, o que quebra e o que acelera · **Beatriz** | Pruning 1/2 — não estruturado · **João Vitor** |
| [9](semana-09_2026-11-13/) | 13/11 | Runtimes de inferência: Ollama, llama.cpp, vLLM, TensorRT-LLM · **Wevton** | Quantização 3/3 — formatos na prática e QAT/QLoRA · **Pedro** |
| — | 20/11 | *Feriado, sem reunião* | |
| [10](semana-10_2026-11-27/) | 27/11 | Como medir sem se enganar: TTFT, TPOT, p50/p95, tok/s por watt, variância · **Pedro** | Kernels/HPC 2/2 — FlashAttention e GEMM de baixa precisão · **João Vitor** |
| [11](semana-11_2026-12-04/) | 04/12 | Hardware pequeno na prática: Jetson, Raspberry, NPU, throttling térmico · **Pedro** | Pruning 2/2 — estruturado e sparsity 2:4 · **Wevton** |
| [12](semana-12_2026-12-11/) | 11/12 | Benchmarks: o que existe, como validar, como não mentir com número · **Marcos** | Compressão 2/2 — destilação · **Beatriz** |
| [13](semana-13_2026-12-18/) | 18/12 | Fecho: cada um apresenta o dossiê da própria máquina · **Todos** | Testbench 2/2 — qualidade sob compressão (15 min) · **Marcos** |

## Por pessoa

### Beatriz

| Semana | Tipo | Tema | Material |
|---|---|---|---|
| [0](semana-00_2026-09-11/) | material | Guia didático sobre SLMs | [Guia didático sobre SLMs (PDF)](semana-00_2026-09-11/material_beatriz_guia-didatico-slms/guia-didatico-slms.pdf) |
| [1](semana-01_2026-09-18/) | atividade | Rodar um modelo local (atividade da semana 1) | [Explicação](semana-01_2026-09-18/atividade_beatriz_rodar-modelo-local/README.md) · [Script](semana-01_2026-09-18/atividade_beatriz_rodar-modelo-local/run_slm.py) |
| [2](semana-02_2026-09-25/) | principal | Docker na prática (oficina) | [Slides (PPTX)](semana-02_2026-09-25/principal_beatriz_docker-na-pratica/docker-na-pratica.pptx) |
| [2](semana-02_2026-09-25/) | keyword | Quantização 1/3 — representação numérica | [Slides (PDF)](semana-02_2026-09-25/keyword_beatriz_quantizacao-1-representacao-numerica/quantizacao-representacao-numerica.pdf) · [Guia de estudo](semana-02_2026-09-25/keyword_beatriz_quantizacao-1-representacao-numerica/guia-de-estudo.md) · [Código: quantização na mão](semana-02_2026-09-25/keyword_beatriz_quantizacao-1-representacao-numerica/quantizacao.py) |
| [6](semana-06_2026-10-23/) | principal | Como um modelo gera texto: prefill, decode, KV cache, sampling | agendada para 23/10 |
| [7](semana-07_2026-10-30/) | keyword | Infraestrutura de inferência 2/2 — KV cache e batching | agendada para 30/10 |
| [8](semana-08_2026-11-06/) | principal | Quantização a fundo: FP16 → FP8 → INT4 → FP4, o que quebra e o que acelera | agendada para 06/11 |
| [12](semana-12_2026-12-11/) | keyword | Compressão 2/2 — destilação | agendada para 11/12 |
| [13](semana-13_2026-12-18/) | principal | Fecho: cada um apresenta o dossiê da própria máquina | agendada para 18/12 |

### João Vitor

| Semana | Tipo | Tema | Material |
|---|---|---|---|
| [3](semana-03_2026-10-02/) | keyword | Kernels/HPC 1/2 — o que é um kernel | [Slides (PDF)](semana-03_2026-10-02/keyword_joao_kernels-1-o-que-e-um-kernel/kernels-o-que-e-um-kernel.pdf) · [Notebook](semana-03_2026-10-02/keyword_joao_kernels-1-o-que-e-um-kernel/kernel_hpc_01_o_que_e_um_kernel.ipynb) |
| [5](semana-05_2026-10-16/) | principal | Transformer por dentro: atenção, pesos, funções de ativação | agendada para 16/10 |
| [7](semana-07_2026-10-30/) | principal | Memória e banda: o roofline, por que decode é lento | agendada para 30/10 |
| [8](semana-08_2026-11-06/) | keyword | Pruning 1/2 — não estruturado | agendada para 06/11 |
| [10](semana-10_2026-11-27/) | keyword | Kernels/HPC 2/2 — FlashAttention e GEMM de baixa precisão | agendada para 27/11 |
| [13](semana-13_2026-12-18/) | principal | Fecho: cada um apresenta o dossiê da própria máquina | agendada para 18/12 |

### Marcos

| Semana | Tipo | Tema | Material |
|---|---|---|---|
| [1](semana-01_2026-09-18/) | keyword | Infraestrutura de inferência 1/2 — servir é diferente de treinar | [Slides (PDF)](semana-01_2026-09-18/keyword_marcos_infraestrutura-de-inferencia-1/infraestrutura-de-inferencia-1.pdf) |
| [4](semana-04_2026-10-09/) | principal | O que é um modelo de linguagem: tokens, vocabulário, embeddings | *sem material* |
| [5](semana-05_2026-10-16/) | keyword | Compressão 1/2 — as quatro famílias | agendada para 16/10 |
| [12](semana-12_2026-12-11/) | principal | Benchmarks: o que existe, como validar, como não mentir com número | agendada para 11/12 |
| [13](semana-13_2026-12-18/) | principal | Fecho: cada um apresenta o dossiê da própria máquina | agendada para 18/12 |
| [13](semana-13_2026-12-18/) | keyword | Testbench 2/2 — qualidade sob compressão (15 min) | agendada para 18/12 |

### Pedro

| Semana | Tipo | Tema | Material |
|---|---|---|---|
| [3](semana-03_2026-10-02/) | principal | Arquitetura de computadores: CPU, GPU, TPU, NPU, LPU — e o que muda para inferência | *sem material* |
| [4](semana-04_2026-10-09/) | keyword | Testbench 1/2 — métricas de desempenho e procedência | *sem material* |
| [9](semana-09_2026-11-13/) | keyword | Quantização 3/3 — formatos na prática e QAT/QLoRA | agendada para 13/11 |
| [10](semana-10_2026-11-27/) | principal | Como medir sem se enganar: TTFT, TPOT, p50/p95, tok/s por watt, variância | agendada para 27/11 |
| [11](semana-11_2026-12-04/) | principal | Hardware pequeno na prática: Jetson, Raspberry, NPU, throttling térmico | agendada para 04/12 |
| [13](semana-13_2026-12-18/) | principal | Fecho: cada um apresenta o dossiê da própria máquina | agendada para 18/12 |

### Wevton

| Semana | Tipo | Tema | Material |
|---|---|---|---|
| [1](semana-01_2026-09-18/) | principal | Debrief: o que deu errado ao rodar um modelo + Git na prática (oficina) | *sem material* |
| [6](semana-06_2026-10-23/) | keyword | Quantização 2/3 — granularidade e outliers | agendada para 23/10 |
| [9](semana-09_2026-11-13/) | principal | Runtimes de inferência: Ollama, llama.cpp, vLLM, TensorRT-LLM | agendada para 13/11 |
| [11](semana-11_2026-12-04/) | keyword | Pruning 2/2 — estruturado e sparsity 2:4 | agendada para 04/12 |
| [13](semana-13_2026-12-18/) | principal | Fecho: cada um apresenta o dossiê da própria máquina | agendada para 18/12 |

## Sem material ainda?

Quem apresentou e ainda não subiu os slides: crie a pasta com o nome que está no README da semana e abra um PR. O passo a passo, inclusive pelo navegador, está no [guia de contribuição](../CONTRIBUTING.md#subindo-a-sua-apresentação).

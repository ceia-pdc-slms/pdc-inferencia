# PDC · Frente de Compressão, Testbench e Infraestrutura de Inferência

**Mapa do time — set a dez/2026**

> **Laboratórios e testbench:** como rodar está no [README da raiz](../README.md#laboratórios-e-testbench-como-rodar). **Material das apresentações:** [`apresentacoes/`](../apresentacoes/), por semana e por pessoa.

Esse documento é como um mapa. Ele diz o que o time faz, o que se espera do time, qual aula é para cada um, quando ela cai e onde buscar material. As fontes estão todas no fim, separadas por aula, para não atrapalhar a leitura.

**Time:** Hugo (líder) · Beatriz · Wevton · João Vitor · Marcos · Pedro
**Reunião interna:** sexta-feira, 1 hora · **Geral do PDC:** terça, 17h

## 1. Por que esse time existe

O PDC inteiro está construindo modelos de linguagem pequenos (SLMs). Nosso time responde uma pergunta específica, e todas as outras frentes dependem dela:

> Esse modelo roda no nosso hardware? Com qual qualidade, a que velocidade, a que custo computacional e energético, e outra pessoa consegue repetir essa medição?

Quando a frente do Daniel ou Lucca treinar um modelo, ela vai querer saber se ele cabe e quanto custa rodar. Quando a frente do Lucca montar um benchmark, ela vai querer rodar o modelo rápido para medir. Somos nós que entregamos isso, e somos nós que construímos as ferramentas que tornam essa resposta confiável, além de otimizar esses modelos e testá-los em essência.

Um norte do PDC, colocado pela coordenação: em julho de 2027, um modelo nosso rodando em pequenas placas e alcançando alta cadência (tokens por segundo).

## 2. A conta que explica o time inteiro

Quando um modelo escreve uma resposta palavra por palavra, o computador não está "fazendo contas" no intervalo entre uma palavra e outra. Ele está lendo o modelo inteiro da memória. A cada token. Em média, um modelo de 1 bilhão de parâmetros em 16 bits ocupa 2 GB, então ele lê 2 GB para produzir um único token.

Daí sai a fórmula que vale para tudo que a gente faz:

```
tokens por segundo ≈ banda de memória ÷ bytes lidos por token
```

É por isso que estudamos quantização, pruning, destilação, kernels e medição. Tudo o que está neste documento serve para atacar os dois lados dessa divisão: diminuir os bytes lidos por token e aproveitar melhor a banda que temos.

## 3. Escopo: seis keywords

O time trabalha com seis assuntos. Eles são os nossos, e são o que vocês vão estudar e apresentar:

`compressão` · `quantization` · `pruning` · `kernels/HPC` · `infraestrutura de inferência` · `testbench`

As keywords das outras frentes do PDC aparecem ancoradas dentro das nossas, quando tocam o nosso trabalho: destilação entra dentro de compressão, porque é um método de compressão; QLoRA entra dentro de quantização; leaderboards entram dentro de testbench. Elas não viram aula própria.

**O que não é nosso:** treinar modelo do zero. Encostamos em treino em dois pontos apenas, quantização dinâmica junto com o pessoal de pré-treino, e destilação, que envolve treinar o modelo aluno.

**A regra de escopo:** se o assunto não muda o que a gente mede ou o que a gente entrega, ele é contexto e cabe num slide. Se muda, vira aula.

## 4. Como funciona a reunião

Sexta-feira, uma hora, sempre com duas apresentações.

| Bloco | Tempo | O que é |
| :--- | :--- | :--- |
| **Apresentação principal** | 25 min | A aula da semana: base ou artigo. Ensina como se faz e o que a gente mede |
| **Apresentação de keyword** | 25 min | Uma parte de uma das seis keywords. Ensina o que é e por que existe |
| **Status** | 10 min | Como está a sua máquina e o que travou |

*Uma vez por mês vamos modificar a reunião para virar discussão de um artigo recente, publicado de janeiro de 2026 para cá. Quem for o curador manda o link no Discord na segunda-feira.*

### As oito regras

1. Toda apresentação principal tenta fechar com um número medido no nosso hardware. Nem sempre vai dar, mas a tentativa é parte do trabalho.
2. Toda apresentação responde explicitamente à pergunta da semana. O último slide é a resposta.
3. Material mínimo é um item por semana. Quem apresenta lê fundo. Todo mundo lê o mínimo e seria ideal levarem uma pergunta escrita para reunião.
4. Quem apresenta traz uma coisa do tema que não entendeu. Isso não é falha, é o formato. Ninguém aqui vai apresentar assunto que domina; todo mundo vai apresentar assunto que acabou de estudar.
5. Ninguém faz principal e keyword na mesma semana.
6. 25 minutos é teto, não meta. Mas aula de 5 minutos não serve — a intenção aqui é aprendizado de verdade.
7. O material aqui contido é um norteador, não oráculo. Dois artigos indicados não são a fronteira do assunto. Espera-se que você traga coisa que ninguém mandou.
8. Faculdade é prioridade. Avisar cedo que a semana não vai dar é combinado, não é problema. Sumir sem avisar é o único problema.

## 5. Calendário: quem apresenta o quê

O material de cada apresentação fica em [`apresentacoes/`](../apresentacoes/): uma pasta por semana e, dentro dela, uma por pessoa.

| # | Data | Apresentação principal | Dono | Keyword | Dono |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | 18/09 | Debrief: o que deu errado ao rodar um modelo + Git na prática (oficina) | Wevton | Infraestrutura de inferência 1/2 — servir é diferente de treinar | Marcos |
| **2** | 25/09 | Docker na prática (oficina) | Wevton | Quantização 1/3 — representação numérica | Beatriz |
| **3** | 02/10 | Arquitetura de computadores: CPU, GPU, TPU, NPU, LPU — e o que muda para inferência | Pedro | Kernels/HPC 1/2 — o que é um kernel | João |
| **4** | 09/10 | O que é um modelo de linguagem: tokens, vocabulário, embeddings | Marcos | Testbench 1/2 — métricas de desempenho e procedência | Pedro |
| **5** | 16/10 | Transformer por dentro: atenção, pesos, funções de ativação | João | Compressão 1/2 — as quatro famílias | Marcos |
| **6** | 23/10 | Como um modelo gera texto: prefill, decode, KV cache, sampling | Beatriz | Quantização 2/3 — granularidade e outliers | Wevton |
| **7** | 30/10 | Memória e banda: o roofline, por que decode é lento | João | Infraestrutura de inferência 2/2 — KV cache e batching | Beatriz |
| **8** | 06/11 | Quantização a fundo: FP16 → FP8 → INT4 → FP4, o que quebra e o que acelera | Beatriz | Pruning 1/2 — não estruturado | João |
| **9** | 13/11 | Runtimes de inferência: Ollama, llama.cpp, vLLM, TensorRT-LLM | Wevton | Quantização 3/3 — formatos na prática e QAT/QLoRA | Pedro |
| **—** | 20/11 | Feriado — sem reunião | | | |
| **10** | 27/11 | Como medir sem se enganar: TTFT, TPOT, p50/p95, tok/s por watt, variância | Pedro | Kernels/HPC 2/2 — FlashAttention e GEMM de baixa precisão | João |
| **11** | 04/12 | Hardware pequeno na prática: Jetson, Raspberry, NPU, throttling térmico | Pedro | Pruning 2/2 — estruturado e sparsity 2:4 | Wevton |
| **12** | 11/12 | Benchmarks: o que existe, como validar, como não mentir com número | Marcos | Compressão 2/2 — destilação | Beatriz |
| **13** | 18/12 | Fecho: cada um apresenta o dossiê da própria máquina | Todos | Testbench 2/2 — qualidade sob compressão (15 min) | Marcos |

São cinco apresentações para cada um em treze semanas. Os assuntos foram distribuídos por afinidade, não por sorteio. Precisa trocar por causa de prova? Avise com uma semana e combine com alguém — isso é procedimento normal.

Git e Docker são oficinas, não apresentações. Todo mundo com o notebook aberto acompanhando. Quem já sabe conduz, quem não sabe aprende fazendo.

## 6. O que cada apresentação precisa responder

A pergunta é combinada antes e o último slide da apresentação é a resposta. Elas foram escolhidas de propósito para não terem resposta pronta na internet: só se responde depois de mexer.

### Apresentações principais

| # | Pergunta a fechar |
| :--- | :--- |
| **1** | O que exatamente quebrou quando você tentou rodar um modelo, e por quê? Traga o erro em tela, não a lembrança dele |
| **2** | Por que "funciona na minha máquina" deixa de ser desculpa depois do Docker? Mostre o mesmo modelo rodando em dois lugares com o mesmo container |
| **3** | O que muda para uma GPU, uma CPU e uma NPU quando o gargalo é ler memória em vez de fazer conta? |
| **4** | Uma mesma frase em português custa quantos tokens a mais que em inglês? O que isso faz com o tempo de resposta que o usuário sente? |
| **5** | Onde estão os parâmetros de um modelo de 1B? Some camada por camada e compare com o tamanho do arquivo em disco |
| **6** | Por que ler uma pergunta de 500 tokens é mais rápido que escrever uma resposta de 500 tokens? |
| **7** | Quantos bytes o nosso hardware lê para gerar um token? Por que é isso, e não os FLOPs, que define o tok/s? |
| **8** | O tok/s aumentou na mesma proporção em que os bytes por token caíram? Se não, o que consumiu o ganho? |
| **9** | Mesmo modelo, mesmo formato, dois runtimes, tok/s diferente. Aponte a causa concreta, não "otimizações" |
| **10** | Medindo cinco vezes, qual foi o desvio? A partir de qual diferença dá para afirmar que um resultado é melhor que o outro? |
| **11** | Depois de 10 minutos de carga contínua, o tok/s da Raspberry caiu quanto? Por quê? |
| **12** | Esse benchmark mede o quê, exatamente? E o que ele esconde? |
| **13** | A sua máquina roda o quê, a que velocidade, com qual consumo, e o que outra pessoa precisa fazer para repetir? |

### Apresentações de keyword

| Keyword | Parte | Sem | Escopo | Pergunta a fechar |
| :--- | :--- | :--- | :--- | :--- |
| **Infraestrutura de inferência** | 1/2 | 1 | Servir é diferente de treinar. O caminho de um pedido: fila, prefill, decode, resposta. O que é um servidor de inferência e por que não é só chamar generate | Que recurso é gargalo ao treinar e irrelevante ao servir, e vice-versa? |
| | 2/2 | 7 | KV cache como estrutura de memória, PagedAttention, batching contínuo, chunked prefill, prefix caching | Por que servir 20 pedidos ao mesmo tempo custa muito menos que 20× servir um? |
| **Quantização** | 1/3 | 2 | Representação numérica: FP32, FP16, BF16, INT8, INT4, FP4. Escala e zero-point. O que exatamente se perde ao usar menos bits | Represente o número 3,7 em INT8 com escala 0,05. Que erro você cometeu, e ele é aceitável? |
| | 2/3 | 6 | Granularidade (por tensor, por canal, por grupo) e os outliers de ativação. O que GPTQ, AWQ e SmoothQuant assumem de diferente. Onde entram os dados de calibração | Por que quantizar ativação é mais difícil que quantizar peso? |
| | 3/3 | 9 | Formatos na prática: GGUF e os k-quants, artefatos AWQ/GPTQ, NVFP4 e MXFP4, bitsandbytes. Onde cada formato roda de verdade. QAT e QLoRA | O modelo ficou 4× menor. Ficou 4× mais rápido? Em qual das nossas máquinas sim, em qual não, e por quê? |
| **Compressão** | 1/2 | 5 | O guarda-chuva: as quatro famílias — quantização, pruning, destilação, compressão de KV — e o que cada uma troca por quê. Fecha com um slide de onde as outras frentes do PDC entram | Se eu preciso caber em 500 MB, qual família eu ataco primeiro e por quê? |
| | 2/2 | 12 | Destilação como método de compressão: professor e aluno, o que o aluno aprende que não está nos dados, por que é o único caminho para um modelo de 0,3B com qualidade | O que se perde numa destilação que uma quantização preservaria? |
| **Pruning** | 1/2 | 8 | Pruning não estruturado: magnitude, Wanda, SparseGPT. O que é esparsidade e como se mede | Por que zerar 50% dos pesos quase nunca dá 2× de velocidade? |
| | 2/2 | 11 | Pruning estruturado: remover cabeças, camadas e dimensões. Sparsity 2:4 e suporte em hardware. A receita Minitron, que junta pruning e destilação | Em qual das nossas três máquinas a esparsidade 2:4 daria ganho real, e em qual seria só economia de disco? |
| **Kernels/HPC** | 1/2 | 3 | O que é um kernel. Threads, blocos, hierarquia de memória. O que é fundir operações e por que fundir economiza tempo sem economizar conta | Uma operação que não reduz nenhum FLOP pode ficar 3× mais rápida. Como? |
| | 2/2 | 10 | Attention fundido: por que FlashAttention existe e o que ele evita. GEMM de baixa precisão e kernels de quantização | Por que um kernel de W4A16 precisa ser escrito à mão em vez de sair do compilador? |
| **Testbench** | 1/2 | 4 | O que é medir: métrica, unidade, variância, procedência. TTFT, TPOT, ITL, throughput, p50/p95, tok/s por watt. Diferença entre benchmark e testbench | Dois números diferentes. A partir de qual diferença dá para afirmar que um é melhor? |
| | 2/2 | 13 | Qualidade sob compressão: o que cada benchmark mede, leaderboards, contaminação, e o corpus de avaliação que a gente monta. Por que perplexidade estável esconde degradação | Um modelo quantizado manteve a perplexidade e quebrou em tool-calling. Que teste teria pego isso? |

## 7. A sua máquina

Além das aulas, cada pessoa ficará responsável por uma máquina física do time. Você vai levá-la para casa, configurar, consertar e aprender na prática o que é operar hardware pequeno.

**Meta até dezembro — o dossiê completo da sua máquina:**

1. Liga e você acessa por SSH.
2. Roda um modelo.
3. Tem um README em [`maquinas/`](../maquinas/) que outra pessoa consegue seguir sozinha.
4. Tem pelo menos uma medição de tok/s registrada, com procedência completa.

Essa é a sua entrega individual do semestre e é o que você vai apresentar na última reunião, em 18/12.

*Procedência completa quer dizer:* máquina, ferramenta, versão, modelo, formato, tamanho de contexto, e como você mediu. Nenhum número existe sem isso. Um tok/s solto não é resultado, é boato.

## 8. Papéis

Além da sua máquina, cada pessoa fica com um assunto transversal. Não é exclusividade: é quem puxa o tema e a quem se pergunta primeiro.

| Pessoa | Papel | Na prática |
| :--- | :--- | :--- |
| **Beatriz** | Receitas de compressão | Decide o que aplicar, em que ordem, e com que perda esperada |
| **Wevton** | Ambiente | Imagens, containers, acesso, reprodutibilidade. Se não roda na máquina, é com ele |
| **João Vitor** | Modelo analítico e kernels | Prevê o tok/s antes de a gente medir, e puxa o lado de baixo nível |
| **Pedro** | Bancada de medição | Como se mede, com que instrumento, com que variância |
| **Marcos** | Corpus e perfis de carga | Define em cima de que prompts e de que carga a gente mede |

*Sobre o papel do João vale um aviso honesto: kernels é o assunto que o Hugo menos domina. Essa trilha o time aprende junto, e é a que mais depende de vocês trazerem material.*

## 9. Como a gente trabalha

- **Discord:** Um canal de dúvidas, um de artigos e um de entregas. No canal de artigos, manda de onde vier: arXiv, blog da NVIDIA, X, LinkedIn, release de biblioteca. Tudo interessa; a gente separa depois o que cabe.
- **Travou? Avise no mesmo dia:** Travar é esperado e faz parte do aprendizado — a maior parte do trabalho com hardware pequeno é justamente resolver erro. Travar em silêncio por uma semana é a única coisa que atrapalha o time.
- **Repositório pdc-inferencia:** Seu primeiro PR é o README da sua máquina da primeira atividade. É assim que Git deixa de ser assunto abstrato.
- **Conversa individual mensal:** Vinte minutos com o Hugo, uma vez por mês, para falar de como está indo — inclusive das partes que não estão indo.
- **Proatividade é a expectativa central:** Isto aqui não é manda-e-obedece. O plano é um norte; se você achar um caminho melhor, traga. Se achar um artigo que muda o que a gente está fazendo, traga.

## 10. O que acontece em janeiro

Setembro a dezembro é ciclo de formação: a demanda das outras frentes chega no começo do ano. A partir de janeiro:

1. O time se divide em duas equipes, com base em quatro meses de trabalho real.
2. Começa a primeira reprodução de paper, uma por equipe. Os dois papers base — AWQ e PagedAttention — já terão sido lidos por todos, nas semanas 8 e 9.
3. Chegam as primeiras demandas do time do Daniel e do time do Lucca. É para isso que essa formação existe.
4. A meta de julho é recalculada com medição real das nossas máquinas na mão.


## Fontes por aula

*Ponto de partida para quem vai apresentar. Não é a fronteira do assunto: parte da sua preparação é achar material que não está aqui.*

### Semana 1 · Git + infraestrutura de inferência
- Documentação oficial do Git — apenas branch, commit, PR e revisão. Não precisa de mais que isso.
- Pro Git (Chacon e Straub), capítulos 2 e 3 — livre e gratuito em [git-scm.com/book/pt-br](https://git-scm.com/book/pt-br).
- Documentação de arquitetura do vLLM: a visão geral de como um servidor de inferência é organizado.
- Post do vLLM sobre *continuous batching* e o que diferencia servir de rodar `generate`.

### Semana 2 · Docker + quantização (representação numérica)
- Docker, guia "Get Started" oficial, partes 1 a 3.
- *What Every Computer Scientist Should Know About Floating-Point Arithmetic* (Goldberg, 1991) — só as seções iniciais; é a referência clássica sobre o que se perde em ponto flutuante.
- Post da NVIDIA sobre formatos de baixa precisão (FP8, FP4, NVFP4) — explica escala, zero-point e por que FP4 precisa de blocos.
- Documentação do `bitsandbytes` sobre INT8 e NF4.

### Semana 3 · Arquitetura de computadores + o que é um kernel
- Curso MIT 6.5940, *TinyML and Efficient Deep Learning Computing*, do Song Han — aulas iniciais. É gratuito no YouTube e é o material mais alinhado com o nosso time que existe. O Song Han é coautor do AWQ e do SmoothQuant.
- Série de aulas do GPU MODE (antigo CUDA MODE) no YouTube — aulas 1 a 3: modelo de memória de GPU, ocupância, profiling.
- Documentação da NVIDIA sobre hierarquia de memória em GPU.
- Material do Jetson e da Raspberry sobre as unidades de processamento disponíveis em cada um.

### Semana 4 · Modelo de linguagem + métricas de desempenho
- Andrej Karpathy, *Deep Dive into LLMs like ChatGPT* (~3h30, YouTube) — o melhor panorama gratuito que existe. Comece por ele.
- Andrej Karpathy, *Let's build the GPT Tokenizer* (~2h) e o repositório `minbpe`.
- Documentação do `benchmark_serving` do vLLM: as definições de TTFT, TPOT, ITL e *throughput*.
- MLPerf Inference — a metodologia, para ver como uma medição séria é especificada.

### Semana 5 · Transformer + compressão (panorama)
- Andrej Karpathy, *Let's build GPT: from scratch, in code, spelled out* (~2h).
- Jay Alammar, *The Illustrated Transformer* (post).
- 3Blue1Brown, série sobre transformers e atenção — visual, curta, excelente para intuição.
- *A Survey on Model Compression for Large Language Models* (TACL, 2024) — o mapa das quatro famílias.
- *Efficient Large Language Models: A Survey* (TMLR, 2024).

### Semana 6 · Geração de texto + quantização (granularidade e outliers)
- Kipply, *Transformer Inference Arithmetic* (post) — leitura obrigatória do time inteiro.
- *LLM.int8()* (arXiv:2208.07339) — por que ativações têm outliers.
- *SmoothQuant* (arXiv:2211.10438).
- *GPTQ* (arXiv:2210.17323) e *AWQ* (arXiv:2306.00978) — ler as seções de método, não o paper inteiro.

### Semana 7 · Roofline + KV cache e batching
- Horace He, *Making Deep Learning Go Brrrr From First Principles* (post). Se você for ler uma coisa só na vida sobre desempenho, é essa.
- *LLM Inference Unveiled: Survey and Roofline Model Insights* (arXiv:2402.16363).
- *Efficient Memory Management for LLM Serving with PagedAttention* (arXiv:2309.06180) — o paper do vLLM.
- *GQA* (arXiv:2305.13245) — por que o KV cache moderno é menor.

### Semana 8 · Quantização a fundo + pruning não estruturado
- *AWQ* (arXiv:2306.00978) — agora inteiro. É o paper base de uma das reproduções de janeiro.
- *The Case for 4-bit Precision: k-bit Inference Scaling Laws* (arXiv:2212.09720).
- *Wanda — A Simple and Effective Pruning Approach* (arXiv:2306.11695).
- *SparseGPT* (arXiv:2301.00774).
- *Wanda++: Pruning LLMs via Regional Gradients* (Findings of ACL, 2025) — a evolução recente do Wanda.

### Semana 9 · Runtimes + formatos e QAT/QLoRA
- Documentação de arquitetura do `llama.cpp` e o guia de formatos GGUF e k-quants.
- Documentação do `llm-compressor` — os fluxos de NVFP4 e MXFP4 para vLLM.
- *QLoRA* (arXiv:2305.14314).
- Notas de release do `llama.cpp` sobre o suporte a NVFP4, adicionado em 2026 — bom exemplo de como acompanhar um projeto vivo.
- Documentação do TensorRT-LLM e do NVIDIA ModelOpt.

### Semana 10 · Como medir + FlashAttention e GEMM de baixa precisão
- *FlashAttention* (arXiv:2205.14135) e *FlashAttention-2* (arXiv:2307.08691).
- *MARLIN* (Frantar et al., PPoPP 2025) — o kernel que faz W4A16 valer a pena.
- *QServe / W4A8KV4* (arXiv:2405.04532) — o que realmente acelera em serving contra o que só economiza memória.
- *Evaluating Quantized Large Language Models* (arXiv:2402.18158).

### Semana 11 · Hardware pequeno + pruning estruturado
- *Minitron — Compact Language Models via Pruning and Knowledge Distillation* (arXiv:2407.14679). É a receita mais próxima do nosso norte de julho.
- *Sheared LLaMA* (arXiv:2310.06694) e *LLM-Pruner* (arXiv:2305.11627).
- *PalmBench* (arXiv:2410.05315) — metodologia de benchmark de modelos comprimidos em plataforma móvel.
- *Sustainable LLM Inference for Edge AI: Evaluating Quantized LLMs for Energy Efficiency, Output Accuracy, and Inference Latency* (ACM TIoT, 2025 — arXiv:2504.03360).
- *Large Language Model Deployment on Resource-Constrained Edge Devices: A Practitioner's Survey* (IEEE CCWC, 2026).
- Documentação de `tegrastats` (Jetson) e de throttling térmico da Raspberry Pi.

### Semana 12 · Benchmarks + destilação
- Documentação do `lm-evaluation-harness` e a lista de tarefas disponíveis.
- *MiniLLM* (arXiv:2306.08543) — destilação com objetivo adequado a geração.
- *Minitron* (arXiv:2407.14679), revisitado como receita de execução.
- *Open Portuguese LLM Leaderboard* — as tarefas em português (ENEM, BLUEX, OAB, ASSIN2, entre outras).

### Semana 13 · Dossiê + qualidade sob compressão
- *Accuracy is Not All You Need* (arXiv:2407.09141) — por que a média de acurácia esconde o que a compressão quebrou.
- *Weight Pruning Amplifies Bias: A Multi-Method Study of Compressed LLMs for Edge AI* (arXiv:2605.08137, 2026).
- *Uncertainty Drives Social Bias Changes in Quantized Large Language Models* (arXiv:2602.06181, 2026).
- *Compression Scaling Laws: Unifying Sparsity and Quantization* (2025).

---

## Fontes vivas: onde achar artigo recente

*Uma vez por mês alguém traz um artigo publicado de 2026 para cá. Estes são os lugares de onde eles saem:*

- **Awesome-LLM-Compression** ([github.com/HuangOwen/Awesome-LLM-Compression](https://github.com/HuangOwen/Awesome-LLM-Compression)) — lista mantida e atualizada de papers de compressão, separada por técnica. É o melhor ponto de partida do time.
- **arXiv**, categorias `cs.LG` e `cs.CL`, com alerta para os termos *quantization, pruning, inference, efficient* e *edge*.
- **Hugging Face Papers** ([huggingface.co/papers](https://huggingface.co/papers)) — o que a comunidade está lendo hoje.
- **Blogs de engenharia:** NVIDIA Developer, vLLM, llama.cpp (notas de release), PyTorch.
- **Conferências que importam para nós:** MLSys, ICLR, NeurIPS, ACL, PPoPP.

### Alguns trabalhos recentes, como exemplo do que procurar
- *Joint Structural Pruning and Mixed-Precision Quantization for LLM Compression* (arXiv:2606.07819, jun/2026) — junta pruning estruturado e quantização de precisão mista em vez de aplicar em sequência.
- *LLM Compression with Jointly Optimizing Architectural and Quantization Choices* (arXiv:2606.04063, jun/2026).
- *EntroLLM: Entropy Encoded Weight Compression for Efficient LLM Inference on Edge Devices* (arXiv:2505.02380).

> Os três apontam para a mesma direção, e ela é exatamente o nosso norte: tratar compressão como um problema conjunto, não como uma pilha de técnicas aplicadas uma depois da outra.

## Vídeos, em ordem de prioridade

1. **Andrej Karpathy** — *Deep Dive into LLMs like ChatGPT* (~3h30). O panorama. Comece por aqui.
2. **Andrej Karpathy** — *Let's build GPT: from scratch, in code, spelled out* (~2h). Construindo um modelo do zero, linha por linha.
3. **Andrej Karpathy** — *Let's build the GPT Tokenizer* (~2h).
4. **MIT 6.5940** — *TinyML and Efficient Deep Learning Computing* (Song Han). Curso inteiro, gratuito. É literalmente a ementa do nosso time.
5. **GPU MODE** (YouTube) — aulas sobre CUDA, Triton e profiling. Para a trilha de kernels.
6. **3Blue1Brown** — série sobre redes neurais, transformers e atenção. Intuição visual, episódios curtos.
7. **Andrej Karpathy** — *Let's reproduce GPT-2 (124M)* (~4h). Para mais adiante, quando a base estiver firme.

> **Sobre o Karpathy, para ninguém repetir informação errada:** ele foi membro fundador da OpenAI e diretor de IA da Tesla, responsável pela visão computacional do piloto automático. Desde 2024 toca a própria empresa de educação, a Eureka Labs. A série dele chama *Neural Networks: Zero to Hero*.
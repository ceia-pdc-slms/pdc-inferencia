# Infraestrutura de inferência 2/2: KV cache e batching

Keyword da semana 7 (30/10/2026) · Beatriz

| Material | O que é |
|---|---|
| [Slides (PDF)](kv-cache-e-batching.pdf) | 26 slides. O último é a resposta da pergunta da semana |
| [Roteiro de fala](roteiro-de-fala.md) | O que dizer em cada slide: como ler cada gráfico, de onde vem cada número, analogias e respostas para perguntas prováveis |
| [`batching.py`](batching.py) | Mede o decode com lotes de 1 a 32 pedidos, calcula o KV cache por modelo e testa um servidor de verdade |
| [Laboratório interativo](../../../laboratorios/batching/kv-cache-e-batching.html) | Lote × tokens por segundo, quantos pedidos cabem na GPU, blocos do PagedAttention, batching estático × contínuo, chunked prefill e prefix caching |

## Pergunta da semana

> Por que servir 20 pedidos ao mesmo tempo custa muito menos que 20× servir um?

**Resposta curta.** No decode, o caro é trazer os pesos da memória, não fazer as contas: cada passo lê o modelo inteiro para gerar um token por pedido. Com 20 pedidos no mesmo passo, os pesos viajam uma vez e servem aos 20, e as contas extras usam um processador que estava ocioso. **O que cresce com o lote é o KV cache, um por pedido**, e é ele que limita quantos pedidos cabem. Por isso os servidores cuidam da memória (PagedAttention, prefix caching) e da agenda (batching contínuo, chunked prefill). O preço: cada pedido fica um pouco mais lento.

## O que cada técnica faz

| Técnica | Problema | Solução |
|---|---|---|
| KV cache como memória | Cada pedido guarda K e V de todos os tokens: 128 KB por token no Llama-3.1-8B, 512 MiB por pedido de 4k | É o que limita o lote. GQA reduz o cache dividindo K/V entre cabeças |
| PagedAttention | Reservar o contexto máximo para cada pedido desperdiça 60% a 80% da memória | KV em blocos de 16 tokens, com tabela de blocos; só o último bloco tem sobra |
| Batching contínuo | No lote estático, quem termina cedo fica preso até o mais longo acabar | O lote é remontado a cada passo: quem termina sai, quem espera entra |
| Chunked prefill | Um prompt longo vira um passo gigante e congela os pedidos que estão escrevendo | O prompt é cortado e misturado com os decodes, até um orçamento de tokens por passo |
| Prefix caching | Pedidos com o mesmo começo recalculam o mesmo KV | Blocos com o mesmo prefixo (hash) são compartilhados |

## Medido no notebook-bia

Intel Core i5-1135G7, só CPU, PyTorch 2.7.1, 03/11/2026. Modelo de teste do `batching.py` (128 M parâmetros, 512 MB em float32), 256 tokens no cache de cada pedido.

| Lote | Um passo | Tok/s no total | Tok/s de cada pedido | Custo por token |
|---|---|---|---|---|
| 1 | 26,3 ms | 38 | 38,1 | 100% |
| 4 | 35,2 ms | 114 | 28,4 | 33% |
| 8 | 43,9 ms | 182 | 22,8 | 21% |
| 20 | 69,2 ms | 289 | 14,4 | 13% |
| 32 | 110,2 ms | 290 | 9,1 | 13% |

Servir 20 juntos custou 13% de servir 20 um de cada vez (7,6× mais barato); cada pedido ficou 2,6× mais lento. Na CPU o ganho para perto de 20 pedidos porque ela fica sem contas; numa GPU, o laboratório estima ~17× para o Llama-3.1-8B numa RTX 4090. Uma segunda rodada deu 7,9×.

Com numpy no lugar do PyTorch, lotes de 2 a 16 saem lentos nesta máquina: o OpenBLAS copia a matriz de pesos inteira antes de multiplicar poucos vetores, e isso esconde o ganho do lote. Por isso o script usa PyTorch quando ele está instalado.

## Como rodar

```bash
# 1. lotes de 1 a 32 (PyTorch se tiver; se não, numpy)
python batching.py

# 2. quanto KV cache cada modelo ocupa e quantos pedidos cabem numa GPU de 24 GB
python batching.py --memoria
python batching.py --memoria --gpu 80 --medio 2000

# 3. um servidor de verdade: 20 pedidos um de cada vez × todos juntos
python batching.py --servidor http://localhost:8080 --pedidos 20
```

Opções: `--lotes 1,8,20` escolhe os lotes; `--contexto 1024` muda os tokens no cache de cada pedido; `--backend numpy` força numpy; `--json arquivo.json` salva as medidas.

Para o modo `--servidor` com o llama.cpp da oficina de Docker, acrescente `--parallel 20` ao comando do serviço `modelo` em [`docker/compose.yaml`](../../../docker/compose.yaml) e aumente o `-c`: o contexto total é dividido entre as vagas (com `-c 8192 --parallel 4`, cada vaga tem 2048 tokens). Sem `--parallel`, o servidor atende poucos pedidos por vez e a diferença some. Para vLLM e Ollama, passe `--modelo <nome-do-modelo>`.

## Para estudar

- Kwon et al., 2023, *Efficient Memory Management for LLM Serving with PagedAttention* (arXiv:2309.06180): o artigo do vLLM, a leitura principal.
- Yu et al., 2022, *Orca: A Distributed Serving System for Transformer-Based Generative Models* (OSDI): batching contínuo (iteration-level scheduling).
- Agrawal et al., 2024, *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve* (OSDI): chunked prefill.
- Ainslie et al., 2023, *GQA* (arXiv:2305.13245): por que o KV cache moderno é menor.

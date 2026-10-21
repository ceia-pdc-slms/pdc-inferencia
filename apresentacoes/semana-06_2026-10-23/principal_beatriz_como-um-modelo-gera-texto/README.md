# Como um modelo gera texto: prefill, decode, KV cache e sampling

Apresentação principal da semana 6 (23/10/2026) · Beatriz

| Material | O que é |
|---|---|
| [Slides (PDF)](como-um-modelo-gera-texto.pdf) | 22 slides. O último é a resposta da pergunta da semana |
| [Slides (PPTX)](como-um-modelo-gera-texto.pptx) | O original, com notas do apresentador em cada slide |
| [`gera_texto.py`](gera_texto.py) | Mede ler × escrever, com e sem KV cache, e demonstra sampling |
| [Laboratório interativo](../../../laboratorios/geracao/geracao-de-texto.html) | O laço token a token, prefill × decode em quatro máquinas, KV cache e sampling |

## Pergunta da semana

> Por que ler uma pergunta de 500 tokens é mais rápido que escrever uma resposta de 500 tokens?

**Resposta curta.** Ler é uma passada só: os 500 tokens da pergunta já existem e passam juntos pelo modelo, então os pesos viajam da memória uma vez e servem a todos. Escrever são 500 passadas em fila, porque cada token depende do anterior, e em cada uma os pesos inteiros viajam para produzir um token só. **Ler é limitado por cálculo; escrever é limitado por banda de memória.** O KV cache evita refazer as contas dos tokens antigos, mas não evita reler os pesos.

## Medido no notebook-bia

Intel Core i5-1135G7, só CPU, 19/10/2026. Prompt de 500 tokens, resposta de 500 tokens, greedy.

| Modelo | Ler 500 (prefill) | Escrever 500 (decode) | Razão | Sem KV cache |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct, PyTorch float32 (`--hf`) | 2,4 s · 206 tok/s | 43,4 s · 11,5 tok/s | 17,9× | 0,4 tok/s (32 tokens) |
| Teste numpy, 128 M pesos aleatórios, 512 MB (padrão) | 1,0 s · 510 tok/s | 22,9 s · 21,8 tok/s | 23,4× | 1,0 tok/s (64 tokens) |

Para comparar: na oficina de Docker (semana 2), o mesmo Qwen2.5-0.5B em Q4_K_M no llama.cpp escreveu a 60,6 tok/s nesta máquina. É mais rápido que o PyTorch porque lê 4 bits por peso em vez de 32.

## Como rodar

```bash
# 1. modelo de teste (precisa só de numpy)
python gera_texto.py

# 2. modelo de verdade (precisa de torch e transformers; baixa ~1 GB na 1ª vez)
python gera_texto.py --hf Qwen/Qwen2.5-0.5B-Instruct

# 3. o llama.cpp da oficina de Docker (docker compose up -d modelo, na pasta docker/)
python gera_texto.py --servidor http://localhost:8080

# demonstração de sampling: temperatura, top-k e top-p no "O gato subiu no ___"
python gera_texto.py --sampling
```

Opções: `--tokens 500` muda o tamanho da pergunta e da resposta; `--sem-cache 32` muda quantos tokens são gerados sem KV cache (é lento de propósito); `--json arquivo.json` salva as medidas.

O resultado depende da máquina. Se você rodar na sua, a razão entre escrever e ler deve continuar bem acima de 1: é esse o ponto.

## Para estudar

- Kipply, *Transformer Inference Arithmetic*: a leitura obrigatória da semana.
- Horace He, *Making Deep Learning Go Brrrr From First Principles*: cálculo, memória e overhead.
- Kwon et al., 2023, *Efficient Memory Management for LLM Serving with PagedAttention* (arXiv:2309.06180).
- Holtzman et al., 2020, *The Curious Case of Neural Text Degeneration*: o artigo do top-p.

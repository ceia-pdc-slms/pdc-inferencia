# Laboratórios

Páginas interativas para aprender mexendo. Abrem direto no navegador (dois cliques no arquivo) ou pela bancada, em `http://localhost:8088`. O hub com todos eles é o [`index.html`](../index.html) da raiz.

| Laboratório | Semana | O que dá para fazer |
|---|---|---|
| [Docker por dentro](docker/docker-por-dentro.html) | 2 · oficina de Docker | Clicar em cada linha do Dockerfile e do compose e ver o que ela faz no seu computador. O ambiente de verdade está em [`docker/`](../docker/) |
| [Compressão: as quatro famílias](compressao/laboratorio-compressao.html) | 5 · Compressão 1/2 | Ver quanto pesa um modelo, o que quantização, pruning, destilação e KV cache fazem com os números, e montar uma receita para caber em 500 MB. As contas estão em [`compressao/calculo.js`](compressao/calculo.js), testadas em [`tests/calculo.test.js`](../tests/calculo.test.js) |
| [Como um modelo gera texto](geracao/geracao-de-texto.html) | 6 · Prefill, decode, KV cache e sampling | Gerar uma frase token por token, comparar o tempo de ler e de escrever em quatro máquinas, ver o KV cache evitando retrabalho e mexer em temperatura, top-k e top-p. As contas estão em [`geracao/geracao.js`](geracao/geracao.js), testadas em [`tests/geracao.test.js`](../tests/geracao.test.js) |

Os três foram feitos pela Beatriz. Como rodar tudo (laboratórios, bancada e testes) está no [README da raiz](../README.md#laboratórios-e-testbench-como-rodar).

## Para adicionar um laboratório

1. Crie uma pasta `laboratorios/<tema>/` com a página. Use `../../assets/tokens.css` e a barra `.appbar` para manter o tema do time.
2. Acrescente uma entrada em `LABS`, no começo do script de [`index.html`](../index.html).
3. Inclua a página na lista `PAGINAS` de [`tests/test_paginas.py`](../tests/test_paginas.py).
4. Ligue o laboratório à semana dele, no README da semana em [`apresentacoes/`](../apresentacoes/).

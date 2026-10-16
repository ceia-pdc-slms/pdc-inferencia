# pdc-inferencia

Repositório da frente de **Compressão, Testbench e Infraestrutura de Inferência** do PDC.

O time responde uma pergunta da qual as outras frentes dependem:

> **Esse modelo roda no nosso hardware? Com qual qualidade, a que velocidade, a que custo computacional e energético — e outra pessoa consegue repetir essa medição?**

Tudo o que entra aqui serve para responder essa pergunta com um número que outra pessoa consegue reproduzir.

## A conta que guia o trabalho

```
tokens por segundo ≈ banda de memória ÷ bytes lidos por token
```

Na geração (decode), o modelo inteiro é lido da memória a cada token. Comprimir diminui os bytes lidos por token; kernels e runtimes melhores aproveitam mais da banda. É isso que a gente estuda e mede.

## Por onde começar

1. Leia o [mapa do time](docs/mapa-do-time.md): o que fazemos, as regras da reunião, o calendário e as fontes por aula.
2. Veja as [apresentações](apresentacoes/): cada semana tem uma pasta, e dentro dela uma pasta por apresentação, com o nome de quem apresentou.
3. Leia o [guia de contribuição](CONTRIBUTING.md): como usar Git aqui, passo a passo, e como subir a sua apresentação.
4. Abra o seu **primeiro PR**: o dossiê da sua máquina em `maquinas/`, a partir do [modelo](maquinas/_modelo.md).

## Estrutura

| Pasta | O que vai lá |
|---|---|
| [`apresentacoes/`](apresentacoes/) | O material de cada reunião de sexta: `semana-NN_data/tipo_pessoa_tema/`. Índice por semana e por pessoa no [README de lá](apresentacoes/README.md) |
| [`maquinas/`](maquinas/) | Um dossiê por máquina física do time: como ligar, acessar, rodar um modelo e medir |
| [`testbench/`](testbench/) | A bancada de medição (o *harness* do plano): servidor, pipelines e resultados com procedência automática |
| [`laboratorios/`](laboratorios/) | Páginas interativas para aprender mexendo: Docker por dentro, compressão em 500 MB |
| [`docker/`](docker/) | Ambiente da oficina de Docker: o modelo em container (llama.cpp) e o `bench.py` |
| [`runs/`](runs/) | Medições feitas fora da bancada (Ollama, llama-bench…), um YAML por medição, com procedência completa |
| [`recipes/`](recipes/) | Receitas de compressão reproduzíveis: quantização, pruning, destilação |
| [`quality/`](quality/) | Avaliação de qualidade: corpus de avaliação e configurações de benchmark |
| [`reports/`](reports/) | Análises que juntam várias medições: comparações, Pareto |
| [`docs/`](docs/) | Mapa do time e decisões |
| [`assets/`](assets/), [`index.html`](index.html), [`tests/`](tests/) | Tema visual do time, hub dos laboratórios e testes automáticos |

## Laboratórios e testbench: como rodar

Tudo abre a partir de [`index.html`](index.html), o hub dos laboratórios e do material de apoio.

| Página | O que é |
|---|---|
| [`index.html`](index.html) | Hub: lista dos laboratórios e do material de apoio |
| [`laboratorios/docker/docker-por-dentro.html`](laboratorios/docker/docker-por-dentro.html) | Oficina de Docker (semana 2) |
| [`laboratorios/compressao/laboratorio-compressao.html`](laboratorios/compressao/laboratorio-compressao.html) | Compressão 1/2: as quatro famílias e a receita para caber em 500 MB (semana 5) |
| [`testbench/index.html`](testbench/index.html) | Bancada de medição: pipelines, execução com log ao vivo, comparação de resultados |

### O que precisa estar instalado

| Para | Precisa |
|---|---|
| Ver os laboratórios | Só um navegador |
| Usar o testbench | Python 3.9 ou mais novo (só biblioteca padrão, nada de `pip install`) **ou** Docker |
| Ter um modelo para medir | Docker Desktop (Windows/macOS) ou Docker Engine (Linux), **aberto e rodando** |
| Rodar os testes | Python; Node 18+ para os testes do laboratório de compressão; Chrome ou Edge para os testes das páginas |

### Jeito 1: tudo em containers (recomendado)

Na raiz do repositório, com o Docker aberto:

```powershell
$env:BANCADA_MAQUINA = "notebook-bia"   # o nome da SUA máquina; vai em todo resultado
docker compose up -d                    # sobe o modelo (llama.cpp) e a bancada
docker compose logs -f modelo           # na 1ª vez baixa ~400 MB; Ctrl+C ao ver "server is listening"
```

Em bash, a primeira linha é `export BANCADA_MAQUINA=notebook-bia`.

Abra **http://localhost:8088**. No testbench, rode primeiro a **Sonda do servidor**, que confere se o modelo responde, e depois **Latência e vazão**. Os resultados vão para `testbench/resultados/`.

Para desligar: `docker compose down` (os pesos do modelo continuam no volume). `docker compose down -v` apaga também os pesos.

### Jeito 2: bancada no Python, modelo no Docker

```bash
cd docker && docker compose up -d modelo && cd ..    # o modelo em http://localhost:8080
python testbench/servidor.py --maquina notebook-bia  # a bancada em http://localhost:8088
```

Nesse jeito a bancada lê o digest da imagem sozinha (`docker image inspect`), e a procedência sai completa.

### Jeito 3: só os laboratórios

Dê dois cliques em `index.html`. As páginas funcionam direto do disco. A exceção é o testbench: sem o servidor ele só compara resultados que você arrastar para a aba **Resultados**.

### Quando algo não funciona

| Sintoma | Causa provável | O que fazer |
|---|---|---|
| "Modelo fora do ar" no testbench, ou `Failed to fetch` / `Connection refused` | O Docker Desktop está fechado ou o container do modelo não subiu | Abra o Docker Desktop; `docker compose ps` deve mostrar `modelo` como `healthy` |
| `modelo` fica em `starting` muito tempo | Primeira subida: baixando os pesos | `docker compose logs -f modelo` e espere |
| `ports are not available` / `bind: Only one usage of each socket address` | A porta 8088 ou a 8080 já está em uso por outro programa | Troque a porta da bancada: `$env:BANCADA_PORTA = "8090"` e suba de novo |
| "A bancada não está rodando" na página do testbench | A página foi aberta direto do disco, ou o servidor caiu | Use o jeito 1 ou o 2 e abra pelo endereço `http://localhost:8088` |
| `maquina-sem-nome` nos resultados | `BANCADA_MAQUINA` não foi definida | Defina antes do `docker compose up`, ou corrija o campo **Máquina** no formulário |

### Testes

Rodam sem modelo nenhum: um servidor falso, compatível com OpenAI e com tempos conhecidos, faz o papel do llama.cpp.

```bash
python -m unittest tests.test_bancada -v   # bancada, bench.py, pipelines e API (~45 s)
node --test tests/calculo.test.js          # contas do laboratório de compressão (<1 s)
python -m unittest tests.test_paginas -v   # abre cada página no Chrome/Edge sem janela (~8 min)
```

O último teste abre as páginas no tema claro e no escuro e percorre todas as interações: o laboratório de compressão clica em cada passo, e o testbench executa a sonda pela interface. Ele falha se aparecer qualquer erro de JavaScript.

### Para adicionar um laboratório

1. Crie a página em `laboratorios/<tema>/`, usando `../../assets/tokens.css` e a barra `.appbar` para manter o padrão.
2. Acrescente uma entrada em `LABS`, no começo do script de `index.html`.
3. Inclua a página na lista `PAGINAS` de `tests/test_paginas.py`.

Para adicionar um pipeline de medição, veja [`testbench/README.md`](testbench/README.md).

## A regra que não se negocia

**Nenhum número existe sem procedência:** máquina, ferramenta, versão, modelo, formato, tamanho de contexto e como foi medido. Um tok/s solto não é resultado, é boato.

A bancada grava a procedência sozinha em cada resultado. Os registros manuais em `runs/` são conferidos automaticamente em todo PR; se faltar algum campo, o PR mostra qual e em qual arquivo (ver [`runs/README.md`](runs/README.md)).

## Quando uma entrega está pronta

- Entrou por um **PR revisado** por pelo menos uma pessoa.
- Tem um **README que outra pessoa segue sozinha**, sem perguntar nada.
- Todo número tem **procedência**.

## O que nunca entra no repositório

Pesos de modelo (`.gguf`, `.safetensors`, `.bin`…), datasets grandes, senhas, tokens, chaves SSH e IPs das máquinas. O `.gitignore` já barra os formatos mais comuns; o resto depende de você. Modelo fica no Hugging Face ou no disco da máquina, e o registro aponta para onde ele está.

## Time

| Pessoa | Papel |
|---|---|
| Hugo | Liderança da frente |
| Beatriz | Receitas de compressão |
| Wevton | Ambiente e reprodutibilidade |
| João Vitor | Modelo analítico e kernels |
| Pedro | Bancada de medição |
| Marcos | Corpus de avaliação e perfis de carga |

Reunião interna: sexta-feira, 1 hora. Travou? Canal de dúvidas no Discord, no mesmo dia.

---

Este repositório começou a partir do `pdc-slms`, o repositório da Beatriz ([@beatrizalmeidaf](https://github.com/beatrizalmeidaf)), que montou o testbench, os laboratórios e o tema visual.

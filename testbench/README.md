# Testbench: a bancada de medição

A bancada é um servidor pequeno (`servidor.py`, só biblioteca padrão do Python) que:

1. serve as páginas do repositório (hub, laboratórios, testbench);
2. lê os **pipelines** de `testbench/pipelines/*.json` e monta o formulário de cada um na interface;
3. valida os parâmetros, roda o comando **sem shell** e acompanha o log ao vivo;
4. guarda cada resultado em `testbench/resultados/<pipeline>/<maquina>_<data>.json`;
5. lista os resultados (inclusive os antigos da oficina, em `docker/resultados/`) para comparar.

A medição roda no servidor, nunca no navegador. Por isso não existe problema de CORS, e o número é o mesmo que o script daria no terminal.

Só uma execução por vez: duas medições simultâneas disputariam CPU e memória e estragariam as duas. A API responde `409` se você tentar.

## Rodar

```bash
python testbench/servidor.py                        # http://localhost:8088
python testbench/servidor.py --maquina notebook-bia --modelo-url http://localhost:8080
```

| Opção | Variável de ambiente | Padrão |
|---|---|---|
| `--porta` | `BANCADA_PORTA` | `8088` |
| `--host` | `BANCADA_HOST` | `127.0.0.1` (use `0.0.0.0` só dentro de container) |
| `--modelo-url` | `BANCADA_MODELO_URL` | `http://localhost:8080` |
| `--maquina` | `BANCADA_MAQUINA` | hostname |
| digest da imagem | `IMAGE_DIGEST` | lido de `docker image inspect`, se o Docker estiver acessível |

> A API executa comandos. Não exponha a bancada na internet nem numa rede que você não controla. Quando ela for para o servidor do time, o próximo passo é colocar autenticação na frente.

## Pipelines

Um pipeline é um JSON. O comando pode ser qualquer programa que receba argumentos e grave um JSON: um script nosso, `llama-bench`, `lm-eval`, um `docker run`…

```jsonc
{
  "id": "latencia",                       // letras minúsculas, números, _ e -
  "nome": "Latência e vazão",
  "categoria": "Desempenho",              // agrupa na lista da esquerda
  "descricao": "…",
  "esquema_resultado": "bench-v1",        // diz à interface como mostrar o resultado
  "comando": ["{python}", "-u", "{raiz}/docker/bench.py"],
  "saida": { "argumento": "--out" },      // a bancada acrescenta --out <arquivo>
  "progresso": "run (\\d+)/(\\d+)",       // regex com (atual, total) lida do log
  "parametros": [
    { "id": "runs", "rotulo": "Execuções", "grupo": "Repetições", "tipo": "inteiro",
      "padrao": 5, "min": 1, "max": 200, "argumento": "--runs", "ajuda": "…" }
  ]
}
```

**Tipos de parâmetro:** `texto`, `texto_longo`, `inteiro`, `real`, `escolha` (com `opcoes`) e `booleano`. O booleano vira uma flag: o argumento entra no comando só quando o valor é verdadeiro.

**Outros campos opcionais:** `obrigatorio`, `min`, `max`, `sugestoes` (lista de valores sugeridos para um campo de texto), `ajuda` e `grupo`.

**Substituições** no `comando` e nos `padrao`:

| Marcador | Vira |
|---|---|
| `{python}` | o Python que roda a bancada |
| `{raiz}` | a raiz do repositório |
| `{modelo_url}` | a URL do modelo configurada na bancada |
| `{maquina}` | o nome da máquina configurado |
| `{imagem_digest}` | o digest da imagem do llama.cpp |

Parâmetro vazio e não obrigatório é omitido do comando, e o programa usa o próprio padrão.

**Esquemas de resultado que a interface já sabe mostrar:**

| Esquema | Como aparece |
|---|---|
| `bench-v1` | Saída do `bench.py`. Cartões, gráfico por execução, teste t de Welch e tabela de procedência |
| `sonda-v1` | Lista de checagens |
| qualquer outro | O JSON formatado |

Para um esquema novo ganhar uma visão própria, edite `visao()` em `testbench/index.html`.

### Adicionar um pipeline

1. Escreva o script, aceitando `--out <arquivo>` e imprimindo o progresso no formato da sua regex.
2. Crie `testbench/pipelines/<id>.json`.
3. Reinicie a bancada. O teste `Manifestos` em `tests/test_bancada.py` valida todos os manifestos do diretório.

## API

| Método | Rota | O que faz |
|---|---|---|
| GET | `/api/estado` | Versão, URL do modelo, máquina, execução em andamento |
| GET | `/api/pipelines` | Manifestos, com os padrões já resolvidos |
| GET | `/api/sonda?url=` | Pergunta `/v1/models` ao servidor do modelo |
| POST | `/api/comando` | `{pipeline, parametros}` → comando montado (sem executar), ou `400` com o erro de validação |
| POST | `/api/execucoes` | `{pipeline, parametros}` → inicia; `409` se já houver uma rodando |
| GET | `/api/execucoes/<id>?desde=n` | Estado, log a partir da linha `n`, progresso e resultado |
| POST | `/api/execucoes/<id>/cancelar` | Interrompe o processo |
| GET | `/api/resultados` | Todos os JSON de resultado, com máquina, modelo e data |

## Testes

```bash
python -m unittest tests.test_bancada -v   # cerca de 45 s
```

Os testes sobem um servidor falso compatível com OpenAI (`tests/servidor_falso.py`), com tempos conhecidos de antemão. Eles conferem quatro coisas:
- que o `bench.py` mede esses tempos;
- que a validação recusa entradas ruins;
- que o comando é montado sem shell;
- que a API funciona de ponta a ponta, inclusive o limite de uma execução por vez e o cancelamento.

Um teste também garante que o manifesto `latencia` cubra todos os argumentos do `bench.py`: se alguém acrescentar um argumento ao script, o teste falha até o manifesto acompanhar.

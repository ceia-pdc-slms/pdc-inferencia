# Oficina: Docker na prática + Testbench

Objetivo: rodar **o mesmo modelo, no mesmo container, em duas máquinas** e medir em cima dele com o `bench.py`.
Os comandos são idênticos nas duas máquinas; só muda o valor de `--machine`.

Pré-requisito: Docker Desktop (Windows/macOS) ou Docker Engine (Linux) em execução.

## 1. Subir o modelo

```bash
docker compose up -d modelo
docker compose logs -f modelo      # Ctrl+C quando aparecer "server is listening"
```

Na primeira vez o Docker baixa a imagem do llama.cpp server e o servidor baixa os pesos
(Qwen2.5 0.5B Instruct, GGUF Q4_K_M, cerca de 400 MB). Os pesos ficam no volume `modelos`.

Teste rápido: abra http://localhost:8080 no navegador ou

```bash
curl http://localhost:8080/v1/models
```

## 2. Anotar o digest da imagem

O digest (sha256) identifica a imagem; a tag `:server` pode mudar com o tempo.

PowerShell:

```powershell
$env:IMAGE_DIGEST = docker image inspect ghcr.io/ggml-org/llama.cpp:server --format "{{index .RepoDigests 0}}"
```

Bash:

```bash
export IMAGE_DIGEST=$(docker image inspect ghcr.io/ggml-org/llama.cpp:server --format '{{index .RepoDigests 0}}')
```

## 3. Medir

```bash
docker compose run --rm bench --machine notebook-bia --notes "na tomada, modo desempenho"
```

O resultado sai em `resultados/resultado_<maquina>_<data>.json`.
Parâmetros extras do `bench.py` (`--runs`, `--warmup`, `--max-tokens`...) podem ir no fim do comando.

## 4. Repetir na máquina B

Copie esta pasta (ou clone o repositório) e repita os passos 1 a 3 trocando `--machine`.

## 5. Comparar

Abra os dois JSON lado a lado, em `procedencia`:

| Campo | Esperado |
|---|---|
| `imagem_docker_digest` | igual (mesma arquitetura) |
| `script_sha256` | igual |
| `como_mediu.prompt_sha256`, `seed`, `temperature` | igual |
| `cpu`, `ram_gb`, `arquitetura` | diferente |
| `resumo.tok_per_s_decode` | diferente |

Se os três primeiros batem, a diferença de velocidade é da máquina, não do ambiente.

Observações:

- Dentro do container, `hostname` é o ID do container; por isso `--machine` é obrigatório.
- No Docker Desktop, `ram_gb` mostra a memória da VM Linux do Docker, não a RAM total do computador.
- amd64 e arm64 (ex.: Raspberry Pi) puxam binários diferentes sob a mesma tag; o campo `arquitetura` registra isso.

## Encerrar

```bash
docker compose down          # mantém os pesos no volume
docker compose down -v       # apaga também os pesos
```

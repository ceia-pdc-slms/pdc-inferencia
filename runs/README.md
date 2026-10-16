# Medições

Cada medição é um arquivo YAML com procedência completa. Número sem procedência não entra.

## Bancada ou registro manual?

| Como você mediu | Onde fica o resultado |
|---|---|
| Pela bancada (`testbench/`, pipelines **Sonda** e **Latência e vazão**) | `testbench/resultados/<pipeline>/<maquina>_<data>.json`, com procedência gravada sozinha. Abra um PR com o JSON |
| Na oficina de Docker (`docker compose run --rm bench`) | `docker/resultados/`. A bancada também lista esses resultados na aba **Resultados** |
| À mão, fora da bancada: `ollama run --verbose`, `llama-bench`, LM Studio, script seu | **Aqui**, em `runs/<nome-da-maquina>/`, seguindo o modelo abaixo |

## Como registrar

1. Copie [`_modelo.yaml`](_modelo.yaml) para `runs/<nome-da-maquina>/<AAAA-MM-DD>_<modelo>_<formato>.yaml`.
   Exemplo: `runs/rpi5-01/2026-10-20_qwen2.5-0.5b_q4_k_m.yaml`.
2. Preencha todos os campos marcados como obrigatórios.
3. Abra um PR.

A máquina precisa ter dossiê em `maquinas/<nome-da-maquina>.md`.

## Verificação automática

Todo PR que mexe em `runs/` ou `maquinas/` passa pelo script [`.github/scripts/validar_runs.py`](../.github/scripts/validar_runs.py). Ele reprova o PR se faltar algum campo de procedência e mostra, no próprio PR, qual campo falta e em qual arquivo.

Para conferir antes de abrir o PR:

```bash
pip install pyyaml
python .github/scripts/validar_runs.py
```

**Campos obrigatórios:** quem mediu e quando; máquina (nome, hardware, sistema); ferramenta (nome, versão, comando exato); modelo (nome, fonte, formato); contexto (tokens de prompt e tokens gerados); método; e pelo menos um valor de decode em tok/s.

## Quantas repetições

Uma medição só já conta para o dossiê. Mas um valor só não diz nada sobre variância: registre **pelo menos três repetições** (o validador avisa quando há menos) e cinco quando for comparar duas coisas. A semana 10 é sobre isso.

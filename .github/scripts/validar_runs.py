#!/usr/bin/env python3
"""Confere se todo registro de medição em runs/ tem procedência completa.

Uso:
    python .github/scripts/validar_runs.py

Sai com código 1 se algum registro estiver incompleto. Arquivos e pastas
que começam com "_" (como runs/_modelo.yaml) são ignorados.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("Falta o PyYAML. Instale com: pip install pyyaml")

RAIZ = Path(__file__).resolve().parents[2]
RUNS = RAIZ / "runs"
MAQUINAS = RAIZ / "maquinas"
NO_GITHUB = os.environ.get("GITHUB_ACTIONS") == "true"
MIN_REPETICOES = 3

OBRIGATORIOS = [
    ("quem",),
    ("data",),
    ("maquina", "nome"),
    ("maquina", "hardware"),
    ("maquina", "sistema"),
    ("ferramenta", "nome"),
    ("ferramenta", "versao"),
    ("ferramenta", "comando"),
    ("modelo", "nome"),
    ("modelo", "fonte"),
    ("modelo", "formato"),
    ("contexto", "tokens_prompt"),
    ("contexto", "tokens_gerados"),
    ("metodo",),
]


def pegar(dados, caminho):
    atual = dados
    for chave in caminho:
        if not isinstance(atual, dict) or chave not in atual:
            return None
        atual = atual[chave]
    return atual


def vazio(valor) -> bool:
    if valor is None:
        return True
    if isinstance(valor, str):
        return not valor.strip()
    if isinstance(valor, (list, dict)):
        return len(valor) == 0
    return False


def numero_positivo(valor) -> bool:
    return isinstance(valor, (int, float)) and not isinstance(valor, bool) and valor > 0


def relatar(nivel: str, arquivo: Path, msg: str) -> None:
    rel = arquivo.relative_to(RAIZ)
    if NO_GITHUB:
        print(f"::{nivel} file={rel}::{msg}")
    else:
        rotulo = "ERRO " if nivel == "error" else "AVISO"
        print(f"{rotulo} {rel}: {msg}")


def validar(arquivo: Path) -> tuple[int, int]:
    erros = avisos = 0

    def erro(msg: str) -> None:
        nonlocal erros
        erros += 1
        relatar("error", arquivo, msg)

    def aviso(msg: str) -> None:
        nonlocal avisos
        avisos += 1
        relatar("warning", arquivo, msg)

    try:
        dados = yaml.safe_load(arquivo.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        erro("YAML inválido: " + " ".join(str(e).split()))
        return erros, avisos

    if not isinstance(dados, dict):
        erro("o arquivo precisa seguir o formato de runs/_modelo.yaml")
        return erros, avisos

    for caminho in OBRIGATORIOS:
        if vazio(pegar(dados, caminho)):
            erro(f"falta o campo obrigatório '{'.'.join(caminho)}'")

    for caminho in (("contexto", "tokens_prompt"), ("contexto", "tokens_gerados")):
        valor = pegar(dados, caminho)
        if not vazio(valor) and not (isinstance(valor, int) and numero_positivo(valor)):
            erro(f"'{'.'.join(caminho)}' precisa ser um número inteiro positivo")

    decode = pegar(dados, ("resultados", "decode_tok_s"))
    if not isinstance(decode, list) or not decode:
        erro("'resultados.decode_tok_s' precisa ser uma lista com pelo menos um valor, ex.: [18.2, 18.0, 18.4]")
    elif not all(numero_positivo(x) for x in decode):
        erro("'resultados.decode_tok_s' só aceita números positivos")
    elif len(decode) < MIN_REPETICOES:
        aviso(
            f"só {len(decode)} repetição(ões) em decode_tok_s; registre pelo menos "
            f"{MIN_REPETICOES} para dar para falar de variância"
        )

    nome = pegar(dados, ("maquina", "nome"))
    if isinstance(nome, str) and nome.strip():
        nome = nome.strip()
        if not (MAQUINAS / f"{nome}.md").is_file():
            erro(f"a máquina '{nome}' não tem dossiê em maquinas/{nome}.md")
        if arquivo.parent.name != nome:
            aviso(f"o arquivo deveria estar em runs/{nome}/, a pasta com o nome da máquina")

    return erros, avisos


def main() -> int:
    arquivos = sorted(
        p
        for p in RUNS.rglob("*")
        if p.is_file()
        and p.suffix in {".yaml", ".yml"}
        and not any(parte.startswith("_") for parte in p.relative_to(RUNS).parts)
    )
    if not arquivos:
        print("Nenhum registro em runs/ ainda.")
        return 0

    total_erros = total_avisos = 0
    for arquivo in arquivos:
        e, a = validar(arquivo)
        total_erros += e
        total_avisos += a

    print(f"\n{len(arquivos)} registro(s) conferido(s): {total_erros} erro(s), {total_avisos} aviso(s).")
    if total_erros:
        print("Corrija os erros acima. O formato de cada campo está em runs/_modelo.yaml.")
    return 1 if total_erros else 0


if __name__ == "__main__":
    sys.exit(main())

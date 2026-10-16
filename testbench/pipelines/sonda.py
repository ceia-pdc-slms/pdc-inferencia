#!/usr/bin/env python3
"""
sonda.py - confere um servidor compatível com OpenAI antes de medir.

Quatro checagens: /health, /v1/models, um pedido curto sem streaming e um com streaming.
Cada uma registra ok/falha, tempo e detalhe. Sai com código 1 se alguma falhar,
mas grava o JSON mesmo assim (o resultado de uma sonda que falhou também é informação).
"""
import argparse, json, platform, socket, sys, time, urllib.request, urllib.error


def pedir(url, corpo=None, timeout=30):
    dados = json.dumps(corpo).encode() if corpo is not None else None
    req = urllib.request.Request(url, data=dados, headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def checar(nome, fn):
    t0 = time.perf_counter()
    try:
        detalhe = fn()
        return {"nome": nome, "ok": True, "ms": round((time.perf_counter() - t0) * 1000, 1), "detalhe": detalhe}
    except urllib.error.HTTPError as e:
        return {"nome": nome, "ok": False, "ms": round((time.perf_counter() - t0) * 1000, 1), "detalhe": f"HTTP {e.code}"}
    except Exception as e:
        return {"nome": nome, "ok": False, "ms": round((time.perf_counter() - t0) * 1000, 1),
                "detalhe": str(getattr(e, "reason", e))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--maquina", required=True)
    ap.add_argument("--timeout", type=float, default=30)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    url, to = a.url.rstrip("/"), a.timeout
    estado = {"modelo": None}

    def health():
        with pedir(url + "/health", timeout=to) as r:
            return r.read(200).decode("utf-8", "replace")

    def modelos():
        with pedir(url + "/v1/models", timeout=to) as r:
            j = json.load(r)
        ids = [d.get("id") for d in j.get("data", [])]
        estado["modelo"] = ids[0] if ids else None
        return ids

    def curto():
        corpo = {"model": estado["modelo"] or "default", "max_tokens": 8, "temperature": 0,
                 "messages": [{"role": "user", "content": "Diga apenas: ok"}]}
        with pedir(url + "/v1/chat/completions", corpo, to) as r:
            j = json.load(r)
        return {"texto": j["choices"][0]["message"]["content"], "uso": j.get("usage")}

    def stream():
        corpo = {"model": estado["modelo"] or "default", "max_tokens": 16, "temperature": 0, "stream": True,
                 "messages": [{"role": "user", "content": "Conte de 1 a 5."}]}
        chunks = 0
        with pedir(url + "/v1/chat/completions", corpo, to) as r:
            for raw in r:
                linha = raw.decode("utf-8", "ignore").strip()
                if linha.startswith("data:") and linha[5:].strip() != "[DONE]":
                    try:
                        obj = json.loads(linha[5:])
                    except json.JSONDecodeError:
                        continue
                    if any((c.get("delta") or {}).get("content") for c in obj.get("choices", [])):
                        chunks += 1
        if not chunks:
            raise RuntimeError("streaming não entregou nenhum pedaço de texto")
        return {"chunks_com_texto": chunks}

    passos = [("health", health), ("modelos", modelos), ("pedido curto", curto), ("streaming", stream)]
    checagens = []
    for i, (nome, fn) in enumerate(passos, 1):
        c = checar(nome, fn)
        checagens.append(c)
        print(f"checagem {i}/{len(passos)}: {nome} {'ok' if c['ok'] else 'FALHOU'} ({c['ms']} ms) {c['detalhe']}", flush=True)

    res = {
        "esquema": "sonda-v1",
        "procedencia": {
            "data_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "maquina": a.maquina, "hostname": socket.gethostname(),
            "so": platform.platform(), "python": platform.python_version(),
            "servidor_url": url, "modelo": estado["modelo"],
        },
        "ok": all(c["ok"] for c in checagens),
        "checagens": checagens,
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"salvo em {a.out}")
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()

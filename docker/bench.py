#!/usr/bin/env python3
"""
bench.py - bancada minima de medicao para o PDC (Testbench 1/2).

Mede um servidor com API compativel com OpenAI (llama.cpp server, vLLM, Ollama...)
usando streaming, e salva TUDO (medidas + procedencia) em um JSON.

Somente biblioteca padrao: roda em qualquer container Python, sem pip install.

Exemplo:
  python bench.py --url http://localhost:8080 --machine "notebook-pedro" \
      --image "sha256:abc..." --runs 5 --warmup 1 --max-tokens 128
"""
import argparse, hashlib, json, os, platform, socket, statistics, subprocess, sys, time
import urllib.request

PROMPT_DEFAULT = "Explique em um paragrafo por que ler o modelo da memoria limita a velocidade de geracao de texto."


def pct(values, p):
    """Percentil por interpolacao linear (p entre 0 e 100)."""
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def one_run(url, model, prompt, max_tokens, temperature, seed):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    stamps = []          # instante de cada chunk com texto
    usage = None
    with urllib.request.urlopen(req) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                if (ch.get("delta") or {}).get("content"):
                    stamps.append(time.perf_counter())
    t_end = time.perf_counter()
    if not stamps:
        raise RuntimeError("nenhum token recebido")

    # Aproximacao: 1 chunk com texto ~ 1 token. Se o servidor informa usage, registramos tambem.
    n_chunks = len(stamps)
    n_tokens = (usage or {}).get("completion_tokens", n_chunks)
    ttft = stamps[0] - t0
    e2e = t_end - t0
    decode_time = stamps[-1] - stamps[0]
    tpot = decode_time / (n_chunks - 1) if n_chunks > 1 else None
    itl = [b - a for a, b in zip(stamps, stamps[1:])]
    return {
        "ttft_s": ttft,
        "tpot_s": tpot,
        "e2e_s": e2e,
        "n_chunks": n_chunks,
        "n_tokens_reported": n_tokens,
        "tok_per_s_decode": (n_chunks - 1) / decode_time if decode_time > 0 else None,
        "tok_per_s_e2e": n_chunks / e2e,
        "itl_s": itl,
    }


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for l in f:
                if l.lower().startswith(("model name", "hardware")):
                    return l.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or None


def ram_gb():
    try:
        with open("/proc/meminfo") as f:
            return round(int(f.readline().split()[1]) / 1024 / 1024, 1)
    except Exception:
        return None


def summarize(runs, key):
    vals = [r[key] for r in runs if r[key] is not None]
    if not vals:
        return None
    return {
        "mean": statistics.mean(vals),
        "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals), "max": max(vals),
        "p50": pct(vals, 50), "p95": pct(vals, 95),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--model", default=None, help="nome do modelo (se vazio, pergunta ao servidor)")
    ap.add_argument("--prompt", default=PROMPT_DEFAULT)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--machine", required=True, help="nome da maquina (ex.: notebook-pedro, rpi5-01)")
    ap.add_argument("--image", default=os.environ.get("IMAGE_DIGEST"), help="digest da imagem Docker")
    ap.add_argument("--quant", default=None, help="formato/quantizacao (ex.: Q4_K_M)")
    ap.add_argument("--ctx", default=None, help="tamanho de contexto do servidor")
    ap.add_argument("--notes", default="", help="modo de energia, temperatura, o que mais importar")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    # modelo
    model = a.model
    server_models = None
    try:
        with urllib.request.urlopen(a.url + "/v1/models", timeout=10) as r:
            server_models = json.load(r)
            if not model and server_models.get("data"):
                model = server_models["data"][0]["id"]
    except Exception as e:
        print(f"aviso: nao consegui ler /v1/models ({e})", file=sys.stderr)
    model = model or "default"

    print(f"modelo={model} | warmup={a.warmup} | runs={a.runs}")
    for i in range(a.warmup):
        one_run(a.url, model, a.prompt, a.max_tokens, a.temperature, a.seed)
        print(f"  warmup {i+1}/{a.warmup} ok (descartado)")

    runs = []
    for i in range(a.runs):
        r = one_run(a.url, model, a.prompt, a.max_tokens, a.temperature, a.seed)
        runs.append(r)
        print(f"  run {i+1}/{a.runs}: TTFT={r['ttft_s']*1000:.0f} ms | "
              f"TPOT={(r['tpot_s'] or 0)*1000:.1f} ms | decode={r['tok_per_s_decode']:.2f} tok/s | "
              f"tokens={r['n_chunks']}")

    all_itl = [x for r in runs for x in r["itl_s"]]
    result = {
        "procedencia": {
            "data_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "maquina": a.machine,
            "hostname": socket.gethostname(),
            "so": platform.platform(),
            "arquitetura": platform.machine(),
            "cpu": cpu_model(),
            "ram_gb": ram_gb(),
            "python": platform.python_version(),
            "imagem_docker_digest": a.image,
            "modelo": model,
            "formato_quantizacao": a.quant,
            "tamanho_contexto": a.ctx,
            "servidor_url": a.url,
            "servidor_modelos_resposta": server_models,
            "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
            "como_mediu": {
                "prompt": a.prompt,
                "prompt_sha256": hashlib.sha256(a.prompt.encode()).hexdigest(),
                "max_tokens": a.max_tokens,
                "temperature": a.temperature,
                "seed": a.seed,
                "warmup_descartado": a.warmup,
                "execucoes": a.runs,
                "token_contado_como": "1 chunk de streaming com texto (aproximacao)",
            },
            "notas": a.notes,
        },
        "resumo": {
            "ttft_s": summarize(runs, "ttft_s"),
            "tpot_s": summarize(runs, "tpot_s"),
            "tok_per_s_decode": summarize(runs, "tok_per_s_decode"),
            "tok_per_s_e2e": summarize(runs, "tok_per_s_e2e"),
            "itl_s_agregado": {"p50": pct(all_itl, 50), "p95": pct(all_itl, 95), "n": len(all_itl)},
        },
        "execucoes": [{k: v for k, v in r.items() if k != "itl_s"} for r in runs],
    }

    out = a.out or f"resultado_{a.machine}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    s = result["resumo"]["tok_per_s_decode"]
    print(f"\ndecode: {s['mean']:.2f} +- {s['stdev']:.2f} tok/s (n={a.runs}) | "
          f"min {s['min']:.2f} max {s['max']:.2f}")
    print(f"salvo em {out}")


if __name__ == "__main__":
    main()

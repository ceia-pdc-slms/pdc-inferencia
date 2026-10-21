"""
Como um modelo gera texto: prefill, decode, KV cache e sampling, medidos na sua máquina.

Pergunta da semana: por que ler uma pergunta de 500 tokens é mais rápido que escrever
uma resposta de 500 tokens?
"""
import argparse
import json
import math
import platform
import random
import sys
import time

TEXTO_BASE = (
    "Um modelo de linguagem recebe um texto, quebra esse texto em tokens e calcula, para cada "
    "posição, a probabilidade do próximo token. Para escrever uma resposta ele repete esse passo "
    "muitas vezes, sempre acrescentando o token escolhido ao final do texto. "
)


def linha(c="-", n=72):
    print(c * n)


def tabela(medidas, n):
    """medidas: lista de (nome, segundos, tokens)."""
    linha()
    print(f"{'fase':<34}{'tokens':>8}{'tempo':>12}{'tok/s':>12}")
    linha()
    for nome, seg, tok in medidas:
        print(f"{nome:<34}{tok:>8}{seg * 1000:>10.0f} ms{tok / seg:>12.1f}")
    linha()
    ler = next(s for nome, s, _ in medidas if nome.startswith("prefill"))
    esc = next(s for nome, s, _ in medidas if nome.startswith("decode"))
    print(f"Escrever {n} tokens levou {esc / ler:.1f}x o tempo de ler {n} tokens.")


# =====================================================================
# 1. Modelo de teste em numpy
# =====================================================================
class Teste:
    """Um decoder estilo GPT, com os mesmos passos de um modelo de verdade:
    embedding -> N blocos (atenção + MLP) -> logits. Pesos aleatórios.

    d=1024, 10 camadas, MLP 4x: ~126 M parâmetros em float32 = ~500 MB,
    o mesmo tamanho em bytes do Qwen2.5-0.5B Q4_K_M (485 MB)."""

    def __init__(self, d=1024, camadas=10, cabecas=16, vocab=2048, seed=0):
        import numpy as np
        self.np = np
        r = np.random.default_rng(seed)
        f = np.float32
        self.d, self.h, self.hd, self.vocab = d, cabecas, d // cabecas, vocab
        s = 1 / math.sqrt(d)
        self.emb = (r.standard_normal((vocab, d)) * 0.02).astype(f)
        self.blocos = []
        for _ in range(camadas):
            self.blocos.append({
                "wqkv": (r.standard_normal((d, 3 * d)) * s).astype(f),  # Q, K e V juntos
                "wo": (r.standard_normal((d, d)) * s).astype(f),
                "w1": (r.standard_normal((d, 4 * d)) * s).astype(f),
                "w2": (r.standard_normal((4 * d, d)) * s / 2).astype(f),
            })
        self.params = self.emb.size + sum(sum(w.size for w in b.values()) for b in self.blocos)

    def _norm(self, x):
        np = self.np
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-5)

    def forward(self, tokens, cache):
        """Processa os `tokens` novos. `cache` é a lista de (K, V) por camada, ou None
        (sem KV cache: o chamador passa o texto inteiro de novo). Devolve os logits do último
        token e o cache atualizado."""
        np = self.np
        T = len(tokens)
        x = self.emb[tokens]                                   # (T, d)
        passado = 0 if cache is None else cache[0][0].shape[1]
        novo_cache = []
        for i, b in enumerate(self.blocos):
            # ---- atenção ----
            qkv = self._norm(x) @ b["wqkv"]                    # 1 matmul para os T tokens juntos
            q, k, v = np.split(qkv, 3, axis=-1)
            q = q.reshape(T, self.h, self.hd).transpose(1, 0, 2)   # (h, T, hd)
            k = k.reshape(T, self.h, self.hd).transpose(1, 0, 2)
            v = v.reshape(T, self.h, self.hd).transpose(1, 0, 2)
            if cache is not None and passado:
                k = np.concatenate([cache[i][0], k], axis=1)   # K e V antigos vêm do cache
                v = np.concatenate([cache[i][1], v], axis=1)
            novo_cache.append((k, v))
            att = q @ k.transpose(0, 2, 1) / math.sqrt(self.hd)    # (h, T, passado+T)
            # máscara causal: cada token só olha para ele mesmo e para trás
            mascara = np.triu(np.ones((T, passado + T), dtype=bool), k=passado + 1)
            att[:, mascara] = -1e9
            att = np.exp(att - att.max(-1, keepdims=True))
            att /= att.sum(-1, keepdims=True)
            o = (att @ v).transpose(1, 0, 2).reshape(T, self.d)
            x = x + o @ b["wo"]
            # ---- MLP ----
            m = self._norm(x) @ b["w1"]
            x = x + np.maximum(m, 0) @ b["w2"]
        logits = self._norm(x[-1]) @ self.emb.T                # embedding compartilhada
        return logits, novo_cache


def rodar_teste(n, sem_cache):
    try:
        import numpy as np
    except ImportError:
        sys.exit("Precisa do numpy: pip install numpy")
    print("Montando o modelo de teste (pesos aleatórios)...")
    m = Teste()
    mb = m.params * 4 / 1e6
    print(f"{m.params / 1e6:.0f} M parâmetros em float32 = {mb:.0f} MB de pesos\n")
    rng = np.random.default_rng(1)
    prompt = list(rng.integers(0, m.vocab, n))

    m.forward(prompt[:8], None)                                # aquecimento

    # PREFILL: os n tokens da pergunta numa passada só
    t0 = time.perf_counter()
    logits, cache = m.forward(prompt, None)
    t_pre = time.perf_counter() - t0

    # DECODE: n tokens, um por vez, reaproveitando o KV cache
    tok = int(logits.argmax())
    t0 = time.perf_counter()
    for _ in range(n):
        logits, cache = m.forward([tok], cache)
        tok = int(logits.argmax())
    t_dec = time.perf_counter() - t0

    # DECODE SEM KV CACHE: cada passo reprocessa o texto inteiro
    seq = prompt + [tok]
    t0 = time.perf_counter()
    for _ in range(sem_cache):
        logits, _ = m.forward(seq, None)
        seq.append(int(logits.argmax()))
    t_sem = time.perf_counter() - t0

    medidas = [(f"prefill (ler {n})", t_pre, n),
               (f"decode com KV cache (escrever {n})", t_dec, n),
               (f"decode SEM KV cache ({sem_cache})", t_sem, sem_cache)]
    tabela(medidas, n)
    kv = 2 * len(m.blocos) * m.d * 4                            # K e V, float32
    print(f"KV cache no fim: {kv * (2 * n) / 1e6:.0f} MB ({kv / 1e3:.0f} KB por token)")
    print(f"Cada passo de decode leu os {mb:.0f} MB de pesos para processar 1 token.")
    print(f"Banda efetiva do decode: ~{mb * n / t_dec / 1e3:.1f} GB/s")
    return {"modo": "teste", "params": m.params, "bytes_pesos": m.params * 4,
            "medidas": [{"fase": a, "segundos": b, "tokens": c} for a, b, c in medidas]}


# =====================================================================
# 2. Modelo de verdade com transformers
# =====================================================================
def rodar_hf(nome, n, sem_cache):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        sys.exit("Precisa de torch e transformers: pip install torch transformers")
    print(f"Carregando {nome} (na primeira vez baixa do Hugging Face)...")
    tok = AutoTokenizer.from_pretrained(nome)
    modelo = AutoModelForCausalLM.from_pretrained(nome, torch_dtype=torch.float32)
    modelo.eval()
    params = sum(p.numel() for p in modelo.parameters())
    print(f"{params / 1e6:.0f} M parâmetros, {torch.get_num_threads()} threads de CPU\n")

    ids = tok(TEXTO_BASE * (n // 40 + 2), return_tensors="pt").input_ids[:, :n]

    with torch.inference_mode():
        modelo(ids[:, :8])                                     # aquecimento

        t0 = time.perf_counter()
        out = modelo(ids, use_cache=True)                      # PREFILL
        t_pre = time.perf_counter() - t0

        cache = out.past_key_values
        prox = out.logits[:, -1].argmax(-1, keepdim=True)
        gerados = [prox]
        t0 = time.perf_counter()
        for _ in range(n):                                     # DECODE com KV cache
            out = modelo(prox, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            prox = out.logits[:, -1].argmax(-1, keepdim=True)
            gerados.append(prox)
        t_dec = time.perf_counter() - t0

        seq = torch.cat([ids, gerados[0]], dim=1)
        t0 = time.perf_counter()
        for _ in range(sem_cache):                             # DECODE sem KV cache
            out = modelo(seq, use_cache=False)
            seq = torch.cat([seq, out.logits[:, -1].argmax(-1, keepdim=True)], dim=1)
        t_sem = time.perf_counter() - t0

    texto = tok.decode(torch.cat(gerados[:40], dim=1)[0])
    print("Começo da resposta (greedy):", repr(texto[:160]), "\n")
    medidas = [(f"prefill (ler {n})", t_pre, n),
               (f"decode com KV cache (escrever {n})", t_dec, n),
               (f"decode SEM KV cache ({sem_cache})", t_sem, sem_cache)]
    tabela(medidas, n)
    return {"modo": "hf", "modelo": nome, "params": params,
            "medidas": [{"fase": a, "segundos": b, "tokens": c} for a, b, c in medidas]}


# =====================================================================
# 3. llama.cpp (servidor da oficina de Docker)
# =====================================================================
def rodar_servidor(url, n, repeticoes=3):
    import urllib.request
    url = url.rstrip("/")
    prompt = TEXTO_BASE * (n // 40 + 2)
    corpo = {"prompt": prompt, "n_predict": n, "temperature": 0, "seed": 42,
             "cache_prompt": False, "ignore_eos": True}
    # a 1ª rodada aquece e descobre quantos tokens o texto tem; depois cortamos para n
    pre, dec = [], []
    for i in range(repeticoes + 1):
        req = urllib.request.Request(url + "/completion", data=json.dumps(corpo).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            t = json.load(r)["timings"]
        if i == 0:
            continue
        pre.append((t["prompt_ms"] / 1000, t["prompt_n"]))
        dec.append((t["predicted_ms"] / 1000, t["predicted_n"]))
    sp = sum(a for a, _ in pre) / len(pre); np_ = pre[0][1]
    sd = sum(a for a, _ in dec) / len(dec); nd = dec[0][1]
    medidas = [(f"prefill (ler {np_})", sp, np_), (f"decode (escrever {nd})", sd, nd)]
    tabela(medidas, n)
    print(f"Média de {repeticoes} rodadas, depois de 1 de aquecimento. Tempos medidos pelo llama.cpp.")
    return {"modo": "servidor", "url": url,
            "medidas": [{"fase": a, "segundos": b, "tokens": c} for a, b, c in medidas]}


# =====================================================================
# 4. Sampling: dos logits ao token escolhido
# =====================================================================
# "O gato subiu no ___": logits inventados, mas com a forma típica de uma distribuição real
EXEMPLO = {"telhado": 4.0, "muro": 3.1, "sofá": 2.6, "árvore": 2.3, "carro": 1.4,
           "armário": 1.1, "computador": 0.3, "foguete": -1.0}


def softmax(logits, temperatura=1.0):
    if temperatura <= 0:                                        # greedy: tudo no maior
        top = max(logits, key=logits.get)
        return {k: float(k == top) for k in logits}
    m = max(logits.values())
    e = {k: math.exp((v - m) / temperatura) for k, v in logits.items()}
    s = sum(e.values())
    return {k: v / s for k, v in e.items()}


def filtrar(probs, top_k=0, top_p=1.0):
    """top-k: fica com os k mais prováveis. top-p: fica com os primeiros até somar p."""
    ordem = sorted(probs.items(), key=lambda kv: -kv[1])
    if top_k:
        ordem = ordem[:top_k]
    if top_p < 1.0:
        acum, corte = 0.0, []
        for k, p in ordem:
            corte.append((k, p))
            acum += p
            if acum >= top_p:
                break
        ordem = corte
    s = sum(p for _, p in ordem)
    return {k: p / s for k, p in ordem}


def sortear(probs, rng):
    r, acum = rng.random(), 0.0
    for k, p in probs.items():
        acum += p
        if r <= acum:
            return k
    return k


def demo_sampling(amostras=20):
    rng = random.Random(42)
    print('Próximo token depois de "O gato subiu no"\n')
    configs = [("greedy (temperatura 0)", 0, 0, 1.0), ("temperatura 0.5", 0.5, 0, 1.0),
               ("temperatura 1.0", 1.0, 0, 1.0), ("temperatura 1.5", 1.5, 0, 1.0),
               ("temperatura 1.0 + top-k 3", 1.0, 3, 1.0), ("temperatura 1.0 + top-p 0.9", 1.0, 0, 0.9)]
    for nome, t, k, p in configs:
        probs = filtrar(softmax(EXEMPLO, t), k, p)
        print(nome)
        for palavra in EXEMPLO:
            pr = probs.get(palavra, 0.0)
            print(f"  {palavra:<11}{'#' * round(pr * 50):<50} {pr * 100:5.1f}%")
        sorteio = [sortear(probs, rng) for _ in range(amostras)]
        print(f"  {amostras} sorteios: {', '.join(sorteio)}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf", metavar="MODELO", help="modelo do Hugging Face, ex.: Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--servidor", metavar="URL", help="llama.cpp server, ex.: http://localhost:8080")
    ap.add_argument("--sampling", action="store_true", help="só a demonstração de sampling")
    ap.add_argument("--tokens", type=int, default=500, help="tokens da pergunta e da resposta (500)")
    ap.add_argument("--sem-cache", type=int, default=32, help="tokens gerados sem KV cache (32)")
    ap.add_argument("--json", metavar="ARQUIVO", help="salva as medidas em JSON")
    a = ap.parse_args()

    if a.sampling:
        demo_sampling()
        return
    print(f"Máquina: {platform.processor() or platform.machine()} · Python {platform.python_version()}")
    if a.servidor:
        res = rodar_servidor(a.servidor, a.tokens)
    elif a.hf:
        res = rodar_hf(a.hf, a.tokens, a.sem_cache)
    else:
        res = rodar_teste(a.tokens, a.sem_cache)
    if a.json:
        res["maquina"] = platform.processor() or platform.machine()
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print("Salvo em", a.json)


if __name__ == "__main__":
    main()

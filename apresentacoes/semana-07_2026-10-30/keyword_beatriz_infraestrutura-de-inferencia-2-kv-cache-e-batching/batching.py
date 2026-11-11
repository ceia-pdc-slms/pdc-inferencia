"""
KV cache e batching, medidos na máquina.

Pergunta da semana: por que servir 20 pedidos ao mesmo tempo custa muito menos que 20x servir um?

  python batching.py                       # mede o decode com lotes de 1 a 32 pedidos
                                           # (PyTorch se tiver; se não, numpy)
  python batching.py --memoria             # quanto KV cache cada pedido ocupa e quantos cabem numa GPU
  python batching.py --servidor http://localhost:8080 --pedidos 20
                                           # 20 pedidos de verdade a um servidor (llama.cpp, vLLM, Ollama):
                                           # um de cada vez x todos juntos
"""
import argparse
import json
import math
import platform
import sys
import threading
import time
import urllib.request


def linha(c="-", n=78):
    print(c * n)


# =====================================================================
# 1. Decode em lote com um modelo de teste em numpy
# =====================================================================
class Teste:
    """Um decoder estilo GPT com pesos aleatórios: embedding -> N blocos (atenção + MLP) -> logits.
    d=1024, 10 camadas, MLP 4x: ~128 M parâmetros em float32 = ~510 MB, o mesmo tamanho em bytes
    do Qwen2.5-0.5B Q4_K_M. É o mesmo modelo do gera_texto.py da semana 6, agora com lote.

    Usa PyTorch se estiver instalado (CPU ou GPU) e numpy se não. Com numpy, o OpenBLAS costuma ser
    lento com lotes pequenos (de 2 a ~16): ele copia a matriz de pesos inteira antes de multiplicar,
    e isso esconde o ganho do lote. O PyTorch não tem esse problema."""

    def __init__(self, d=1024, camadas=10, cabecas=16, vocab=2048, seed=0, backend="auto"):
        self.torch = None
        if backend in ("auto", "torch"):
            try:
                import torch
                self.torch = torch
            except ImportError:
                if backend == "torch":
                    raise
        import numpy as np
        self.np = np
        r = np.random.default_rng(seed)
        f = np.float32
        self.d, self.h, self.hd, self.vocab = d, cabecas, d // cabecas, vocab
        s = 1 / math.sqrt(d)
        self.emb = (r.standard_normal((vocab, d)) * 0.02).astype(f)
        self.blocos = [{
            "wqkv": (r.standard_normal((d, 3 * d)) * s).astype(f),
            "wo": (r.standard_normal((d, d)) * s).astype(f),
            "w1": (r.standard_normal((d, 4 * d)) * s).astype(f),
            "w2": (r.standard_normal((4 * d, d)) * s / 2).astype(f),
        } for _ in range(camadas)]
        self.params = self.emb.size + sum(sum(w.size for w in b.values()) for b in self.blocos)
        self.bytes = self.params * 4
        if self.torch:
            t = self.torch
            self.emb = t.from_numpy(self.emb)
            self.blocos = [{k: t.from_numpy(w) for k, w in b.items()} for b in self.blocos]
        self.nome = "PyTorch " + self.torch.__version__ if self.torch else "numpy " + np.__version__

    def _norm(self, x):
        return x / ((x * x).mean(-1, keepdims=True) + 1e-5) ** 0.5

    def zeros(self, forma):
        return self.torch.zeros(forma) if self.torch else self.np.zeros(forma, self.np.float32)

    def novo_cache(self, lote, contexto):
        """O KV cache como estrutura de memória: por camada, um K e um V de forma
        (lote, cabeças, posições, dimensão). Reservado inteiro de uma vez, como faziam os
        servidores antes do PagedAttention."""
        forma = (lote, self.h, contexto, self.hd)
        return [[self.zeros(forma), self.zeros(forma)] for _ in self.blocos]

    def passo(self, tokens, cache, pos):
        """Um passo de decode para o lote inteiro: `tokens` tem um token por pedido.
        Cada matriz de pesos é lida UMA vez e multiplica os B vetores juntos."""
        B, d = len(tokens), self.d
        x = self.emb[tokens]                                            # (B, d)
        for (K, V), b in zip(cache, self.blocos):
            qkv = self._norm(x) @ b["wqkv"]                             # pesos lidos 1x para B tokens
            q, k, v = (qkv[:, i * d:(i + 1) * d].reshape(B, self.h, 1, self.hd) for i in range(3))
            K[:, :, pos:pos + 1], V[:, :, pos:pos + 1] = k, v           # guarda no cache
            Kp, Vp = K[:, :, :pos + 1], V[:, :, :pos + 1]               # cada pedido lê o SEU KV
            a = (q @ Kp.swapaxes(-1, -2)) / math.sqrt(self.hd)
            a = a - a.max(-1, keepdims=True)[0] if self.torch else a - a.max(-1, keepdims=True)
            a = self.torch.exp(a) if self.torch else self.np.exp(a)
            a = a / a.sum(-1, keepdims=True)
            o = (a @ Vp).reshape(B, d)
            x = x + o @ b["wo"]
            hdn = self._norm(x) @ b["w1"]
            x = x + (hdn.clamp(min=0) if self.torch else self.np.maximum(hdn, 0)) @ b["w2"]
        logits = self._norm(x) @ self.emb.T
        return logits.argmax(-1)


def medir_lote(modelo, lote, contexto, passos):
    """Enche o cache até `contexto` e mede `passos` passos de decode com `lote` pedidos."""
    import numpy as np
    cache = modelo.novo_cache(lote, contexto + passos + 1)
    r = np.random.default_rng(1)
    for K, V in cache:                                     # finge um prompt de `contexto` tokens já lido
        for M in (K, V):
            ruido = r.standard_normal((lote, modelo.h, contexto, modelo.hd), dtype=np.float32) * 0.1
            M[:, :, :contexto] = modelo.torch.from_numpy(ruido) if modelo.torch else ruido
    tok = r.integers(0, modelo.vocab, lote)
    if modelo.torch:
        tok = modelo.torch.from_numpy(tok)
    tok = modelo.passo(tok, cache, contexto)               # aquecimento
    t0 = time.perf_counter()
    for i in range(passos):
        tok = modelo.passo(tok, cache, contexto + 1 + i)
    return (time.perf_counter() - t0) / passos


def modo_teste(a):
    m = Teste(backend=a.backend)
    if m.torch:
        m.torch.set_grad_enabled(False)
    lotes = [int(x) for x in a.lotes.split(",")]
    print(f"Modelo de teste: {m.params / 1e6:.0f} M parâmetros, {m.bytes / 1e6:.0f} MB em float32 · {m.nome}")
    if not m.torch:
        print("Aviso: sem PyTorch. Com numpy, lotes pequenos podem sair lentos por causa do OpenBLAS.")
    print(f"Contexto de cada pedido: {a.contexto} tokens · {a.passos} passos de decode por medida")
    kv_tok = 2 * len(m.blocos) * m.d * 4
    print(f"KV cache: {kv_tok / 1024:.0f} KB por token por pedido (2 x {len(m.blocos)} camadas x {m.d} x 4 bytes)")
    linha()
    print(f"{'lote':>5}{'passo':>11}{'tok/s total':>14}{'tok/s cada':>13}{'custo/token':>14}{'KV no lote':>14}")
    linha()
    res = []
    base = None
    for B in lotes:
        s = medir_lote(m, B, a.contexto, a.passos)
        base = base or s
        kv = kv_tok * B * a.contexto
        res.append({"lote": B, "segundos_por_passo": s, "tok_s_total": B / s, "tok_s_por_pedido": 1 / s})
        print(f"{B:>5}{s * 1000:>9.1f} ms{B / s:>14.1f}{1 / s:>13.1f}{s / B / base:>13.2f}x{kv / 1e6:>11.0f} MB")
    linha()
    um = next((r for r in res if r["lote"] == 1), None)
    vinte = next((r for r in res if r["lote"] == 20), None)
    if um and vinte:
        sozinho = 20 * um["segundos_por_passo"]
        juntos = vinte["segundos_por_passo"]
        print(f"Um token para cada um de 20 pedidos: um de cada vez {sozinho * 1000:.0f} ms, "
              f"todos juntos {juntos * 1000:.0f} ms.")
        print(f"Servir 20 juntos custou {juntos / sozinho * 100:.0f}% de servir 20 um de cada vez "
              f"({sozinho / juntos:.1f}x mais barato).")
        print(f"Cada pedido ficou {um['tok_s_por_pedido'] / vinte['tok_s_por_pedido']:.1f}x mais lento: "
              f"é o preço, em latência, do lote.")
    return {"modo": "teste", "backend": m.nome, "params": m.params, "contexto": a.contexto, "medidas": res}


# =====================================================================
# 2. KV cache como estrutura de memória
# =====================================================================
# config.json públicos no Hugging Face. kvh = cabeças de K/V (GQA), hd = dimensão da cabeça.
MODELOS = [
    ("Qwen2.5-0.5B", 0.494e9, 24, 2, 64),
    ("Llama-3.2-1B", 1.236e9, 16, 8, 64),
    ("Qwen2.5-3B", 3.086e9, 36, 2, 128),
    ("Llama-3.1-8B", 8.030e9, 32, 8, 128),
]


def kv_por_token(camadas, kvh, hd, bytes_=2):
    return 2 * camadas * kvh * hd * bytes_          # K e V, todas as camadas, FP16


def modo_memoria(a):
    gb = a.gpu
    util = 0.9                                       # como o --gpu-memory-utilization do vLLM
    print(f"GPU de {gb} GB, {util:.0%} para o servidor · pesos em FP16 · KV cache em FP16")
    print(f"Cada pedido pode ir até {a.contexto} tokens; em média usa {a.medio}. Blocos de {a.bloco} tokens.")
    linha()
    print(f"{'modelo':<14}{'KV/token':>10}{'KV/pedido':>12}{'livre p/ KV':>13}"
          f"{'reserva contígua':>18}{'em blocos':>11}")
    linha()
    for nome, p, cam, kvh, hd in MODELOS:
        kv = kv_por_token(cam, kvh, hd)
        pesos = p * 2
        livre = gb * 1e9 * util - pesos
        if livre <= 0:
            print(f"{nome:<14}{kv / 1024:>8.0f} KB  o modelo nem cabe")
            continue
        contig = int(livre // (kv * a.contexto))
        pag = int(livre // (kv * math.ceil(a.medio / a.bloco) * a.bloco))
        print(f"{nome:<14}{kv / 1024:>8.0f} KB{kv * a.contexto / 2**20:>9.0f} MiB{livre / 1e9:>10.1f} GB"
              f"{contig:>12} pedidos{pag:>8} pedidos")
    linha()
    print("Reserva contígua: cada pedido reserva o contexto máximo, mesmo usando menos.")
    print("Em blocos (PagedAttention): cada pedido ocupa só os blocos que preencheu.")
    return {"modo": "memoria", "gpu_gb": gb}


# =====================================================================
# 3. Servidor de verdade: N pedidos, um de cada vez x todos juntos
# =====================================================================
PERGUNTAS = [
    "Explique em poucas frases o que é um KV cache.",
    "Liste três usos de um modelo de linguagem pequeno.",
    "O que é quantização de pesos?",
    "Por que a memória limita a velocidade de um modelo?",
    "Escreva um haicai sobre servidores.",
]


def pedir(url, modelo, prompt, tokens):
    corpo = json.dumps({"model": modelo, "prompt": prompt, "max_tokens": tokens,
                        "temperature": 0, "seed": 42}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/completions", corpo, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    return d.get("usage", {}).get("completion_tokens", tokens)


def modo_servidor(a):
    n, tokens = a.pedidos, a.tokens
    prompts = [PERGUNTAS[i % len(PERGUNTAS)] for i in range(n)]
    print(f"Servidor {a.servidor} · {n} pedidos de até {tokens} tokens · temperatura 0")
    pedir(a.servidor, a.modelo, "Olá", 4)                         # aquecimento

    t0 = time.perf_counter()
    gerados_seq = sum(pedir(a.servidor, a.modelo, p, tokens) for p in prompts)
    seq = time.perf_counter() - t0

    gerados = [0] * n
    def um(i):
        gerados[i] = pedir(a.servidor, a.modelo, prompts[i], tokens)
    ths = [threading.Thread(target=um, args=(i,)) for i in range(n)]
    t0 = time.perf_counter()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    par = time.perf_counter() - t0
    linha()
    print(f"{'':<28}{'tempo':>10}{'tokens':>10}{'tok/s total':>14}")
    linha()
    print(f"{'um de cada vez':<28}{seq:>8.1f} s{gerados_seq:>10}{gerados_seq / seq:>14.1f}")
    print(f"{'todos juntos':<28}{par:>8.1f} s{sum(gerados):>10}{sum(gerados) / par:>14.1f}")
    linha()
    print(f"Todos juntos levou {par / seq * 100:.0f}% do tempo ({seq / par:.1f}x mais rápido).")
    print("Se deu perto de 1x, o servidor está atendendo um por vez. No llama.cpp, suba com")
    print(f"--parallel {n} e um -c grande o bastante: o contexto total é dividido entre as vagas.")
    return {"modo": "servidor", "url": a.servidor, "pedidos": n, "tokens": tokens,
            "sequencial_s": seq, "paralelo_s": par, "tokens_seq": gerados_seq, "tokens_par": sum(gerados)}


def main():
    ap = argparse.ArgumentParser(description="KV cache e batching na sua máquina")
    ap.add_argument("--lotes", default="1,2,4,8,16,20,32", help="tamanhos de lote a medir")
    ap.add_argument("--contexto", type=int, default=256, help="tokens que cada pedido já tem no cache")
    ap.add_argument("--passos", type=int, default=8, help="passos de decode por medida")
    ap.add_argument("--backend", choices=["auto", "torch", "numpy"], default="auto",
                    help="auto usa PyTorch se estiver instalado")
    ap.add_argument("--memoria", action="store_true", help="tabela de KV cache por modelo")
    ap.add_argument("--gpu", type=float, default=24, help="memória da GPU em GB (com --memoria)")
    ap.add_argument("--medio", type=int, default=1000, help="tokens que um pedido usa em média (com --memoria)")
    ap.add_argument("--bloco", type=int, default=16, help="tokens por bloco (com --memoria)")
    ap.add_argument("--servidor", help="URL de um servidor compatível com OpenAI, ex.: http://localhost:8080")
    ap.add_argument("--modelo", default="modelo", help="nome do modelo no servidor (vLLM e Ollama exigem)")
    ap.add_argument("--pedidos", type=int, default=20)
    ap.add_argument("--tokens", type=int, default=64, help="tokens por resposta (com --servidor)")
    ap.add_argument("--json", help="salva as medidas neste arquivo")
    a = ap.parse_args()

    print(f"{platform.node()} · {platform.processor() or platform.machine()} · Python {platform.python_version()}")
    if a.memoria:
        if a.contexto == 256:
            a.contexto = 4096
        r = modo_memoria(a)
    elif a.servidor:
        r = modo_servidor(a)
    else:
        r = modo_teste(a)
    if a.json:
        r["maquina"] = platform.node()
        r["data"] = time.strftime("%Y-%m-%d %H:%M")
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)
        print(f"Salvo em {a.json}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

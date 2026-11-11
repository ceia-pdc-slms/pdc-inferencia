"""
Quantização a fundo: FP16 -> FP8 -> INT4 -> FP4, o que quebra e o que acelera.

Pergunta da semana: o tok/s aumentou na mesma proporção em que os bytes por token caíram?
Se não, o que consumiu o ganho?

  python quantizacao_a_fundo.py                 # 1. o erro de cada formato, nos pesos reais do Qwen2.5-0.5B
                                                #    (ou em pesos sintéticos, se o modelo não estiver no cache)
  python quantizacao_a_fundo.py --exemplo       # 0. a conta na mão: 8 números, um outlier, INT4 por tensor x por grupo
  python quantizacao_a_fundo.py --qualidade     # 2. perplexidade e respostas do modelo com cada formato (precisa de transformers)
  python quantizacao_a_fundo.py --decode --lotes 1,64
                                                # 3. um passo de decode inteiro (24 camadas + lm_head) no ONNX Runtime,
                                                #    por formato: bytes x tempo, no decode e no prefill
  python quantizacao_a_fundo.py --ponta-a-ponta # 3b. o modelo de verdade: quanto do token é multiplicação por pesos
  python quantizacao_a_fundo.py --velocidade    # 3c. os kernels do PyTorch (FP16, FP8, INT8, INT4): bytes x tempo
  python quantizacao_a_fundo.py --gguf pasta/com/ggufs
                                                # 4. dentro dos GGUF: em que formato cada tensor fica, bytes por token
  python quantizacao_a_fundo.py --llama-bench caminho/llama-bench --gguf pasta/com/ggufs
                                                # 4b. tok/s de verdade no llama.cpp x bytes

Precisa de: numpy e torch (>= 2.6, CPU serve). --qualidade e --ponta-a-ponta também precisam de transformers;
--decode precisa de onnx e onnxruntime. Os modos que carregam o modelo inteiro conferem a RAM livre antes.
O modelo é lido do cache do Hugging Face (Qwen/Qwen2.5-0.5B-Instruct); --modelo aponta outro caminho.
"""
import argparse
import glob
import json
import math
import os
import platform
import subprocess
import sys
import time

import numpy as np

try:
    import torch
except ImportError:  # o modo --exemplo e o modo --llama-bench funcionam sem torch
    torch = None


def linha(c="-", n=86):
    print(c * n)


# =====================================================================
# 0. Os formatos: como cada um arredonda um número
# =====================================================================
# Cada formato recebe um tensor float32 e devolve (tensor "desquantizado", bits por peso).
# Os bits por peso contam também as escalas guardadas junto: é o tamanho real no disco e na memória.

FP4_GRADE = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)  # E2M1: os 8 valores positivos


def _fp4_arredonda(x):
    """Arredonda para o valor E2M1 mais próximo (satura em ±6). x é torch.Tensor."""
    g = torch.from_numpy(FP4_GRADE)
    a = x.abs().clamp(max=6.0)
    i = torch.bucketize(a, (g[1:] + g[:-1]) / 2)  # índice do vizinho mais perto
    return torch.sign(x) * g[i]


def _grupos(w, g):
    """[N, K] -> [N, K/g, g]. Completa K com zeros se não for múltiplo de g."""
    n, k = w.shape
    pad = (-k) % g
    if pad:
        w = torch.nn.functional.pad(w, (0, pad))
    return w.reshape(n, -1, g), k


def _volta(wg, k):
    n = wg.shape[0]
    return wg.reshape(n, -1)[:, :k]


def q_fp16(w):
    return w.half().float(), 16.0


def q_bf16(w):
    return w.bfloat16().float(), 16.0


def q_fp8_e4m3(w):
    # uma escala por tensor: o maior |peso| vira 448, o maior número do E4M3
    s = w.abs().max().clamp(min=1e-12) / 448.0
    return (w / s).to(torch.float8_e4m3fn).float() * s, 8.0


def q_fp8_e5m2(w):
    s = w.abs().max().clamp(min=1e-12) / 57344.0
    return (w / s).to(torch.float8_e5m2).float() * s, 8.0


def q_int8_canal(w):
    # simétrico, uma escala FP16 por linha (canal de saída)
    s = w.abs().amax(1, keepdim=True).clamp(min=1e-12) / 127
    return (w / s).round().clamp(-127, 127) * s, 8 + 16 / w.shape[1]


def q_int4_tensor(w):
    # simétrico, UMA escala para o tensor inteiro: o maior peso manda em todos
    s = w.abs().max().clamp(min=1e-12) / 7
    return (w / s).round().clamp(-8, 7) * s, 4.0


def q_int4_canal(w):
    s = w.abs().amax(1, keepdim=True).clamp(min=1e-12) / 7
    return (w / s).round().clamp(-8, 7) * s, 4 + 16 / w.shape[1]


def q_int4_g32(w):
    # como o Q4_0 do llama.cpp: blocos de 32 pesos, uma escala FP16 por bloco = 4,5 bits por peso
    wg, k = _grupos(w, 32)
    s = wg.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 7
    return _volta((wg / s).round().clamp(-8, 7) * s, k), 4 + 16 / 32


def q_fp4_tensor(w):
    s = w.abs().max().clamp(min=1e-12) / 6
    return _fp4_arredonda(w / s) * s, 4.0


def q_mxfp4(w):
    # OCP MX: blocos de 32, escala E8M0 (uma potência de 2, 8 bits) = 4,25 bits por peso
    wg, k = _grupos(w, 32)
    amax = wg.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    s = torch.pow(2.0, torch.floor(torch.log2(amax)) - 2)  # 2 = maior expoente do E2M1 (6 = 1,5 x 2^2)
    return _volta(_fp4_arredonda(wg / s) * s, k), 4 + 8 / 32


def q_nvfp4(w):
    # NVIDIA NVFP4: blocos de 16, escala em FP8 E4M3 + uma escala FP32 por tensor = 4,5 bits por peso
    wg, k = _grupos(w, 16)
    glob_s = w.abs().max().clamp(min=1e-12) / (6 * 448)
    amax = wg.abs().amax(-1, keepdim=True)
    s = (amax / 6 / glob_s).to(torch.float8_e4m3fn).float().clamp(min=2 ** -9) * glob_s
    return _volta(_fp4_arredonda(wg / s) * s, k), 4 + 8 / 16


FORMATOS = [
    # nome,               função,         família
    ("FP16",              q_fp16,         "float"),
    ("BF16",              q_bf16,         "float"),
    ("FP8 E4M3",          q_fp8_e4m3,     "float"),
    ("FP8 E5M2",          q_fp8_e5m2,     "float"),
    ("INT8 por canal",    q_int8_canal,   "int"),
    ("INT4 por tensor",   q_int4_tensor,  "int"),
    ("INT4 por canal",    q_int4_canal,   "int"),
    ("INT4 grupo 32",     q_int4_g32,     "int"),
    ("FP4 por tensor",    q_fp4_tensor,   "float"),
    ("MXFP4 (bloco 32)",  q_mxfp4,        "float"),
    ("NVFP4 (bloco 16)",  q_nvfp4,        "float"),
]
POR_NOME = {n: f for n, f, _ in FORMATOS}


def erro(w, wq):
    """Erro relativo (%) e SQNR (dB): quanto do sinal sobra depois de arredondar."""
    e = (w - wq).norm() / w.norm()
    return 100 * float(e), float(-20 * math.log10(max(float(e), 1e-12)))


# =====================================================================
# 0b. A conta na mão
# =====================================================================
def exemplo():
    v = np.array([0.12, -0.31, 0.05, 0.27, -0.08, 0.19, -0.22, 8.0], dtype=np.float32)
    print("Oito pesos. O último é um outlier (acontece em todo LLM).")
    print("  pesos:            ", " ".join(f"{x:6.2f}" for x in v))
    linha()
    s = np.abs(v).max() / 7
    q = np.clip(np.round(v / s), -8, 7)
    print(f"INT4 por tensor: escala = maior |peso| / 7 = {np.abs(v).max():.2f} / 7 = {s:.3f}")
    print("  inteiros:         ", " ".join(f"{int(x):6d}" for x in q))
    print("  de volta:         ", " ".join(f"{x:6.2f}" for x in q * s))
    print("  -> os 7 pesos pequenos viraram 0. Só o outlier sobreviveu.")
    linha()
    a, b = v[:4], v[4:]
    sa, sb = np.abs(a).max() / 7, np.abs(b).max() / 7
    qa, qb = np.clip(np.round(a / sa), -8, 7), np.clip(np.round(b / sb), -8, 7)
    print(f"INT4 em grupos de 4: cada grupo tem a sua escala ({sa:.3f} e {sb:.3f})")
    print("  inteiros:         ", " ".join(f"{int(x):6d}" for x in np.r_[qa, qb]))
    print("  de volta:         ", " ".join(f"{x:6.2f}" for x in np.r_[qa * sa, qb * sb]))
    print("  -> o grupo sem outlier volta quase igual. O preço: uma escala a mais (16 bits) por grupo.")
    linha()
    g = FP4_GRADE
    print("FP4 (E2M1) só tem estes valores positivos:", " ".join(f"{x:g}" for x in g))
    print("  degraus de 0,5 perto do zero, de 2 perto do topo: erra pouco nos pequenos, muito nos grandes.")
    print("INT4 tem 16 degraus iguais: -8 ... 7.")


# =====================================================================
# 1. O erro de cada formato nos pesos reais
# =====================================================================
def acha_modelo(caminho=None):
    if caminho:
        return caminho
    raizes = [os.environ.get("HF_HUB_CACHE"), os.environ.get("HF_HOME") and os.path.join(os.environ["HF_HOME"], "hub"),
              os.path.expanduser("~/.cache/huggingface/hub"), "D:/hf_cache/hub"]
    for r in filter(None, raizes):
        achados = glob.glob(os.path.join(r, "models--Qwen--Qwen2.5-0.5B-Instruct", "snapshots", "*", "model.safetensors"))
        if achados:
            return os.path.dirname(achados[0])
    return None


def carrega_pesos(pasta):
    from safetensors.torch import load_file
    return {k: v.float() for k, v in load_file(os.path.join(pasta, "model.safetensors")).items()}


def pesos_sinteticos():
    """Se o modelo não estiver no disco: matrizes gaussianas com alguns canais outliers, como num LLM."""
    g = torch.Generator().manual_seed(0)
    out = {}
    for i in range(4):
        w = torch.randn(896, 896, generator=g) * 0.02
        w[:, torch.randint(0, 896, (4,), generator=g)] *= 30  # canais de entrada com valores grandes
        out[f"sintetico.{i}.weight"] = w
    return out


def formatos(pasta):
    if pasta:
        pesos = carrega_pesos(pasta)
        print(f"Pesos de verdade: {pasta}")
        alvo = {k: v for k, v in pesos.items() if k.endswith(".weight") and v.ndim == 2 and "layers." in k}
    else:
        print("Modelo não encontrado no cache: usando pesos sintéticos com outliers (passe --modelo para usar um real).")
        alvo = pesos_sinteticos()
    n = sum(v.numel() for v in alvo.values())
    print(f"{len(alvo)} matrizes das camadas, {n / 1e6:.0f} M pesos. Erro somado sobre todas.")
    amax = max(float(v.abs().max()) for v in alvo.values())
    rms = math.sqrt(sum(float((v ** 2).sum()) for v in alvo.values()) / n)
    print(f"Maior |peso| = {amax:.3f}; peso típico (RMS) = {rms:.4f}; razão = {amax / rms:.0f}x")
    linha()
    print(f"{'formato':<18}{'bits/peso':>10}{'MB p/ 358 M':>13}{'erro rel.':>11}{'SQNR':>9}   o que acontece")
    linha()
    res = []
    for nome, f, _ in FORMATOS:
        num = den = 0.0
        bits = 0.0
        for w in alvo.values():
            wq, b = f(w)
            num += float(((w - wq) ** 2).sum())
            den += float((w ** 2).sum())
            bits += b * w.numel()
        e = math.sqrt(num / den)
        bpp = bits / n
        sqnr = -20 * math.log10(max(e, 1e-12))
        nota = ("praticamente sem perda" if e < 0.01 else "perda pequena" if e < 0.05 else
                "perda visível" if e < 0.15 else "quebra: os pesos pequenos somem")
        print(f"{nome:<18}{bpp:>10.2f}{bpp * 357.8e6 / 8 / 1e6:>13.0f}{100 * e:>10.2f}%{sqnr:>8.1f} dB   {nota}")
        res.append({"formato": nome, "bits_por_peso": round(bpp, 3), "erro_rel_pct": round(100 * e, 3), "sqnr_db": round(sqnr, 2)})
    linha()
    print("SQNR = 20 log10(|sinal| / |erro|). Cada bit a mais vale ~6 dB.")
    print("358 M = os pesos das camadas lineares do Qwen2.5-0.5B (sem o embedding, que tem mais 136 M).")
    return res


# =====================================================================
# 2. Qualidade: o modelo inteiro com cada formato
# =====================================================================
TEXTO = (
    "A quantização troca a precisão dos números de um modelo por espaço e velocidade. Um peso guardado em "
    "dezesseis bits ocupa dois bytes; em quatro bits, meio byte. Para gerar cada token, o modelo precisa ler "
    "todos os pesos da memória, então um arquivo quatro vezes menor deveria, em princípio, gerar texto quatro "
    "vezes mais rápido. Na prática o ganho é menor. Parte da conta não encolhe: a tabela de vocabulário costuma "
    "ficar em oito bits, as anotações da atenção continuam em dezesseis, e o processador ainda precisa "
    "desempacotar os números pequenos antes de multiplicar. Quando o desempacotamento é feito por um código "
    "bem escrito, quase não custa nada. Quando não é, ele consome tudo o que a memória economizou. "
    "O Brasil tem vinte e seis estados e um distrito federal. A capital é Brasília, inaugurada em 1960. "
    "O rio Amazonas é o maior do mundo em volume de água, e a floresta que leva o seu nome cobre boa parte "
    "do norte do país. Em Goiás, o cerrado domina a paisagem, com árvores baixas, troncos tortos e raízes "
    "profundas que buscam água durante a seca. "
    "def soma_quadrados(lista):\n    total = 0\n    for x in lista:\n        total += x * x\n    return total\n"
    "Quantization reduces the number of bits used to store each weight. Large language models contain a few "
    "outlier values that are much larger than the rest, and a single scale per tensor wastes most of the "
    "available levels on them. Splitting the tensor into small groups, each with its own scale, keeps the error low."
)
PERGUNTAS = ["A capital do Brasil é", "Quanto é 12 vezes 12? Resposta:", "def fatorial(n):"]


def qualidade(pasta, nomes, tudo=False):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        sys.exit("O modo --qualidade precisa de transformers: pip install transformers")
    if not pasta:
        sys.exit("Modelo não encontrado. Baixe com: huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct  (ou passe --modelo)")
    livre = memoria_livre_gb()
    if livre is not None and livre < 3.5:
        sys.exit(f"Só {livre:.1f} GB de RAM livre; este modo precisa de ~3,5 GB (o modelo em FP32 tem 2 GB). Feche programas e tente de novo.")
    import gc
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(pasta)
    ids = tok(TEXTO, return_tensors="pt").input_ids
    print(f"Perplexidade sobre um texto fixo de {ids.shape[1]} tokens (português, código e inglês).")
    print("Perplexidade = em média, entre quantos tokens o modelo fica em dúvida. Menor é melhor.")
    print("Respostas: decodificação gulosa (sempre o token mais provável), 12 tokens.")
    linha()
    res = []
    original = None
    for nome in ["FP32 (original)"] + nomes:
        m = AutoModelForCausalLM.from_pretrained(pasta, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        m.eval()
        if nome != "FP32 (original)":
            f = POR_NOME[nome]
            with torch.no_grad():
                for k, mod in m.named_modules():
                    if isinstance(mod, torch.nn.Linear) and (tudo or "layers." in k):
                        mod.weight.copy_(f(mod.weight.data)[0])
                if tudo:  # o embedding é o mesmo tensor do lm_head no Qwen2.5-0.5B
                    e = m.get_input_embeddings().weight
                    e.copy_(f(e.data)[0])
        with torch.no_grad():
            logits = m(ids).logits[0, :-1]
            nll = torch.nn.functional.cross_entropy(logits, ids[0, 1:])
            ppl = math.exp(float(nll))
            respostas = []
            for p in PERGUNTAS:
                x = tok(p, return_tensors="pt").input_ids
                y = m.generate(x, max_new_tokens=12, do_sample=False, pad_token_id=tok.eos_token_id)
                respostas.append(tok.decode(y[0, x.shape[1]:], skip_special_tokens=True).replace("\n", "⏎"))
        if original is None:
            original = ppl
        print(f"{nome:<18} perplexidade {ppl:8.2f}  ({100 * (ppl / original - 1):+6.1f}%)")
        for p, r in zip(PERGUNTAS, respostas):
            print(f"    {p!r:<36} -> {r[:60]!r}")
        res.append({"formato": nome, "perplexidade": round(ppl, 3), "respostas": respostas})
        del m, logits
        gc.collect()
    linha()
    print("Só os pesos das camadas foram arredondados" + (" (e o embedding/lm_head)." if tudo else "; o embedding ficou em FP32 (use --tudo)."))
    print("Ativações e KV cache continuam em FP32 aqui. Num servidor, eles também podem ser quantizados.")
    return res


# =====================================================================
# 3. Velocidade: as camadas lineares de um passo de decode, com cada kernel
# =====================================================================
# Formas das matrizes do Qwen2.5-0.5B (24 camadas): q, k, v, o, gate, up, down
FORMAS = [(896, 896), (128, 896), (128, 896), (896, 896), (4864, 896), (4864, 896), (896, 4864)]
CAMADAS = 24


def _cronometra(f, reps):
    for _ in range(3):
        f()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def _prepara(kernel, w):
    """Devolve (função que faz y = x @ w.T, bytes lidos dos pesos)."""
    n, k = w.shape
    F = torch.nn.functional
    ops = torch.ops.aten
    if kernel == "FP32":
        return (lambda x: F.linear(x, w)), w.numel() * 4
    if kernel in ("BF16", "FP16"):
        dt = torch.bfloat16 if kernel == "BF16" else torch.float16
        wd = w.to(dt)
        return (lambda x: F.linear(x.to(dt), wd)), wd.numel() * 2
    if kernel == "FP8 (convertido a cada passo)":
        w8 = w.to(torch.float8_e4m3fn)  # sem kernel FP8 na CPU: guarda em 1 byte, converte para FP32 e multiplica
        return (lambda x: F.linear(x, w8.float())), w8.numel()
    if kernel == "INT8 (kernel int8pack)":
        s = w.abs().amax(1).clamp(min=1e-12) / 127
        q = (w / s[:, None]).round().clamp(-127, 127).to(torch.int8)
        return (lambda x: ops._weight_int8pack_mm(x, q, s)), q.numel() + s.numel() * 4
    if kernel.startswith("INT4"):
        g = 32
        wg = w.reshape(n, k // g, g)
        mn, mx = wg.amin(-1), wg.amax(-1)
        sc = (mx - mn).clamp(min=1e-6) / 15
        zr = mn + sc * 8
        qi = ((wg - mn[..., None]) / sc[..., None]).round().clamp(0, 15).to(torch.int32).reshape(n, k)
        packed = ops._convert_weight_to_int4pack_for_cpu(qi, 1)
        sz = torch.stack([sc.T, zr.T], -1).contiguous()
        dt = torch.bfloat16 if "bf16" in kernel else torch.float32
        szd = sz.to(dt)
        return (lambda x: ops._weight_int4pack_mm_for_cpu(x.to(dt), packed, g, szd)), packed.numel() + sz.numel() * 2
    if kernel == "INT4 desempacotado em Python":
        # o jeito ingênuo: guardar 4 bits, mas reconstruir a matriz FP32 inteira antes de multiplicar
        s = w.abs().amax(1, keepdim=True) / 7
        q = ((w / s).round().clamp(-8, 7) + 8).to(torch.uint8)
        packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()

        def f(x):
            lo = (packed & 15).float() - 8
            hi = (packed >> 4).float() - 8
            wd = torch.stack([lo, hi], -1).reshape(n, k) * s
            return F.linear(x, wd)
        return f, packed.numel() + s.numel() * 2
    raise ValueError(kernel)


KERNELS = ["FP32", "BF16", "FP16", "FP8 (convertido a cada passo)", "INT8 (kernel int8pack)",
           "INT4 (kernel int4pack, fp32)", "INT4 (kernel int4pack, bf16)", "INT4 desempacotado em Python"]


def banda_leitura(mb=512, reps=5):
    """GB/s de leitura da memória: somar um tensor grande que não cabe no cache."""
    a = torch.ones(mb * 2 ** 20 // 4, dtype=torch.float32)
    t = _cronometra(lambda: a.sum(), reps)
    return a.numel() * 4 / t / 1e9


def velocidade(lotes, reps, kernels):
    if torch is None:
        sys.exit("O modo --velocidade precisa de torch.")
    g = torch.Generator().manual_seed(0)
    mats = [torch.randn(n, k, generator=g) * 0.02 for (n, k) in FORMAS]
    bw = banda_leitura()
    print(f"CPU: {platform.processor() or platform.machine()} · {torch.get_num_threads()} threads · torch {torch.__version__}")
    print(f"Banda de leitura medida (somar 512 MB): {bw:.1f} GB/s")
    print(f"Um passo de decode = {CAMADAS} camadas x 7 matrizes do Qwen2.5-0.5B ({sum(n * k for n, k in FORMAS) * CAMADAS / 1e6:.0f} M pesos).")
    print("As 7 matrizes de cada camada são distintas; as 24 camadas reusam as mesmas 7 (o tempo é por camada x 24).")
    res = []
    for m in lotes:
        linha("=")
        print(f"Lote de {m} token(s) por passo" + ("  (decode: 1 token por vez)" if m == 1 else "  (prefill: o prompt inteiro de uma vez)"))
        linha()
        print(f"{'kernel':<32}{'MB lidos':>9}{'ms/passo':>10}{'bytes vs FP32':>15}{'tempo vs FP32':>15}{'GB/s efetivo':>14}")
        linha()
        base_b = base_t = None
        for kname in kernels:
            try:
                prep = [_prepara(kname, w) for w in mats]
            except Exception as e:  # kernel ausente nesta versão do torch
                print(f"{kname:<32}  indisponível: {str(e)[:50]}")
                continue
            x = [torch.randn(m, w.shape[1], generator=g) for w in mats]

            def passo():
                for (f, _), xi in zip(prep, x):
                    f(xi)
            t = _cronometra(passo, reps) * CAMADAS
            b = sum(bb for _, bb in prep) * CAMADAS
            if base_b is None:
                base_b, base_t = b, t
            print(f"{kname:<32}{b / 1e6:>9.0f}{t * 1e3:>10.1f}{b / base_b:>14.2f}x{t / base_t:>14.2f}x{b / t / 1e9:>14.1f}")
            res.append({"lote": m, "kernel": kname, "mb": round(b / 1e6, 1), "ms": round(t * 1e3, 2)})
        linha()
        print(f"Se só os bytes importassem, o tempo cairia junto com os bytes e o GB/s efetivo ficaria perto de {bw:.0f}.")
    return {"banda_gbs": round(bw, 2), "medidas": res}


# =====================================================================
# 3b. Um passo de decode inteiro no ONNX Runtime: a medida que responde a pergunta
# =====================================================================
# O grafo tem todas as multiplicações por pesos de um passo do Qwen2.5-0.5B:
# 24 camadas (q, k, v, o, gate, up, down), cada uma com os seus próprios pesos, e o lm_head
# (896 x 151.936, o mesmo tensor do embedding). Fica de fora só a atenção sobre o KV cache,
# que com contexto curto lê ~1% dos bytes do passo.
# Kernels: MatMul (FP32, FP16) e MatMulNBits (INT8/INT4/INT2 em blocos de 32, o mesmo desenho do Q4_0).
#   accuracy_level=4: o kernel converte a ativação para INT8 e multiplica em inteiro (rápido)
#   accuracy_level=0: o kernel desempacota os pesos para FP32 e multiplica em FP32 (lento)
D_MODELO, D_MLP, D_KV, VOCAB = 896, 4864, 128, 151936

CONFIGS = {
    # nome: (formato das camadas, formato do lm_head)
    "FP32":                      ("fp32", "fp32"),
    "FP16":                      ("fp16", "fp16"),
    "INT8":                      ("int8", "int8"),
    "INT4 + lm_head INT8":       ("int4", "int8"),
    "INT4 tudo":                 ("int4", "int4"),
    "INT4 tudo, sem conta int":  ("int4_a0", "int4_a0"),
}
CAMADAS_MEDIDAS = 6  # as 24 camadas são iguais: mede 6 (pesos distintos) e multiplica por 4. Cabe em ~2 GB de RAM


def _pack_nbits(w, bits, bloco=32):
    """[N, K] float32 -> (blob uint8 [N, K/bloco, bloco*bits/8], escalas float32 [N*K/bloco]). Simétrico, zero = 2^(bits-1)."""
    n, k = w.shape
    wg = w.reshape(n, k // bloco, bloco)
    qmax = 2 ** (bits - 1) - 1
    s = np.abs(wg).max(-1) / qmax
    s[s == 0] = 1
    q = (np.clip(np.round(wg / s[..., None]), -qmax - 1, qmax) + 2 ** (bits - 1)).astype(np.uint8)
    por_byte = 8 // bits
    blob = np.zeros((n, k // bloco, bloco // por_byte), dtype=np.uint8)
    for j in range(por_byte):
        blob |= (q[..., j::por_byte] << (bits * j)).astype(np.uint8)
    return blob, s.astype(np.float32).reshape(-1)


class _Grafo:
    """Monta o grafo ONNX matriz por matriz. Cada peso é sorteado, convertido para o formato e descartado:
    só a versão no formato final fica na memória."""

    def __init__(self, seed=0):
        from onnx import helper, numpy_helper, TensorProto
        self.H, self.nh, self.T = helper, numpy_helper, TensorProto
        self.nos, self.inits, self.saidas, self.bytes, self.i = [], [], [], 0, 0
        self.r = np.random.default_rng(seed)

    def _nome(self, p):
        self.i += 1
        return f"{p}{self.i}"

    def linear(self, x, n, k, fmt):
        """y = x @ W.T, com W [n, k] sorteado e guardado no formato pedido."""
        H, nh = self.H, self.nh
        w = self.r.standard_normal((n, k), dtype=np.float32)
        w *= 1 / math.sqrt(k)
        y = self._nome("y")
        if fmt in ("fp32", "fp16"):
            W = self._nome("W")
            wt = np.ascontiguousarray(w.T if fmt == "fp32" else w.T.astype(np.float16))
            del w
            self.inits.append(nh.from_array(wt, W))
            self.bytes += wt.nbytes
            del wt
            if fmt == "fp32":
                self.nos.append(H.make_node("MatMul", [x, W], [y]))
            else:
                x16, y16 = self._nome("x16_"), self._nome("y16_")
                self.nos += [H.make_node("Cast", [x], [x16], to=self.T.FLOAT16),
                             H.make_node("MatMul", [x16, W], [y16]),
                             H.make_node("Cast", [y16], [y], to=self.T.FLOAT)]
        else:
            bits = {"int8": 8, "int4": 4, "int4_a0": 4, "int2": 2}[fmt]
            acc = 0 if fmt == "int4_a0" else 4
            blob, s = _pack_nbits(w, bits)
            del w
            B, S = self._nome("B"), self._nome("S")
            self.inits += [nh.from_array(blob, B), nh.from_array(s, S)]
            self.nos.append(H.make_node("MatMulNBits", [x, B, S], [y], domain="com.microsoft",
                                        K=k, N=n, bits=bits, block_size=32, accuracy_level=acc))
            self.bytes += blob.nbytes + s.nbytes
        return y

    def op(self, tipo, ins):
        y = self._nome("t")
        self.nos.append(self.H.make_node(tipo, ins, [y]))
        return y

    def sessao(self, pasta, threads):
        import gc
        import onnx
        import onnxruntime as ort
        H, T = self.H, self.T
        g = H.make_graph(self.nos, "decode", [H.make_tensor_value_info("x", T.FLOAT, [None, D_MODELO])],
                         [H.make_tensor_value_info(s, T.FLOAT, None) for s in self.saidas], self.inits)
        self.inits, self.nos = [], []
        m = H.make_model(g, opset_imports=[H.make_opsetid("", 17), H.make_opsetid("com.microsoft", 1)])
        m.ir_version = 9
        del g
        arq = os.path.join(pasta, f"g{self.i}.onnx")
        onnx.save_model(m, arq, save_as_external_data=True, all_tensors_to_one_file=True,
                        location=os.path.basename(arq) + ".dados")
        del m  # libera a cópia dos pesos antes de a sessão carregar a sua
        gc.collect()
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        return ort.InferenceSession(arq, so, providers=["CPUExecutionProvider"])


def _grafo_decode(fmt_c, fmt_lm, partes):
    G = _Grafo()
    h = "x"
    if "camadas" in partes:
        for _ in range(CAMADAS_MEDIDAS):
            q = G.linear(h, D_MODELO, D_MODELO, fmt_c)
            G.saidas += [G.linear(h, D_KV, D_MODELO, fmt_c), G.linear(h, D_KV, D_MODELO, fmt_c)]  # K e V iriam para o cache
            h = G.op("Add", [h, G.linear(q, D_MODELO, D_MODELO, fmt_c)])
            mlp = G.op("Mul", [G.linear(h, D_MLP, D_MODELO, fmt_c), G.linear(h, D_MLP, D_MODELO, fmt_c)])
            h = G.op("Add", [h, G.linear(mlp, D_MODELO, D_MLP, fmt_c)])
    if "lm_head" in partes:
        G.saidas.append(G.linear(h, VOCAB, D_MODELO, fmt_lm))
    else:
        G.saidas.append(h)
    return G


def memoria_livre_gb():
    """RAM livre em GB (Windows, Linux, macOS). None se não der para saber."""
    try:
        if sys.platform == "win32":
            import ctypes

            class M(ctypes.Structure):
                _fields_ = [("l", ctypes.c_ulong), ("c", ctypes.c_ulong), ("tot", ctypes.c_ulonglong),
                            ("livre", ctypes.c_ulonglong), ("a", ctypes.c_ulonglong), ("b", ctypes.c_ulonglong),
                            ("v", ctypes.c_ulonglong), ("vl", ctypes.c_ulonglong), ("e", ctypes.c_ulonglong)]
            m = M()
            m.l = ctypes.sizeof(M)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.livre / 2 ** 30
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2 ** 30
    except Exception:
        return None


def decode_ort(lotes, reps, threads, nomes):
    try:
        import onnx  # noqa: F401
        import onnxruntime as ort
    except ImportError:
        sys.exit("O modo --decode precisa de onnx e onnxruntime: pip install onnx onnxruntime")
    import tempfile
    livre = memoria_livre_gb()
    if livre is not None and livre < 2.5:
        sys.exit(f"Só {livre:.1f} GB de RAM livre; este modo precisa de ~2,5 GB. Feche programas e tente de novo.")
    print(f"CPU: {platform.processor() or platform.machine()} · onnxruntime {ort.__version__} · {threads or 'todas as'} threads")
    if torch is not None:
        print(f"Banda de leitura medida (somar 256 MB): {banda_leitura(256):.1f} GB/s")
    fator = CAMADAS / CAMADAS_MEDIDAS
    print(f"Um passo = 24 camadas x 7 matrizes ({sum(n * k for n, k in FORMAS) * CAMADAS / 1e6:.0f} M pesos)"
          f" + lm_head ({VOCAB * D_MODELO / 1e6:.0f} M pesos).")
    print(f"Medidas {CAMADAS_MEDIDAS} camadas com pesos distintos (não cabem no cache) e o tempo delas x {fator:.0f};"
          " o lm_head é medido inteiro.")
    res = []
    with tempfile.TemporaryDirectory() as tmp:
        for nome in nomes:
            fmt_c, fmt_lm = CONFIGS[nome]
            r = {"config": nome}
            for partes, chave in ((("camadas",), "camadas"), (("lm_head",), "lm_head")):
                G = _grafo_decode(fmt_c, fmt_lm, partes)
                f_ = fator if chave == "camadas" else 1
                r[f"mb_{chave}"] = round(G.bytes * f_ / 1e6, 1)
                s = G.sessao(tmp, threads)
                for m_ in lotes:
                    x = np.random.default_rng(1).standard_normal((m_, D_MODELO), dtype=np.float32)
                    n_rep = reps if m_ == 1 else max(2, reps // 2)
                    t = _cronometra(lambda: s.run(None, {"x": x}), n_rep)
                    r[f"ms_{chave}_{m_}"] = round(t * 1e3 * f_, 2)
                del s, G
                for arq in glob.glob(os.path.join(tmp, "*")):
                    os.remove(arq)
            r["mb_total"] = round(r["mb_camadas"] + r["mb_lm_head"], 1)
            for m_ in lotes:
                r[f"ms_total_{m_}"] = round(r[f"ms_camadas_{m_}"] + r[f"ms_lm_head_{m_}"], 2)
            res.append(r)
            print(f"  medido: {nome}", flush=True)
    mostra_decode(res, lotes)
    return res


def mostra_decode(res, lotes):
    base = res[0]
    for m_ in lotes:
        linha("=")
        rotulo = "decode: 1 token por passo" if m_ == 1 else f"prefill: {m_} tokens por passo"
        print(f"Lote de {m_} ({rotulo})" + ("" if m_ == 1 else "  -> tok/s = tokens processados por segundo"))
        linha()
        print(f"{'config':<26}{'MB/passo':>9}{'bytes caíram':>14}{'ms/passo':>10}{'tok/s':>8}{'tok/s subiu':>13}{'aproveitado':>13}{'lm_head':>9}{'GB/s':>7}")
        linha()
        for r in res:
            mb, ms = r["mb_total"], r[f"ms_total_{m_}"]
            queda = base["mb_total"] / mb
            subida = base[f"ms_total_{m_}"] / ms
            aprov = (subida - 1) / (queda - 1) if queda > 1.001 else float("nan")
            lm_frac = r[f"ms_lm_head_{m_}"] / (r[f"ms_lm_head_{m_}"] + r[f"ms_camadas_{m_}"])
            print(f"{r['config']:<26}{mb:>9.0f}{queda:>13.2f}x{ms:>10.1f}{1000 * m_ / ms:>8.1f}{subida:>12.2f}x"
                  f"{('' if math.isnan(aprov) else f'{100 * aprov:.0f}%'):>13}{100 * lm_frac:>8.0f}%{mb / ms:>7.1f}")
        linha()
    print("'aproveitado' = quanto do ganho prometido pelos bytes virou velocidade: (tok/s subiu - 1) / (bytes caíram - 1).")
    print("'lm_head' = fração do tempo do passo gasta na última matriz (vocabulário). GB/s = bytes de pesos / tempo.")


# =====================================================================
# 3c. Dentro de um GGUF: em que formato cada tensor fica de verdade
# =====================================================================
# tipo ggml: (nome, pesos por bloco, bytes por bloco)
GGML = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22),
        7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
        12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 20: ("IQ4_NL", 32, 18), 23: ("IQ4_XS", 256, 136), 30: ("BF16", 1, 2)}


def le_gguf(arq):
    """Lê só o cabeçalho de um .gguf (não carrega pesos): lista de (nome, forma, tipo, bytes)."""
    import struct
    with open(arq, "rb") as f:
        def u(fmt):
            return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]

        def s():
            return f.read(u("Q")).decode("utf-8", "replace")
        tam = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

        def valor(t):
            if t == 8:
                return s()
            if t == 9:
                ti, n = u("I"), u("Q")
                return [valor(ti) for _ in range(n)] if ti in (8, 9) else f.read(tam[ti] * n) and None
            return f.read(tam[t])
        if f.read(4) != b"GGUF":
            raise ValueError(f"{arq} não é GGUF")
        u("I")
        n_t, n_kv = u("Q"), u("Q")
        for _ in range(n_kv):
            s()
            valor(u("I"))
        tensores = []
        for _ in range(n_t):
            nome = s()
            forma = [u("Q") for _ in range(u("I"))]
            tipo = u("I")
            u("Q")
            n = math.prod(forma)
            tn, bl, by = GGML.get(tipo, (f"tipo{tipo}", 1, 0))
            tensores.append((nome, forma, tn, n * by // bl))
    return tensores


def gguf_info(pasta):
    arqs = sorted(glob.glob(os.path.join(pasta, "*.gguf")), key=os.path.getsize, reverse=True)
    res = []
    base = None
    print(f"{'arquivo':<34}{'MB':>7}{'bits/peso':>10}{'camadas':>22}{'embedding':>12}{'saída (lm_head)':>18}{'emb+saída':>11}")
    linha("-", 114)
    for arq in arqs:
        t = le_gguf(arq)
        total = sum(b for *_, b in t)
        n = sum(math.prod(f) for _, f, _, _ in t)
        camadas = {}
        for nome, forma, tn, b in t:
            if nome.startswith("blk.") and len(forma) == 2:
                camadas[tn] = camadas.get(tn, 0) + b
        emb = [(tn, b) for nome, _, tn, b in t if nome == "token_embd.weight"]
        out = [(tn, b) for nome, _, tn, b in t if nome == "output.weight"]
        c_txt = "+".join(sorted(camadas, key=camadas.get, reverse=True)[:2])
        e_txt = f"{emb[0][0]}" if emb else "-"
        o_txt = f"{out[0][0]}" if out else "(= embedding)"
        eb = sum(b for _, b in emb + out)
        base = base or total
        print(f"{os.path.basename(arq):<34}{total / 1e6:>7.0f}{total * 8 / n:>10.2f}{c_txt:>22}{e_txt:>12}{o_txt:>18}{100 * eb / total:>10.0f}%")
        res.append({"arquivo": os.path.basename(arq), "bytes": total, "params": n, "camadas": camadas,
                    "embedding": emb, "saida": out})
    linha("-", 114)
    print("No decode, cada token lê as camadas inteiras e a matriz de saída inteira. Do embedding, só 1 linha.")
    return res


# =====================================================================
# 3d. Ponta a ponta: num modelo de verdade, quanto do tempo do token é multiplicação por pesos?
# =====================================================================
def ponta_a_ponta(pasta, contextos, n_gen):
    """Roda o Qwen2.5-0.5B de verdade (transformers, FP32) e separa o tempo de cada token em duas partes:
    as multiplicações por pesos (nn.Linear, incluindo o lm_head), que a quantização encolhe,
    e todo o resto (atenção sobre o KV cache, normalizações, RoPE, escolha do token, Python), que ela não toca."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        sys.exit("O modo --ponta-a-ponta precisa de transformers: pip install transformers")
    if not pasta:
        sys.exit("Modelo não encontrado. Baixe com: huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct  (ou passe --modelo)")
    livre = memoria_livre_gb()
    if livre is not None and livre < 3.5:
        sys.exit(f"Só {livre:.1f} GB de RAM livre; este modo precisa de ~3,5 GB (o modelo em FP32 tem 2 GB). Feche programas e tente de novo.")
    tok = AutoTokenizer.from_pretrained(pasta)
    m = AutoModelForCausalLM.from_pretrained(pasta, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    m.eval()
    gasto = [0.0]

    def envolve(mod):
        orig = mod.forward

        def f(x):
            t = time.perf_counter()
            y = orig(x)
            gasto[0] += time.perf_counter() - t
            return y
        mod.forward = f
    for mod in m.modules():
        if isinstance(mod, torch.nn.Linear):
            envolve(mod)
    base = tok(TEXTO, return_tensors="pt").input_ids[0]
    print(f"Qwen2.5-0.5B em FP32 (transformers {__import__('transformers').__version__}, torch {torch.__version__}, {torch.get_num_threads()} threads)")
    print(f"{n_gen} tokens gerados um a um com KV cache, depois de um prompt de N tokens.")
    linha()
    print(f"{'prompt':>8}{'ms/token':>10}{'pesos (Linear)':>16}{'resto':>9}{'resto %':>9}")
    linha()
    res = []
    with torch.no_grad():
        for ctx in contextos:
            ids = base.repeat(ctx // len(base) + 1)[:ctx][None]
            out = m(ids, use_cache=True)
            past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
            for _ in range(3):  # aquecimento
                o = m(nxt, past_key_values=past, use_cache=True)
                past, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
            gasto[0] = 0.0
            t0 = time.perf_counter()
            for _ in range(n_gen):
                o = m(nxt, past_key_values=past, use_cache=True)
                past, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
            total = (time.perf_counter() - t0) / n_gen
            lin = gasto[0] / n_gen
            print(f"{ctx:>8}{total * 1e3:>10.1f}{lin * 1e3:>13.1f} ms{(total - lin) * 1e3:>6.1f} ms{100 * (1 - lin / total):>8.0f}%")
            res.append({"contexto": ctx, "ms_token": round(total * 1e3, 2), "ms_linear": round(lin * 1e3, 2)})
            del past, out, o
    linha()
    print("Só a parte 'pesos' encolhe com a quantização dos pesos. Se ela ficasse 6x mais rápida, o token ficaria:")
    for r in res:
        novo = r["ms_linear"] / 6 + (r["ms_token"] - r["ms_linear"])
        print(f"  prompt {r['contexto']:>5}: {r['ms_token']:.1f} -> {novo:.1f} ms  =  {r['ms_token'] / novo:.2f}x, não 6x")
    return res


# =====================================================================
# 4. A mesma pergunta no llama.cpp: llama-bench nos GGUF
# =====================================================================
def llama_bench(exe, pasta, threads, n_gen, reps):
    arquivos = sorted(glob.glob(os.path.join(pasta, "*.gguf")), key=os.path.getsize, reverse=True)
    if not arquivos:
        sys.exit(f"Nenhum .gguf em {pasta}")
    res = []
    for arq in arquivos:
        cmd = [exe, "-m", arq, "-p", "512", "-n", str(n_gen), "-r", str(reps), "-o", "json"]
        if threads:
            cmd += ["-t", str(threads)]
        print("rodando:", os.path.basename(arq), flush=True)
        saida = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        try:
            dados = json.loads(saida.stdout[saida.stdout.index("["):])
        except ValueError:
            print(saida.stderr[-800:])
            continue
        r = {"arquivo": os.path.basename(arq)}
        for d in dados:
            r["bytes"] = d["model_size"]
            r["params"] = d["model_n_params"]
            r["tipo"] = d.get("model_type", "")
            if d.get("n_gen", 0) > 0:
                r["tg"], r["tg_sd"] = d["avg_ts"], d["stddev_ts"]
            else:
                r["pp"], r["pp_sd"] = d["avg_ts"], d["stddev_ts"]
            r["build"] = d.get("build_commit", "")
            r["cpu"] = d.get("cpu_info", "")
            r["threads"] = d.get("n_threads")
        res.append(r)
    return res


def mostra_bench(res):
    if not res:
        return
    base = res[0]
    print(f"CPU: {base.get('cpu')} · {base.get('threads')} threads · llama.cpp {base.get('build')}")
    linha()
    print(f"{'arquivo':<34}{'bits/peso':>10}{'MB':>7}{'bytes caíram':>14}{'decode tok/s':>15}{'tok/s subiu':>13}{'aproveitado':>13}{'prefill tok/s':>15}")
    linha()
    for r in res:
        bpp = r["bytes"] * 8 / r["params"]
        queda = base["bytes"] / r["bytes"]
        subida = r["tg"] / base["tg"]
        aprov = (subida - 1) / (queda - 1) if queda > 1.001 else float("nan")
        print(f"{r['arquivo']:<34}{bpp:>10.2f}{r['bytes'] / 1e6:>7.0f}{queda:>13.2f}x{r['tg']:>10.1f} ±{r['tg_sd']:<4.1f}{subida:>12.2f}x"
              f"{('' if math.isnan(aprov) else f'{100 * aprov:.0f}%'):>13}{r.get('pp', float('nan')):>15.0f}")
    linha()
    print("'aproveitado' = quanto do ganho prometido pelos bytes virou tok/s: (subida - 1) / (queda - 1).")
    print("bits/peso = tamanho do arquivo / número de parâmetros: inclui escalas e as camadas que ficam maiores.")


# =====================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exemplo", action="store_true", help="a conta na mão com 8 números")
    ap.add_argument("--qualidade", action="store_true", help="perplexidade e respostas com cada formato")
    ap.add_argument("--decode", action="store_true", help="um passo de decode inteiro no ONNX Runtime, por formato")
    ap.add_argument("--configs", default="todas", help="--decode: configurações, separadas por vírgula")
    ap.add_argument("--ponta-a-ponta", action="store_true", help="o modelo de verdade: quanto do token é multiplicação por pesos")
    ap.add_argument("--contextos", default="64,1024,4096", help="--ponta-a-ponta: tamanhos de prompt")
    ap.add_argument("--velocidade", action="store_true", help="os kernels do PyTorch: bytes x tempo, por kernel")
    ap.add_argument("--llama-bench", help="caminho do executável llama-bench")
    ap.add_argument("--gguf", help="pasta com os .gguf: sozinho, mostra o que há dentro; com --llama-bench, mede")
    ap.add_argument("--modelo", help="pasta com config.json e model.safetensors")
    ap.add_argument("--formatos", default="FP16,FP8 E4M3,INT8 por canal,INT4 por tensor,INT4 grupo 32,FP4 por tensor,MXFP4 (bloco 32),NVFP4 (bloco 16)",
                    help="formatos do --qualidade, separados por vírgula")
    ap.add_argument("--tudo", action="store_true", help="--qualidade: quantiza também o embedding/lm_head")
    ap.add_argument("--lotes", default="1,64", help="--decode e --velocidade: tokens por passo")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--n-gen", type=int, default=128, help="--llama-bench: tokens gerados por medida")
    ap.add_argument("--json", help="salva as medidas neste arquivo")
    a = ap.parse_args()

    if a.threads and torch is not None:
        torch.set_num_threads(a.threads)
    saida = {"maquina": platform.node(), "cpu": platform.processor(), "python": platform.python_version(),
             "torch": getattr(torch, "__version__", None), "data": time.strftime("%Y-%m-%d %H:%M")}
    if a.exemplo:
        exemplo()
        return
    if a.gguf and not a.llama_bench:
        saida["gguf"] = gguf_info(a.gguf)
    elif a.llama_bench:
        if not a.gguf:
            sys.exit("Passe --gguf com a pasta dos arquivos .gguf")
        r = llama_bench(a.llama_bench, a.gguf, a.threads, a.n_gen, a.reps if a.reps != 7 else 3)
        mostra_bench(r)
        saida["llama_bench"] = r
    elif a.ponta_a_ponta:
        saida["ponta_a_ponta"] = ponta_a_ponta(acha_modelo(a.modelo), [int(x) for x in a.contextos.split(",")], a.n_gen)
    elif a.decode:
        nomes = list(CONFIGS) if a.configs == "todas" else [c.strip() for c in a.configs.split(",")]
        saida["decode"] = decode_ort([int(x) for x in a.lotes.split(",")], a.reps, a.threads, nomes)
    elif a.velocidade:
        saida["velocidade"] = velocidade([int(x) for x in a.lotes.split(",")], a.reps, KERNELS)
    elif a.qualidade:
        nomes = [n.strip() for n in a.formatos.split(",") if n.strip()]
        for n in nomes:
            if n not in POR_NOME:
                sys.exit(f"Formato desconhecido: {n}. Opções: {', '.join(POR_NOME)}")
        saida["qualidade"] = qualidade(acha_modelo(a.modelo), nomes, a.tudo)
    else:
        if torch is None:
            sys.exit("Este modo precisa de torch. Sem torch, rode --exemplo.")
        saida["formatos"] = formatos(acha_modelo(a.modelo))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(saida, f, ensure_ascii=False, indent=2)
        print("salvo em", a.json)


if __name__ == "__main__":
    main()

/* Contas do laboratório "Quantização a fundo". Sem DOM: roda no navegador (window.Quantizacao)
   e no Node (require), onde é testado por tests/quantizacao.test.js.

   Três partes:
     1. formatos de ponto flutuante bit a bit (FP32, FP16, BF16, FP8 E4M3/E5M2, FP4 E2M1)
     2. quantização com escala: por tensor, por grupo, INT ou FP4
     3. a conta do tok/s: bytes por token, kernel e o que não encolhe
   As medidas de MEDIDAS vêm de quantizacao_a_fundo.py (semana 8), rodado no notebook-bia. */
(function (raiz, fabrica) {
  if (typeof module === "object" && module.exports) module.exports = fabrica();
  else raiz.Quantizacao = fabrica();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // ---------------- 1. ponto flutuante ----------------
  // e = bits de expoente, m = bits de mantissa. inf: o formato tem infinito (senão, satura no máximo).
  var FLOATS = [
    { id: "FP32", nome: "FP32", e: 8, m: 23, bias: 127, inf: true, uso: "treino clássico, referência" },
    { id: "FP16", nome: "FP16", e: 5, m: 10, bias: 15, inf: true, uso: "pesos e ativações na inferência" },
    { id: "BF16", nome: "BF16", e: 8, m: 7, bias: 127, inf: true, uso: "treino moderno: mesma faixa do FP32" },
    { id: "E4M3", nome: "FP8 E4M3", e: 4, m: 3, bias: 7, inf: false, maxFixo: 448, uso: "pesos e ativações em FP8" },
    { id: "E5M2", nome: "FP8 E5M2", e: 5, m: 2, bias: 15, inf: true, uso: "gradientes: mais faixa, menos precisão" },
    { id: "E2M1", nome: "FP4 E2M1", e: 2, m: 1, bias: 1, inf: false, uso: "MXFP4 e NVFP4, com escala por bloco" },
  ];
  function float(id) { for (var i = 0; i < FLOATS.length; i++) if (FLOATS[i].id === id) return FLOATS[i]; throw new Error("formato desconhecido: " + id); }

  function bits(f) { return 1 + f.e + f.m; }
  function maximo(f) {
    if (f.maxFixo) return f.maxFixo;
    var eMax = (f.inf ? (1 << f.e) - 2 : (1 << f.e) - 1) - f.bias;
    return (2 - Math.pow(2, -f.m)) * Math.pow(2, eMax);
  }
  function menorNormal(f) { return Math.pow(2, 1 - f.bias); }
  function menorPositivo(f) { return Math.pow(2, 1 - f.bias - f.m); }

  // arredonda para o vizinho mais perto (empate: mantissa par). Acima do máximo: infinito ou satura.
  function arredonda(x, f) {
    if (x === 0 || !isFinite(x)) return x;
    var s = x < 0 ? -1 : 1, a = Math.abs(x), mx = maximo(f);
    var e = Math.max(Math.floor(Math.log2(a)), 1 - f.bias);
    var passo = Math.pow(2, e - f.m), k = a / passo, r = Math.round(k);
    if (Math.abs(k - Math.floor(k) - 0.5) < 1e-12 && r % 2 === 1) r -= 1;
    var v = r * passo;
    if (v > mx) {
      if (f.inf && a >= mx + Math.pow(2, Math.floor(Math.log2(mx)) - f.m) / 2) return s * Infinity;
      v = mx;
    }
    return s * v;
  }

  // os campos de bits de um valor já representável
  function campos(x, f) {
    var sinal = x < 0 || Object.is(x, -0) ? 1 : 0, a = Math.abs(x), ex, man;
    if (!isFinite(a)) { ex = (1 << f.e) - 1; man = 0; }
    else if (a === 0) { ex = 0; man = 0; }
    else {
      var e = Math.floor(Math.log2(a));
      if (e < 1 - f.bias) { ex = 0; man = Math.round(a / menorPositivo(f)); }
      else { ex = e + f.bias; man = Math.round((a / Math.pow(2, e) - 1) * Math.pow(2, f.m)); }
      if (man >= Math.pow(2, f.m)) { man = 0; ex += 1; }
    }
    var bin = function (v, n) { var t = ""; for (var i = n - 1; i >= 0; i--) t += Math.floor(v / Math.pow(2, i)) % 2; return t; };
    return { sinal: String(sinal), expoente: bin(ex, f.e), mantissa: f.m <= 23 ? bin(man, f.m) : "", subnormal: ex === 0 && a > 0 };
  }

  // todos os valores positivos de um formato pequeno (até 8 bits)
  function grade(f) {
    if (bits(f) > 8) throw new Error("grade só para formatos de até 8 bits");
    var vs = [0];
    for (var ex = 0; ex < (1 << f.e); ex++) for (var man = 0; man < (1 << f.m); man++) {
      var v = ex === 0 ? man * menorPositivo(f) : (1 + man / (1 << f.m)) * Math.pow(2, ex - f.bias);
      if (v > 0 && v <= maximo(f)) vs.push(v);
    }
    return vs.sort(function (a, b) { return a - b; }).filter(function (v, i, l) { return i === 0 || v !== l[i - 1]; });
  }

  // ---------------- 2. quantização com escala ----------------
  var FP4 = [0, 0.5, 1, 1.5, 2, 3, 4, 6];
  function fp4(x) {
    var a = Math.min(Math.abs(x), 6), melhor = 0;
    for (var i = 1; i < FP4.length; i++) if (Math.abs(FP4[i] - a) < Math.abs(FP4[melhor] - a)) melhor = i;
    return (x < 0 ? -1 : 1) * FP4[melhor];
  }

  /* v: lista de números. op = { tipo: "int" | "fp4", bits (int), grupo (0 = tensor inteiro) }
     Devolve os inteiros (ou códigos FP4), os valores de volta, as escalas e o erro. */
  function quantiza(v, op) {
    var g = op.grupo > 0 ? op.grupo : v.length, q = [], volta = [], escalas = [];
    var qmax = op.tipo === "fp4" ? 6 : Math.pow(2, op.bits - 1) - 1;
    for (var i = 0; i < v.length; i += g) {
      var bloco = v.slice(i, i + g), amax = 0;
      for (var j = 0; j < bloco.length; j++) amax = Math.max(amax, Math.abs(bloco[j]));
      var s = amax > 0 ? amax / qmax : 1;
      escalas.push(s);
      for (j = 0; j < bloco.length; j++) {
        var c = op.tipo === "fp4" ? fp4(bloco[j] / s) : Math.max(-qmax - 1, Math.min(qmax, Math.round(bloco[j] / s)));
        q.push(c); volta.push(c * s);
      }
    }
    var num = 0, den = 0, zerados = 0;
    for (i = 0; i < v.length; i++) {
      num += Math.pow(v[i] - volta[i], 2); den += v[i] * v[i];
      if (v[i] !== 0 && volta[i] === 0) zerados++;
    }
    var bitsCodigo = op.tipo === "fp4" ? 4 : op.bits;
    return {
      q: q, volta: volta, escalas: escalas, zerados: zerados,
      erroRel: den > 0 ? Math.sqrt(num / den) : 0,
      bitsPorPeso: bitsCodigo + 16 * escalas.length / v.length,  // escala em FP16
    };
  }

  // ---------------- 3. a conta do tok/s ----------------
  // bits por peso de cada formato de armazenamento, contando as escalas
  var ARMAZ = [
    { id: "FP32", nome: "FP32", bits: 32 },
    { id: "FP16", nome: "FP16 / BF16", bits: 16 },
    { id: "FP8", nome: "FP8", bits: 8 },
    { id: "Q8_0", nome: "INT8, bloco 32 (Q8_0)", bits: 8.5 },
    { id: "Q4_0", nome: "INT4, bloco 32 (Q4_0)", bits: 4.5 },
    { id: "NVFP4", nome: "NVFP4, bloco 16", bits: 4.5 },
    { id: "MXFP4", nome: "MXFP4, bloco 32", bits: 4.25 },
    { id: "INT4", nome: "INT4 sem escala (só teoria)", bits: 4 },
  ];
  function armaz(id) { for (var i = 0; i < ARMAZ.length; i++) if (ARMAZ[i].id === id) return ARMAZ[i]; throw new Error("desconhecido: " + id); }

  // Qwen2.5-0.5B: 358 M pesos nas camadas, 136 M na matriz de saída (o mesmo tensor do embedding)
  var QWEN = { camadas: 357.8e6, saida: 136.1e6, kvPorToken: 24 * 2 * 128 * 2 };  // KV em FP16: 12 KB por token

  /* c = { camadas, saida (ids de ARMAZ), gbs (banda efetiva do kernel, GB/s),
           fixoMs (custo por token que não depende dos pesos), ctx (tokens no contexto),
           ctxUs (µs a mais por token de contexto: atenção sobre o KV cache) }
     Devolve o tempo de um token separado em partes e os tok/s. */
  function token(c, m) {
    m = m || QWEN;
    var bCam = m.camadas * armaz(c.camadas).bits / 8, bSai = m.saida * armaz(c.saida).bits / 8;
    var bKV = m.kvPorToken * (c.ctx || 0);
    var msPesos = (bCam + bSai) / (c.gbs * 1e9) * 1e3;
    var msResto = (c.fixoMs || 0) + (c.ctx || 0) * (c.ctxUs || 0) / 1e3;
    var ms = msPesos + msResto;
    return { bytesCamadas: bCam, bytesSaida: bSai, bytesKV: bKV, bytesPesos: bCam + bSai,
             msPesos: msPesos, msResto: msResto, ms: ms, tokS: 1000 / ms, fracResto: msResto / ms };
  }

  /* compara com uma base: quanto os bytes caíram, quanto o tok/s subiu e quanto do ganho foi aproveitado */
  function compara(base, novo) {
    var queda = base.bytesPesos / novo.bytesPesos, subida = novo.tokS / base.tokS;
    return { queda: queda, subida: subida, aproveitado: queda > 1.0001 ? (subida - 1) / (queda - 1) : NaN };
  }

  /* a cascata: do que o nome promete ao tok/s de verdade */
  function cascata(base, novo, m) {
    m = m || QWEN;
    var b = token(base, m), n = token(novo, m), mesmoKernel = token(Object.assign({}, novo, { gbs: base.gbs }), m);
    var semResto = token(Object.assign({}, novo, { fixoMs: 0, ctx: 0 }), m), baseSemResto = token(Object.assign({}, base, { fixoMs: 0, ctx: 0 }), m);
    return {
      nome: armaz(base.camadas).bits / Math.max(1, Math.round(armaz(novo.camadas).bits)),  // "16 -> 4 bits"
      escalas: armaz(base.camadas).bits / armaz(novo.camadas).bits,
      porToken: b.bytesPesos / n.bytesPesos,
      kernel: semResto.tokS / baseSemResto.tokS,
      tokS: n.tokS / b.tokS,
      mesmoKernel: mesmoKernel.tokS / b.tokS,
    };
  }

  // ---------------- medidas (quantizacao_a_fundo.py, notebook-bia, 10/11/2026) ----------------
  var MEDIDAS = {
    maquina: "notebook-bia · Intel Core i5-1135G7 · 4 núcleos · só CPU",
    // 1. erro nos 358 M pesos das camadas do Qwen2.5-0.5B, e perplexidade do modelo com cada formato
    formatos: [
      { nome: "FP16", bits: 16, erro: 0.00, ppl: 17.98 },
      { nome: "FP8 E4M3", bits: 8, erro: 2.65, ppl: 17.94 },
      { nome: "FP8 E5M2", bits: 8, erro: 5.27, ppl: 19.42 },
      { nome: "INT8 por canal", bits: 8.01, erro: 1.00, ppl: 17.99 },
      { nome: "INT4 por tensor", bits: 4, erro: 75.76, ppl: 37222237 },
      { nome: "INT4 por canal", bits: 4.01, erro: 17.90, ppl: 44.94 },
      { nome: "INT4 grupo 32", bits: 4.5, erro: 10.25, ppl: 22.74 },
      { nome: "FP4 por tensor", bits: 4, erro: 52.42, ppl: 758563 },
      { nome: "MXFP4 (bloco 32)", bits: 4.25, erro: 11.71, ppl: 22.66 },
      { nome: "NVFP4 (bloco 16)", bits: 4.5, erro: 9.45, ppl: 19.76 },
    ],
    pplOriginal: 17.98,
    // 2. um passo de decode (24 camadas + lm_head) no ONNX Runtime 1.24.1, 4 threads
    decode: [
      { config: "FP32", mb: 1976, ms1: 71.1, ms64: 339.0 },
      { config: "FP16", mb: 988, ms1: 74.3, ms64: 341.6 },
      { config: "INT8", mb: 556, ms1: 18.1, ms64: 318.2 },
      { config: "INT4 + lm_head INT8", mb: 377, ms1: 13.6, ms64: 371.2 },
      { config: "INT4 tudo", mb: 309, ms1: 12.8, ms64: 320.6 },
      { config: "INT4 tudo, sem conta int", mb: 309, ms1: 28.9, ms64: 465.2 },
    ],
    // 3. o modelo de verdade (transformers 4.49, FP32): ms por token, e quanto disso é nn.Linear
    pontaAPonta: [
      { ctx: 64, ms: 107.4, linear: 76.6 },
      { ctx: 1024, ms: 137.5, linear: 78.9 },
      { ctx: 4096, ms: 206.4, linear: 76.4 },
    ],
    // 4. dentro dos GGUF oficiais do Qwen2.5-0.5B-Instruct: MB lidos por token (camadas + saída)
    gguf: [
      { arquivo: "F16", mb: 1260, bpp: 16.0, camadas: 716, saida: 272 },
      { arquivo: "Q8_0", mb: 670, bpp: 8.5, camadas: 380, saida: 145 },
      { arquivo: "Q4_K_M", mb: 485, bpp: 6.16, camadas: 247, saida: 145 },
      { arquivo: "Q4_0", mb: 423, bpp: 5.37, camadas: 201, saida: 145 },
      { arquivo: "Q2_K", mb: 409, bpp: 5.2, camadas: 188, saida: 145 },
    ],
    // ajuste do "resto" medido: ms fixos + µs por token de contexto
    restoFixoMs: 29.2, restoCtxUs: 24.6,
    // GB/s efetivos medidos por kernel (bytes de pesos / tempo das multiplicações)
    kernels: [
      { id: "bom", nome: "Kernel com conta inteira (MatMulNBits nível 4)", gbs: 25 },
      { id: "desempacota", nome: "Kernel que desempacota para FP32 (nível 0)", gbs: 10.7 },
      { id: "ruim", nome: "Kernel sem otimização (PyTorch int4 na CPU)", gbs: 0.13 },
    ],
  };

  return {
    FLOATS: FLOATS, float: float, bits: bits, maximo: maximo, menorNormal: menorNormal, menorPositivo: menorPositivo,
    arredonda: arredonda, campos: campos, grade: grade,
    FP4: FP4, fp4: fp4, quantiza: quantiza,
    ARMAZ: ARMAZ, armaz: armaz, QWEN: QWEN, token: token, compara: compara, cascata: cascata,
    MEDIDAS: MEDIDAS,
  };
});

/* Contas do laboratório de compressão. Sem DOM: roda no navegador (window.Calculo)
   e no Node (require), onde é testado por tests/calculo.test.js. */
(function (raiz, fabrica) {
  if (typeof module === "object" && module.exports) module.exports = fabrica();
  else raiz.Calculo = fabrica();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // Configurações públicas (config.json no Hugging Face). Todos com embedding compartilhada
  // entre entrada e saída (tie_word_embeddings). emb = vocab × hidden.
  var MODELOS = [
    { id: "smol135", nome: "SmolLM2-135M", total: 134.5e6, vocab: 49152, hidden: 576, camadas: 30, kvh: 3, hd: 64 },
    { id: "smol360", nome: "SmolLM2-360M", total: 361.8e6, vocab: 49152, hidden: 960, camadas: 32, kvh: 5, hd: 64 },
    { id: "qwen05", nome: "Qwen2.5-0.5B", total: 494.0e6, vocab: 151936, hidden: 896, camadas: 24, kvh: 2, hd: 64, oficina: true },
    { id: "llama1b", nome: "Llama-3.2-1B", total: 1235.8e6, vocab: 128256, hidden: 2048, camadas: 16, kvh: 8, hd: 64 },
    { id: "qwen15", nome: "Qwen2.5-1.5B", total: 1543.7e6, vocab: 151936, hidden: 1536, camadas: 28, kvh: 2, hd: 128 },
    { id: "qwen3b", nome: "Qwen2.5-3B", total: 3085.9e6, vocab: 151936, hidden: 2048, camadas: 36, kvh: 2, hd: 128 },
  ];

  // bits efetivos por peso, já contando as escalas de cada bloco (médias citadas pelo llama.cpp)
  var FORMATOS = [
    { id: "F16", bits: 16, nome: "FP16" },
    { id: "Q8_0", bits: 8.5, nome: "Q8_0 · 8 bits" },
    { id: "Q6_K", bits: 6.56, nome: "Q6_K · 6 bits" },
    { id: "Q5_K_M", bits: 5.69, nome: "Q5_K_M · 5 bits" },
    { id: "Q4_K_M", bits: 4.89, nome: "Q4_K_M · 4 bits" },
    { id: "Q3_K_M", bits: 3.91, nome: "Q3_K_M · 3 bits" },
    { id: "Q2_K", bits: 2.96, nome: "Q2_K · 2 bits" },
  ];
  var KV = { f16: { bits: 16, nome: "FP16" }, q8: { bits: 8.5, nome: "Q8_0" }, q4: { bits: 4.5, nome: "Q4_0" } };

  function modelo(id) { for (var i = 0; i < MODELOS.length; i++) if (MODELOS[i].id === id) return MODELOS[i]; throw new Error("modelo desconhecido: " + id); }
  function formato(id) { for (var i = 0; i < FORMATOS.length; i++) if (FORMATOS[i].id === id) return FORMATOS[i]; throw new Error("formato desconhecido: " + id); }

  function kvPorToken(m, kvBits) { return 2 * m.camadas * m.kvh * m.hd * (kvBits == null ? 16 : kvBits) / 8; }

  /* Memória para servir o modelo, em bytes.
     c = { modelo, formato, poda (0..1, estruturada, só nos blocos), kv ("f16"|"q8"|"q4"),
           ctx, pedidos, janela (tokens guardados; 0 = todos), buffersMB, embSeparada } */
  function memoria(c) {
    var m = typeof c.modelo === "string" ? modelo(c.modelo) : c.modelo;
    var f = formato(c.formato || "F16");
    var poda = c.poda || 0;
    var emb = m.vocab * m.hidden;
    var blocos = (m.total - emb) * (1 - poda);
    // Em GGUF quantizado abaixo de 8 bits, entrada e saída costumam ser guardadas
    // separadas e em ~8 bits: é por isso que o arquivo real é maior que a conta ingênua.
    var embSeparada = c.embSeparada !== false && f.bits < 8.5;
    var bytesEmb = embSeparada ? 2 * emb * 8.5 / 8 : emb * f.bits / 8;
    var pesos = blocos * f.bits / 8 + bytesEmb;
    var kvb = KV[c.kv || "f16"].bits;
    var porTok = kvPorToken(m, kvb);
    var tokens = c.janela ? Math.min(c.janela, c.ctx) : c.ctx;
    var kv = porTok * tokens * (c.pedidos || 1);
    var buffers = (c.buffersMB == null ? 60 : c.buffersMB) * 1e6;
    return { modelo: m, formato: f, params: blocos + emb, emb: emb, blocos: blocos, pesos: pesos, bytesEmb: bytesEmb,
             kvPorToken: porTok, kv: kv, buffers: buffers, total: pesos + kv + buffers };
  }

  /* Receita: aplica as famílias em ordem (quantizar → comprimir KV → podar → destilar) e
     devolve a cascata, passo a passo, do modelo em FP16 até o resultado.
     base = { modelo, ctx, pedidos, buffersMB }; passos = { formato, kv, poda, aluno } */
  var ORDEM = ["quant", "kv", "poda", "dest"];
  function receita(base, passos) {
    var c = { modelo: base.modelo, formato: "F16", kv: "f16", poda: 0, ctx: base.ctx, pedidos: base.pedidos, buffersMB: base.buffersMB };
    var inicio = memoria(c);
    var cascata = [{ id: "base", rotulo: "Original (FP16)", total: inicio.total, mem: inicio }];
    function aplicar(id, rotulo, muda) {
      var antes = memoria(c).total;
      muda(c);
      var mem = memoria(c);
      cascata.push({ id: id, rotulo: rotulo, total: mem.total, economia: antes - mem.total, mem: mem });
    }
    if (passos.formato && passos.formato !== "F16") aplicar("quant", "Quantização " + formato(passos.formato).id, function (c) { c.formato = passos.formato; });
    if (passos.kv && passos.kv !== "f16") aplicar("kv", "KV em " + KV[passos.kv].nome, function (c) { c.kv = passos.kv; });
    if (passos.poda) aplicar("poda", "Pruning " + Math.round(passos.poda * 100) + "%", function (c) { c.poda = passos.poda; });
    if (passos.aluno && passos.aluno !== (typeof base.modelo === "string" ? base.modelo : base.modelo.id))
      aplicar("dest", "Destilar → " + modelo(passos.aluno).nome, function (c) { c.modelo = passos.aluno; });
    return { inicio: inicio, fim: memoria(c), cascata: cascata, config: c };
  }

  /* O que cada passo custa para aplicar e o risco para a qualidade (regra de bolso, não medida). */
  function custo(passos) {
    var out = [];
    if (passos.formato && passos.formato !== "F16") {
      var b = formato(passos.formato).bits;
      out.push({ familia: "quant", treino: false,
        risco: b >= 8 ? "baixo" : b >= 5 ? "baixo" : b >= 4.5 ? "moderado" : "alto",
        texto: b >= 8 ? "Sem treino. Perda praticamente imperceptível." : b >= 5 ? "Sem treino. Perda pequena." :
               b >= 4.5 ? "Sem treino. Ponto de equilíbrio comum; medir qualidade." : "Sem treino, mas modelos pequenos degradam muito abaixo de 4 bits." });
    }
    if (passos.kv && passos.kv !== "f16") out.push({ familia: "kv", treino: false, risco: passos.kv === "q8" ? "baixo" : "moderado",
      texto: passos.kv === "q8" ? "Uma opção do servidor. Perda pequena." : "Uma opção do servidor. Pode perder detalhes em contexto longo." });
    if (passos.poda) out.push({ familia: "poda", treino: true, risco: passos.poda > 0.25 ? "alto" : "moderado",
      texto: "Precisa de re-treino curto (ou destilação) para recuperar a qualidade." });
    if (passos.aluno) out.push({ familia: "dest", treino: true, risco: "moderado",
      texto: "Precisa treinar o aluno: GPU, dados e dias. O modelo final é outro modelo." });
    return out;
  }

  // teto de decode: cada passo lê os pesos uma vez e o cache de todos os pedidos
  function tetoTokS(mem, bandaGBs, pedidos) {
    var bytes = mem.pesos + mem.kv;
    return { porPedido: bandaGBs * 1e9 / bytes, total: bandaGBs * 1e9 / bytes * (pedidos || 1), bytesPorPasso: bytes };
  }

  /* ---- auxiliares das visualizações ---- */
  // quantização uniforme assimétrica (min-max): o que GGUF faz por bloco
  function quantizar(xs, bits) {
    var lo = Math.min.apply(null, xs), hi = Math.max.apply(null, xs), niveis = Math.pow(2, bits);
    var escala = (hi - lo) / (niveis - 1) || 1;
    var q = xs.map(function (x) { return Math.round((x - lo) / escala); });
    var deq = q.map(function (k) { return lo + k * escala; });
    var erro = Math.sqrt(xs.reduce(function (s, x, i) { return s + Math.pow(x - deq[i], 2); }, 0) / xs.length);
    return { q: q, deq: deq, escala: escala, minimo: lo, niveis: niveis, erroRMS: erro };
  }
  function softmax(logits, T) {
    var e = logits.map(function (l) { return Math.exp(l / (T || 1)); }), s = e.reduce(function (a, b) { return a + b; }, 0);
    return e.map(function (x) { return x / s; });
  }
  function kl(p, q) { return p.reduce(function (s, pi, i) { return pi > 0 ? s + pi * Math.log(pi / Math.max(q[i], 1e-12)) : s; }, 0); }
  // gerador determinístico (as figuras ficam iguais em toda máquina)
  function gerador(semente) {
    var s = semente || 1;
    return function () { s = (s * 16807) % 2147483647; return (s - 1) / 2147483646; };
  }
  function gauss(rnd) { var u = Math.max(rnd(), 1e-9), v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

  return { MODELOS: MODELOS, FORMATOS: FORMATOS, KV: KV, ORDEM: ORDEM, modelo: modelo, formato: formato,
           kvPorToken: kvPorToken, memoria: memoria, receita: receita, custo: custo, tetoTokS: tetoTokS,
           quantizar: quantizar, softmax: softmax, kl: kl, gerador: gerador, gauss: gauss };
});

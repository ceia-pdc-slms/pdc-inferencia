/* Contas do laboratório de geração de texto. Sem DOM: roda no navegador (window.Geracao)
   e no Node (require), onde é testado por tests/geracao.test.js.

   O modelo de tempo é o roofline mais simples possível: cada fase leva o MAIOR entre
   o tempo de fazer as contas e o tempo de trazer os bytes da memória.
     contas  = 2 × parâmetros × tokens          (uma multiplicação e uma soma por peso)
     bytes   = pesos (uma vez por passada) + KV cache lido
   Serve para ordem de grandeza, não para prever o número exato de uma máquina. */
(function (raiz, fabrica) {
  if (typeof module === "object" && module.exports) module.exports = fabrica();
  else raiz.Geracao = fabrica();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // config.json públicos no Hugging Face. kvh = cabeças de K/V (GQA), hd = dimensão da cabeça.
  // amarrada = a mesma matriz serve de embedding de entrada e de saída (tie_word_embeddings).
  var MODELOS = [
    { id: "qwen05", nome: "Qwen2.5-0.5B", params: 494.0e6, vocab: 151936, hidden: 896, amarrada: true, camadas: 24, kvh: 2, hd: 64, oficina: true },
    { id: "llama1b", nome: "Llama-3.2-1B", params: 1235.8e6, vocab: 128256, hidden: 2048, amarrada: true, camadas: 16, kvh: 8, hd: 64 },
    { id: "qwen3b", nome: "Qwen2.5-3B", params: 3085.9e6, vocab: 151936, hidden: 2048, amarrada: true, camadas: 36, kvh: 2, hd: 128 },
    { id: "llama8b", nome: "Llama-3.1-8B", params: 8030.3e6, vocab: 128256, hidden: 4096, amarrada: false, camadas: 32, kvh: 8, hd: 128 },
  ];

  // bits efetivos por peso (com as escalas dos blocos, médias do llama.cpp)
  var FORMATOS = [
    { id: "F16", nome: "FP16", bits: 16 },
    { id: "Q8_0", nome: "Q8_0", bits: 8.5 },
    { id: "Q4_K_M", nome: "Q4_K_M", bits: 4.89 },
  ];

  /* Máquinas: banda e capacidade de cálculo EFETIVAS, não o pico do datasheet.
     O notebook foi calibrado com a medida da oficina de Docker (Qwen2.5-0.5B Q4_K_M no
     llama.cpp: 60,6 tok/s de decode). A banda efetiva sai bem abaixo do pico porque absorve
     o que esta conta simples não vê: embedding guardada em mais bits, overhead do runtime.
     As outras máquinas usam a mesma fração do pico (~35%): são estimativas. */
  var MAQUINAS = [
    { id: "rpi5", nome: "Raspberry Pi 5", banda: 7.5e9, flops: 0.08e12, pico: "LPDDR4X, 17 GB/s de pico" },
    { id: "notebook", nome: "Notebook (i5-1135G7)", banda: 22e9, flops: 0.3e12, pico: "DDR4, ~50 GB/s de pico", medida: true },
    { id: "jetson", nome: "Jetson Orin Nano", banda: 30e9, flops: 4e12, pico: "LPDDR5, 68 GB/s de pico" },
    { id: "rtx4090", nome: "GPU RTX 4090", banda: 440e9, flops: 80e12, pico: "GDDR6X, 1 008 GB/s de pico" },
  ];

  function acha(lista, id) { for (var i = 0; i < lista.length; i++) if (lista[i].id === id) return lista[i]; throw new Error("desconhecido: " + id); }
  function modelo(id) { return acha(MODELOS, id); }
  function formato(id) { return acha(FORMATOS, id); }
  function maquina(id) { return acha(MAQUINAS, id); }

  // Em GGUF quantizado abaixo de 8 bits, a embedding de entrada e a de saída ficam separadas e
  // em ~8 bits (mesma conta do laboratório de compressão). Por isso o arquivo é maior que a conta ingênua.
  function partes(m, f) {
    var emb = m.vocab * m.hidden;
    var blocos = m.params - emb * (m.amarrada ? 1 : 2);
    var bitsEmb = f.bits < 8.5 ? 8.5 : f.bits;
    return { emb: emb, blocos: blocos, bitsEmb: bitsEmb, copiasEmb: f.bits < 8.5 || !m.amarrada ? 2 : 1 };
  }
  // tamanho do arquivo de pesos
  function bytesPesos(m, f) { var p = partes(m, f); return p.blocos * f.bits / 8 + p.copiasEmb * p.emb * p.bitsEmb / 8; }
  // o que uma passada lê: todos os blocos + a cabeça de saída inteira. Da embedding de entrada
  // só sai a linha de cada token, que é desprezível.
  function bytesLidos(m, f) { var p = partes(m, f); return p.blocos * f.bits / 8 + p.emb * p.bitsEmb / 8; }
  // K e V, em todas as camadas e cabeças de KV, em FP16
  function kvPorToken(m, bytes) { return 2 * m.camadas * m.kvh * m.hd * (bytes == null ? 2 : bytes); }

  /* Tempo das duas fases.
     c = { modelo, formato, maquina, prompt (tokens), resposta (tokens) } */
  function tempos(c) {
    var m = modelo(c.modelo), f = formato(c.formato), q = maquina(c.maquina);
    var P = bytesLidos(m, f), kv = kvPorToken(m);
    var contasTok = 2 * m.params;

    // PREFILL: uma passada com os `prompt` tokens; os pesos são lidos uma vez
    var preContas = contasTok * c.prompt;
    var preBytes = P + kv * c.prompt;                       // escreve o KV do prompt
    var preT = Math.max(preContas / q.flops, preBytes / q.banda);

    // DECODE: `resposta` passadas de 1 token; cada uma lê os pesos e o KV acumulado
    var decT = 0, decBytes = 0;
    for (var i = 0; i < c.resposta; i++) {
      var b = P + kv * (c.prompt + i);
      decBytes += b;
      decT += Math.max(contasTok / q.flops, b / q.banda);
    }
    return {
      pesos: P, arquivo: bytesPesos(m, f), kvPorToken: kv,
      prefill: { segundos: preT, contas: preContas, bytes: preBytes, intensidade: preContas / preBytes,
                 limite: preContas / q.flops >= preBytes / q.banda ? "cálculo" : "memória",
                 tokPorSeg: c.prompt / preT },
      decode: { segundos: decT, contas: contasTok * c.resposta, bytes: decBytes, intensidade: contasTok * c.resposta / decBytes,
                limite: contasTok / q.flops >= P / q.banda ? "cálculo" : "memória",
                tokPorSeg: c.resposta / decT, porToken: decT / Math.max(1, c.resposta) },
      razao: (decT / Math.max(1, c.resposta)) / (preT / Math.max(1, c.prompt)),
      kvFinal: kv * (c.prompt + c.resposta),
    };
  }

  /* Quantas vezes um token passa pelo modelo para escrever `resposta` tokens depois de um
     prompt de `prompt` tokens: com cache, só o token novo; sem cache, o texto inteiro. */
  function passagens(prompt, resposta, comCache) {
    if (comCache) return prompt + resposta;                 // prefill + 1 por passo
    var t = prompt;                                         // o prefill
    for (var i = 1; i < resposta; i++) t += prompt + i;     // cada passo reprocessa tudo
    return t;
  }

  // ---------------- sampling ----------------
  var EXEMPLOS = [
    { id: "gato", frase: "O gato subiu no", logits: [["telhado", 4.0], ["muro", 3.1], ["sofá", 2.6], ["árvore", 2.3], ["carro", 1.4], ["armário", 1.1], ["computador", 0.3], ["foguete", -1.0]] },
    { id: "capital", frase: "A capital do Brasil é", logits: [["Brasília", 7.5], ["a", 3.2], ["uma", 2.4], ["Rio", 2.0], ["São", 1.6], ["Salvador", 0.9], ["linda", 0.5], ["Buenos", -0.5]] },
    { id: "historia", frase: "Era uma vez um", logits: [["menino", 2.6], ["rei", 2.5], ["dragão", 2.3], ["homem", 2.2], ["gato", 2.0], ["robô", 1.8], ["castelo", 1.2], ["planeta", 0.9]] },
  ];

  function softmax(logits, T) {
    if (T <= 0) {
      var top = 0;
      for (var i = 1; i < logits.length; i++) if (logits[i] > logits[top]) top = i;
      return logits.map(function (_, j) { return j === top ? 1 : 0; });
    }
    var mx = Math.max.apply(null, logits);
    var e = logits.map(function (v) { return Math.exp((v - mx) / T); });
    var s = e.reduce(function (a, b) { return a + b; }, 0);
    return e.map(function (v) { return v / s; });
  }

  /* Aplica top-k, top-p e min-p e renormaliza. Devolve as probabilidades na ordem original
     (zero para os cortados). k = 0 desliga top-k; p = 1 desliga top-p; minp = 0 desliga min-p. */
  function filtrar(probs, k, p, minp) {
    var idx = probs.map(function (_, i) { return i; }).sort(function (a, b) { return probs[b] - probs[a]; });
    var fica = {};
    var mx = probs[idx[0]], acum = 0;
    for (var r = 0; r < idx.length; r++) {
      var i = idx[r];
      if (k && r >= k) break;
      if (minp && probs[i] < minp * mx) break;
      fica[i] = true;
      acum += probs[i];
      if (p < 1 && acum >= p - 1e-12) break;
    }
    var s = 0;
    probs.forEach(function (v, i) { if (fica[i]) s += v; });
    return probs.map(function (v, i) { return fica[i] ? v / s : 0; });
  }

  function distribuicao(logits, cfg) {
    return filtrar(softmax(logits, cfg.temperatura), cfg.topK || 0, cfg.topP == null ? 1 : cfg.topP, cfg.minP || 0);
  }

  // gerador pseudoaleatório com semente (mulberry32): mesma seed, mesmos sorteios
  function rng(seed) {
    var a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      var t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function sortear(probs, r) {
    var x = r(), acum = 0, ultimo = 0;
    for (var i = 0; i < probs.length; i++) {
      if (probs[i] <= 0) continue;
      ultimo = i; acum += probs[i];
      if (x < acum) return i;
    }
    return ultimo;
  }

  return {
    MODELOS: MODELOS, FORMATOS: FORMATOS, MAQUINAS: MAQUINAS, EXEMPLOS: EXEMPLOS,
    modelo: modelo, formato: formato, maquina: maquina,
    bytesPesos: bytesPesos, bytesLidos: bytesLidos, kvPorToken: kvPorToken, tempos: tempos, passagens: passagens,
    softmax: softmax, filtrar: filtrar, distribuicao: distribuicao, rng: rng, sortear: sortear,
  };
});

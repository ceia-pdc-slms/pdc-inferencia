/* Contas do laboratório de KV cache e batching. Sem DOM: roda no navegador (window.Batching)
   e no Node (require), onde é testado por tests/batching.test.js.

   Reaproveita os modelos, formatos e máquinas do laboratório de geração de texto (semana 6):
   no navegador, carregue ../geracao/geracao.js antes deste arquivo.

   O modelo de tempo é o mesmo roofline simples: um passo leva o MAIOR entre
     contas / capacidade de cálculo      contas = 2 × parâmetros × tokens no passo
     bytes  / banda de memória           bytes  = pesos (uma vez por passo) + KV cache lido
   Serve para ordem de grandeza, não para prever o número exato de uma máquina. */
(function (raiz, fabrica) {
  if (typeof module === "object" && module.exports) module.exports = fabrica(require("../geracao/geracao.js"));
  else raiz.Batching = fabrica(raiz.Geracao);
})(typeof self !== "undefined" ? self : this, function (G) {
  "use strict";

  // memórias para a parte do KV cache. util = fração que o servidor pode usar (o resto é do sistema
  // e de buffers de ativação). É o mesmo papel do --gpu-memory-utilization do vLLM (padrão 0,9).
  var MEMORIAS = [
    { id: "jetson", nome: "Jetson Orin Nano · 8 GB", gb: 8, util: 0.7 },
    { id: "notebook", nome: "Notebook · 16 GB", gb: 16, util: 0.6 },
    { id: "rtx4090", nome: "RTX 4090 · 24 GB", gb: 24, util: 0.9 },
    { id: "h100", nome: "H100 · 80 GB", gb: 80, util: 0.9 },
  ];
  function memoria(id) { for (var i = 0; i < MEMORIAS.length; i++) if (MEMORIAS[i].id === id) return MEMORIAS[i]; throw new Error("desconhecida: " + id); }

  // ---------------- 1. batching: um passo de decode com B pedidos ----------------
  /* c = { modelo, formato, maquina, lote (B), contexto (tokens que cada pedido já tem) } */
  function passo(c) {
    var m = G.modelo(c.modelo), f = G.formato(c.formato), q = G.maquina(c.maquina);
    var B = Math.max(1, c.lote), ctx = c.contexto == null ? 0 : c.contexto;
    var pesos = G.bytesLidos(m, f), kv = G.kvPorToken(m) * ctx * B;
    var contas = 2 * m.params * B;
    var tCalc = contas / q.flops, tMem = (pesos + kv) / q.banda;
    var s = Math.max(tCalc, tMem);
    return {
      segundos: s, pesos: pesos, kv: kv, contas: contas,
      limite: tCalc >= tMem ? "cálculo" : "memória",
      tokPorSegTotal: B / s, tokPorSegPorPedido: 1 / s,
      fracPesos: pesos / (pesos + kv),
    };
  }

  /* Curva para vários tamanhos de lote, com o custo de cada token relativo ao lote de 1. */
  function curva(c, lotes) {
    var base = passo(Object.assign({}, c, { lote: 1 }));
    return lotes.map(function (B) {
      var p = passo(Object.assign({}, c, { lote: B }));
      return { lote: B, segundos: p.segundos, tokPorSegTotal: p.tokPorSegTotal, tokPorSegPorPedido: p.tokPorSegPorPedido,
               limite: p.limite, custoPorToken: (p.segundos / B) / base.segundos };
    });
  }

  /* 20 pedidos de `tokens` tokens: um depois do outro × todos juntos (só o decode). */
  function vinteVezesUm(c, pedidos, tokens) {
    var umSo = 0, juntos = 0;
    for (var i = 0; i < tokens; i++) {
      var ctx = (c.contexto || 0) + i;
      umSo += passo(Object.assign({}, c, { lote: 1, contexto: ctx })).segundos;
      juntos += passo(Object.assign({}, c, { lote: pedidos, contexto: ctx })).segundos;
    }
    return { umDeCada: umSo * pedidos, juntos: juntos, economia: (umSo * pedidos) / juntos };
  }

  /* Lote em que o passo deixa de ser limitado por memória: daí em diante, mais pedidos custam conta. */
  function loteDeVirada(c) {
    for (var B = 1; B <= 4096; B++) if (passo(Object.assign({}, c, { lote: B })).limite === "cálculo") return B;
    return Infinity;
  }

  // ---------------- 2. KV cache como memória: quantos pedidos cabem ----------------
  /* c = { modelo, formato, memoria, contexto (máximo por pedido), medio (tokens que um pedido usa
     de verdade), bloco (tokens por bloco), modo: "contigua" | "paginada" } */
  function cabem(c) {
    var m = G.modelo(c.modelo), f = G.formato(c.formato), mem = memoria(c.memoria);
    var bloco = c.bloco || 16;
    var total = mem.gb * 1e9 * mem.util;
    var pesos = G.bytesPesos(m, f);
    var livre = Math.max(0, total - pesos);
    var porToken = G.kvPorToken(m);
    var usado = Math.min(c.medio, c.contexto);
    var reservado = c.modo === "paginada" ? Math.ceil(usado / bloco) * bloco : c.contexto;
    var porPedido = reservado * porToken;
    var n = Math.floor(livre / porPedido);
    return {
      total: total, pesos: pesos, livre: livre, porToken: porToken, porPedido: porPedido,
      pedidos: n, aproveitamento: reservado ? usado / reservado : 1,
      cabeModelo: pesos <= total,
    };
  }

  // ---------------- 3. PagedAttention: um alocador de blocos ----------------
  /* Memória física = `nBlocos` blocos de `bloco` tokens. Cada pedido tem uma tabela de blocos:
     posição lógica i -> bloco físico tabela[i]. Os blocos não precisam ser vizinhos. */
  function novoPaginador(nBlocos, bloco) {
    var livres = [];
    for (var i = 0; i < nBlocos; i++) livres.push(i);
    return { bloco: bloco, nBlocos: nBlocos, livres: livres, dono: new Array(nBlocos).fill(null), pedidos: {}, ordem: [] };
  }
  // pega um bloco livre "espalhado" (determinístico): mostra que eles não precisam ser vizinhos
  function pegaBloco(P, id) {
    if (!P.livres.length) return null;
    var k = (P.ordem.length * 7 + P.livres.length * 3) % P.livres.length;
    var b = P.livres.splice(k, 1)[0];
    P.dono[b] = id;
    return b;
  }
  /* Acrescenta `n` tokens ao pedido `id` (cria se não existe). Devolve false se faltar memória. */
  function acrescentar(P, id, n) {
    var r = P.pedidos[id];
    if (!r) { r = P.pedidos[id] = { tokens: 0, tabela: [] }; P.ordem.push(id); }
    var precisa = Math.ceil((r.tokens + n) / P.bloco) - r.tabela.length;
    if (precisa > P.livres.length) return false;
    for (var i = 0; i < precisa; i++) r.tabela.push(pegaBloco(P, id));
    r.tokens += n;
    return true;
  }
  function liberar(P, id) {
    var r = P.pedidos[id];
    if (!r) return;
    r.tabela.forEach(function (b) { P.dono[b] = null; P.livres.push(b); });
    P.livres.sort(function (a, b) { return a - b; });
    delete P.pedidos[id];
    P.ordem = P.ordem.filter(function (x) { return x !== id; });
  }
  function resumoPaginador(P) {
    var usados = 0, tokens = 0;
    P.ordem.forEach(function (id) { usados += P.pedidos[id].tabela.length; tokens += P.pedidos[id].tokens; });
    var capacidade = usados * P.bloco;
    return { blocosUsados: usados, blocosLivres: P.livres.length, tokens: tokens,
             desperdicio: capacidade ? (capacidade - tokens) / capacidade : 0 };
  }

  // ---------------- 4. batching estático × contínuo ----------------
  /* pedidos = [{ id, tokens }], todos na fila no instante 0. `vagas` = quantos cabem no lote.
     Devolve a grade passo × vaga (id ou null), o passo em que cada pedido terminou e métricas. */
  function agenda(pedidos, vagas, modo) {
    var grade = [], fim = {}, inicio = {};
    if (modo === "estatico") {
      for (var g = 0; g < pedidos.length; g += vagas) {
        var grupo = pedidos.slice(g, g + vagas);
        var dur = Math.max.apply(null, grupo.map(function (p) { return p.tokens; }));
        var t0 = grade.length;
        for (var s = 0; s < dur; s++) {
          var linha = [];
          for (var v = 0; v < vagas; v++) {
            var p = grupo[v];
            linha.push(p && s < p.tokens ? p.id : (p ? "ocioso" : null));
          }
          grade.push(linha);
        }
        // a resposta só é entregue quando o lote inteiro termina
        grupo.forEach(function (p) { inicio[p.id] = t0; fim[p.id] = t0 + dur; });
      }
    } else {
      var fila = pedidos.slice(), ativos = new Array(vagas).fill(null), falta = {};
      while (fila.length || ativos.some(function (a) { return a !== null; })) {
        for (var v2 = 0; v2 < vagas; v2++) if (ativos[v2] === null && fila.length) {
          var n = fila.shift(); ativos[v2] = n.id; falta[n.id] = n.tokens; inicio[n.id] = grade.length;
        }
        grade.push(ativos.slice());
        for (var v3 = 0; v3 < vagas; v3++) if (ativos[v3] !== null) {
          if (--falta[ativos[v3]] === 0) { fim[ativos[v3]] = grade.length; ativos[v3] = null; }
        }
      }
    }
    var uteis = pedidos.reduce(function (a, p) { return a + p.tokens; }, 0);
    var soma = 0; pedidos.forEach(function (p) { soma += fim[p.id]; });
    return { grade: grade, passos: grade.length, ocupacao: uteis / (grade.length * vagas),
             fim: fim, inicio: inicio, terminoMedio: soma / pedidos.length };
  }

  // gerador com semente (o mesmo do laboratório de geração), para os pedidos de exemplo
  function pedidosExemplo(n, seed, min, max) {
    var r = G.rng(seed == null ? 7 : seed), lista = [];
    min = min || 2; max = max || 12;
    for (var i = 0; i < n; i++) lista.push({ id: String.fromCharCode(65 + (i % 26)) + (i >= 26 ? Math.floor(i / 26) : ""), tokens: min + Math.floor(r() * (max - min + 1)) });
    return lista;
  }

  // ---------------- 5. chunked prefill ----------------
  /* `ativos` pedidos estão escrevendo (decode). No passo 3, chega um prompt de `prompt` tokens.
     Sem chunked prefill, ele entra inteiro num passo só, e todo mundo espera esse passo.
     Com chunked prefill, ele é cortado em pedaços de até `orcamento - ativos` tokens por passo.
     c = { modelo, formato, maquina, ativos, contexto, prompt, orcamento, chunked, passos } */
  function chunked(c) {
    var m = G.modelo(c.modelo), f = G.formato(c.formato), q = G.maquina(c.maquina);
    var pesos = G.bytesLidos(m, f), kvTok = G.kvPorToken(m);
    var falta = c.prompt, chegou = 3, t = 0, passos = [], ttft = null;
    var nPassos = c.passos || 24;
    for (var s = 0; s < nPassos || falta > 0; s++) {
      var pref = 0;
      if (s >= chegou && falta > 0) pref = c.chunked ? Math.min(falta, Math.max(1, c.orcamento - c.ativos)) : falta;
      var toks = c.ativos + pref;
      var dur = Math.max(2 * m.params * toks / q.flops, (pesos + kvTok * c.ativos * c.contexto + kvTok * pref) / q.banda);
      t += dur;
      falta -= pref;
      if (pref && falta === 0) ttft = t;          // fim do prefill do prompt longo
      passos.push({ inicio: t - dur, segundos: dur, prefill: pref, decode: c.ativos });
      if (s > 400) break;
    }
    var maior = 0, normal = Infinity;
    passos.forEach(function (p) { if (p.segundos > maior) maior = p.segundos; if (!p.prefill && p.segundos < normal) normal = p.segundos; });
    var t0 = passos[chegou] ? passos[chegou].inicio : 0;
    return { passos: passos, maiorPasso: maior, passoNormal: normal, ttft: ttft === null ? null : ttft - t0 };
  }

  // ---------------- 6. prefix caching ----------------
  /* `pedidos` pedidos com o mesmo começo de `prefixo` tokens (prompt de sistema, documento)
     e `proprio` tokens diferentes cada um. Só blocos CHEIOS do prefixo podem ser compartilhados. */
  function prefixo(c) {
    var m = G.modelo(c.modelo), kvTok = G.kvPorToken(m), bloco = c.bloco || 16;
    var comp = Math.floor(c.prefixo / bloco) * bloco;          // tokens em blocos compartilháveis
    var semCache = c.pedidos * (c.prefixo + c.proprio);
    var comCache = comp + c.pedidos * (c.prefixo - comp + c.proprio);
    return {
      prefillSem: semCache, prefillCom: comCache, economiaPrefill: semCache / comCache,
      kvSem: semCache * kvTok, kvCom: comCache * kvTok, compartilhados: comp,
    };
  }

  return {
    MEMORIAS: MEMORIAS, memoria: memoria,
    passo: passo, curva: curva, vinteVezesUm: vinteVezesUm, loteDeVirada: loteDeVirada,
    cabem: cabem,
    novoPaginador: novoPaginador, acrescentar: acrescentar, liberar: liberar, resumoPaginador: resumoPaginador,
    agenda: agenda, pedidosExemplo: pedidosExemplo,
    chunked: chunked, prefixo: prefixo,
  };
});
